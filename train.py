#!/usr/bin/env python3
import os
import yaml
import json
import pickle
import random
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
import time
import copy
from fvcore.nn import FlopCountAnalysis

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from scipy.ndimage import distance_transform_edt, binary_erosion

from curvFormer_model import create_layernode_curvatureformer

try:
    from oct_loader import RetinaDataset
    OCT_LOADER_AVAILABLE = True
except Exception:
    OCT_LOADER_AVAILABLE = False


DEFAULT_CONFIG = {
    "training": {
        "seed": 42,
        "num_classes": 12,
        "lr": 1e-4,
        "batch_size": 1,
        "epochs": 100,
        "weight_decay": 0.01,
        "model_size": "S",
        "img_size": [32, 256, 256],
        "spacing_dhw": [1.0, 1.0, 1.0],
        "print_every": 1,

        "w_seg": 0.5,
        "w_bnd": 0.5,

        "dice_weight": 0.5,
        "ce_weight": 0.5,

        "bce_pos_weight": None,
        "boundary_thickness": 1,

        "strict_mask_values": True,
        "auto_ce_weights": False,
    },
    "data": {
        "dataset_path": "data/oct.p",
    },
    "model": {
        "use_curvature": True,
        "use_learned_curvature": False,
        "dropout": 0.1,
    },
}


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    merged = copy.deepcopy(DEFAULT_CONFIG)
    for k in ["training", "data", "model"]:
        if k in cfg and cfg[k] is not None:
            merged[k].update(cfg[k])

    return merged


def setup_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


ABLATIONS = {
    "full": {
        "model": {},
        "training": {},
        "desc": "Full model (curvature + boundary-guided seg)",
    },
    "no_curvature": {
        "model": {"use_curvature": False},
        "training": {},
        "desc": "Disable curvature bottleneck",
    },
    "seg_only": {
        "model": {},
        "training": {"w_bnd": 0.0, "w_seg": 1.0},
        "desc": "Segmentation-only loss (no boundary supervision)",
    },
}


def apply_ablation_to_config(cfg: Dict, ablation_name: str) -> Dict:
    if ablation_name not in ABLATIONS:
        raise ValueError(f"Unknown ablation '{ablation_name}'. Options: {list(ABLATIONS.keys())}")
    out = copy.deepcopy(cfg)
    a = ABLATIONS[ablation_name]
    out["model"].update(a.get("model", {}))
    out["training"].update(a.get("training", {}))
    return out


def _ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def _parse_seeds(seeds_str: str) -> List[int]:
    parts = [p.strip() for p in seeds_str.replace(";", ",").split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def load_data_retina(dataset_path: str, batch_size: int, seed: int, img_size: Tuple[int, int, int], num_classes: int, strict_mask_values: bool):
    if not OCT_LOADER_AVAILABLE:
        raise RuntimeError("oct_loader.RetinaDataset is not available. Please put RetinaDataset in oct_loader.py")

    with open(dataset_path, "rb") as f:
        data_dict = pickle.load(f)

    keys = list(data_dict.keys())
    train_keys, temp = train_test_split(keys, test_size=0.2, random_state=seed, shuffle=True)
    val_keys, test_keys = train_test_split(temp, test_size=0.5, random_state=seed, shuffle=True)

    print(f"Split (seed={seed}) -> Train: {len(train_keys)} | Val: {len(val_keys)} | Test: {len(test_keys)}")

    train_data = {k: data_dict[k] for k in train_keys}
    val_data = {k: data_dict[k] for k in val_keys}
    test_data = {k: data_dict[k] for k in test_keys}

    D, H, W = int(img_size[0]), int(img_size[1]), int(img_size[2])

    train_ds = RetinaDataset(
        train_data,
        width=W,
        height=H,
        desired_depth=D,
        allowed_mask_values=range(int(num_classes)),
        strict_mask_values=bool(strict_mask_values),
        debug=False,
    )
    val_ds = RetinaDataset(
        val_data,
        width=W,
        height=H,
        desired_depth=D,
        allowed_mask_values=range(int(num_classes)),
        strict_mask_values=bool(strict_mask_values),
        debug=False,
    )
    test_ds = RetinaDataset(
        test_data,
        width=W,
        height=H,
        desired_depth=D,
        allowed_mask_values=range(int(num_classes)),
        strict_mask_values=bool(strict_mask_values),
        debug=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader, train_keys, val_keys, test_keys


def mask_to_boundary(mask: torch.Tensor, thickness: int = 1) -> torch.Tensor:
    b = torch.zeros_like(mask, dtype=torch.bool)

    diff = mask[:, 1:, :, :] != mask[:, :-1, :, :]
    b[:, 1:, :, :] |= diff
    b[:, :-1, :, :] |= diff

    diff = mask[:, :, 1:, :] != mask[:, :, :-1, :]
    b[:, :, 1:, :] |= diff
    b[:, :, :-1, :] |= diff

    diff = mask[:, :, :, 1:] != mask[:, :, :, :-1]
    b[:, :, :, 1:] |= diff
    b[:, :, :, :-1] |= diff

    b = b.unsqueeze(1).float()  # (B,1,D,H,W)

    if thickness is not None and thickness > 1:
        k = 2 * thickness + 1
        B, C, D, H, W = b.shape
        b2 = b.reshape(B * D, 1, H, W)
        b2 = F.max_pool2d(b2, kernel_size=k, stride=1, padding=k // 2)
        b = b2.reshape(B, 1, D, H, W)

    return (b > 0).float()


def unpack_batch(batch: Any, device: torch.device, boundary_thickness: int):
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        images, masks = batch
        images = images.to(device, non_blocking=True).float()   # (B,1,D,H,W)
        masks = masks.to(device, non_blocking=True).long()      # (B,D,H,W)
        boundary_gt = mask_to_boundary(masks, thickness=boundary_thickness).to(device)
        return images, masks, boundary_gt
    raise TypeError(f"Unexpected batch format: {type(batch)} len={len(batch) if hasattr(batch,'__len__') else 'NA'}")


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-5, exclude_bg: bool = True):
        super().__init__()
        self.num_classes = int(num_classes)
        self.smooth = float(smooth)
        self.exclude_bg = bool(exclude_bg)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(targets, self.num_classes).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        inter = (probs * onehot).sum(dim=dims)
        union = probs.sum(dim=dims) + onehot.sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        if self.exclude_bg and self.num_classes >= 2:
            dice = dice[1:]
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int, dice_weight: float, ce_weight: float, ce_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.dw = float(dice_weight)
        self.cw = float(ce_weight)
        self.dice = DiceLoss(num_classes=num_classes, exclude_bg=True)
        self.ce = nn.CrossEntropyLoss(weight=ce_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dw * self.dice(logits, targets) + self.cw * self.ce(logits, targets)


class SegBoundaryLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        w_seg: float,
        w_bnd: float,
        dice_weight: float,
        ce_weight: float,
        ce_weights: Optional[torch.Tensor],
        bce_pos_weight: Optional[float],
    ):
        super().__init__()
        self.w_seg = float(w_seg)
        self.w_bnd = float(w_bnd)
        self.seg = DiceCELoss(num_classes=num_classes, dice_weight=dice_weight, ce_weight=ce_weight, ce_weights=ce_weights)

        if bce_pos_weight is not None:
            self.register_buffer("_posw", torch.tensor(float(bce_pos_weight), dtype=torch.float32))
            self.bce = nn.BCEWithLogitsLoss(pos_weight=self._posw)
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, seg_logits, masks, boundary_logits, boundary_gt):
        ls = self.seg(seg_logits, masks)
        lb = self.bce(boundary_logits, boundary_gt)
        total = self.w_seg * ls + self.w_bnd * lb
        return total, {"seg": float(ls.detach().cpu().item()), "bnd": float(lb.detach().cpu().item())}


@torch.no_grad()
def compute_fg_iou_dice(preds: torch.Tensor, masks: torch.Tensor, num_classes: int):
    device = preds.device
    intersection = torch.zeros(num_classes, device=device, dtype=torch.float32)
    union = torch.zeros(num_classes, device=device, dtype=torch.float32)
    pred_sum = torch.zeros(num_classes, device=device, dtype=torch.float32)
    gt_sum = torch.zeros(num_classes, device=device, dtype=torch.float32)

    for c in range(num_classes):
        pc = (preds == c)
        gc = (masks == c)
        intersection[c] = (pc & gc).sum().float()
        union[c] = (pc | gc).sum().float()
        pred_sum[c] = pc.sum().float()
        gt_sum[c] = gc.sum().float()

    iou = intersection / (union + 1e-8)
    dice = (2 * intersection) / (pred_sum + gt_sum + 1e-8)

    if num_classes >= 2:
        fg = slice(1, num_classes)
        present = union[fg] > 0
        mIoU_fg = float(iou[fg][present].mean().item()) if present.any() else 0.0
        mDice_fg = float(dice[fg][present].mean().item()) if present.any() else 0.0
    else:
        mIoU_fg = float(iou.mean().item())
        mDice_fg = float(dice.mean().item())

    return iou, dice, mIoU_fg, mDice_fg


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    er = binary_erosion(mask)
    return mask ^ er


def hd95_binary(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    pred_s = _surface_voxels(pred)
    gt_s = _surface_voxels(gt)
    if pred_s.sum() == 0 or gt_s.sum() == 0:
        return float("inf")

    dt_gt = distance_transform_edt(~gt_s, sampling=spacing)
    dt_pr = distance_transform_edt(~pred_s, sampling=spacing)

    d_pr_to_gt = dt_gt[pred_s]
    d_gt_to_pr = dt_pr[gt_s]
    d = np.concatenate([d_pr_to_gt, d_gt_to_pr], axis=0)
    if d.size == 0:
        return 0.0
    return float(np.percentile(d, 95))


def mean_hd95(preds: torch.Tensor, gts: torch.Tensor, num_classes: int, spacing=(1.0, 1.0, 1.0)):
    preds_np = preds.detach().cpu().numpy()
    gts_np = gts.detach().cpu().numpy()

    per_class = [None] * num_classes
    for c in range(1, num_classes):
        vals = []
        for b in range(preds_np.shape[0]):
            p = (preds_np[b] == c)
            g = (gts_np[b] == c)
            if (p.sum() == 0) and (g.sum() == 0):
                continue
            vals.append(hd95_binary(p, g, spacing=spacing))
        if vals:
            per_class[c] = float(np.mean(vals))

    fg_vals = [v for v in per_class[1:] if v is not None and np.isfinite(v)]
    mean_fg = float(np.mean(fg_vals)) if fg_vals else float("inf")
    return mean_fg, per_class


def create_model(config: Dict, device: torch.device):
    tr = config["training"]
    md = config["model"]

    img_size = tuple(int(x) for x in tr.get("img_size", [32, 256, 256]))

    model = create_layernode_curvatureformer(
        num_classes=int(tr["num_classes"]),
        in_channels=1,
        model_size=str(tr.get("model_size", "S")),
        img_size=img_size,
        use_curvature=bool(md.get("use_curvature", True)),
        use_learned_curvature=bool(md.get("use_learned_curvature", False)),
        dropout=float(md.get("dropout", 0.1)),
    ).to(device)

    n_params = model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters())

    # ---- GFLOPs (fvcore) ----
    gflops = None
    try:
        model.eval()
        D, H, W = img_size
        dummy = torch.randn(1, 1, D, H, W, device=device)
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy).total()
        gflops = float(flops) / 1e9
    except Exception as e:
        print(f"[GFLOPs] failed: {type(e).__name__}: {e}", flush=True)

    if gflops is not None:
        print(
            f"Model params: {n_params/1e6:.2f}M | model_size={tr.get('model_size','S')} | "
            f"img_size={img_size} | use_curvature={md.get('use_curvature',True)} | GFLOPs={gflops:.3f}",
            flush=True
        )
    else:
        print(
            f"Model params: {n_params/1e6:.2f}M | model_size={tr.get('model_size','S')} | "
            f"img_size={img_size} | use_curvature={md.get('use_curvature',True)}",
            flush=True
        )

    return model, int(n_params)



@torch.no_grad()
def compute_ce_weights_from_loader(train_loader, num_classes: int, device: torch.device, boundary_thickness: int):
    counts = torch.zeros(num_classes, device=device)
    for batch in train_loader:
        images, masks, boundary_gt = unpack_batch(batch, device=device, boundary_thickness=boundary_thickness)
        m = masks.reshape(-1)
        counts += torch.bincount(m, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    freq = counts / counts.sum()
    w = 1.0 / (freq + 1e-12)
    w = w / w.mean()
    return w


def train_epoch(model, loader, criterion, device, num_classes, boundary_thickness: int):
    model.train()
    loss_sum = 0.0
    seg_sum = 0.0
    bnd_sum = 0.0
    acc_num = 0
    acc_den = 0
    miou_list = []
    mdice_list = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images, masks, boundary_gt = unpack_batch(batch, device, boundary_thickness)

        optimizer = None
        raise RuntimeError("train_epoch called without optimizer (bug).")


def train_epoch_with_opt(model, loader, criterion, optimizer, device, num_classes, boundary_thickness: int):
    model.train()
    loss_sum = 0.0
    seg_sum = 0.0
    bnd_sum = 0.0
    acc_num = 0
    acc_den = 0
    miou_list = []
    mdice_list = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images, masks, boundary_gt = unpack_batch(batch, device, boundary_thickness)

        optimizer.zero_grad(set_to_none=True)
        seg_logits, boundary_logits = model(images)

        loss, parts = criterion(seg_logits, masks, boundary_logits, boundary_gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += float(loss.detach().cpu().item())
        seg_sum += float(parts["seg"])
        bnd_sum += float(parts["bnd"])

        preds = seg_logits.argmax(dim=1)
        acc_num += (preds == masks).sum().item()
        acc_den += masks.numel()

        _, _, miou_fg, mdice_fg = compute_fg_iou_dice(preds, masks, num_classes)
        miou_list.append(miou_fg)
        mdice_list.append(mdice_fg)

        pbar.set_postfix(loss=f"{loss_sum/max(1,len(miou_list)):.4f}", miou=f"{miou_fg:.4f}", mdice=f"{mdice_fg:.4f}")

    n = max(1, len(loader))
    return {
        "loss": loss_sum / n,
        "seg_loss": seg_sum / n,
        "bnd_loss": bnd_sum / n,
        "accuracy": float(acc_num / max(1, acc_den)),
        "mIoU_fg": float(np.mean(miou_list)) if miou_list else 0.0,
        "mDice_fg": float(np.mean(mdice_list)) if mdice_list else 0.0,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, boundary_thickness: int, desc: str):
    model.eval()
    loss_sum = 0.0
    seg_sum = 0.0
    bnd_sum = 0.0
    acc_num = 0
    acc_den = 0
    iou_all = []
    dice_all = []
    miou_list = []
    mdice_list = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        images, masks, boundary_gt = unpack_batch(batch, device, boundary_thickness)

        seg_logits, boundary_logits = model(images)
        loss, parts = criterion(seg_logits, masks, boundary_logits, boundary_gt)

        loss_sum += float(loss.detach().cpu().item())
        seg_sum += float(parts["seg"])
        bnd_sum += float(parts["bnd"])

        preds = seg_logits.argmax(dim=1)
        acc_num += (preds == masks).sum().item()
        acc_den += masks.numel()

        iou, dice, miou_fg, mdice_fg = compute_fg_iou_dice(preds, masks, num_classes)
        iou_all.append(iou.detach().cpu().numpy())
        dice_all.append(dice.detach().cpu().numpy())
        miou_list.append(miou_fg)
        mdice_list.append(mdice_fg)

        pbar.set_postfix(loss=f"{loss.item():.4f}", miou=f"{miou_fg:.4f}", mdice=f"{mdice_fg:.4f}")

    n = max(1, len(loader))
    iou_pc = np.mean(np.stack(iou_all, axis=0), axis=0) if iou_all else np.zeros((num_classes,), dtype=float)
    dice_pc = np.mean(np.stack(dice_all, axis=0), axis=0) if dice_all else np.zeros((num_classes,), dtype=float)

    return {
        "loss": loss_sum / n,
        "seg_loss": seg_sum / n,
        "bnd_loss": bnd_sum / n,
        "accuracy": float(acc_num / max(1, acc_den)),
        "mIoU_fg": float(np.mean(miou_list)) if miou_list else 0.0,
        "mDice_fg": float(np.mean(mdice_list)) if mdice_list else 0.0,
        "iou_per_class": iou_pc,
        "dice_per_class": dice_pc,
    }


@torch.no_grad()
def test_with_hd95(model, loader, criterion, device, num_classes, boundary_thickness: int, spacing=(1.0, 1.0, 1.0)):
    model.eval()
    met = validate(model, loader, criterion, device, num_classes, boundary_thickness, desc="Test")
    hd95_values = []
    hd95_pc_all = []

    pbar = tqdm(loader, desc="HD95", leave=False)
    for batch in pbar:
        images, masks, boundary_gt = unpack_batch(batch, device, boundary_thickness)
        seg_logits, _ = model(images)
        preds = seg_logits.argmax(dim=1)
        hd95_fg, hd95_pc = mean_hd95(preds, masks, num_classes=num_classes, spacing=spacing)
        hd95_values.append(hd95_fg)
        hd95_pc_all.append(hd95_pc)

    hd95_mean = float(np.mean(hd95_values)) if hd95_values else float("inf")
    hd95_per_class = [None] * num_classes
    for c in range(1, num_classes):
        vals = [pc[c] for pc in hd95_pc_all if pc[c] is not None and np.isfinite(pc[c])]
        if vals:
            hd95_per_class[c] = float(np.mean(vals))

    met["HD95_fg"] = float(hd95_mean)
    met["hd95_per_class"] = hd95_per_class
    return met


def train_model(model, train_loader, val_loader, config, device, save_dir):
    trc = config["training"]
    epochs = int(trc["epochs"])
    lr = float(trc["lr"])
    wd = float(trc.get("weight_decay", 0.01))
    num_classes = int(trc["num_classes"])
    print_every = int(trc.get("print_every", 1))
    boundary_thickness = int(trc.get("boundary_thickness", 1))

    ce_weights = None
    if bool(trc.get("auto_ce_weights", False)):
        ce_weights = compute_ce_weights_from_loader(train_loader, num_classes=num_classes, device=device, boundary_thickness=boundary_thickness)
        print("CE weights (auto):", [float(x) for x in ce_weights.detach().cpu().tolist()])

    criterion = SegBoundaryLoss(
        num_classes=num_classes,
        w_seg=float(trc.get("w_seg", 0.5)),
        w_bnd=float(trc.get("w_bnd", 0.5)),
        dice_weight=float(trc.get("dice_weight", 0.5)),
        ce_weight=float(trc.get("ce_weight", 0.5)),
        ce_weights=ce_weights,
        bce_pos_weight=trc.get("bce_pos_weight", None),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train": {"loss": [], "seg": [], "bnd": [], "mIoU_fg": [], "mDice_fg": [], "acc": []},
        "val": {"loss": [], "seg": [], "bnd": [], "mIoU_fg": [], "mDice_fg": [], "acc": []},
        "best": {"epoch": -1, "val_mIoU_fg": -1.0},
    }

    best_val = -1.0
    best_epoch = -1
    best_state = None
    ckpt_path = os.path.join(save_dir, "best_model.pth")

    start = time.time()
    for ep in range(1, epochs + 1):
        tr = train_epoch_with_opt(model, train_loader, criterion, optimizer, device, num_classes, boundary_thickness)
        va = validate(model, val_loader, criterion, device, num_classes, boundary_thickness, desc="Val")

        history["train"]["loss"].append(float(tr["loss"]))
        history["train"]["seg"].append(float(tr["seg_loss"]))
        history["train"]["bnd"].append(float(tr["bnd_loss"]))
        history["train"]["mIoU_fg"].append(float(tr["mIoU_fg"]))
        history["train"]["mDice_fg"].append(float(tr["mDice_fg"]))
        history["train"]["acc"].append(float(tr["accuracy"]))

        history["val"]["loss"].append(float(va["loss"]))
        history["val"]["seg"].append(float(va["seg_loss"]))
        history["val"]["bnd"].append(float(va["bnd_loss"]))
        history["val"]["mIoU_fg"].append(float(va["mIoU_fg"]))
        history["val"]["mDice_fg"].append(float(va["mDice_fg"]))
        history["val"]["acc"].append(float(va["accuracy"]))

        lr_now = float(optimizer.param_groups[0]["lr"])
        if (ep % max(1, print_every) == 0) or (ep == 1) or (ep == epochs):
            print(
                f"Epoch {ep:03d}/{epochs:03d} | lr={lr_now:.6g} | "
                f"train: loss={tr['loss']:.4f} seg={tr['seg_loss']:.4f} bnd={tr['bnd_loss']:.4f} miou={tr['mIoU_fg']:.4f} mdice={tr['mDice_fg']:.4f} acc={tr['accuracy']:.4f} | "
                f"val: loss={va['loss']:.4f} seg={va['seg_loss']:.4f} bnd={va['bnd_loss']:.4f} miou={va['mIoU_fg']:.4f} mdice={va['mDice_fg']:.4f} acc={va['accuracy']:.4f}"
            )

        if float(va["mIoU_fg"]) > best_val:
            best_val = float(va["mIoU_fg"])
            best_epoch = int(ep)
            history["best"] = {"epoch": best_epoch, "val_mIoU_fg": best_val}
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)
            print(f"  ✓ New best at epoch {ep}: val mIoU_fg={best_val:.4f} -> saved best_model.pth")

        scheduler.step()

    train_time_sec = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        torch.save(model.state_dict(), ckpt_path)

    return history, best_val, best_epoch, train_time_sec, ckpt_path


def run_one_experiment(base_config: Dict[str, Any], ablation_name: str, seed: int, device: torch.device, out_root: str):
    cfg = apply_ablation_to_config(base_config, ablation_name)
    cfg["training"]["seed"] = int(seed)

    setup_seed(int(seed))

    img_size = tuple(int(x) for x in cfg["training"].get("img_size", [32, 256, 256]))
    num_classes = int(cfg["training"]["num_classes"])
    strict_mask_values = bool(cfg["training"].get("strict_mask_values", True))

    train_loader, val_loader, test_loader, train_keys, val_keys, test_keys = load_data_retina(
        dataset_path=cfg["data"]["dataset_path"],
        batch_size=int(cfg["training"]["batch_size"]),
        seed=int(seed),
        img_size=img_size,
        num_classes=num_classes,
        strict_mask_values=strict_mask_values,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _ensure_dir(os.path.join(out_root, f"{ablation_name}_seed{seed}_{ts}"))

    model, n_params = create_model(cfg, device)

    history, best_val, best_epoch, train_time_sec, ckpt_path = train_model(
        model, train_loader, val_loader, cfg, device, run_dir
    )

    boundary_thickness = int(cfg["training"].get("boundary_thickness", 1))
    spacing = tuple(float(x) for x in cfg["training"].get("spacing_dhw", [1.0, 1.0, 1.0]))

    ce_weights = None
    if bool(cfg["training"].get("auto_ce_weights", False)):
        ce_weights = compute_ce_weights_from_loader(train_loader, num_classes=num_classes, device=device, boundary_thickness=boundary_thickness)

    test_criterion = SegBoundaryLoss(
        num_classes=num_classes,
        w_seg=float(cfg["training"].get("w_seg", 0.5)),
        w_bnd=float(cfg["training"].get("w_bnd", 0.5)),
        dice_weight=float(cfg["training"].get("dice_weight", 0.5)),
        ce_weight=float(cfg["training"].get("ce_weight", 0.5)),
        ce_weights=ce_weights,
        bce_pos_weight=cfg["training"].get("bce_pos_weight", None),
    ).to(device)

    test_metrics = test_with_hd95(
        model, test_loader, test_criterion, device,
        num_classes=num_classes,
        boundary_thickness=boundary_thickness,
        spacing=spacing,
    )

    results = {
        "ablation": ablation_name,
        "ablation_desc": ABLATIONS[ablation_name]["desc"],
        "seed": int(seed),
        "num_params": int(n_params),
        "train_time_minutes": float(train_time_sec / 60.0),
        "best_epoch": int(best_epoch),
        "best_val_mIoU_fg": float(best_val),
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "mIoU_fg": float(test_metrics["mIoU_fg"]),
            "mDice_fg": float(test_metrics["mDice_fg"]),
            "HD95_fg": float(test_metrics["HD95_fg"]),
            "loss": float(test_metrics["loss"]),
            "seg_loss": float(test_metrics["seg_loss"]),
            "bnd_loss": float(test_metrics["bnd_loss"]),
            "dice_per_class": test_metrics["dice_per_class"].tolist(),
            "iou_per_class": test_metrics["iou_per_class"].tolist(),
            "hd95_per_class": test_metrics["hd95_per_class"],
        },
        "splits": {"train_keys": train_keys, "val_keys": val_keys, "test_keys": test_keys},
        "paths": {"run_dir": run_dir, "best_model": ckpt_path},
        "config": cfg,
    }

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\nTEST SUMMARY")
    print(f"mIoU_fg  : {results['test_metrics']['mIoU_fg']:.4f}")
    print(f"mDice_fg : {results['test_metrics']['mDice_fg']:.4f}")
    print(f"HD95_fg  : {results['test_metrics']['HD95_fg']:.4f}")
    print(f"Accuracy : {results['test_metrics']['accuracy']:.4f}")
    print(f"Saved: {run_dir}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_dir, results


def _nanify_inf(x: float) -> float:
    if x is None:
        return float("nan")
    x = float(x)
    if not np.isfinite(x):
        return float("nan")
    return x


def _mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.array([_nanify_inf(v) for v in values], dtype=float)
    return {
        "mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"),
        "std": float(np.nanstd(arr, ddof=1)) if np.isfinite(arr).sum() > 1 else 0.0,
    }


def aggregate_results_by_ablation(all_results: List[Dict], num_classes: int) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict]] = {}
    for r in all_results:
        grouped.setdefault(r["ablation"], []).append(r)

    summary: Dict[str, Any] = {}
    for ablation, runs in grouped.items():
        seeds = [int(r.get("seed", -1)) for r in runs]
        best_val = [float(r["best_val_mIoU_fg"]) for r in runs]
        test_miou = [float(r["test_metrics"]["mIoU_fg"]) for r in runs]
        test_mdice = [float(r["test_metrics"]["mDice_fg"]) for r in runs]
        test_hd95 = [float(r["test_metrics"]["HD95_fg"]) for r in runs]
        test_acc = [float(r["test_metrics"]["accuracy"]) for r in runs]
        train_time = [float(r.get("train_time_minutes", np.nan)) for r in runs]

        dice_pc = np.stack([np.asarray(r["test_metrics"]["dice_per_class"], dtype=np.float64) for r in runs], axis=0)
        iou_pc = np.stack([np.asarray(r["test_metrics"]["iou_per_class"], dtype=np.float64) for r in runs], axis=0)

        hd95_pc_rows = []
        for r in runs:
            hd95_list = r["test_metrics"]["hd95_per_class"]
            if isinstance(hd95_list, list):
                row = []
                for v in hd95_list:
                    row.append(_nanify_inf(v))
                hd95_pc_rows.append(np.asarray(row, dtype=np.float64))
            else:
                hd95_pc_rows.append(np.full((num_classes,), np.nan, dtype=np.float64))
        hd95_pc = np.stack(hd95_pc_rows, axis=0)

        dice_pc_stats = [{"mean": float(np.nanmean(dice_pc[:, c])), "std": float(np.nanstd(dice_pc[:, c], ddof=1) if len(runs) >= 2 else 0.0)} for c in range(num_classes)]
        iou_pc_stats = [{"mean": float(np.nanmean(iou_pc[:, c])), "std": float(np.nanstd(iou_pc[:, c], ddof=1) if len(runs) >= 2 else 0.0)} for c in range(num_classes)]
        hd95_pc_stats = [{"mean": float(np.nanmean(hd95_pc[:, c])), "std": float(np.nanstd(hd95_pc[:, c], ddof=1) if len(runs) >= 2 else 0.0)} for c in range(num_classes)]

        summary[ablation] = {
            "ablation_desc": ABLATIONS[ablation]["desc"],
            "seeds": seeds,
            "n_runs": len(runs),
            "best_val_mIoU_fg": _mean_std(best_val),
            "test_mIoU_fg": _mean_std(test_miou),
            "test_mDice_fg": _mean_std(test_mdice),
            "test_HD95_fg": _mean_std(test_hd95),
            "test_accuracy": _mean_std(test_acc),
            "train_time_minutes": _mean_std(train_time),
            "test_dice_per_class_mean_std": dice_pc_stats,
            "test_iou_per_class_mean_std": iou_pc_stats,
            "test_hd95_per_class_mean_std": hd95_pc_stats,
            "per_run": [
                {
                    "seed": int(r["seed"]),
                    "best_val_mIoU_fg": float(r["best_val_mIoU_fg"]),
                    "test_mIoU_fg": float(r["test_metrics"]["mIoU_fg"]),
                    "test_mDice_fg": float(r["test_metrics"]["mDice_fg"]),
                    "test_HD95_fg": float(r["test_metrics"]["HD95_fg"]),
                    "test_accuracy": float(r["test_metrics"]["accuracy"]),
                    "run_dir": r["paths"]["run_dir"],
                } for r in runs
            ],
        }

    return summary


def print_multiseed_table(agg: Dict[str, Any]):
    print("\n" + "=" * 90)
    print("MULTI-SEED SUMMARY (mean ± std)")
    print("=" * 90)
    for ablation, s in agg.items():
        m1, s1 = s["best_val_mIoU_fg"]["mean"], s["best_val_mIoU_fg"]["std"]
        m2, s2 = s["test_mIoU_fg"]["mean"], s["test_mIoU_fg"]["std"]
        m3, s3 = s["test_mDice_fg"]["mean"], s["test_mDice_fg"]["std"]
        m4, s4 = s["test_HD95_fg"]["mean"], s["test_HD95_fg"]["std"]
        seeds = ",".join(str(x) for x in s["seeds"])
        print(f"- {ablation} | seeds [{seeds}]")
        print(f"    best val mIoU_fg : {m1:.4f} ± {s1:.4f}")
        print(f"    test mIoU_fg     : {m2:.4f} ± {s2:.4f}")
        print(f"    test mDice_fg    : {m3:.4f} ± {s3:.4f}")
        print(f"    test HD95_fg     : {m4:.4f} ± {s4:.4f}")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/layernode_v4_retinadataset")

    parser.add_argument("--ablation", type=str, default="full", choices=list(ABLATIONS.keys()))
    parser.add_argument("--ablation_study", action="store_true")

    parser.add_argument("--seeds", type=str, default="0,42,512")
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--no_multi_seed", action="store_true")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    args = parser.parse_args()

    base_config = load_config(args.config)

    if args.epochs is not None:
        base_config["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        base_config["training"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        base_config["training"]["lr"] = float(args.lr)
    if args.weight_decay is not None:
        base_config["training"]["weight_decay"] = float(args.weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_multi_seed = bool(args.multi_seed) or (not args.no_multi_seed)
    seeds = _parse_seeds(args.seeds) if use_multi_seed else [int(base_config["training"].get("seed", 42))]

    ablations_to_run = list(ABLATIONS.keys()) if args.ablation_study else [args.ablation]

    print("=" * 90)
    print("LayerNODE-CurvatureFormer V4 | RetinaDataset | Train/Val/Test + Multi-seed")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Dataset: {base_config['data']['dataset_path']}")
    print(f"img_size: {tuple(base_config['training'].get('img_size', [32,256,256]))}")
    print(f"num_classes: {base_config['training'].get('num_classes', 12)}")
    print(f"Ablations: {ablations_to_run}")
    print(f"Multi-seed: {use_multi_seed} | seeds: {seeds}")
    print("=" * 90)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_root = _ensure_dir(os.path.join(args.output_dir, f"study_{timestamp}"))

    all_runs: List[str] = []
    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        print("\n" + "#" * 90)
        print(f"RUN SEED = {seed}")
        print("#" * 90)

        seed_root = _ensure_dir(os.path.join(study_root, f"seed_{int(seed):03d}"))

        for ablation in ablations_to_run:
            run_dir, res = run_one_experiment(base_config, ablation, int(seed), device, seed_root)
            all_runs.append(run_dir)
            all_results.append(res)

    study_summary = {
        "study_root": study_root,
        "runs": all_runs,
        "seeds": seeds,
        "ablations": ablations_to_run,
        "results": [
            {
                "ablation": r["ablation"],
                "seed": int(r["seed"]),
                "best_val_mIoU_fg": float(r["best_val_mIoU_fg"]),
                "test_mIoU_fg": float(r["test_metrics"]["mIoU_fg"]),
                "test_mDice_fg": float(r["test_metrics"]["mDice_fg"]),
                "test_HD95_fg": float(r["test_metrics"]["HD95_fg"]),
                "test_accuracy": float(r["test_metrics"]["accuracy"]),
                "num_params": int(r["num_params"]),
                "run_dir": r["paths"]["run_dir"],
                "best_model": r["paths"]["best_model"],
            } for r in all_results
        ],
        "config_used": base_config,
    }
    with open(os.path.join(study_root, "study_summary.json"), "w") as f:
        json.dump(study_summary, f, indent=2)

    agg = aggregate_results_by_ablation(all_results, num_classes=int(base_config["training"]["num_classes"]))
    with open(os.path.join(study_root, "multi_seed_summary_mean_std.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print_multiseed_table(agg)

    print("\n" + "=" * 90)
    print("DONE.")
    print(f"Study outputs: {study_root}")
    print("Saved: study_summary.json + multi_seed_summary_mean_std.json + each run's results.json/history.json")
    print("=" * 90)


if __name__ == "__main__":
    main()
