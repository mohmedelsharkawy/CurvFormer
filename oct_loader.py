import os
from pathlib import Path
import re
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom


class RetinaDataset(Dataset):
    """
    data: dict[str -> list[tuple(img_path, mask_path)]]
          e.g. {'patient_001': [(img1, mask1), ...], ...}

    Returns:
      img_t:  FloatTensor, shape (1, D, H, W) in [0,1]
      mask_t: LongTensor,  shape (D, H, W)   with integer labels
    """

    def __init__(
        self,
        data,
        width=256,
        height=256,
        desired_depth=32,
        allowed_mask_values=range(12),  # 0..11
        strict_mask_values=True,
        debug=False,
    ):
        self.data = data
        self.keys = list(self.data.keys())
        self.width = int(width)
        self.height = int(height)
        self.desired_depth = int(desired_depth)
        self.allowed_mask_values = set(int(v) for v in allowed_mask_values)
        self.strict_mask_values = bool(strict_mask_values)
        self.debug = debug

    def __len__(self):
        return len(self.keys)

    # ---------- helpers ----------

    @staticmethod
    def _assert_exists(p: str | Path):
        if not os.path.exists(str(p)):
            raise FileNotFoundError(f"Path does not exist: {p}")

    @staticmethod
    def _q_index_from_path(p: str | Path) -> int:
        """Extract K from qK.png or qK_seg.png. Returns large number if not found."""
        name = os.path.basename(str(p))
        m = re.search(r"q(\d+)", name)
        return int(m.group(1)) if m else 10**9

    def read_image_gray(self, path: str | Path) -> np.ndarray:
        self._assert_exists(path)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            raise RuntimeError(f"Failed reading image: {path}")
        return img.astype(np.float32)

    def read_mask_any(self, path: str | Path) -> np.ndarray:
        self._assert_exists(path)
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.size == 0:
            raise RuntimeError(f"Failed reading mask: {path}")

        # Reduce to single channel if needed
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask.astype(np.float32)

    @staticmethod
    def resize_img(img_2d: np.ndarray, width: int, height: int) -> np.ndarray:
        out = cv2.resize(img_2d, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
        if out is None or out.size == 0:
            raise RuntimeError("cv2.resize produced empty image (img).")
        return out.astype(np.float32)

    @staticmethod
    def resize_mask(mask_2d: np.ndarray, width: int, height: int) -> np.ndarray:
        out = cv2.resize(mask_2d, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        if out is None or out.size == 0:
            raise RuntimeError("cv2.resize produced empty image (mask).")
        return out.astype(np.float32)

    @staticmethod
    def _resample_depth_exact(vol_3d: np.ndarray, target_depth: int, is_mask: bool) -> np.ndarray:
        """Resample along depth axis to exactly target_depth (crop/pad if zoom is off by 1)."""
        if vol_3d.ndim != 3:
            raise ValueError(f"Expected (D,H,W), got {vol_3d.shape}")
        D, H, W = vol_3d.shape
        if D <= 0:
            raise ValueError("Volume has non-positive depth.")
        if D == target_depth:
            return vol_3d

        zoom_factor = target_depth / float(D)
        order = 0 if is_mask else 1
        res = zoom(vol_3d, (zoom_factor, 1.0, 1.0), order=order).astype(np.float32)

        # Enforce exact depth
        if res.shape[0] > target_depth:
            res = res[:target_depth, :, :]
        elif res.shape[0] < target_depth:
            pad = target_depth - res.shape[0]
            res = np.pad(res, ((0, pad), (0, 0), (0, 0)), mode="edge")

        return res.astype(np.float32)

    def _check_mask_values(self, mask_2d_or_3d: np.ndarray, pid: str, slice_idx: int | None = None, stage=""):
        u = np.unique(mask_2d_or_3d).astype(np.int64)
        invalid = set(u.tolist()) - self.allowed_mask_values
        if invalid:
            msg = f"Invalid mask values for key={pid}"
            if slice_idx is not None:
                msg += f", slice={slice_idx}"
            msg += f" {stage}: invalid={sorted(invalid)}; unique={sorted(set(u.tolist()))}"
            if self.strict_mask_values:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

    # ---------- main ----------

    def __getitem__(self, idx: int):
        pid = self.keys[idx]
        patient_list = self.data[pid]

        if not isinstance(patient_list, (list, tuple)) or len(patient_list) == 0:
            raise RuntimeError(f"No slices for key={pid} at idx={idx}")

        # IMPORTANT: sort by q-index to ensure correct depth ordering
        patient_list = sorted(patient_list, key=lambda pair: self._q_index_from_path(pair[0]))

        img_slices, mask_slices = [], []

        for s, pair in enumerate(patient_list):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise RuntimeError(f"Invalid (img, mask) pair at key={pid}, slice={s}: {pair}")

            img_path, mask_path = pair

            img = self.read_image_gray(img_path)
            mask = self.read_mask_any(mask_path)

            img_r = self.resize_img(img, self.width, self.height)
            mask_r = self.resize_mask(mask, self.width, self.height)

            if np.any(mask_r < 0):
                warnings.warn(f"Negative mask values at key={pid}, slice={s}; clipping to [0, inf).")
                mask_r = np.clip(mask_r, 0, None)

            # Check values BEFORE depth resample
            mask_r_int = np.rint(mask_r).astype(np.int64)
            self._check_mask_values(mask_r_int, pid=pid, slice_idx=s, stage="(per-slice)")

            img_slices.append(img_r)
            mask_slices.append(mask_r)

        img_v = np.stack(img_slices, axis=0).astype(np.float32)   # (D,H,W)
        mask_v = np.stack(mask_slices, axis=0).astype(np.float32) # (D,H,W)

        if self.debug:
            print(f"[DEBUG] key={pid} raw vol shapes -> img: {img_v.shape}, mask: {mask_v.shape}")

        # Depth resample to desired_depth
        img_v = self._resample_depth_exact(img_v, self.desired_depth, is_mask=False)
        mask_v = self._resample_depth_exact(mask_v, self.desired_depth, is_mask=True)

        if self.debug:
            print(f"[DEBUG] key={pid} after depth resample -> img: {img_v.shape}, mask: {mask_v.shape}")

        # Normalize image to [0,1] per-volume
        vmin, vmax = float(img_v.min()), float(img_v.max())
        if vmax > vmin:
            img_v = (img_v - vmin) / (vmax - vmin)
        else:
            warnings.warn(f"Constant image volume for key={pid}; leaving as zeros.")
            img_v = np.zeros_like(img_v, dtype=np.float32)

        # Integer labels + check AFTER depth resample
        mask_v = np.rint(mask_v).astype(np.int64)
        self._check_mask_values(mask_v, pid=pid, slice_idx=None, stage="(after depth resample)")

        img_t = torch.from_numpy(img_v[None, ...].astype(np.float32))  # (1,D,H,W)
        mask_t = torch.from_numpy(mask_v.astype(np.int64))             # (D,H,W)

        return img_t, mask_t
