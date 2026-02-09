

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


# =============================================================================
# Utility Functions
# =============================================================================

def gaussian_kernel_2d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones(1, 1, 1, 1, device=device, dtype=dtype)

    radius = int(3 * sigma + 0.5)
    radius = max(radius, 1)

    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
    return kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1])


def sobel_kernels_2d(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        device=device, dtype=dtype
    ).view(1, 1, 3, 3) / 8.0

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        device=device, dtype=dtype
    ).view(1, 1, 3, 3) / 8.0

    return sobel_x, sobel_y


def _safe_gn_groups(num_channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return g


# =============================================================================
# Structure Tensor Curvature Descriptor
# =============================================================================

class StructureTensorDescriptor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        integration_sigma: float = 1.5,
        gradient_sigma: float = 0.8,
        axial_weight_init: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.integration_sigma = integration_sigma
        self.gradient_sigma = gradient_sigma

        self.scalar_proj = nn.Sequential(
            nn.Conv3d(in_channels, 16, 1),
            nn.GELU(),
            nn.Conv3d(16, 1, 1)
        )

        self.grad_h_weight = nn.Parameter(torch.tensor(1.0))
        self.grad_d_weight = nn.Parameter(torch.tensor(axial_weight_init))

        self.feature_net = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, padding=1),
            nn.GroupNorm(_safe_gn_groups(16, 4), 16),
            nn.GELU(),
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.GroupNorm(_safe_gn_groups(8, 2), 8),
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=1),
        )

        self.head_lambda = nn.Parameter(torch.zeros(num_heads))

        hidden = max(in_channels // 4, 8)
        self.channel_gate = nn.Sequential(
            nn.Linear(in_channels + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid()
        )
        self.gate_strength = nn.Parameter(torch.tensor(0.1))

    def _compute_structure_tensor_2d(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, H, W = f.shape
        device, dtype = f.device, f.dtype

        f_slices = f.view(B * D, 1, H, W)
        sobel_x, sobel_y = sobel_kernels_2d(device, dtype)

        if self.gradient_sigma > 0:
            g_kernel = gaussian_kernel_2d(self.gradient_sigma, device, dtype)
            pad_g = g_kernel.shape[-1] // 2
            f_slices = F.pad(f_slices, (pad_g, pad_g, pad_g, pad_g), mode="replicate")
            f_slices = F.conv2d(f_slices, g_kernel)

        f_pad = F.pad(f_slices, (1, 1, 1, 1), mode="replicate")
        fx = F.conv2d(f_pad, sobel_x)
        fy = F.conv2d(f_pad, sobel_y)

        Jxx = fx * fx
        Jyy = fy * fy
        Jxy = fx * fy

        if self.integration_sigma > 0:
            int_kernel = gaussian_kernel_2d(self.integration_sigma, device, dtype)
            pad_i = int_kernel.shape[-1] // 2
            Jxx = F.conv2d(F.pad(Jxx, (pad_i, pad_i, pad_i, pad_i), mode="replicate"), int_kernel)
            Jyy = F.conv2d(F.pad(Jyy, (pad_i, pad_i, pad_i, pad_i), mode="replicate"), int_kernel)
            Jxy = F.conv2d(F.pad(Jxy, (pad_i, pad_i, pad_i, pad_i), mode="replicate"), int_kernel)

        Sxx = Jxx.view(B, D, H, W)
        Syy = Jyy.view(B, D, H, W)
        Sxy = Jxy.view(B, D, H, W)
        return Sxx, Syy, Sxy

    def _eigenvalues_2x2(self, Sxx: torch.Tensor, Syy: torch.Tensor, Sxy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trace = Sxx + Syy
        det = Sxx * Syy - Sxy * Sxy
        disc = torch.clamp(trace * trace - 4.0 * det, min=0.0)
        sqrt_disc = torch.sqrt(disc + 1e-8)

        lambda1 = (trace + sqrt_disc) / 2.0
        lambda2 = (trace - sqrt_disc) / 2.0

        lambda1 = torch.clamp(lambda1, min=0.0)
        lambda2 = torch.clamp(lambda2, min=0.0)
        return lambda1, lambda2

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        B, C, D, H, W = x.shape
        N = D * H * W

        f = self.scalar_proj(x).squeeze(1)  # (B,D,H,W)

        Sxx, Syy, Sxy = self._compute_structure_tensor_2d(f)
        lambda1, lambda2 = self._eigenvalues_2x2(Sxx, Syy, Sxy)

        edge_strength = lambda1 - lambda2
        coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)

        f_pad_h = F.pad(f, (0, 0, 1, 1, 0, 0), mode="replicate")
        fh = (f_pad_h[:, :, 2:, :] - f_pad_h[:, :, :-2, :]) / 2.0  # (B,D,H,W)

        f_pad_d = F.pad(f, (0, 0, 0, 0, 1, 1), mode="replicate")
        fd = (f_pad_d[:, 2:, :, :] - f_pad_d[:, :-2, :, :]) / 2.0  # (B,D,H,W)

        grad_h = torch.abs(fh) * torch.abs(self.grad_h_weight)
        grad_d = torch.abs(fd) * torch.abs(self.grad_d_weight)

        features = torch.stack([edge_strength, coherence, grad_h, grad_d], dim=1)  # (B,4,D,H,W)
        features = features / (features.std(dim=(2, 3, 4), keepdim=True) + 1e-6)

        curvature = self.feature_net(features)  # (B,1,D,H,W)

        curv_flat = curvature.view(B, -1)
        curv_mean = curv_flat.mean(dim=1, keepdim=True)
        curv_std = curv_flat.std(dim=1, keepdim=True) + 1e-6
        curv_norm = (curv_flat - curv_mean) / curv_std  # (B,N)

        curv_tokens = curv_norm.unsqueeze(-1)  # (B,N,1)
        head_scales = self.head_lambda.view(1, 1, self.num_heads)  # (1,1,heads)
        bias = torch.tanh(curv_tokens * head_scales)  # (B,N,heads)

        curv_global = F.adaptive_avg_pool3d(curvature, 1).view(B, 1)
        x_global = F.adaptive_avg_pool3d(x, 1).view(B, C)
        gate_input = torch.cat([x_global, curv_global], dim=1)  # (B,C+1)
        gate = self.channel_gate(gate_input)  # (B,C)
        gate = 1.0 + self.gate_strength * (2.0 * gate - 1.0)

        if return_intermediates:
            inter = {
                "scalar_field": f.detach(),
                "lambda1": lambda1.detach(),
                "lambda2": lambda2.detach(),
                "edge_strength": edge_strength.detach(),
                "coherence": coherence.detach(),
                "grad_h": grad_h.detach(),
                "grad_d": grad_d.detach(),
                "curvature": curvature.detach(),
                "head_lambda": self.head_lambda.detach(),
                "grad_h_weight": self.grad_h_weight.detach(),
                "grad_d_weight": self.grad_d_weight.detach(),
            }
            return (bias, gate), inter

        return bias, gate


# =============================================================================
# Learned Curvature Descriptor (optional)
# =============================================================================

class LearnedCurvatureDescriptor(nn.Module):
    def __init__(self, in_channels: int, num_heads: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels

        self.edge_detector = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, hidden_dim, kernel_size=(1, k, k),
                          padding=(0, k // 2, k // 2), groups=min(in_channels, hidden_dim)),
                nn.Conv3d(hidden_dim, hidden_dim // 2, 1),
                nn.GroupNorm(_safe_gn_groups(hidden_dim // 2, 4), hidden_dim // 2),
                nn.GELU(),
            )
            for k in [3, 5, 7]
        ])

        self.axial_detector = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(hidden_dim, hidden_dim // 2, 1),
            nn.GroupNorm(_safe_gn_groups(hidden_dim // 2, 4), hidden_dim // 2),
            nn.GELU(),
        )

        total_features = (hidden_dim // 2) * 4
        self.fusion = nn.Sequential(
            nn.Conv3d(total_features, hidden_dim, 3, padding=1),
            nn.GroupNorm(_safe_gn_groups(hidden_dim, 8), hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, 1, 1),
        )

        self.head_proj = nn.Linear(1, num_heads)

        self.channel_gate = nn.Sequential(
            nn.Linear(in_channels + 1, in_channels),
            nn.Sigmoid()
        )
        self.gate_strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        B, C, D, H, W = x.shape

        lateral_features = [det(x) for det in self.edge_detector]
        axial_features = self.axial_detector(x)
        all_features = torch.cat(lateral_features + [axial_features], dim=1)
        curvature = self.fusion(all_features)  # (B,1,D,H,W)

        curv_flat = curvature.view(B, -1)
        curv_mean = curv_flat.mean(dim=1, keepdim=True)
        curv_std = curv_flat.std(dim=1, keepdim=True) + 1e-6
        curv_norm = (curv_flat - curv_mean) / curv_std

        bias = self.head_proj(curv_norm.unsqueeze(-1))  # (B,N,heads)
        bias = torch.tanh(bias)

        curv_global = F.adaptive_avg_pool3d(curvature, 1).view(B, 1)
        x_global = F.adaptive_avg_pool3d(x, 1).view(B, C)
        gate_input = torch.cat([x_global, curv_global], dim=1)
        gate = self.channel_gate(gate_input)
        gate = 1.0 + self.gate_strength * (2.0 * gate - 1.0)

        if return_intermediates:
            return (bias, gate), {"curvature": curvature.detach(), "bias_sample": bias[0].detach()}
        return bias, gate


# =============================================================================
# Axial Attention
# =============================================================================

class SimplifiedAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        bias_on_axes: Tuple[bool, bool, bool] = (True, True, False),
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.bias_on_axes = bias_on_axes

        self.qkv_d = nn.Linear(dim, dim * 3)
        self.qkv_h = nn.Linear(dim, dim * 3)
        self.qkv_w = nn.Linear(dim, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _attention_1d(self, x: torch.Tensor, qkv_layer: nn.Module, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        qkv = qkv_layer(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if bias is not None:
            attn = attn + bias.permute(0, 2, 1).unsqueeze(2)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        return out

    def forward(self, x: torch.Tensor, shape: Tuple[int, int, int], curvature_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        D, H, W = shape

        x = x.view(B, D, H, W, C)
        curv = curvature_bias.view(B, D, H, W, self.num_heads) if curvature_bias is not None else None

        x_d = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, D, C)
        bias_d = None
        if curv is not None and self.bias_on_axes[0]:
            bias_d = curv.permute(0, 2, 3, 1, 4).reshape(B * H * W, D, self.num_heads)
        x_d = self._attention_1d(x_d, self.qkv_d, bias_d)
        x = x_d.view(B, H, W, D, C).permute(0, 3, 1, 2, 4)

        x_h = x.permute(0, 1, 3, 2, 4).reshape(B * D * W, H, C)
        bias_h = None
        if curv is not None and self.bias_on_axes[1]:
            bias_h = curv.permute(0, 1, 3, 2, 4).reshape(B * D * W, H, self.num_heads)
        x_h = self._attention_1d(x_h, self.qkv_h, bias_h)
        x = x_h.view(B, D, W, H, C).permute(0, 1, 3, 2, 4)

        x_w = x.reshape(B * D * H, W, C)
        bias_w = None
        if curv is not None and self.bias_on_axes[2]:
            bias_w = curv.reshape(B * D * H, W, self.num_heads)
        x_w = self._attention_1d(x_w, self.qkv_w, bias_w)
        x = x_w.view(B, D, H, W, C)

        x = x.view(B, N, C)
        return self.proj(x)


class CurvatureTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimplifiedAxialAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, shape: Tuple[int, int, int], curvature_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), shape, curvature_bias)
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# Bottleneck
# =============================================================================

class CurvatureBottleneckV4(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_blocks: int = 2,
        max_spatial: Tuple[int, int, int] = (32, 64, 64),
        dropout: float = 0.1,
        use_learned_curvature: bool = False,
    ):
        super().__init__()
        self.dim = dim

        self.pos_d = nn.Parameter(torch.randn(1, max_spatial[0], 1, 1, dim) * 0.02)
        self.pos_h = nn.Parameter(torch.randn(1, 1, max_spatial[1], 1, dim) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(1, 1, 1, max_spatial[2], dim) * 0.02)

        self.curvature = LearnedCurvatureDescriptor(dim, num_heads) if use_learned_curvature else StructureTensorDescriptor(dim, num_heads)

        self.blocks = nn.ModuleList([
            CurvatureTransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        B, C, D, H, W = x.shape
        shape = (D, H, W)

        if return_intermediates:
            (curv_bias, ch_gate), curv_inter = self.curvature(x, return_intermediates=True)
        else:
            curv_bias, ch_gate = self.curvature(x, return_intermediates=False)
            curv_inter = None

        x = x * ch_gate.view(B, C, 1, 1, 1)

        x_seq = x.flatten(2).transpose(1, 2)  # (B,N,C)

        pos = self.pos_d[:, :D] + self.pos_h[:, :, :H] + self.pos_w[:, :, :, :W]
        pos = pos.reshape(1, D * H * W, C)
        x_seq = x_seq + pos

        for block in self.blocks:
            x_seq = block(x_seq, shape, curv_bias)

        x_out = x_seq.transpose(1, 2).view(B, C, D, H, W)

        if return_intermediates:
            return x_out, {"curvature": curv_inter, "channel_gate": ch_gate.detach()}
        return x_out


# =============================================================================
# Encoder / Decoder Blocks (OCT-aware: do NOT downsample depth)
# =============================================================================

class ConvBlock(nn.Module):
    # anisotropic by default: no depth mixing
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int, int] = (1, 3, 3),
        max_gn_groups: int = 8
    ):
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)  # (0,1,1) for (1,3,3)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False)
        g = _safe_gn_groups(out_ch, max_gn_groups)
        self.norm = nn.GroupNorm(g, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch)      # (1,3,3)
        self.conv2 = ConvBlock(out_ch, out_ch)     # (1,3,3)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv3d(out_ch, out_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        skip = self.conv2(x)
        x = self.downsample(skip) if self.downsample is not None else skip
        return x, skip


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1 = ConvBlock(out_ch + skip_ch, out_ch)  # (1,3,3)
        self.conv2 = ConvBlock(out_ch, out_ch)            # (1,3,3)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# =============================================================================
# Main Model + Deep Supervision
# =============================================================================

class LayerNODECurvatureFormerV4(nn.Module):
    CONFIGS = {
        "XS": {"channels": [16, 32, 64, 128], "heads": 4, "blocks": 1},
        "S":  {"channels": [24, 48, 96, 192], "heads": 4, "blocks": 2},
        "M":  {"channels": [32, 64, 128, 256], "heads": 8, "blocks": 3},
        "L":  {"channels": [48, 96, 192, 384], "heads": 8, "blocks": 4},
    }

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 12,
        model_size: str = "S",
        img_size: Tuple[int, int, int] = (11, 256, 256),
        use_curvature: bool = True,
        use_learned_curvature: bool = False,
        dropout: float = 0.1,
        deep_supervision: bool = True,
        ds_levels: int = 2,
    ):
        super().__init__()

        cfg = self.CONFIGS.get(model_size.upper(), self.CONFIGS["S"])
        channels = cfg["channels"]
        num_heads = cfg["heads"]
        num_blocks = cfg["blocks"]

        self.use_curvature = bool(use_curvature)
        self.deep_supervision = bool(deep_supervision)
        self.ds_levels = int(ds_levels)

        self.input_proj = nn.Sequential(
            ConvBlock(in_channels, channels[0]),
            ConvBlock(channels[0], channels[0]),
        )

        self.encoders = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoders.append(EncoderStage(channels[i], channels[i + 1], downsample=True))

        if self.use_curvature:
            n_stages = len(channels) - 1
            max_spatial = (
                img_size[0] + 4,
                img_size[1] // (2 ** n_stages) + 4,
                img_size[2] // (2 ** n_stages) + 4,
            )
            self.bottleneck = CurvatureBottleneckV4(
                dim=channels[-1],
                num_heads=num_heads,
                num_blocks=num_blocks,
                max_spatial=max_spatial,
                dropout=dropout,
                use_learned_curvature=use_learned_curvature,
            )
        else:
            self.bottleneck = nn.Sequential(
                ConvBlock(channels[-1], channels[-1]),
                ConvBlock(channels[-1], channels[-1]),
            )

        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_ch = channels[-(i + 1)]
            skip_ch = channels[-(i + 1)]
            out_ch = channels[-(i + 2)]
            self.decoders.append(DecoderStage(in_ch, skip_ch, out_ch))

        bnd_hidden = max(8, channels[0] // 2)
        self.boundary_head = nn.Sequential(
            ConvBlock(channels[0], bnd_hidden),
            ConvBlock(bnd_hidden, bnd_hidden),
            nn.Conv3d(bnd_hidden, 1, kernel_size=1)
        )

        self.seg_head = nn.Conv3d(channels[0] + 1, num_classes, kernel_size=1)

        self.aux_refine = nn.ModuleList()
        self.aux_seg_heads = nn.ModuleList()
        for i in range(len(self.encoders)):
            skip_ch = channels[i + 1]
            self.aux_refine.append(ConvBlock(skip_ch, skip_ch))
            self.aux_seg_heads.append(nn.Conv3d(skip_ch, num_classes, kernel_size=1))

        if self.ds_levels <= 0:
            self.ds_weights = []
        else:
            base = [0.2, 0.1, 0.05, 0.025]
            self.ds_weights = base[: max(1, min(self.ds_levels, len(base)))]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
        return_aux: bool = False,
    ):
        inter = {} if return_intermediates else None

        x0 = self.input_proj(x)
        full_shape = x0.shape[2:]  # (D,H,W)

        skips: List[torch.Tensor] = []
        x_enc = x0
        for enc in self.encoders:
            x_enc, skip = enc(x_enc)
            skips.append(skip)

        aux_seg_logits: List[torch.Tensor] = []
        if self.deep_supervision and (return_aux or return_intermediates):
            k = max(0, min(self.ds_levels, len(skips)))
            if k > 0:
                start = len(skips) - k
                for i in range(start, len(skips)):
                    s = skips[i]
                    s = self.aux_refine[i](s)
                    aux = self.aux_seg_heads[i](s)
                    if aux.shape[2:] != full_shape:
                        aux = F.interpolate(aux, size=full_shape, mode="trilinear", align_corners=False)
                    aux_seg_logits.append(aux)

        if isinstance(self.bottleneck, CurvatureBottleneckV4):
            if return_intermediates:
                x_enc, bottle_inter = self.bottleneck(x_enc, return_intermediates=True)
                inter["bottleneck"] = bottle_inter
            else:
                x_enc = self.bottleneck(x_enc)
        else:
            x_enc = self.bottleneck(x_enc)

        x_dec = x_enc
        for i, dec in enumerate(self.decoders):
            x_dec = dec(x_dec, skips[-(i + 1)])

        boundary_logits = self.boundary_head(x_dec)
        bnd_prob = torch.sigmoid(boundary_logits)
        seg_in = torch.cat([x_dec, bnd_prob], dim=1)
        seg_logits = self.seg_head(seg_in)

        if return_intermediates:
            inter["seg_logits"] = seg_logits.detach()
            inter["boundary_logits"] = boundary_logits.detach()
            if self.deep_supervision:
                inter["aux_seg_logits"] = [t.detach() for t in aux_seg_logits]
                inter["ds_weights"] = list(self.ds_weights)
            return (seg_logits, boundary_logits), inter

        if return_aux:
            return seg_logits, boundary_logits, aux_seg_logits

        return seg_logits, boundary_logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Factory
# =============================================================================

def create_model(
    num_classes: int = 12,
    in_channels: int = 1,
    model_size: str = "S",
    img_size: Tuple[int, int, int] = (11, 256, 256),
    use_curvature: bool = True,
    use_learned_curvature: bool = False,
    dropout: float = 0.1,
    deep_supervision: bool = True,
    ds_levels: int = 2,
    **kwargs
) -> LayerNODECurvatureFormerV4:
    return LayerNODECurvatureFormerV4(
        in_channels=in_channels,
        num_classes=num_classes,
        model_size=model_size,
        img_size=img_size,
        use_curvature=use_curvature,
        use_learned_curvature=use_learned_curvature,
        dropout=dropout,
        deep_supervision=deep_supervision,
        ds_levels=ds_levels,
    )


LayerNODECurvatureFormer = LayerNODECurvatureFormerV4
create_layernode_curvatureformer = create_model


# =============================================================================
# THOP profiling wrapper (main seg head only)
# =============================================================================

try:
    from thop import profile
except Exception:
    profile = None


class _THOPWrapper(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        seg_logits, boundary_logits = self.m(x)
        return seg_logits


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(
        num_classes=12,
        model_size="S",
        img_size=(11, 256, 256),
        use_curvature=True,
        deep_supervision=True,
        ds_levels=2,
    ).to(device)

    x = torch.randn(1, 1, 11, 256, 256, device=device)
    wrapped = _THOPWrapper(model).to(device).eval()

    with torch.no_grad():
        seg, bnd, aux = model(x, return_aux=True)
        if profile is not None:
            flops, params = profile(wrapped, inputs=(x,), verbose=False)
        else:
            flops, params = None, None

    print("seg:", tuple(seg.shape))
    print("bnd:", tuple(bnd.shape))
    print("aux heads:", len(aux), [tuple(a.shape) for a in aux])
    print("ds_weights:", model.ds_weights)
    print("params:", model.count_parameters())
    if flops is not None:
        print(f"FLOPs:   {flops/1e9:.3f} GFLOPs")
    else:
        print("FLOPs:   (thop not installed)")
