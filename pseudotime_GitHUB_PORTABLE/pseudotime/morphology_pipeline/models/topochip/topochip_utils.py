"""Utility helpers for the TopoChip generative model."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def lazy_conv_nd(dims: int, *args, **kwargs):
    if dims == 1:
        return nn.LazyConv1d(*args, **kwargs)
    if dims == 2:
        return nn.LazyConv2d(*args, **kwargs)
    if dims == 3:
        return nn.LazyConv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def resize(im: torch.Tensor, num_blocks: int) -> torch.Tensor:
    size = im.shape[-1]
    factor = 2 ** num_blocks
    desired = int(np.ceil(size / factor) * factor)
    if desired == size:
        return im
    return F.interpolate(im, size=desired, mode="bilinear", align_corners=False)


def undo_resize(reference: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    target = reference.shape[-1]
    if im.shape[-1] == target:
        return im
    return F.interpolate(im, size=target, mode="bilinear", align_corners=False)


def prepare_cfg_conditioning(cond: torch.Tensor, prob_no_guidance: float = 0.0) -> torch.Tensor:
    gate = torch.ones(cond.shape[0], 1, cond.shape[2], cond.shape[3], device=cond.device, dtype=cond.dtype)
    conditioned = torch.cat([cond, gate], dim=1)
    if prob_no_guidance > 0.0:
        mask = torch.rand(cond.shape[0], device=cond.device) < prob_no_guidance
        if mask.any():
            conditioned[mask] = 0
    return conditioned


def normalize_background_per_image(background: torch.Tensor) -> torch.Tensor:
    b_min = background.amin(dim=(-1, -2), keepdim=True)
    b_max = background.amax(dim=(-1, -2), keepdim=True)
    denom = (b_max - b_min).clamp_min(1e-6)
    scaled = (background - b_min) / denom
    return scaled * 2.0 - 1.0


def normalize_with_stats(background: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    low = low.to(background.device, background.dtype)
    high = high.to(background.device, background.dtype)
    scaled = (background - low) / (high - low).clamp_min(1e-6)
    scaled = torch.clamp(scaled, 0.0, 1.0)
    return scaled * 2.0 - 1.0


def sample_prior(shape: torch.Size, device: torch.device) -> torch.Tensor:
    return torch.randn(shape, device=device)


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
