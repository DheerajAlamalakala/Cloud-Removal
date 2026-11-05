"""Fixed Sentinel-2 natural-color display transform shared by training previews."""
from __future__ import annotations

import torch


def display_rgb(rgb: torch.Tensor, exposure: float = 2.5) -> torch.Tensor:
    """Convert linear reflectance RGB (N,3,H,W) to display-referred sRGB in [0,1]."""
    linear = (rgb.float().clamp(0, 1) * exposure).clamp(0, 1)
    return torch.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * linear.clamp(min=0.0031308).pow(1 / 2.4) - 0.055,
    ).clamp(0, 1)
