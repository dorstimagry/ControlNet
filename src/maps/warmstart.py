"""Warm-start logic for iterative map reconstruction."""

from __future__ import annotations

from typing import Optional

import torch


def warmstart_sample(
    prev_map: Optional[torch.Tensor],
    rho: float,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Initialize reconstruction from previous estimate.
    
    If previous map exists:
        X_init = rho * X_prev + (1-rho) * noise
    Otherwise:
        X_init = noise
    
    Args:
        prev_map: Previous map estimate (normalized), shape (B, C, H, W)
        rho: Mixing coefficient in [0, 1] (1 = full warmstart, 0 = pure noise)
        noise: Noise sample, shape (B, C, H, W)
    
    Returns:
        Initial sample for reconstruction
    """
    if prev_map is None or rho <= 0.0:
        return noise
    
    if rho >= 1.0:
        return prev_map
    
    # Mix previous map and noise
    return rho * prev_map + (1.0 - rho) * noise

