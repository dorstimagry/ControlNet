"""Integration utilities for SysID with SAC training."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .encoder import ContextEncoder, FeatureBuilder
from .normalization import RunningNorm


def compute_z_online(
    encoder: ContextEncoder,
    feature_builder: FeatureBuilder,
    encoder_norm: RunningNorm,
    h_prev: torch.Tensor,
    v_t: float,
    u_t: float,
    device: torch.device,
    update_norm_stats: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute z_t online during environment interaction.
    
    Args:
        encoder: Context encoder
        feature_builder: Feature builder with history
        encoder_norm: Normalization for encoder features
        h_prev: Previous hidden state of shape (hidden_dim,)
        v_t: Current speed
        u_t: Current action
        device: Device for computation
        update_norm_stats: Whether to update normalization statistics (default False for evaluation)
    
    Returns:
        Tuple of (h_t, z_t):
            - h_t: New hidden state of shape (hidden_dim,)
            - z_t: Dynamics latent of shape (z_dim,)
    """
    encoder.eval()
    
    with torch.no_grad():
        # Build features
        o_t = feature_builder.build_features(v_t, u_t).to(device)
        
        # Normalize (use frozen stats during evaluation)
        o_t_norm = encoder_norm(o_t, update_stats=update_norm_stats)
        
        # Update encoder
        h_t, z_t = encoder.step(o_t_norm, h_prev)
    
    return h_t.squeeze(0), z_t.squeeze(0)


def batch_compute_z(
    encoder: ContextEncoder,
    encoder_norm: RunningNorm,
    v_seq: np.ndarray,
    u_seq: np.ndarray,
    dt: float,
    device: torch.device
) -> np.ndarray:
    """Compute z for a batch of sequences (for replay buffer initialization).
    
    Args:
        encoder: Context encoder
        encoder_norm: Normalization for encoder features
        v_seq: Speed sequences of shape (batch, seq_len)
        u_seq: Action sequences of shape (batch, seq_len)
        dt: Environment timestep
        device: Device for computation
    
    Returns:
        Latent sequence of shape (batch, seq_len, z_dim)
    """
    encoder.eval()
    feature_builder = FeatureBuilder(dt=dt)
    
    with torch.no_grad():
        # Build features
        v_torch = torch.from_numpy(v_seq).float().to(device)
        u_torch = torch.from_numpy(u_seq).float().to(device)
        
        features = feature_builder.build_features_batch(v_torch, u_torch)
        
        # Normalize
        batch_size, seq_len, feat_dim = features.shape
        features_flat = features.reshape(-1, feat_dim)
        features_norm = encoder_norm(features_flat, update_stats=False)
        features_norm = features_norm.reshape(batch_size, seq_len, feat_dim)
        
        # Encode
        _, z_seq, _ = encoder(features_norm)
    
    return z_seq.cpu().numpy()

