"""Dataset utilities for sequence sampling from replay buffer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


@dataclass
class SequenceBatch:
    """Container for sequence data used in SysID training.
    
    Each sequence consists of:
        - Burn-in window: [t-B, t] for computing z_t
        - Rollout window: [t, t+H] for prediction
    """
    
    # Speeds (including burn-in and rollout)
    v_seq: torch.Tensor  # (batch, burn_in + horizon + 1)
    
    # Actions (including burn-in and rollout)
    u_seq: torch.Tensor  # (batch, burn_in + horizon)
    
    # Metadata
    burn_in: int  # Number of burn-in steps
    horizon: int  # Number of rollout steps
    anchor_idx: torch.Tensor  # (batch,) anchor time indices in original buffer
    episode_ids: torch.Tensor  # (batch,) episode IDs
    
    def to(self, device: torch.device) -> SequenceBatch:
        """Move batch to device."""
        return SequenceBatch(
            v_seq=self.v_seq.to(device),
            u_seq=self.u_seq.to(device),
            burn_in=self.burn_in,
            horizon=self.horizon,
            anchor_idx=self.anchor_idx.to(device),
            episode_ids=self.episode_ids.to(device),
        )


def find_valid_anchor_indices(
    episode_ids: np.ndarray,
    step_indices: np.ndarray,
    burn_in: int,
    horizon: int,
    buffer_size: int
) -> List[int]:
    """Find valid anchor indices for sequence sampling.
    
    An anchor index i is valid if:
        - i - burn_in >= 0 (have enough history)
        - i + horizon < buffer_size (have enough future)
        - All steps in [i-burn_in, i+horizon] are in the same episode
    
    Args:
        episode_ids: Episode IDs for each step in buffer
        step_indices: Step indices within episode for each step
        burn_in: Number of burn-in steps
        horizon: Number of rollout steps
        buffer_size: Current buffer size
    
    Returns:
        List of valid anchor indices
    """
    valid_indices = []
    
    for i in range(buffer_size):
        # Check buffer boundaries
        if i < burn_in:
            continue
        if i + horizon >= buffer_size:
            continue
        
        # Check episode continuity
        # Need: step_indices[i-burn_in:i+horizon+1] to be continuous in same episode
        window_start = i - burn_in
        window_end = i + horizon + 1
        
        # All episode IDs in window must match
        ep_id = episode_ids[i]
        if not np.all(episode_ids[window_start:window_end] == ep_id):
            continue
        
        # Step indices must be continuous
        expected_steps = np.arange(
            step_indices[window_start],
            step_indices[window_start] + (window_end - window_start)
        )
        actual_steps = step_indices[window_start:window_end]
        if not np.array_equal(actual_steps, expected_steps):
            continue
        
        valid_indices.append(i)
    
    return valid_indices


def sample_sequences(
    speed_buf: np.ndarray,
    action_buf: np.ndarray,
    episode_id_buf: np.ndarray,
    step_in_episode_buf: np.ndarray,
    buffer_size: int,
    batch_size: int,
    burn_in: int,
    horizon: int,
    rng: np.random.Generator | None = None
) -> SequenceBatch | None:
    """Sample sequences from replay buffer for SysID training.
    
    Args:
        speed_buf: Speed buffer of shape (capacity,)
        action_buf: Action buffer of shape (capacity, action_dim)
        episode_id_buf: Episode ID buffer of shape (capacity,)
        step_in_episode_buf: Step index buffer of shape (capacity,)
        buffer_size: Current buffer size
        batch_size: Number of sequences to sample
        burn_in: Number of burn-in steps
        horizon: Number of rollout steps
        rng: Random number generator (default: np.random.default_rng())
    
    Returns:
        SequenceBatch or None if not enough valid sequences
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Find valid anchor indices
    valid_indices = find_valid_anchor_indices(
        episode_id_buf[:buffer_size],
        step_in_episode_buf[:buffer_size],
        burn_in,
        horizon,
        buffer_size
    )
    
    if len(valid_indices) < batch_size:
        return None  # Not enough valid sequences
    
    # Sample anchor indices
    sampled_anchors = rng.choice(valid_indices, size=batch_size, replace=False)
    
    # Extract sequences
    v_seqs = []
    u_seqs = []
    episode_ids = []
    
    for anchor in sampled_anchors:
        # Extract v[anchor - burn_in : anchor + horizon + 1]
        v_start = anchor - burn_in
        v_end = anchor + horizon + 1
        v_seq = speed_buf[v_start:v_end]
        
        # Extract u[anchor - burn_in : anchor + horizon]
        # Note: u needs one less element than v (u[t] produces v[t+1])
        u_start = anchor - burn_in
        u_end = anchor + horizon
        u_seq = action_buf[u_start:u_end, 0]  # Extract first action dim
        
        v_seqs.append(v_seq)
        u_seqs.append(u_seq)
        episode_ids.append(episode_id_buf[anchor])
    
    # Stack into tensors
    v_seqs = torch.from_numpy(np.stack(v_seqs, axis=0)).float()
    u_seqs = torch.from_numpy(np.stack(u_seqs, axis=0)).float()
    anchor_idx = torch.from_numpy(sampled_anchors).long()
    episode_ids = torch.from_numpy(np.array(episode_ids)).long()
    
    return SequenceBatch(
        v_seq=v_seqs,
        u_seq=u_seqs,
        burn_in=burn_in,
        horizon=horizon,
        anchor_idx=anchor_idx,
        episode_ids=episode_ids,
    )

