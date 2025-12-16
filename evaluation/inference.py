"""Lightweight inference wrapper for external usage.

This module provides a stateful wrapper around trained SAC policies,
making it easy to run inference from external projects without needing
to manage observation history manually.

Usage from external project:
    from DiffDynamics.evaluation.inference import PolicyInferenceWrapper
    
    wrapper = PolicyInferenceWrapper("/path/to/checkpoint.pt")
    wrapper.reset(initial_speed=0.0)
    
    # Each timestep:
    action = wrapper.get_action(current_speed, target_profile)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from evaluation.policy_loader import load_policy_from_checkpoint, select_action


class PolicyInferenceWrapper:
    """Stateful wrapper for running trained policy inference.
    
    This class handles:
    - Loading the trained policy from a checkpoint
    - Tracking speed history (for acceleration/jerk estimation)
    - Tracking previous action (for action smoothness)
    - Constructing observations in the correct format
    - Auto-detecting observation format from checkpoint (legacy vs new)
    
    Attributes:
        env_cfg: The environment configuration from the checkpoint
        preview_steps: Number of future target speeds expected in the profile
        dt: Control timestep (seconds)
        obs_format: Either "legacy" or "full" indicating the observation format
    """
    
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        deterministic: bool = True,
    ):
        """Initialize the inference wrapper.
        
        Args:
            checkpoint_path: Path to the trained SAC checkpoint (.pt file)
            device: Torch device ("cpu" or "cuda")
            deterministic: If True, use mean action; if False, sample from policy
        """
        self.device = torch.device(device)
        self.policy, self.env_cfg, self.horizon = load_policy_from_checkpoint(
            Path(checkpoint_path), device=self.device
        )
        self.deterministic = deterministic
        
        # Load checkpoint to get obs_dim from metadata
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        meta = checkpoint.get("meta", {})
        saved_obs_dim = int(meta.get("obs_dim", 0))
        
        # Compute expected preview steps from config
        self.preview_steps = max(int(round(self.env_cfg.preview_horizon_s / self.env_cfg.dt)), 1)
        self.dt = self.env_cfg.dt
        
        # Detect observation format from saved obs_dim
        # Legacy format: [speed, refs...] -> obs_dim = 1 + preview_steps
        # Full format: [speed, prev_speed, prev_prev_speed, prev_action, refs...] -> obs_dim = 4 + preview_steps
        expected_full = 4 + self.preview_steps
        expected_legacy = 1 + self.preview_steps
        
        if saved_obs_dim == expected_full:
            self.obs_format = "full"
            self.obs_dim = expected_full
        elif saved_obs_dim == expected_legacy:
            self.obs_format = "legacy"
            self.obs_dim = expected_legacy
        elif saved_obs_dim > 0:
            # Infer preview_steps from saved obs_dim
            # Try full format first
            if saved_obs_dim > 4:
                self.obs_format = "full"
                self.preview_steps = saved_obs_dim - 4
                self.obs_dim = saved_obs_dim
            else:
                self.obs_format = "legacy"
                self.preview_steps = saved_obs_dim - 1
                self.obs_dim = saved_obs_dim
        else:
            # Default to full format
            self.obs_format = "full"
            self.obs_dim = expected_full
        
        # State tracking (used for full format)
        self._prev_speed: float = 0.0
        self._prev_prev_speed: float = 0.0
        self._prev_action: float = 0.0
    
    def reset(self, initial_speed: float = 0.0) -> None:
        """Reset internal state for a new episode.
        
        Call this at the start of each new control episode to clear history.
        
        Args:
            initial_speed: Initial vehicle speed (m/s)
        """
        self._prev_speed = initial_speed
        self._prev_prev_speed = initial_speed
        self._prev_action = 0.0
    
    def _build_refs(self, current_speed: float, target_profile: Sequence[float]) -> np.ndarray:
        """Build padded reference array from target profile."""
        refs = np.zeros(self.preview_steps, dtype=np.float32)
        profile_len = len(target_profile)
        for i in range(self.preview_steps):
            if i < profile_len:
                refs[i] = float(target_profile[i])
            else:
                # Repeat last available value or current speed
                refs[i] = refs[i - 1] if i > 0 else float(current_speed)
        return refs
    
    def _build_observation(self, current_speed: float, refs: np.ndarray) -> np.ndarray:
        """Build observation array based on detected format."""
        if self.obs_format == "legacy":
            # Legacy format: [speed, refs...]
            return np.concatenate([
                np.array([current_speed], dtype=np.float32),
                refs,
            ])
        else:
            # Full format: [speed, prev_speed, prev_prev_speed, prev_action, refs...]
            return np.concatenate([
                np.array([
                    current_speed,
                    self._prev_speed,
                    self._prev_prev_speed,
                    self._prev_action,
                ], dtype=np.float32),
                refs,
            ])
    
    def get_action(
        self,
        current_speed: float,
        target_profile: Sequence[float],
    ) -> float:
        """Get control action given current speed and upcoming target speeds.
        
        Args:
            current_speed: Current vehicle speed (m/s)
            target_profile: Target speeds for the next preview_steps timesteps (m/s).
                           Should have at least `preview_steps` elements.
                           If shorter, the last value is repeated.
        
        Returns:
            Action value in [-1, 1]:
                - Negative values = braking
                - Positive values = throttle
                - 0 = coast
        """
        refs = self._build_refs(current_speed, target_profile)
        obs = self._build_observation(current_speed, refs)
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_array, _ = select_action(self.policy, obs_tensor, deterministic=self.deterministic)
        action = float(action_array[0])
        
        # Update state for next call (used in full format)
        self._prev_prev_speed = self._prev_speed
        self._prev_speed = current_speed
        self._prev_action = action
        
        return action
    
    def get_action_with_stats(
        self,
        current_speed: float,
        target_profile: Sequence[float],
    ) -> tuple[float, dict]:
        """Get control action along with policy statistics.
        
        Same as get_action(), but also returns policy statistics for debugging.
        
        Args:
            current_speed: Current vehicle speed (m/s)
            target_profile: Target speeds for the next preview_steps timesteps (m/s)
        
        Returns:
            Tuple of (action, stats) where:
                - action: Action value in [-1, 1]
                - stats: Dict with 'pre_tanh_mean', 'log_std', 'plan'
        """
        refs = self._build_refs(current_speed, target_profile)
        obs = self._build_observation(current_speed, refs)
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_array, stats = select_action(self.policy, obs_tensor, deterministic=self.deterministic)
        action = float(action_array[0])
        
        # Update state for next call (used in full format)
        self._prev_prev_speed = self._prev_speed
        self._prev_speed = current_speed
        self._prev_action = action
        
        return action, stats
    
    @property
    def action_bounds(self) -> tuple[float, float]:
        """Get the action bounds (low, high)."""
        return (self.env_cfg.action_low, self.env_cfg.action_high)


__all__ = ["PolicyInferenceWrapper"]

