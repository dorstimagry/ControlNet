"""Observation buffer with time-decayed weights for map reconstruction."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Observation:
    """Single observation constraint.
    
    Attributes:
        i_u: Bin index for u
        i_v: Bin index for v
        a_dyn: Dynamics acceleration (gravity-compensated)
        timestamp: Observation time (seconds or steps)
    """
    i_u: int
    i_v: int
    a_dyn: float
    timestamp: float


class ObservationBuffer:
    """Ring buffer for storing map observations with time decay.
    
    Stores point constraints (bin, value, timestamp) and computes
    time-decayed weights for use in guided reconstruction.
    
    Weight decay formula: w_i(t) = max(w_min, exp(-lambda * (t - t_i)))
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        lambda_decay: float = 0.3,
        w_min: float = 0.2,
    ):
        """Initialize observation buffer.
        
        Args:
            capacity: Maximum number of observations to store
            lambda_decay: Decay rate (1/time_units)
            w_min: Minimum weight floor
        """
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if lambda_decay < 0:
            raise ValueError(f"lambda_decay must be non-negative, got {lambda_decay}")
        if not (0 <= w_min <= 1):
            raise ValueError(f"w_min must be in [0, 1], got {w_min}")
        
        self.capacity = capacity
        self.lambda_decay = lambda_decay
        self.w_min = w_min
        
        self.buffer: deque[Observation] = deque(maxlen=capacity)
    
    def add(self, i_u: int, i_v: int, a_dyn: float, timestamp: float) -> None:
        """Add observation to buffer.
        
        Args:
            i_u: Bin index for u
            i_v: Bin index for v
            a_dyn: Dynamics acceleration (gravity-compensated)
            timestamp: Observation time
        """
        obs = Observation(i_u=i_u, i_v=i_v, a_dyn=a_dyn, timestamp=timestamp)
        self.buffer.append(obs)
    
    def reset(self) -> None:
        """Clear all observations."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Return number of observations in buffer."""
        return len(self.buffer)
    
    def get_weights(self, t_now: float) -> np.ndarray:
        """Compute time-decayed weights for all observations.
        
        Args:
            t_now: Current time
        
        Returns:
            Array of weights, shape (N_obs,)
        """
        if len(self.buffer) == 0:
            return np.array([], dtype=np.float32)
        
        timestamps = np.array([obs.timestamp for obs in self.buffer], dtype=np.float32)
        dt = t_now - timestamps
        weights = np.exp(-self.lambda_decay * dt)
        weights = np.maximum(weights, self.w_min)
        
        return weights
    
    def get_observations(self) -> List[Observation]:
        """Get all observations as list.
        
        Returns:
            List of observations
        """
        return list(self.buffer)
    
    def get_filtered_observations(
        self,
        t_now: float,
        patch_size_u: int,
        patch_size_v: int,
        grid_shape: Tuple[int, int],
        seed: int | None = None,
    ) -> Tuple[List[Observation], np.ndarray]:
        """Get observations filtered by patch-based uniform sampling.
        
        Divides the map into patches and ensures uniform spatial distribution
        by capping observations per patch based on coverage ratio.
        
        Algorithm:
        1. Divide map into patches of size (patch_size_u, patch_size_v)
        2. Group observations by their patch
        3. Compute coverage ratio: n_patches_observed / n_patches_total
        4. Calculate continuous cap per patch:
           cap = max(1, 1 + (target_obs_per_patch - 1) * coverage_ratio)
        5. Randomly select up to cap observations per patch
        
        Args:
            t_now: Current time (for weight computation)
            patch_size_u: Patch size in u direction
            patch_size_v: Patch size in v direction
            grid_shape: (N_u, N_v) - grid dimensions
            seed: Random seed for reproducible selection
        
        Returns:
            Tuple of (filtered_observations, filtered_weights)
        """
        if len(self.buffer) == 0:
            return [], np.array([], dtype=np.float32)
        
        N_u, N_v = grid_shape
        
        # Ensure patch size doesn't exceed grid size (treat entire grid as one patch if needed)
        effective_patch_size_u = min(patch_size_u, N_u)
        effective_patch_size_v = min(patch_size_v, N_v)
        
        # Compute number of patches
        n_patches_u = (N_u + effective_patch_size_u - 1) // effective_patch_size_u  # Ceiling division
        n_patches_v = (N_v + effective_patch_size_v - 1) // effective_patch_size_v
        n_patches_total = n_patches_u * n_patches_v
        
        # Group observations by patch
        patches: dict[Tuple[int, int], List[Tuple[int, Observation]]] = defaultdict(list)
        all_weights = self.get_weights(t_now)
        
        for idx, obs in enumerate(self.buffer):
            # Clamp indices to valid range
            i_u_clamped = max(0, min(obs.i_u, N_u - 1))
            i_v_clamped = max(0, min(obs.i_v, N_v - 1))
            patch_u = i_u_clamped // effective_patch_size_u
            patch_v = i_v_clamped // effective_patch_size_v
            patch_id = (patch_u, patch_v)
            patches[patch_id].append((idx, obs))
        
        n_patches_observed = len(patches)
        
        if n_patches_observed == 0:
            return [], np.array([], dtype=np.float32)
        
        # Compute continuous cap per patch
        total_observations = len(self.buffer)
        coverage_ratio = n_patches_observed / n_patches_total
        
        # Target: uniform distribution across all patches
        target_obs_per_patch = total_observations / n_patches_total
        
        # Continuous scaling: cap = 1 + (target - 1) * coverage_ratio
        # When coverage=0: cap=1 (minimum)
        # When coverage=1: cap=target (full uniform)
        cap_per_patch = max(1.0, 1.0 + (target_obs_per_patch - 1.0) * coverage_ratio)
        cap_per_patch = int(np.ceil(cap_per_patch))  # Round up to integer
        
        # Filter observations per patch
        filtered_obs: List[Observation] = []
        filtered_weights: List[float] = []
        
        # Use deterministic random generator for reproducibility
        rng = np.random.default_rng(seed)
        
        for patch_id, patch_data in sorted(patches.items()):  # Sort for determinism
            patch_indices, patch_observations = zip(*patch_data)
            patch_weights = [all_weights[idx] for idx in patch_indices]
            
            if len(patch_observations) <= cap_per_patch:
                # Use all observations in this patch
                filtered_obs.extend(patch_observations)
                filtered_weights.extend(patch_weights)
            else:
                # Randomly select cap_per_patch observations
                # Use patch_id as seed component for reproducibility
                patch_rng = np.random.default_rng(
                    seed=(hash(patch_id) + (seed if seed is not None else 0)) % (2**32)
                )
                selected_indices = patch_rng.choice(
                    len(patch_observations),
                    size=cap_per_patch,
                    replace=False
                )
                filtered_obs.extend([patch_observations[i] for i in selected_indices])
                filtered_weights.extend([patch_weights[i] for i in selected_indices])
        
        return filtered_obs, np.array(filtered_weights, dtype=np.float32)
    
    def compute_sparse_gradient(
        self,
        X: np.ndarray,
        t_now: float,
        sigma_meas: float,
    ) -> np.ndarray:
        """Compute sparse gradient of measurement energy.
        
        Energy: E(X) = sum_i [w_i / (2*sigma^2)] * (X[bin_i] - a_i)^2
        Gradient: dE/dX[bin_i] = [w_i / sigma^2] * (X[bin_i] - a_i)
        
        Args:
            X: Current map estimate, shape (N_u, N_v)
            t_now: Current time
            sigma_meas: Measurement noise std (m/s^2)
        
        Returns:
            Gradient array, same shape as X
        """
        if len(self.buffer) == 0:
            return np.zeros_like(X, dtype=np.float32)
        
        grad = np.zeros_like(X, dtype=np.float32)
        weights = self.get_weights(t_now)
        
        sigma_sq = sigma_meas ** 2
        
        for obs, w in zip(self.buffer, weights):
            residual = X[obs.i_u, obs.i_v] - obs.a_dyn
            grad[obs.i_u, obs.i_v] += (w / sigma_sq) * residual
        
        return grad

