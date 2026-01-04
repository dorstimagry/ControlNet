"""Energy and gradient computation for measurement constraints."""

from __future__ import annotations

from typing import List

import numpy as np

from .buffer import Observation


def compute_energy(
    X: np.ndarray,
    observations: List[Observation],
    weights: np.ndarray,
    sigma_meas: float,
) -> float:
    """Compute measurement constraint energy.
    
    E(X) = (1/N) * sum_i [w_i / (2*sigma^2)] * (X[bin_i] - a_i)^2
    
    Normalized by number of observations to prevent energy explosion.
    
    Args:
        X: Map estimate, shape (N_u, N_v)
        observations: List of observations
        weights: Per-observation weights, shape (N_obs,)
        sigma_meas: Measurement noise std (m/s^2)
    
    Returns:
        Energy value (scalar)
    """
    if len(observations) == 0:
        return 0.0
    
    energy = 0.0
    sigma_sq = sigma_meas ** 2
    
    for obs, w in zip(observations, weights):
        residual = X[obs.i_u, obs.i_v] - obs.a_dyn
        energy += (w / (2.0 * sigma_sq)) * (residual ** 2)
    
    # Normalize by number of observations
    energy /= len(observations)
    
    return float(energy)


def compute_energy_gradient(
    X: np.ndarray,
    observations: List[Observation],
    weights: np.ndarray,
    sigma_meas: float,
) -> np.ndarray:
    """Compute gradient of measurement constraint energy.
    
    dE/dX[bin_i] = (1/N) * [w_i / sigma^2] * (X[bin_i] - a_i)
    
    Normalized by number of observations to prevent gradient explosion.
    
    Args:
        X: Map estimate, shape (N_u, N_v)
        observations: List of observations
        weights: Per-observation weights, shape (N_obs,)
        sigma_meas: Measurement noise std (m/s^2)
    
    Returns:
        Gradient array, same shape as X
    """
    if len(observations) == 0:
        return np.zeros_like(X, dtype=np.float32)
    
    grad = np.zeros_like(X, dtype=np.float32)
    sigma_sq = sigma_meas ** 2
    
    for obs, w in zip(observations, weights):
        residual = X[obs.i_u, obs.i_v] - obs.a_dyn
        grad[obs.i_u, obs.i_v] += (w / sigma_sq) * residual
    
    # Normalize by number of observations
    grad /= len(observations)
    
    return grad

