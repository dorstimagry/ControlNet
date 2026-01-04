"""Gravity compensation for measured acceleration."""

from __future__ import annotations

import numpy as np


GRAVITY = 9.80665  # m/s^2


class GravityCompensator:
    """Compensate for gravity effects in measured acceleration.
    
    Converts measured longitudinal acceleration (along motion direction)
    to dynamics-only acceleration (flat ground equivalent).
    
    Convention: theta > 0 means nose-up (uphill).
    Gravity contributes: -g*sin(theta) along forward axis.
    Therefore: a_dyn = a_meas + g*sin(theta)
    """
    
    def __init__(self, g: float = GRAVITY):
        """Initialize gravity compensator.
        
        Args:
            g: Gravitational acceleration (m/s^2)
        """
        self.g = g
    
    def compensate(self, a_meas: float | np.ndarray, theta: float | np.ndarray) -> float | np.ndarray:
        """Compute dynamics acceleration from measured acceleration.
        
        Args:
            a_meas: Measured longitudinal acceleration (m/s^2)
            theta: Pitch angle in radians (positive = nose-up)
        
        Returns:
            Dynamics-only acceleration (m/s^2)
        """
        return a_meas + self.g * np.sin(theta)
    
    def __call__(self, a_meas: float | np.ndarray, theta: float | np.ndarray) -> float | np.ndarray:
        """Shorthand for compensate()."""
        return self.compensate(a_meas, theta)


def compensate_gravity(a_meas: float | np.ndarray, theta: float | np.ndarray, g: float = GRAVITY) -> float | np.ndarray:
    """Functional interface for gravity compensation.
    
    Args:
        a_meas: Measured longitudinal acceleration (m/s^2)
        theta: Pitch angle in radians (positive = nose-up)
        g: Gravitational acceleration (m/s^2)
    
    Returns:
        Dynamics-only acceleration (m/s^2)
    """
    return a_meas + g * np.sin(theta)

