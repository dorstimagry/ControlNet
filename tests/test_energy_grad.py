"""Unit tests for energy and gradient computation."""

import numpy as np
import pytest

from src.maps.buffer import Observation
from src.maps.energy import compute_energy, compute_energy_gradient


def test_energy_empty():
    """Test energy with no observations."""
    X = np.ones((100, 100), dtype=np.float32)
    observations = []
    weights = np.array([])
    sigma_meas = 0.5
    
    energy = compute_energy(X, observations, weights, sigma_meas)
    assert np.isclose(energy, 0.0)


def test_energy_single_observation():
    """Test energy with single observation."""
    X = np.zeros((100, 100), dtype=np.float32)
    X[5, 10] = 3.0
    
    observations = [Observation(i_u=5, i_v=10, a_dyn=2.0, timestamp=0.0)]
    weights = np.array([1.0])
    sigma_meas = 0.5
    
    energy = compute_energy(X, observations, weights, sigma_meas)
    
    # E = (w / (2*sigma^2)) * (X - a)^2
    # E = (1.0 / (2 * 0.25)) * (3.0 - 2.0)^2 = 2.0
    expected = (1.0 / (2.0 * 0.5 ** 2)) * (3.0 - 2.0) ** 2
    assert np.isclose(energy, expected)


def test_energy_multiple_observations():
    """Test energy with multiple observations."""
    X = np.zeros((100, 100), dtype=np.float32)
    X[5, 10] = 3.0
    X[6, 11] = 4.0
    
    observations = [
        Observation(i_u=5, i_v=10, a_dyn=2.0, timestamp=0.0),
        Observation(i_u=6, i_v=11, a_dyn=3.5, timestamp=0.0),
    ]
    weights = np.array([1.0, 0.8])
    sigma_meas = 0.5
    
    energy = compute_energy(X, observations, weights, sigma_meas)
    
    # E = sum_i (w_i / (2*sigma^2)) * (X_i - a_i)^2
    term1 = (1.0 / (2.0 * 0.25)) * (3.0 - 2.0) ** 2
    term2 = (0.8 / (2.0 * 0.25)) * (4.0 - 3.5) ** 2
    expected = term1 + term2
    assert np.isclose(energy, expected)


def test_gradient_empty():
    """Test gradient with no observations."""
    X = np.ones((100, 100), dtype=np.float32)
    observations = []
    weights = np.array([])
    sigma_meas = 0.5
    
    grad = compute_energy_gradient(X, observations, weights, sigma_meas)
    
    assert grad.shape == X.shape
    assert np.allclose(grad, 0.0)


def test_gradient_single_observation():
    """Test gradient with single observation."""
    X = np.zeros((100, 100), dtype=np.float32)
    X[5, 10] = 3.0
    
    observations = [Observation(i_u=5, i_v=10, a_dyn=2.0, timestamp=0.0)]
    weights = np.array([1.0])
    sigma_meas = 0.5
    
    grad = compute_energy_gradient(X, observations, weights, sigma_meas)
    
    assert grad.shape == X.shape
    
    # Only non-zero at observation location
    grad_copy = grad.copy()
    grad_copy[5, 10] = 0.0
    assert np.allclose(grad_copy, 0.0)
    
    # dE/dX = (w / sigma^2) * (X - a)
    expected = (1.0 / (0.5 ** 2)) * (3.0 - 2.0)
    assert np.isclose(grad[5, 10], expected)


def test_gradient_finite_difference():
    """Test gradient using finite difference approximation."""
    X = np.random.randn(10, 10).astype(np.float32)
    
    observations = [
        Observation(i_u=2, i_v=3, a_dyn=1.5, timestamp=0.0),
        Observation(i_u=5, i_v=7, a_dyn=-0.5, timestamp=0.0),
    ]
    weights = np.array([1.0, 0.8])
    sigma_meas = 0.5
    
    # Compute analytical gradient
    grad = compute_energy_gradient(X, observations, weights, sigma_meas)
    
    # Compute numerical gradient using finite differences
    eps = 1e-4
    for obs in observations:
        i_u, i_v = obs.i_u, obs.i_v
        
        # Perturb X slightly
        X_plus = X.copy()
        X_plus[i_u, i_v] += eps
        
        X_minus = X.copy()
        X_minus[i_u, i_v] -= eps
        
        # Finite difference: (E(X+eps) - E(X-eps)) / (2*eps)
        energy_plus = compute_energy(X_plus, observations, weights, sigma_meas)
        energy_minus = compute_energy(X_minus, observations, weights, sigma_meas)
        
        numerical_grad = (energy_plus - energy_minus) / (2.0 * eps)
        
        # Compare with analytical gradient
        assert np.isclose(grad[i_u, i_v], numerical_grad, rtol=1e-3)


def test_gradient_multiple_observations_same_bin():
    """Test that gradients sum when multiple observations hit same bin."""
    X = np.zeros((100, 100), dtype=np.float32)
    X[5, 10] = 3.0
    
    observations = [
        Observation(i_u=5, i_v=10, a_dyn=2.0, timestamp=0.0),
        Observation(i_u=5, i_v=10, a_dyn=2.5, timestamp=0.0),
    ]
    weights = np.array([1.0, 1.0])
    sigma_meas = 0.5
    
    grad = compute_energy_gradient(X, observations, weights, sigma_meas)
    
    # Gradients should sum
    grad1 = (1.0 / (0.5 ** 2)) * (3.0 - 2.0)
    grad2 = (1.0 / (0.5 ** 2)) * (3.0 - 2.5)
    expected = grad1 + grad2
    
    assert np.isclose(grad[5, 10], expected)


def test_gradient_different_weights():
    """Test that weights affect gradient magnitude."""
    X = np.zeros((100, 100), dtype=np.float32)
    X[5, 10] = 3.0
    
    observations = [Observation(i_u=5, i_v=10, a_dyn=2.0, timestamp=0.0)]
    sigma_meas = 0.5
    
    # Test with different weights
    for weight in [0.1, 0.5, 1.0, 2.0]:
        weights = np.array([weight])
        grad = compute_energy_gradient(X, observations, weights, sigma_meas)
        
        expected = (weight / (0.5 ** 2)) * (3.0 - 2.0)
        assert np.isclose(grad[5, 10], expected)


def test_gradient_different_sigma():
    """Test that sigma affects gradient magnitude."""
    X = np.zeros((100, 100), dtype=np.float32)
    X[5, 10] = 3.0
    
    observations = [Observation(i_u=5, i_v=10, a_dyn=2.0, timestamp=0.0)]
    weights = np.array([1.0])
    
    # Test with different sigma values
    for sigma in [0.1, 0.5, 1.0, 2.0]:
        grad = compute_energy_gradient(X, observations, weights, sigma)
        
        expected = (1.0 / (sigma ** 2)) * (3.0 - 2.0)
        assert np.isclose(grad[5, 10], expected)

