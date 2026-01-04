"""Unit tests for ObservationBuffer."""

import numpy as np
import pytest

from src.maps.buffer import Observation, ObservationBuffer


def test_buffer_initialization():
    """Test basic initialization."""
    buffer = ObservationBuffer(capacity=100, lambda_decay=0.3, w_min=0.2)
    assert buffer.capacity == 100
    assert buffer.lambda_decay == 0.3
    assert buffer.w_min == 0.2
    assert len(buffer) == 0


def test_buffer_invalid_params():
    """Test that invalid parameters raise errors."""
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        ObservationBuffer(capacity=0)
    
    with pytest.raises(ValueError, match="lambda_decay must be non-negative"):
        ObservationBuffer(lambda_decay=-0.1)
    
    with pytest.raises(ValueError, match="w_min must be in"):
        ObservationBuffer(w_min=1.5)


def test_buffer_add():
    """Test adding observations."""
    buffer = ObservationBuffer(capacity=10)
    
    buffer.add(i_u=5, i_v=10, a_dyn=2.5, timestamp=1.0)
    assert len(buffer) == 1
    
    buffer.add(i_u=6, i_v=11, a_dyn=3.0, timestamp=2.0)
    assert len(buffer) == 2
    
    obs_list = buffer.get_observations()
    assert len(obs_list) == 2
    assert obs_list[0].i_u == 5
    assert obs_list[0].i_v == 10
    assert obs_list[0].a_dyn == 2.5
    assert obs_list[0].timestamp == 1.0


def test_buffer_reset():
    """Test buffer reset."""
    buffer = ObservationBuffer(capacity=10)
    
    buffer.add(i_u=5, i_v=10, a_dyn=2.5, timestamp=1.0)
    buffer.add(i_u=6, i_v=11, a_dyn=3.0, timestamp=2.0)
    assert len(buffer) == 2
    
    buffer.reset()
    assert len(buffer) == 0


def test_buffer_capacity():
    """Test that buffer respects capacity limit."""
    buffer = ObservationBuffer(capacity=3)
    
    # Add 5 observations
    for i in range(5):
        buffer.add(i_u=i, i_v=i, a_dyn=float(i), timestamp=float(i))
    
    # Should only keep last 3
    assert len(buffer) == 3
    
    obs_list = buffer.get_observations()
    # Should have observations 2, 3, 4 (oldest evicted)
    assert obs_list[0].i_u == 2
    assert obs_list[1].i_u == 3
    assert obs_list[2].i_u == 4


def test_weight_decay_immediate():
    """Test that weights are 1.0 at observation time."""
    buffer = ObservationBuffer(capacity=10, lambda_decay=0.5, w_min=0.1)
    
    buffer.add(i_u=5, i_v=10, a_dyn=2.5, timestamp=1.0)
    
    # At the same time as observation
    weights = buffer.get_weights(t_now=1.0)
    assert len(weights) == 1
    assert np.isclose(weights[0], 1.0)


def test_weight_decay_over_time():
    """Test that weights decay exponentially over time."""
    lambda_decay = 0.5
    buffer = ObservationBuffer(capacity=10, lambda_decay=lambda_decay, w_min=0.0)
    
    buffer.add(i_u=5, i_v=10, a_dyn=2.5, timestamp=0.0)
    
    # Check decay at various times
    t_vals = np.array([0.0, 1.0, 2.0, 3.0])
    for t in t_vals:
        weights = buffer.get_weights(t_now=t)
        expected = np.exp(-lambda_decay * t)
        assert np.isclose(weights[0], expected)


def test_weight_decay_floor():
    """Test that weights don't go below w_min."""
    buffer = ObservationBuffer(capacity=10, lambda_decay=0.5, w_min=0.2)
    
    buffer.add(i_u=5, i_v=10, a_dyn=2.5, timestamp=0.0)
    
    # At very late time, weight should be at floor
    weights = buffer.get_weights(t_now=100.0)
    assert np.isclose(weights[0], 0.2)


def test_weight_decay_multiple_observations():
    """Test weights for multiple observations with different ages."""
    buffer = ObservationBuffer(capacity=10, lambda_decay=0.3, w_min=0.1)
    
    # Add observations at different times
    buffer.add(i_u=0, i_v=0, a_dyn=1.0, timestamp=0.0)
    buffer.add(i_u=1, i_v=1, a_dyn=2.0, timestamp=1.0)
    buffer.add(i_u=2, i_v=2, a_dyn=3.0, timestamp=2.0)
    
    # Get weights at t=2.0
    weights = buffer.get_weights(t_now=2.0)
    
    assert len(weights) == 3
    
    # Oldest observation should have lowest weight
    assert weights[0] < weights[1] < weights[2]
    
    # Most recent should be ~1.0
    assert np.isclose(weights[2], 1.0)


def test_sparse_gradient_empty_buffer():
    """Test gradient with no observations."""
    buffer = ObservationBuffer(capacity=10)
    
    X = np.ones((100, 100), dtype=np.float32)
    grad = buffer.compute_sparse_gradient(X, t_now=0.0, sigma_meas=0.5)
    
    # Should be all zeros
    assert grad.shape == X.shape
    assert np.allclose(grad, 0.0)


def test_sparse_gradient_single_observation():
    """Test gradient with single observation."""
    buffer = ObservationBuffer(capacity=10, lambda_decay=0.0, w_min=1.0)
    
    # Add observation
    i_u, i_v = 5, 10
    a_dyn = 2.0
    buffer.add(i_u=i_u, i_v=i_v, a_dyn=a_dyn, timestamp=0.0)
    
    # Create map with value 3.0 at observation location
    X = np.zeros((100, 100), dtype=np.float32)
    X[i_u, i_v] = 3.0
    
    sigma_meas = 0.5
    grad = buffer.compute_sparse_gradient(X, t_now=0.0, sigma_meas=sigma_meas)
    
    # Gradient should be non-zero only at observation location
    assert grad.shape == X.shape
    
    # All other locations should be zero
    grad_copy = grad.copy()
    grad_copy[i_u, i_v] = 0.0
    assert np.allclose(grad_copy, 0.0)
    
    # Gradient at observation: (w / sigma^2) * (X[bin] - a_dyn)
    # w=1.0, X=3.0, a_dyn=2.0, sigma=0.5
    expected_grad = (1.0 / (0.5 ** 2)) * (3.0 - 2.0)
    assert np.isclose(grad[i_u, i_v], expected_grad)


def test_sparse_gradient_multiple_observations_same_bin():
    """Test gradient when multiple observations hit same bin."""
    buffer = ObservationBuffer(capacity=10, lambda_decay=0.0, w_min=1.0)
    
    # Add two observations at same location
    i_u, i_v = 5, 10
    buffer.add(i_u=i_u, i_v=i_v, a_dyn=2.0, timestamp=0.0)
    buffer.add(i_u=i_u, i_v=i_v, a_dyn=2.5, timestamp=0.0)
    
    X = np.zeros((100, 100), dtype=np.float32)
    X[i_u, i_v] = 3.0
    
    sigma_meas = 0.5
    grad = buffer.compute_sparse_gradient(X, t_now=0.0, sigma_meas=sigma_meas)
    
    # Gradients should sum
    grad1 = (1.0 / (0.5 ** 2)) * (3.0 - 2.0)
    grad2 = (1.0 / (0.5 ** 2)) * (3.0 - 2.5)
    expected_grad = grad1 + grad2
    
    assert np.isclose(grad[i_u, i_v], expected_grad)

