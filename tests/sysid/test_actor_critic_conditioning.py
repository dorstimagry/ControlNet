"""Unit tests for actor/critic conditioning with z."""

import pytest
import torch
import numpy as np

from training.train_sac import GaussianPolicy, QNetwork


def test_policy_backward_compatibility():
    """Test that policy works with z_dim=0 (backward compatible)."""
    obs_dim = 10
    action_dim = 1
    action_low = np.array([-1.0])
    action_high = np.array([1.0])
    
    policy = GaussianPolicy(obs_dim, action_dim, action_low, action_high, z_dim=0)
    
    obs = torch.randn(4, obs_dim)
    action, log_prob, mu_action = policy.sample(obs)
    
    assert action.shape == (4, action_dim)
    assert log_prob.shape == (4, 1)
    assert mu_action.shape == (4, action_dim)


def test_policy_with_z():
    """Test policy with z augmentation."""
    obs_dim = 10
    z_dim = 12
    action_dim = 1
    action_low = np.array([-1.0])
    action_high = np.array([1.0])
    
    policy = GaussianPolicy(obs_dim, action_dim, action_low, action_high, z_dim=z_dim)
    
    # Input should be obs + z
    obs_aug = torch.randn(4, obs_dim + z_dim)
    action, log_prob, mu_action = policy.sample(obs_aug)
    
    assert action.shape == (4, action_dim)
    assert log_prob.shape == (4, 1)
    assert mu_action.shape == (4, action_dim)


def test_policy_action_bounds():
    """Test that policy respects action bounds."""
    obs_dim = 10
    action_dim = 1
    action_low = np.array([-2.0])
    action_high = np.array([3.0])
    
    policy = GaussianPolicy(obs_dim, action_dim, action_low, action_high, z_dim=0)
    
    obs = torch.randn(100, obs_dim)
    action, _, _ = policy.sample(obs)
    
    # Actions should be within bounds
    assert torch.all(action >= action_low[0])
    assert torch.all(action <= action_high[0])


def test_qnetwork_backward_compatibility():
    """Test Q-network with z_dim=0."""
    obs_dim = 10
    action_dim = 1
    
    q_net = QNetwork(obs_dim, action_dim, z_dim=0)
    
    obs = torch.randn(4, obs_dim)
    action = torch.randn(4, action_dim)
    
    q_value = q_net(obs, action)
    assert q_value.shape == (4, 1)


def test_qnetwork_with_z():
    """Test Q-network with z augmentation."""
    obs_dim = 10
    z_dim = 12
    action_dim = 1
    
    q_net = QNetwork(obs_dim, action_dim, z_dim=z_dim)
    
    # Input should be obs + z
    obs_aug = torch.randn(4, obs_dim + z_dim)
    action = torch.randn(4, action_dim)
    
    q_value = q_net(obs_aug, action)
    assert q_value.shape == (4, 1)


def test_policy_gradients_with_z():
    """Test that gradients flow through policy with z."""
    obs_dim = 10
    z_dim = 12
    action_dim = 1
    action_low = np.array([-1.0])
    action_high = np.array([1.0])
    
    policy = GaussianPolicy(obs_dim, action_dim, action_low, action_high, z_dim=z_dim)
    
    obs_aug = torch.randn(4, obs_dim + z_dim, requires_grad=True)
    action, log_prob, _ = policy.sample(obs_aug)
    
    loss = log_prob.sum()
    loss.backward()
    
    # Check gradients
    for param in policy.parameters():
        assert param.grad is not None


def test_qnetwork_gradients_with_z():
    """Test that gradients flow through Q-network with z."""
    obs_dim = 10
    z_dim = 12
    action_dim = 1
    
    q_net = QNetwork(obs_dim, action_dim, z_dim=z_dim)
    
    obs_aug = torch.randn(4, obs_dim + z_dim, requires_grad=True)
    action = torch.randn(4, action_dim, requires_grad=True)
    
    q_value = q_net(obs_aug, action)
    loss = q_value.sum()
    loss.backward()
    
    # Check gradients
    for param in q_net.parameters():
        assert param.grad is not None
    
    assert obs_aug.grad is not None
    assert action.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

