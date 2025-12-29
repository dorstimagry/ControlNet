"""Unit tests for RunningNorm."""

import pytest
import torch

from src.sysid import RunningNorm


def test_running_norm_initialization():
    """Test RunningNorm initialization."""
    norm = RunningNorm(dim=4, eps=1e-6, clip=10.0)
    
    assert norm.mean.shape == (4,)
    assert norm.var.shape == (4,)
    assert torch.allclose(norm.mean, torch.zeros(4))
    assert torch.allclose(norm.var, torch.ones(4))
    assert norm.count.item() == 0


def test_running_norm_update():
    """Test that statistics update correctly."""
    norm = RunningNorm(dim=2)
    
    # Add batch
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    norm.update(x)
    
    assert norm.count.item() == 2
    # Mean should be [2.0, 3.0]
    assert torch.allclose(norm.mean, torch.tensor([2.0, 3.0]))


def test_running_norm_normalize():
    """Test normalization."""
    norm = RunningNorm(dim=2, eps=0.0, clip=10.0)
    
    # Update with known statistics
    x1 = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    norm.update(x1)
    
    # Normalize
    x2 = torch.tensor([[1.0, 1.0]])
    x_norm = norm.normalize(x2, update_stats=False)
    
    # Should be (1.0 - 1.0) / std = 0.0
    assert x_norm.shape == (1, 2)
    assert torch.allclose(x_norm, torch.zeros(1, 2), atol=1e-5)


def test_running_norm_clipping():
    """Test that values are clipped."""
    norm = RunningNorm(dim=1, eps=1e-6, clip=2.0)
    
    # Add data with mean=0, std=1
    x = torch.randn(1000, 1)
    norm.update(x)
    
    # Normalize extreme value
    x_extreme = torch.tensor([[100.0]])
    x_norm = norm.normalize(x_extreme, update_stats=False)
    
    # Should be clipped to max 2.0
    assert x_norm[0, 0] <= 2.0


def test_running_norm_reset():
    """Test reset functionality."""
    norm = RunningNorm(dim=3)
    
    # Update
    x = torch.randn(10, 3)
    norm.update(x)
    
    assert norm.count.item() > 0
    
    # Reset
    norm.reset()
    
    assert norm.count.item() == 0
    assert torch.allclose(norm.mean, torch.zeros(3))
    assert torch.allclose(norm.var, torch.ones(3))


def test_running_norm_online_updates():
    """Test online updates match batch computation."""
    norm_online = RunningNorm(dim=2)
    
    data = torch.randn(100, 2)
    
    # Online updates
    for i in range(100):
        norm_online.update(data[i:i+1])
    
    # Batch computation
    batch_mean = data.mean(dim=0)
    batch_var = data.var(dim=0, unbiased=False)
    
    # Should be close
    assert torch.allclose(norm_online.mean, batch_mean, atol=1e-5)
    assert torch.allclose(norm_online.var, batch_var, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

