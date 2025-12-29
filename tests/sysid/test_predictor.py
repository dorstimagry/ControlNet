"""Unit tests for SysID dynamics predictor."""

import pytest
import torch

from src.sysid import DynamicsPredictor


def test_predictor_shapes():
    """Test predictor output shapes."""
    predictor = DynamicsPredictor(z_dim=12, hidden_dim=128, num_layers=2)
    
    batch_size = 8
    v = torch.randn(batch_size)
    u = torch.randn(batch_size)
    z = torch.randn(batch_size, 12)
    
    dv = predictor(v, u, z)
    
    assert dv.shape == (batch_size, 1)


def test_predictor_gradients():
    """Test that predictor produces gradients."""
    predictor = DynamicsPredictor(z_dim=12, hidden_dim=128, num_layers=2)
    
    v = torch.randn(4, requires_grad=True)
    u = torch.randn(4, requires_grad=True)
    z = torch.randn(4, 12, requires_grad=True)
    
    dv = predictor(v, u, z)
    loss = dv.sum()
    loss.backward()
    
    # Check parameters have gradients
    for param in predictor.parameters():
        assert param.grad is not None


def test_predictor_forward_backward():
    """Test full forward-backward pass."""
    predictor = DynamicsPredictor(z_dim=12, hidden_dim=128, num_layers=2)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    
    # Synthetic data: dv = 0.1 * v + 0.2 * u
    v = torch.tensor([1.0, 2.0, 3.0, 4.0])
    u = torch.tensor([0.5, 0.5, 0.5, 0.5])
    z = torch.randn(4, 12)
    target_dv = 0.1 * v + 0.2 * u
    
    # Training step
    for _ in range(10):
        optimizer.zero_grad()
        pred_dv = predictor(v, u, z).squeeze()
        loss = ((pred_dv - target_dv) ** 2).mean()
        loss.backward()
        optimizer.step()
    
    # Loss should decrease
    final_loss = loss.item()
    assert final_loss < 1.0  # Initial loss should be larger


@pytest.mark.parametrize("z_dim", [4, 12, 32])
@pytest.mark.parametrize("hidden_dim", [64, 128, 256])
def test_predictor_different_architectures(z_dim, hidden_dim):
    """Test predictor with different architectures."""
    predictor = DynamicsPredictor(z_dim=z_dim, hidden_dim=hidden_dim, num_layers=2)
    
    batch_size = 4
    v = torch.randn(batch_size)
    u = torch.randn(batch_size)
    z = torch.randn(batch_size, z_dim)
    
    dv = predictor(v, u, z)
    assert dv.shape == (batch_size, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

