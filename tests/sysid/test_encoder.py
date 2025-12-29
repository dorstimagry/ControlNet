"""Unit tests for SysID encoder components."""

import pytest
import torch
import numpy as np

from src.sysid import ContextEncoder, FeatureBuilder


def test_feature_builder_shapes():
    """Test that FeatureBuilder produces correct shapes."""
    builder = FeatureBuilder(dt=0.1)
    
    # Single step
    features = builder.build_features(v_t=5.0, u_t=0.5)
    assert features.shape == (4,)
    assert isinstance(features, torch.Tensor)
    
    # Features should be: [v_t, u_{t-1}, dv_t, du_{t-1}]
    assert features[0] == 5.0  # v_t
    
def test_feature_builder_batch():
    """Test batch feature building."""
    builder = FeatureBuilder(dt=0.1)
    
    batch_size = 8
    seq_len = 20
    v_seq = torch.randn(batch_size, seq_len)
    u_seq = torch.randn(batch_size, seq_len)
    
    features = builder.build_features_batch(v_seq, u_seq)
    assert features.shape == (batch_size, seq_len, 4)


def test_feature_builder_history():
    """Test that FeatureBuilder maintains correct history."""
    builder = FeatureBuilder(dt=0.1)
    builder.reset()
    
    # First step
    f1 = builder.build_features(v_t=1.0, u_t=0.1)
    # dv should be 10.0 (1.0 - 0.0) / 0.1, du should be 0.1 - 0.0
    assert abs(f1[2] - 10.0) < 1e-5
    
    # Second step
    f2 = builder.build_features(v_t=1.5, u_t=0.2)
    # dv should be 5.0 (1.5 - 1.0) / 0.1
    assert abs(f2[2] - 5.0) < 1e-5
    # u_prev should be 0.1
    assert abs(f2[1] - 0.1) < 1e-5


def test_encoder_reset():
    """Test encoder reset functionality."""
    encoder = ContextEncoder(input_dim=4, hidden_dim=32, z_dim=8)
    
    # Single batch
    h = encoder.reset(batch_size=1)
    assert h.shape == (1, 32)
    assert torch.allclose(h, torch.zeros_like(h))
    
    # Multiple batch
    h = encoder.reset(batch_size=5)
    assert h.shape == (5, 32)


def test_encoder_step():
    """Test single-step encoder update."""
    encoder = ContextEncoder(input_dim=4, hidden_dim=32, z_dim=8)
    
    h_prev = encoder.reset(batch_size=1)
    o_t = torch.randn(1, 4)
    
    h_t, z_t = encoder.step(o_t, h_prev)
    
    assert h_t.shape == (1, 32)
    assert z_t.shape == (1, 8)
    
    # Hidden state should have changed
    assert not torch.allclose(h_t, h_prev)


def test_encoder_forward_sequence():
    """Test encoder forward pass on sequences."""
    encoder = ContextEncoder(input_dim=4, hidden_dim=32, z_dim=8)
    
    batch_size = 4
    seq_len = 10
    o_seq = torch.randn(batch_size, seq_len, 4)
    
    h_seq, z_seq, h_final = encoder(o_seq)
    
    assert h_seq.shape == (batch_size, seq_len, 32)
    assert z_seq.shape == (batch_size, seq_len, 8)
    assert h_final.shape == (batch_size, 32)
    
    # Final hidden should match last in sequence
    assert torch.allclose(h_final, h_seq[:, -1, :])


def test_encoder_gradients():
    """Test that encoder produces gradients."""
    encoder = ContextEncoder(input_dim=4, hidden_dim=32, z_dim=8)
    
    o_seq = torch.randn(2, 5, 4, requires_grad=True)
    _, z_seq, _ = encoder(o_seq)
    
    loss = z_seq.sum()
    loss.backward()
    
    # Check that encoder parameters have gradients
    for param in encoder.parameters():
        assert param.grad is not None
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad))


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [5, 20, 50])
def test_encoder_different_sizes(batch_size, seq_len):
    """Test encoder with different batch sizes and sequence lengths."""
    encoder = ContextEncoder(input_dim=4, hidden_dim=32, z_dim=8)
    
    o_seq = torch.randn(batch_size, seq_len, 4)
    h_seq, z_seq, h_final = encoder(o_seq)
    
    assert h_seq.shape == (batch_size, seq_len, 32)
    assert z_seq.shape == (batch_size, seq_len, 8)
    assert h_final.shape == (batch_size, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

