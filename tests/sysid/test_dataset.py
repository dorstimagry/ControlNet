"""Unit tests for SysID dataset and sequence sampling."""

import pytest
import numpy as np

from src.sysid.dataset import find_valid_anchor_indices, sample_sequences


def test_find_valid_anchor_indices_simple():
    """Test finding valid anchors in simple case."""
    # Single episode
    episode_ids = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    step_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    burn_in = 2
    horizon = 3
    buffer_size = 10
    
    valid = find_valid_anchor_indices(episode_ids, step_indices, burn_in, horizon, buffer_size)
    
    # Valid anchors: 2, 3, 4, 5, 6
    # (need 2 steps before, 3 steps after, plus 1 for v[t+H])
    assert len(valid) > 0
    assert 2 in valid
    assert 6 in valid
    assert 0 not in valid  # Too early
    assert 7 not in valid  # Too late (7 + 3 >= 10)


def test_find_valid_anchor_indices_multiple_episodes():
    """Test with multiple episodes."""
    # Two episodes
    episode_ids = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    step_indices = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    burn_in = 1
    horizon = 2
    buffer_size = 10
    
    valid = find_valid_anchor_indices(episode_ids, step_indices, burn_in, horizon, buffer_size)
    
    # Episode 0: valid anchors at 1, 2
    # Episode 1: valid anchors at 6, 7
    assert 1 in valid
    assert 2 in valid
    assert 6 in valid
    assert 7 in valid
    
    # Should not cross episode boundaries
    assert 4 not in valid  # Would cross into episode 1
    assert 5 not in valid  # Would need data from episode 0


def test_sample_sequences_basic():
    """Test basic sequence sampling."""
    buffer_size = 100
    burn_in = 5
    horizon = 10
    batch_size = 4
    
    # Create single episode data
    speed_buf = np.linspace(0, 10, buffer_size).astype(np.float32)
    action_buf = np.random.randn(buffer_size, 1).astype(np.float32)
    episode_id_buf = np.zeros(buffer_size, dtype=np.int32)
    step_in_episode_buf = np.arange(buffer_size, dtype=np.int32)
    
    batch = sample_sequences(
        speed_buf=speed_buf,
        action_buf=action_buf,
        episode_id_buf=episode_id_buf,
        step_in_episode_buf=step_in_episode_buf,
        buffer_size=buffer_size,
        batch_size=batch_size,
        burn_in=burn_in,
        horizon=horizon,
        rng=np.random.default_rng(42)
    )
    
    assert batch is not None
    assert batch.v_seq.shape == (batch_size, burn_in + horizon + 1)
    assert batch.u_seq.shape == (batch_size, burn_in + horizon)
    assert batch.burn_in == burn_in
    assert batch.horizon == horizon


def test_sample_sequences_insufficient_data():
    """Test that sampling returns None when insufficient data."""
    buffer_size = 10
    burn_in = 5
    horizon = 10
    batch_size = 4
    
    # Not enough data for the requested horizon
    speed_buf = np.zeros(buffer_size, dtype=np.float32)
    action_buf = np.zeros((buffer_size, 1), dtype=np.float32)
    episode_id_buf = np.zeros(buffer_size, dtype=np.int32)
    step_in_episode_buf = np.arange(buffer_size, dtype=np.int32)
    
    batch = sample_sequences(
        speed_buf=speed_buf,
        action_buf=action_buf,
        episode_id_buf=episode_id_buf,
        step_in_episode_buf=step_in_episode_buf,
        buffer_size=buffer_size,
        batch_size=batch_size,
        burn_in=burn_in,
        horizon=horizon
    )
    
    # Should return None (not enough valid sequences)
    assert batch is None


def test_sample_sequences_episode_boundaries():
    """Test that sequences don't cross episode boundaries."""
    # Two short episodes
    buffer_size = 20
    episode_ids = np.array([0]*10 + [1]*10, dtype=np.int32)
    step_indices = np.array(list(range(10)) + list(range(10)), dtype=np.int32)
    
    speed_buf = np.arange(buffer_size, dtype=np.float32)
    action_buf = np.ones((buffer_size, 1), dtype=np.float32)
    
    burn_in = 2
    horizon = 3
    batch_size = 2
    
    batch = sample_sequences(
        speed_buf=speed_buf,
        action_buf=action_buf,
        episode_id_buf=episode_ids,
        step_in_episode_buf=step_indices,
        buffer_size=buffer_size,
        batch_size=batch_size,
        burn_in=burn_in,
        horizon=horizon,
        rng=np.random.default_rng(42)
    )
    
    if batch is not None:
        # Check that each sequence is from a single episode
        for i in range(batch_size):
            anchor = batch.anchor_idx[i].item()
            episode_id = episode_ids[anchor]
            
            # All indices in window should have same episode ID
            window_start = anchor - burn_in
            window_end = anchor + horizon + 1
            window_episodes = episode_ids[window_start:window_end]
            
            assert np.all(window_episodes == episode_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

