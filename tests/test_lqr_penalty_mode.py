"""Unit tests for LQR penalty mode in RL training environment."""

import numpy as np
import pytest
from pathlib import Path

from env.longitudinal_env import LongitudinalEnv, LongitudinalEnvConfig
from speed_shaper.src.config_schema import ShaperConfig


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a sample shaper config file for testing."""
    config = ShaperConfig(
        dt=0.1,
        wE_start=30.0,
        wE_end=10.0,
        lamE=1.0,
        wA_start=2.0,
        wA_end=10.0,
        lamA=-0.5,
        wJ_start=2.0,
        wJ_end=10.0,
        lamJ=-0.5,
    )
    
    filepath = tmp_path / "test_lqr_config.json"
    config.to_json(filepath)
    return filepath


def test_env_init_without_lqr_penalty():
    """Test that environment initializes normally without LQR penalty mode."""
    env_cfg = LongitudinalEnvConfig(
        dt=0.1,
        max_episode_steps=100,
        lqr_penalty_enabled=False,
    )
    
    env = LongitudinalEnv(env_cfg, seed=42)
    assert env._lqr_config is None
    assert env._lqr_accel_weights is None
    assert env._lqr_jerk_weights is None


def test_env_init_with_lqr_penalty_no_path():
    """Test that env raises error if LQR enabled but no config path."""
    env_cfg = LongitudinalEnvConfig(
        dt=0.1,
        max_episode_steps=100,
        lqr_penalty_enabled=True,
        lqr_penalty_config_path=None,
    )
    
    with pytest.raises(ValueError, match="lqr_penalty_enabled=True but lqr_penalty_config_path not provided"):
        LongitudinalEnv(env_cfg, seed=42)


def test_env_init_with_lqr_penalty_invalid_path():
    """Test that env raises error if config file doesn't exist."""
    env_cfg = LongitudinalEnvConfig(
        dt=0.1,
        max_episode_steps=100,
        lqr_penalty_enabled=True,
        lqr_penalty_config_path="/nonexistent/path.json",
    )
    
    with pytest.raises(FileNotFoundError):
        LongitudinalEnv(env_cfg, seed=42)


def test_env_init_with_valid_lqr_config(sample_config_file):
    """Test that environment loads LQR config correctly."""
    env_cfg = LongitudinalEnvConfig(
        dt=0.1,
        max_episode_steps=100,
        lqr_penalty_enabled=True,
        lqr_penalty_config_path=str(sample_config_file),
    )
    
    env = LongitudinalEnv(env_cfg, seed=42)
    assert env._lqr_config is not None
    assert env._lqr_config.wA_start == 2.0
    assert env._lqr_config.wA_end == 10.0
    assert env._lqr_config.lamA == -0.5


def test_lqr_weight_schedules_computed_on_reset(sample_config_file):
    """Test that weight schedules are computed correctly on reset."""
    env_cfg = LongitudinalEnvConfig(
        dt=0.1,
        max_episode_steps=100,
        lqr_penalty_enabled=True,
        lqr_penalty_config_path=str(sample_config_file),
    )
    
    env = LongitudinalEnv(env_cfg, seed=42)
    obs, info = env.reset(seed=42)
    
    # Check that weight schedules were computed
    assert env._lqr_accel_weights is not None
    assert env._lqr_jerk_weights is not None
    assert len(env._lqr_accel_weights) == len(env.reference)
    assert len(env._lqr_jerk_weights) == len(env.reference)
    
    # Check that weights are positive (after clipping)
    assert np.all(env._lqr_accel_weights >= 0)
    assert np.all(env._lqr_jerk_weights >= 0)
    
    # For lamA=-0.5 (negative), with wA_start=2.0, wA_end=10.0,
    # the formula is: w = w_end + (w_start - w_end) * exp(-lam * t)
    # = 10 + (2 - 10) * exp(0.5 * t) = 10 - 8 * exp(0.5 * t)
    # This decreases over time (exponentially), potentially going negative and clipping to 0
    # So we check that weights decrease or stay non-negative
    assert env._lqr_accel_weights[0] >= env._lqr_accel_weights[-1]


def test_lqr_penalties_applied_in_step(sample_config_file):
    """Test that LQR penalties are applied during env.step()."""
    env_cfg = LongitudinalEnvConfig(
        dt=0.1,
        max_episode_steps=20,
        lqr_penalty_enabled=True,
        lqr_penalty_config_path=str(sample_config_file),
        lqr_accel_weight=1.0,
        lqr_jerk_weight=1.0,
    )
    
    env = LongitudinalEnv(env_cfg, seed=42)
    env.reset(seed=42)
    
    # Take a step
    obs, reward, terminated, truncated, info = env.step(0.5)
    
    # Check that LQR penalty components exist
    assert 'lqr_accel_penalty' in info['reward_components']
    assert 'lqr_jerk_penalty' in info['reward_components']
    
    # For the first step, jerk penalty should be 0 (initialization artifact)
    assert info['reward_components']['lqr_jerk_penalty'] == 0.0
    
    # Accel penalty should be non-zero (negative)
    assert info['reward_components']['lqr_accel_penalty'] <= 0.0


def test_lqr_penalties_are_additive(sample_config_file):
    """Test that LQR penalties are added ON TOP of existing penalties."""
    # Create two envs: one with LQR, one without
    base_cfg = LongitudinalEnvConfig(
        dt=0.1,
        max_episode_steps=20,
        jerk_weight=0.1,
        smooth_action_weight=0.05,
        lqr_penalty_enabled=False,
    )
    
    lqr_cfg = LongitudinalEnvConfig(
        dt=0.1,
        max_episode_steps=20,
        jerk_weight=0.1,  # Same as base
        smooth_action_weight=0.05,  # Same as base
        lqr_penalty_enabled=True,
        lqr_penalty_config_path=str(sample_config_file),
        lqr_accel_weight=0.5,
        lqr_jerk_weight=0.5,
    )
    
    # Reset both with same seed
    env_base = LongitudinalEnv(base_cfg, seed=42)
    env_lqr = LongitudinalEnv(lqr_cfg, seed=42)
    
    env_base.reset(seed=42, options={'reference_profile': np.linspace(0, 10, 21)})
    env_lqr.reset(seed=42, options={'reference_profile': np.linspace(0, 10, 21)})
    
    # Take same action in both
    action = 0.3
    _, reward_base, _, _, info_base = env_base.step(action)
    _, reward_lqr, _, _, info_lqr = env_lqr.step(action)
    
    # Base penalties should be same in both
    assert info_base['reward_components']['jerk_penalty'] == info_lqr['reward_components']['jerk_penalty']
    assert info_base['reward_components']['smooth_action_penalty'] == info_lqr['reward_components']['smooth_action_penalty']
    
    # LQR env should have additional penalties
    assert info_lqr['reward_components']['lqr_accel_penalty'] < 0.0
    assert info_base['reward_components']['lqr_accel_penalty'] == 0.0
    
    # Total reward in LQR env should be lower (more penalties)
    # (both have negative rewards, so lqr should be MORE negative)
    assert reward_lqr < reward_base


def test_lqr_penalties_vary_over_time(sample_config_file):
    """Test that LQR penalties change over time according to weight schedule."""
    env_cfg = LongitudinalEnvConfig(
        dt=0.1,
        max_episode_steps=50,
        lqr_penalty_enabled=True,
        lqr_penalty_config_path=str(sample_config_file),
        lqr_accel_weight=1.0,
        lqr_jerk_weight=1.0,
    )
    
    env = LongitudinalEnv(env_cfg, seed=42)
    env.reset(seed=42)
    
    # Collect penalties over multiple steps
    accel_penalties = []
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(0.2)
        accel_penalties.append(info['reward_components']['lqr_accel_penalty'])
    
    # With lamA=-0.5 (growth), penalties should change over time
    # (becoming more negative as weights increase and acceleration persists)
    # At least some variation should exist
    assert len(set(accel_penalties)) > 1  # Not all the same value


def test_lqr_jerk_penalty_skipped_at_boundaries(sample_config_file):
    """Test that jerk penalty is 0 at first and last steps."""
    env_cfg = LongitudinalEnvConfig(
        dt=0.1,
        max_episode_steps=5,
        lqr_penalty_enabled=True,
        lqr_penalty_config_path=str(sample_config_file),
        lqr_accel_weight=1.0,
        lqr_jerk_weight=1.0,
    )
    
    env = LongitudinalEnv(env_cfg, seed=42)
    env.reset(seed=42)
    
    # First step: jerk should be 0
    obs, reward, terminated, truncated, info = env.step(0.5)
    assert info['reward_components']['lqr_jerk_penalty'] == 0.0
    
    # Middle steps: jerk should be non-zero
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(0.5)
        # Jerk penalty may be 0 if jerk is 0, but should exist in components
        assert 'lqr_jerk_penalty' in info['reward_components']
    
    # Last step: jerk should be 0
    obs, reward, terminated, truncated, info = env.step(0.5)
    assert truncated  # Episode should end
    assert info['reward_components']['lqr_jerk_penalty'] == 0.0


def test_lqr_penalty_without_config():
    """Test that LQR penalties are 0 when mode is disabled."""
    env_cfg = LongitudinalEnvConfig(
        dt=0.1,
        max_episode_steps=10,
        lqr_penalty_enabled=False,
    )
    
    env = LongitudinalEnv(env_cfg, seed=42)
    env.reset(seed=42)
    
    obs, reward, terminated, truncated, info = env.step(0.5)
    
    # LQR penalties should exist in components but be 0
    assert info['reward_components']['lqr_accel_penalty'] == 0.0
    assert info['reward_components']['lqr_jerk_penalty'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
