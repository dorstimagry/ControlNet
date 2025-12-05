"""Unit tests for the longitudinal environment."""

from __future__ import annotations

import numpy as np

from env.longitudinal_env import LongitudinalEnv, LongitudinalEnvConfig
from utils.data_utils import ReferenceTrajectoryGenerator
from utils.dynamics import RandomizationConfig, sample_sensor_noise, sample_vehicle_params


def test_env_reset_and_step_runs() -> None:
    """Ensure reset/step produce well-formed outputs."""

    generator = ReferenceTrajectoryGenerator(min_speed=2.0, max_speed=2.0)
    randomization = RandomizationConfig(
        mass_range=(1500.0, 1500.0),
        rolling_coeff_range=(0.01, 0.01),
        drag_area_range=(0.8, 0.8),
        actuator_tau_range=(0.1, 0.1),
        grade_range_deg=(0.0, 0.0),
        speed_noise_range=(0.01, 0.01),
        accel_noise_range=(0.02, 0.02),
    )
    env = LongitudinalEnv(
        LongitudinalEnvConfig(max_episode_steps=32, dt=0.05),
        randomization=randomization,
        reference_generator=generator,
        seed=7,
    )

    obs, info = env.reset()
    expected_dim = 1 + env.preview_steps
    assert obs.shape == (expected_dim,)
    assert "reference_speed" in info

    step_out = env.step(np.array([0.0], dtype=np.float32))
    assert len(step_out) == 5
    next_obs, reward, terminated, truncated, info = step_out
    assert next_obs.shape == (expected_dim,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "speed" in info
    assert "acceleration" in info
    assert "wheel_speed" in info
    assert "V_cmd" in info  # Commanded voltage (replaces throttle_angle)
    assert "motor_voltage" in info
    assert "action_value" in info


def test_randomization_sampling_within_bounds() -> None:
    """Randomized vehicle params and noise must respect configured ranges."""

    rng = np.random.default_rng(1234)
    config = RandomizationConfig(
        mass_range=(1000.0, 1200.0),
        rolling_coeff_range=(0.01, 0.015),
        drag_area_range=(0.7, 0.9),
        actuator_tau_range=(0.05, 0.06),
        grade_range_deg=(-1.0, 1.0),
        speed_noise_range=(0.02, 0.03),
        accel_noise_range=(0.05, 0.07),
    )

    params = sample_vehicle_params(rng, config)
    assert config.mass_range[0] <= params.mass <= config.mass_range[1]
    assert config.drag_area_range[0] <= params.drag_area <= config.drag_area_range[1]
    assert config.rolling_coeff_range[0] <= params.rolling_coeff <= config.rolling_coeff_range[1]
    assert config.actuator_tau_range[0] <= params.actuator_tau <= config.actuator_tau_range[1]
    speed_noise, accel_noise = sample_sensor_noise(rng, config)
    assert config.speed_noise_range[0] <= speed_noise <= config.speed_noise_range[1]
    assert config.accel_noise_range[0] <= accel_noise <= config.accel_noise_range[1]


def test_env_accepts_custom_reference_profile() -> None:
    env = LongitudinalEnv(LongitudinalEnvConfig(max_episode_steps=64), seed=3)
    profile = np.linspace(0.0, 5.0, num=50, dtype=np.float32)
    obs, info = env.reset(options={"reference_profile": profile, "initial_speed": 0.0})
    assert env.reference.shape[0] == 50
    assert info["reference_speed"] == profile[0]
    assert obs.shape == (1 + env.preview_steps,)


def test_profile_feasibility_integration() -> None:
    """Test that environment makes profiles feasible."""
    env = LongitudinalEnv(LongitudinalEnvConfig(max_episode_steps=64), seed=42)

    # Reset to generate a feasible profile
    obs, info = env.reset()

    # Check that feasibility info is included
    assert "profile_feasible" in info
    assert "max_profile_adjustment" in info
    assert "original_reference" in info
    assert "feasible_reference" in info

    # Check that profiles are stored
    assert hasattr(env, "_original_reference")
    assert hasattr(env, "_feasible_reference")
    assert hasattr(env, "_vehicle_caps")

    # Check that feasible profile is accessible
    assert len(env.reference) == len(env._original_reference)
    assert len(env.reference) == len(env._feasible_reference)


def test_profile_feasibility_custom_profile() -> None:
    """Test feasibility with custom reference profile."""
    env = LongitudinalEnv(LongitudinalEnvConfig(max_episode_steps=64), seed=3)

    # Create an aggressive profile that should be modified
    profile = np.linspace(0.0, 20.0, num=50, dtype=np.float32)  # Fast acceleration

    obs, info = env.reset(options={"reference_profile": profile})

    # Profile should be made feasible
    assert "profile_feasible" in info
    assert len(info["original_reference"]) == len(info["feasible_reference"])

    # Feasible profile should be different from original for aggressive profiles
    if not info["profile_feasible"]:
        assert info["max_profile_adjustment"] > 0.01  # Some adjustment made


def test_feasible_profile_bounds_check() -> None:
    """Test that feasible profiles respect vehicle acceleration limits."""
    from utils.data_utils import feasible_accel_bounds

    env = LongitudinalEnv(LongitudinalEnvConfig(max_episode_steps=64), seed=123)

    obs, info = env.reset()

    # Check that all accelerations in the feasible profile are within bounds
    dt = env.config.dt
    v_profile = env.reference
    grade_profile = np.zeros_like(v_profile)  # Assume flat for this test

    a_req = np.diff(v_profile) / dt

    violations = 0
    for k in range(len(a_req)):
        a_min, a_max = feasible_accel_bounds(
            v_profile[k], grade_profile[k], env._vehicle_caps, safety_margin=0.9
        )
        if not (a_min - 1e-3 <= a_req[k] <= a_max + 1e-3):
            violations += 1

    assert violations == 0, f"Found {violations} acceleration bound violations in feasible profile"


def test_post_feasibility_smoothing() -> None:
    """Test the post-feasibility smoothing feature."""
    # Test with post-feasibility smoothing enabled
    config_smooth = LongitudinalEnvConfig(
        max_episode_steps=50,
        post_feasibility_smoothing=True,
        post_feasibility_alpha=0.8
    )
    env_smooth = LongitudinalEnv(config_smooth, seed=42)

    # Test with smoothing disabled
    config_no_smooth = LongitudinalEnvConfig(
        max_episode_steps=50,
        post_feasibility_smoothing=False
    )
    env_no_smooth = LongitudinalEnv(config_no_smooth, seed=42)

    obs_smooth, _ = env_smooth.reset()
    obs_no_smooth, _ = env_no_smooth.reset()

    # Both should start at the same initial speed (feasibility may adjust)
    # But the smoothed version should have lower jerk

    dt = 0.1
    accel_smooth = np.diff(env_smooth.reference) / dt
    accel_no_smooth = np.diff(env_no_smooth.reference) / dt

    jerk_smooth = np.diff(accel_smooth) / dt if len(accel_smooth) > 1 else np.array([0])
    jerk_no_smooth = np.diff(accel_no_smooth) / dt if len(accel_no_smooth) > 1 else np.array([0])

    max_jerk_smooth = np.max(np.abs(jerk_smooth)) if len(jerk_smooth) > 0 else 0
    max_jerk_no_smooth = np.max(np.abs(jerk_no_smooth)) if len(jerk_no_smooth) > 0 else 0

    # Smoothing should reduce maximum jerk (though exact amount may vary)
    assert max_jerk_smooth <= max_jerk_no_smooth * 1.1, \
        f"Post-feasibility smoothing should reduce jerk: {max_jerk_smooth} vs {max_jerk_no_smooth}"

    # Test that config is properly set
    assert config_smooth.post_feasibility_smoothing == True
    assert config_smooth.post_feasibility_alpha == 0.8
    assert config_no_smooth.post_feasibility_smoothing == False


def test_force_initial_speed_zero() -> None:
    """Test the force_initial_speed_zero configuration option."""
    # Test with force_initial_speed_zero = True
    config = LongitudinalEnvConfig(max_episode_steps=50, force_initial_speed_zero=True)
    env = LongitudinalEnv(config, seed=42)

    obs, info = env.reset()

    # Should always start at zero speed
    assert env.reference[0] == 0.0, f"Initial reference speed should be 0, got {env.reference[0]}"
    assert info["adjusted_initial_speed"] == 0.0, "Adjusted initial speed should be 0"
    assert info["initial_target_adjusted"] == True, "Should report adjustment made"

    # Test with custom profile
    custom_profile = np.linspace(15.0, 10.0, 50)  # Start at 15 m/s
    obs2, info2 = env.reset(options={"reference_profile": custom_profile})

    assert env.reference[0] == 0.0, "Custom profile should also start at 0"
    assert info2["original_initial_speed"] == 15.0, "Should remember original speed"
    assert info2["adjusted_initial_speed"] == 0.0, "Should adjust to 0"

    # Test with force_initial_speed_zero = False (default)
    config_default = LongitudinalEnvConfig(max_episode_steps=50, force_initial_speed_zero=False)
    env_default = LongitudinalEnv(config_default, seed=42)

    obs3, info3 = env_default.reset()
    # Default behavior should not force zero (unless infeasible)
    # We can't predict exact value, but it should be recorded
    assert "original_initial_speed" in info3
    assert "adjusted_initial_speed" in info3


