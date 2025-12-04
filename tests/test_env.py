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


