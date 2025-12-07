"""Gym-compatible longitudinal control environment for SAC training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:  # pragma: no cover - exercised indirectly during CI
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback for classic gym
    import gym
    from gym import spaces

from utils.data_utils import ReferenceTrajectoryGenerator
from generator.adapter import create_reference_generator, extended_params_to_vehicle_capabilities
from utils.dynamics import (
    RandomizationConfig,
    VehicleParams,
    ExtendedPlant,
    ExtendedPlantParams,
    ExtendedPlantRandomization,
    ExtendedPlantState,
    aerodynamic_drag,
    grade_force,
    rolling_resistance,
    sample_extended_params,
    sample_sensor_noise,
    sample_vehicle_params,
)


ObservationFn = Callable[[int, np.random.Generator | None], np.ndarray]


@dataclass(slots=True)
class LongitudinalEnvConfig:
    """Hyper-parameters that define the longitudinal environment."""

    dt: float = 0.1
    max_episode_steps: int = 512
    action_low: float = -1.0
    action_high: float = 1.0
    max_speed: float = 40.0
    max_position: float = 5_000.0
    preview_horizon_s: float = 3.0
    use_extended_plant: bool = True
    plant_substeps: int = 2
    track_weight: float = 1.0
    horizon_penalty_weight: float = 0.0  # Weight for future horizon penalties (0 = only current, 1.0 = equal weight)
    horizon_penalty_decay: float = 0.9  # Exponential decay factor for future penalties (1.0 = no decay)
    jerk_weight: float = 0.1
    action_weight: float = 0.01
    voltage_weight: float = 1e-4
    brake_weight: float = 1e-3
    smooth_action_weight: float = 0.05
    negative_speed_weight: float = 1.0  # Penalty weight for negative vehicle speeds
    accel_filter_alpha: float = 0.1  # Exponential smoothing factor for acceleration (0 = no smoothing, 1 = instant)
    base_reward_clip: float = 10000.0
    # Deprecated parameters (kept for backward compatibility with config files)
    force_initial_speed_zero: bool = False  # Ignored - new generator handles feasibility
    post_feasibility_smoothing: bool = False  # Ignored - new generator handles feasibility
    post_feasibility_alpha: float = 0.8  # Ignored - new generator handles feasibility


class LongitudinalEnv(gym.Env):
    """Implements the longitudinal dynamics environment with extended plant."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: LongitudinalEnvConfig | None = None,
        *,
        randomization: RandomizationConfig | None = None,
        reference_generator: ReferenceTrajectoryGenerator | ObservationFn | None = None,
        generator_config: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or LongitudinalEnvConfig()
        self.randomization = randomization or RandomizationConfig()
        self.generator_config = generator_config
        self._rng = np.random.default_rng(seed)

        if reference_generator is None:
            # Generator will be initialized after preview_steps
            self._reference_callable: ObservationFn | None = None
        elif isinstance(reference_generator, ReferenceTrajectoryGenerator):
            self.reference_generator = reference_generator
            self._reference_callable = None
        else:
            self.reference_generator = None
            self._reference_callable = reference_generator

        self.preview_steps = max(int(round(self.config.preview_horizon_s / self.config.dt)), 1)
        if reference_generator is None:
            self.reference_generator = create_reference_generator(
                dt=self.config.dt,
                prediction_horizon=self.preview_steps,
                generator_config=self.generator_config
            )

        self.action_space = spaces.Box(
            low=np.array([self.config.action_low], dtype=np.float32),
            high=np.array([self.config.action_high], dtype=np.float32),
        )
        obs_dim = 1 + self.preview_steps  # [speed] + [current_ref + future_refs]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.params: VehicleParams | None = None
        self.extended_params: ExtendedPlantParams | None = None
        # Create extended randomization from config if available
        extended_config = {}
        if generator_config and 'vehicle_randomization' in generator_config:
            extended_config = generator_config
        self.extended_random = ExtendedPlantRandomization.from_config(extended_config)
        self.extended: ExtendedPlant | None = None
        self.reference: np.ndarray | None = None
        self.grade_profile: np.ndarray | None = None
        self._speed_noise_std: float = 0.05
        self._accel_noise_std: float = 0.1
        self._ref_idx: int = 0
        self._step_count: int = 0
        self.speed: float = 0.0
        self.prev_speed: float = 0.0
        self.position: float = 0.0
        self._prev_action: float = 0.0
        self._prev_accel: float = 0.0
        self._filtered_accel: float = 0.0  # Filtered acceleration for jerk calculation
        self._last_state: ExtendedPlantState | None = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def seed(self, seed: int | None = None) -> None:  # pragma: no cover
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}
        self.params = sample_vehicle_params(self._rng, self.randomization)
        self.extended_params = sample_extended_params(self._rng, self.extended_random)
        self._speed_noise_std, self._accel_noise_std = sample_sensor_noise(self._rng, self.randomization)

        reference_profile = options.get("reference_profile")
        if reference_profile is not None:
            profile = np.asarray(reference_profile, dtype=np.float32)
            if profile.ndim != 1 or profile.size < 2:
                raise ValueError("reference_profile must be a 1-D sequence with at least two entries")
            self.reference = profile
            # For custom reference profiles, assume flat grade
            self.grade_profile = np.zeros(len(profile), dtype=np.float32)
        else:
            profile_length = int(options.get("profile_length", self.config.max_episode_steps))
            profile_length = max(profile_length, 4)
            self.reference = self._generate_reference(profile_length)
        # Store reference profiles for backward compatibility
        self._original_reference = self.reference.copy()
        self._vehicle_caps = None  # Not used with new generator
        self._feasible_reference = self.reference.copy()

        initial_speed = float(options.get("initial_speed", self.reference[0]))
        self.speed = max(0.0, initial_speed)
        self.prev_speed = self.speed  # Start with consistent prev_speed
        self.position = 0.0
        self._prev_action = 0.0
        self._prev_accel = 0.0  # Start with zero acceleration
        self._filtered_accel = 0.0  # Start filtered accel at zero
        self._ref_idx = 0
        self._step_count = 0
        if self.config.use_extended_plant:
            self.extended = ExtendedPlant(self.extended_params)
            self._last_state = self.extended.reset(speed=self.speed, position=self.position)
        else:
            self.extended = None
            self._last_state = None

        obs = self._build_observation()
        info = {
            "reference_speed": float(self.reference[self._ref_idx]),
            "profile_feasible": True,
            "max_profile_adjustment": 0.0,
            "original_reference": self.reference.copy(),
            "feasible_reference": self.reference.copy(),
        }
        return obs, info

    def step(self, action: float | np.ndarray):
        assert self.reference is not None, "reset() must be called before step()"

        action_scalar = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        action_value = float(np.clip(action_scalar, self.config.action_low, self.config.action_high))
        plant_state: ExtendedPlantState
        if self.config.use_extended_plant and self.extended is not None:
            # Get current grade for this time step
            current_grade = float(self.grade_profile[self._ref_idx]) if self.grade_profile is not None else None
            plant_state = self.extended.step(action_value, self.config.dt, self.config.plant_substeps, grade_rad=current_grade)
            self.speed = plant_state.speed
            self.position = plant_state.position
            self._last_state = plant_state
            accel = plant_state.acceleration
        else:
            throttle_cmd = max(action_value, 0.0)
            brake_cmd = max(-action_value, 0.0)
            tau = max(self.params.actuator_tau, 1e-3)
            alpha = self.config.dt / tau
            self._prev_action += alpha * (action_value - self._prev_action)
            commanded_accel = throttle_cmd * 3.5 - brake_cmd * 6.0
            drag_force = aerodynamic_drag(self.speed, self.params)
            rolling_force = rolling_resistance(self.params)
            grade_force_value = grade_force(self.params)
            drive_force = self.params.mass * commanded_accel
            net_force = drive_force - drag_force - rolling_force - grade_force_value
            accel = net_force / self.params.mass

        # Apply exponential smoothing to acceleration
        alpha = self.config.accel_filter_alpha
        self._filtered_accel = alpha * accel + (1 - alpha) * self._filtered_accel

        # Compute jerk from filtered acceleration
        jerk = (self._filtered_accel - self._prev_accel) / max(self.config.dt, 1e-6)

        # Update speed and position (only for simple plant, extended plant handles this internally)
        if not self.config.use_extended_plant:
            self.speed = np.clip(self.speed + self.config.dt * accel, 0.0, self.config.max_speed * 1.5)
            self.position += self.config.dt * self.speed
            plant_state = ExtendedPlantState(
                speed=self.speed,
                position=self.position,
                acceleration=accel,
                wheel_speed=self.speed / 0.3,
                brake_torque=brake_cmd * 1000.0,
                slip_ratio=0.0,
                action=action_value,
                motor_current=0.0,
                back_emf_voltage=0.0,
                V_cmd=throttle_cmd * motor_V_max if 'motor_V_max' in globals() else 0.0,
                drive_torque=0.0,
                tire_force=0.0,
                drag_force=drag_force,
                rolling_force=rolling_force,
                grade_force=grade_force_value,
                net_force=net_force,
                held_by_brakes=False,
            )
            self._last_state = plant_state

        # Current speed tracking penalty
        speed_error = self.speed - float(self.reference[self._ref_idx])
        reward = -self.config.track_weight * (speed_error**2)

        # Horizon penalty: encourage anticipation of future speed changes
        # Penalize deviations from future reference speeds with exponential decay
        if self.config.horizon_penalty_weight > 0.0:
            horizon_penalty = 0.0
            decay_factor = 1.0

            for i in range(1, min(self.preview_steps + 1, len(self.reference) - self._ref_idx)):
                future_idx = self._ref_idx + i
                future_ref = float(self.reference[future_idx])

                # Penalize absolute deviation between current speed and future reference
                # This encourages being at the right speed for upcoming changes
                speed_deviation = abs(self.speed - future_ref)
                horizon_penalty -= decay_factor * speed_deviation

                # Apply exponential decay for future timesteps
                decay_factor *= self.config.horizon_penalty_decay

            reward += self.config.horizon_penalty_weight * horizon_penalty

        # Other penalties
        reward -= self.config.jerk_weight * abs(jerk)
        reward -= self.config.action_weight * (abs(action_value))
        reward -= self.config.smooth_action_weight * abs(action_value - self._prev_action)
        if self.config.use_extended_plant and plant_state is not None:
            reward -= self.config.brake_weight * abs(plant_state.brake_torque)

        # Negative speed penalty
        if self.speed < 0.0:
            reward -= self.config.negative_speed_weight * abs(self.speed)
        reward = float(np.clip(reward, -self.config.base_reward_clip, self.config.base_reward_clip))

        self.prev_speed = self.speed
        self._prev_accel = self._filtered_accel  # Use filtered accel for jerk calculation consistency
        self._prev_action = action_value
        self._step_count += 1

        # For fixed-length episodes, increment reference index or stay at last value
        if self._ref_idx < len(self.reference) - 1:
            self._ref_idx += 1
        # If we've reached the end, stay at the last reference value

        obs = self._build_observation()

        # For fixed-length episodes, only terminate when max steps reached
        # Remove early termination due to speed/position/reference limits
        terminated = False
        truncated = bool(self._step_count >= self.config.max_episode_steps)

        info = {
            "speed_error": speed_error,
            "reference_speed": float(self.reference[self._ref_idx]),
            "speed": plant_state.speed,
            "acceleration": self._filtered_accel,  # Use filtered acceleration for logging/plots
            "raw_acceleration": plant_state.acceleration,  # Keep raw for debugging if needed
            "wheel_speed": plant_state.wheel_speed,
            "brake_torque": plant_state.brake_torque,
            "brake_torque_max": self.extended_params.brake.T_br_max,  # Maximum brake torque
            "slip_ratio": plant_state.slip_ratio,
            "back_emf_voltage": plant_state.back_emf_voltage,
            "motor_current": plant_state.motor_current,
            "V_cmd": plant_state.V_cmd,
            "V_max": self.extended_params.motor.V_max,  # Max motor voltage for percentage calculation
            "action_value": action_value,
            "jerk": jerk,
            "drive_torque": plant_state.drive_torque,
            "tire_force": plant_state.tire_force,
            "drag_force": plant_state.drag_force,
            "rolling_force": plant_state.rolling_force,
            "grade_force": plant_state.grade_force,
            "net_force": plant_state.net_force,
        }

        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - visualization hook
        return {
            "speed": self.speed,
            "position": self.position,
            "reference_speed": None if self.reference is None else float(self.reference[self._ref_idx]),
        }

    def close(self):  # pragma: no cover - compatibility hook
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_reference(self, length: int) -> np.ndarray:
        if self._reference_callable is not None:
            profile = self._reference_callable(length, self._rng)
            # For callable references, assume flat grade
            self.grade_profile = np.zeros(length, dtype=np.float32)
        else:
            assert self.reference_generator is not None
            # Convert extended_params to VehicleCapabilities if available
            vehicle = None
            if self.extended_params is not None:
                import torch
                device = torch.device('cpu')  # Generator uses CPU by default
                vehicle = extended_params_to_vehicle_capabilities(self.extended_params, device=device)
            result = self.reference_generator.sample(length, self._rng, vehicle=vehicle)
            if isinstance(result, tuple):
                # New interface: returns (speed_profile, grade_profile)
                profile, grade_profile = result
                self.grade_profile = np.asarray(grade_profile, dtype=np.float32)
            else:
                # Legacy interface: only speed profile
                profile = result
                self.grade_profile = np.zeros(length, dtype=np.float32)

        if not isinstance(profile, np.ndarray):
            profile = np.asarray(profile, dtype=np.float32)
        return profile.astype(np.float32)

    def _build_observation(self) -> np.ndarray:
        assert self.reference is not None
        state = self._last_state
        speed_meas = (state.speed if state else self.speed) + self._rng.normal(0.0, self._speed_noise_std)

        # Only include speed and reference speeds (current + future)
        base = np.array([speed_meas], dtype=np.float32)

        preview = np.empty(self.preview_steps, dtype=np.float32)
        last_idx = len(self.reference) - 1
        for idx in range(self.preview_steps):
            ref_idx = min(self._ref_idx + idx, last_idx)
            preview[idx] = float(self.reference[ref_idx])
        return np.concatenate([base, preview]).astype(np.float32)


__all__ = ["LongitudinalEnv", "LongitudinalEnvConfig"]
