"""Gym-compatible longitudinal control environment for SAC training."""

from __future__ import annotations

from collections import deque
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
    zero_speed_error_weight: float = 0.25  # Penalty weight for speed error when target speed is 0
    zero_speed_throttle_weight: float = 0.25  # Penalty weight for throttle usage when target speed is 0
    accel_filter_alpha: float = 0.1  # Exponential smoothing factor for acceleration (0 = no smoothing, 1 = instant)
    base_reward_clip: float = 10000.0
    # Oscillation and overshoot penalty configuration
    oscillation_weight: float = 0.0  # Weight for oscillatory switching penalty (λ_osc)
    overshoot_weight: float = 0.0  # Weight for overshoot crossing penalty (λ_over)
    oscillation_epsilon: float = 0.05  # Smoothing parameter for action sign (ε)
    oscillation_error_scale: float = 0.3  # Error scale for proximity gate (e_s in m/s)
    oscillation_ref_scale: float = 0.3  # Reference rate scale for stationarity gate (r_s in m/s²)
    overshoot_crossing_scale: float = 0.02  # Crossing detection scale (c_s)
    # Comfort penalty annealing configuration
    comfort_anneal_enabled: bool = False  # Enable weight annealing for jerk/smooth_action penalties
    comfort_anneal_start_mult: float = 0.0  # Starting multiplier (0 = no penalty at start)
    comfort_anneal_end_mult: float = 1.0  # Ending multiplier (1 = full penalty at end)
    comfort_anneal_steps: int = 500000  # Number of steps to anneal over
    # Violent profile training mode configuration
    violent_profile_mode: bool = False  # Enable/disable violent profile mode (non-smooth profiles in observations)
    filter_reward_mode: bool = False  # Enable/disable filter reward mode (standard profiles in observations, filtered profiles for reward)
    reward_filter_use_lookahead: bool = False  # Enable/disable look-ahead mode for reward filter
    reward_filter_lookahead_s: float = 3.0  # Look-ahead time for reward filter (seconds, must be <= preview_horizon_s)
    reward_filter_freq_cutoff: float = 0.6  # Filter cutoff frequency for reward (Hz)
    reward_filter_zeta: float = 0.9  # Filter damping ratio for reward
    reward_filter_dt: float = 0.1  # Filter timestep for reward (should match env dt)
    reward_filter_rate_max: float = 15.0  # Default max acceleration for reward filter (m/s²) - overridden per episode based on vehicle
    reward_filter_rate_neg_max: float = 20.0  # Default max deceleration for reward filter (m/s²) - overridden per episode based on vehicle
    reward_filter_jerk_max: float = 12.0  # Default max jerk for reward filter (m/s³)
    
    # Initial speed randomization
    initial_speed_error_range: float = 0.0  # Maximum absolute error for random initial speed (m/s). When > 0, initial speed is sampled from [reference[0] - error_range, reference[0] + error_range]. 0.0 = disabled (use reference[0])
    
    # Initial action randomization
    randomize_initial_action: bool = False  # If True, initialize the first previous action to a random value in [action_low, action_high]. If False, initialize to 0.0.
    
    # Sliding-window mean speed error penalty (bias elimination)
    bias_penalty_enabled: bool = False  # Enable sliding-window mean error penalty
    bias_penalty_window_s: float = 2.0  # Window length in seconds (T_window)
    bias_penalty_weight: float = 0.0  # Weight for bias penalty (w_bias), typically 0.1-1.0 × track_weight
    bias_penalty_v_scale: float = 15.0  # Speed scale for normalization (typical operating speed)
    bias_penalty_v_ref_dot_threshold: float = 0.2  # Reference speed change threshold for gating (m/s²)
    bias_penalty_a_threshold: float = 0.3  # Vehicle acceleration threshold for gating (m/s²)
    
    # Maximum absolute acceleration and jerk penalties (H-infinity norm)
    max_acc_penalty_weight: float = 0.0  # Weight for maximum absolute acceleration penalty (only when |acc| > |target_acc|)
    max_jerk_penalty_weight: float = 0.0  # Weight for maximum absolute jerk penalty (only when |jerk| > |target_jerk|)
    
    # LQR penalty mode (speed shaper integration)
    lqr_penalty_enabled: bool = False  # Enable LQR penalty mode using speed shaper config
    lqr_penalty_config_path: str | None = None  # Path to speed shaper config JSON file
    lqr_error_weight: float = 0.1  # Global multiplier for LQR error (tracking) penalty
    lqr_accel_weight: float = 0.1  # Global multiplier for LQR acceleration penalty
    lqr_jerk_weight: float = 0.1  # Global multiplier for LQR jerk penalty
    
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
        
        # Validate reward filter lookahead doesn't exceed preview horizon
        if self.config.reward_filter_use_lookahead:
            if self.config.reward_filter_lookahead_s > self.config.preview_horizon_s:
                raise ValueError(
                    f"reward_filter_lookahead_s ({self.config.reward_filter_lookahead_s}) "
                    f"cannot exceed preview_horizon_s ({self.config.preview_horizon_s})"
                )
        
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
        # Observation: [speed, prev_speed, prev_prev_speed, prev_action, speed_error, grade] + [refs]
        # This allows the agent to estimate acceleration, jerk, and action smoothness
        # Speed error is always included regardless of mode
        # Grade (road grade) is included to help the agent account for road slope
        obs_dim = 6 + self.preview_steps
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.params: VehicleParams | None = None
        self.extended_params: ExtendedPlantParams | None = None
        # Create extended randomization from config if available
        extended_config = {}
        if generator_config and 'vehicle_randomization' in generator_config:
            extended_config = generator_config
        self.extended_random = ExtendedPlantRandomization.from_config(extended_config)
        self.extended: ExtendedPlant | None = None
        self.reference: np.ndarray | None = None  # Current reference (raw or filtered based on mode)
        self.raw_reference: np.ndarray | None = None  # Raw (non-smooth) profile
        self.filtered_reference: np.ndarray | None = None  # Filtered profile for reward
        self.grade_profile: np.ndarray | None = None
        self._speed_noise_std: float = 0.05
        self._accel_noise_std: float = 0.1
        self._ref_idx: int = 0
        self._step_count: int = 0
        self._global_step_count: int = 0  # Global step counter for weight annealing (persists across episodes)
        self.speed: float = 0.0
        self.prev_speed: float = 0.0
        self._prev_prev_speed: float = 0.0  # Speed from 2 steps ago (for jerk estimation)
        self.position: float = 0.0
        self._prev_action: float = 0.0
        self._prev_accel: float = 0.0
        self._filtered_accel: float = 0.0  # Filtered acceleration for jerk calculation
        self._prev_jerk: float = 0.0  # Previous jerk value (for last step continuity)
        self._prev_target_jerk: float = 0.0  # Previous target jerk value (for last step continuity)
        self._last_state: ExtendedPlantState | None = None
        # Tracking variables for oscillation and overshoot penalties
        self._prev_action_sign: float = 0.0  # Previous smooth action sign for oscillation penalty
        self._prev_error: float = 0.0  # Previous tracking error for overshoot penalty
        self._prev_ref_speed: float = 0.0  # Previous reference speed for stationarity gate
        
        # Initialize sliding-window error buffer for bias penalty
        self._bias_error_buffer: deque[float] | None = None
        self._bias_window_steps: int = 0
        if self.config.bias_penalty_enabled:
            self._bias_window_steps = max(1, int(self.config.bias_penalty_window_s / self.config.dt))
            self._bias_error_buffer = deque(maxlen=self._bias_window_steps)
        
        # Initialize reward filter if violent profile mode or filter reward mode is enabled
        self._reward_filter = None
        if self.config.violent_profile_mode or self.config.filter_reward_mode:
            from generator.lpf import SecondOrderLPF
            import torch
            # Create a single-element batch filter for reward filtering
            # Use config values as defaults (will be updated per episode based on vehicle capabilities)
            rate_max = torch.tensor([self.config.reward_filter_rate_max], device=torch.device('cpu'))
            rate_neg_max = torch.tensor([self.config.reward_filter_rate_neg_max], device=torch.device('cpu'))
            jerk_max = torch.tensor([self.config.reward_filter_jerk_max], device=torch.device('cpu'))
            self._reward_filter = SecondOrderLPF(
                batch_size=1,
                freq_cutoff=self.config.reward_filter_freq_cutoff,
                zeta=self.config.reward_filter_zeta,
                dt=self.config.reward_filter_dt,
                rate_max=rate_max,
                rate_neg_max=rate_neg_max,
                jerk_max=jerk_max,
                device=torch.device('cpu')
            )
        
        # Load LQR penalty configuration if enabled
        self._lqr_config: 'ShaperConfig | None' = None
        self._lqr_error_weights: np.ndarray | None = None
        self._lqr_accel_weights: np.ndarray | None = None
        self._lqr_jerk_weights: np.ndarray | None = None
        
        if self.config.lqr_penalty_enabled:
            if not self.config.lqr_penalty_config_path:
                raise ValueError("lqr_penalty_enabled=True but lqr_penalty_config_path not provided")
            from speed_shaper.src.config_schema import ShaperConfig
            from pathlib import Path
            config_path = Path(self.config.lqr_penalty_config_path)
            self._lqr_config = ShaperConfig.from_json(config_path)
            # Weight schedules will be computed per-episode in reset()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def seed(self, seed: int | None = None) -> None:  # pragma: no cover
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def set_global_step(self, step: int) -> None:
        """Set the global step count for weight annealing.
        
        Call this from the training loop to update the annealing progress.
        
        Args:
            step: Current global training step count
        """
        self._global_step_count = step

    def _compute_vehicle_lpf_limits(self, vehicle) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute conservative LPF rate limits based on vehicle feasibility constraints.
        
        Uses representative operating conditions to compute conservative bounds that ensure
        feasibility across typical driving scenarios. Similar to BatchTargetGenerator._compute_vehicle_lpf_limits().
        
        Args:
            vehicle: VehicleCapabilities object
            
        Returns:
            Tuple of (rate_max, rate_neg_max) tensors [1] (m/s²)
        """
        import torch
        # Representative conditions for conservative limit calculation
        v_rep = torch.full_like(vehicle.m, 15.0)  # Representative speed (m/s) - highway cruising
        grade_rep = torch.zeros_like(vehicle.m)    # Flat road assumption

        # Compute resistive forces at representative conditions
        F_drag = 0.5 * vehicle.C_dA * v_rep**2  # Aerodynamic drag
        F_roll = vehicle.C_r * vehicle.m * 9.81  # Rolling resistance
        F_resist = F_drag + F_roll  # Total resistance (flat road)

        # Maximum drive acceleration (torque-limited, conservative)
        T_wheel_max = vehicle.i_max * vehicle.K_t * vehicle.eta_gb * vehicle.gear_ratio
        a_drive_max_raw = (T_wheel_max / vehicle.r_w - F_resist) / vehicle.m

        # Maximum braking acceleration (regenerative + mechanical braking)
        # Use mechanical braking as conservative limit (regenerative may vary)
        a_brake_min_raw = -vehicle.T_brake_max / vehicle.r_w / vehicle.m

        # Apply safety margins for conservative operation
        safety_factor = 0.85  # Conservative safety margin

        # Ensure reasonable bounds and positive values
        rate_max = torch.clamp(a_drive_max_raw * safety_factor, min=1.0, max=15.0)
        rate_neg_max = torch.clamp(-a_brake_min_raw * safety_factor, min=1.0, max=20.0)

        return rate_max, rate_neg_max

    def get_comfort_anneal_multiplier(self) -> float:
        """Compute the current annealing multiplier for comfort penalties.
        
        Returns a value that linearly interpolates from start_mult to end_mult
        over the configured annealing steps.
        
        Returns:
            float: Current multiplier in [start_mult, end_mult]
        """
        if not self.config.comfort_anneal_enabled:
            return 1.0  # No annealing, use full weights
        
        start = self.config.comfort_anneal_start_mult
        end = self.config.comfort_anneal_end_mult
        total_steps = max(self.config.comfort_anneal_steps, 1)
        
        # Linear interpolation, clamped to [start, end]
        progress = min(self._global_step_count / total_steps, 1.0)
        return start + progress * (end - start)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}
        self.params = sample_vehicle_params(self._rng, self.randomization)
        self.extended_params = sample_extended_params(self._rng, self.extended_random)
        self._speed_noise_std, self._accel_noise_std = sample_sensor_noise(self._rng, self.randomization)

        # Convert extended_params to VehicleCapabilities if available (for reward filter and max speed enforcement)
        vehicle = None
        if self.extended_params is not None:
            import torch
            device = torch.device('cpu')
            vehicle = extended_params_to_vehicle_capabilities(self.extended_params, device=device)
            
            # Update reward filter limits based on vehicle capabilities if violent mode or filter reward mode is enabled
            if (self.config.violent_profile_mode or self.config.filter_reward_mode) and self._reward_filter is not None:
                # Compute vehicle-specific LPF limits
                rate_max, rate_neg_max = self._compute_vehicle_lpf_limits(vehicle)
                
                # Update reward filter limits (keep jerk_max from config)
                self._reward_filter.rate_max = rate_max.to(self._reward_filter.device)
                self._reward_filter.rate_neg_max = rate_neg_max.to(self._reward_filter.device)

        reference_profile = options.get("reference_profile")
        if reference_profile is not None:
            profile = np.asarray(reference_profile, dtype=np.float32)
            if profile.ndim != 1 or profile.size < 2:
                raise ValueError("reference_profile must be a 1-D sequence with at least two entries")
            # For custom reference profiles, use same profile for both raw and filtered
            self.raw_reference = profile.copy()
            self.filtered_reference = profile.copy()
            self.reference = profile  # Will be set based on mode below
            # For custom reference profiles, assume flat grade
            self.grade_profile = np.zeros(len(profile), dtype=np.float32)
        else:
            profile_length = int(options.get("profile_length", self.config.max_episode_steps))
            profile_length = max(profile_length, 4)
            filtered_profile, grade_profile, raw_profile = self._generate_reference(profile_length)
            self.raw_reference = raw_profile
            self.filtered_reference = filtered_profile
            self.grade_profile = grade_profile
        
        # Enforce max speed on raw profile before filtering (if violent mode and vehicle available)
        if self.config.violent_profile_mode and vehicle is not None:
            from generator.feasibility import compute_max_feasible_speed
            
            # Get speed_limit_safety_factor from generator config
            speed_limit_safety_factor = 0.8  # Default
            if self.generator_config and 'generator' in self.generator_config:
                speed_limit_safety_factor = self.generator_config['generator'].get('speed_limit_safety_factor', 0.8)
            
            # Compute max feasible speed with safety factor
            v_max_theoretical = compute_max_feasible_speed(vehicle)  # [B] tensor
            v_max_feasible = speed_limit_safety_factor * v_max_theoretical.item()  # Scalar
            
            # Clamp raw profile to [0, v_max_feasible]
            self.raw_reference = np.clip(self.raw_reference, 0.0, v_max_feasible)
        
        # Set reference based on mode (before determining initial speed)
        if self.config.violent_profile_mode:
            self.reference = self.raw_reference
        else:
            self.reference = self.filtered_reference
        
        # Store reference profiles for backward compatibility
        self._original_reference = self.reference.copy()
        self._vehicle_caps = None  # Not used with new generator
        self._feasible_reference = self.reference.copy()

        # Determine initial speed: explicit override > randomization > default
        if "initial_speed" in options:
            # Explicit override - use provided value
            initial_speed = float(options["initial_speed"])
        elif self.config.initial_speed_error_range > 0:
            # Randomize around reference[0]
            error = self._rng.uniform(-self.config.initial_speed_error_range, 
                                       self.config.initial_speed_error_range)
            initial_speed = self.reference[0] + error
        else:
            # Default behavior
            initial_speed = self.reference[0]
        
        # Apply reward filter based on mode (after initial speed is determined)
        # Initialize filter with vehicle's initial speed for smooth error closure
        if self.config.filter_reward_mode and self._reward_filter is not None:
            # Filter reward mode: filter the standard generator profile for reward calculation
            # Standard filtered_profile is used for observations, filtered version for rewards
            import torch
            standard_profile_tensor = torch.from_numpy(self.filtered_reference).unsqueeze(0)  # [1, T]
            filtered_tensor = torch.zeros_like(standard_profile_tensor)
            
            # Reset filter state with vehicle's initial speed (not reference speed)
            # This ensures smooth closure of initial speed error
            initial_y = torch.tensor([[initial_speed]], device=torch.device('cpu'), dtype=torch.float32)
            self._reward_filter.reset(initial_y=initial_y)
            
            # Determine look-ahead steps (within preview horizon)
            lookahead_steps = 0
            if self.config.reward_filter_use_lookahead:
                # Convert lookahead time to steps, but don't exceed preview horizon
                lookahead_steps_float = self.config.reward_filter_lookahead_s / self.config.dt
                max_lookahead_steps = min(self.preview_steps, len(self.filtered_reference) - 1)
                lookahead_steps = min(int(lookahead_steps_float), max_lookahead_steps)
            
            # Process each timestep through the filter
            for t in range(len(self.filtered_reference)):
                # Use look-ahead reference if enabled, otherwise use current-time reference
                if self.config.reward_filter_use_lookahead and t + lookahead_steps < len(self.filtered_reference):
                    input_idx = t + lookahead_steps
                else:
                    input_idx = t
                
                u_t = standard_profile_tensor[:, input_idx:input_idx+1]  # [1, 1]
                filtered_y = self._reward_filter.update(u_t)
                filtered_tensor[:, t] = filtered_y.squeeze(0)
            
            # Store original filtered profile for observations, filtered version for rewards
            self._observation_reference = self.filtered_reference.copy()  # Standard profile for observations
            self.filtered_reference = filtered_tensor.squeeze(0).cpu().numpy().astype(np.float32)  # Filtered for rewards
            self.reference = self._observation_reference  # Use standard profile for observations
        elif self.config.violent_profile_mode and self._reward_filter is not None:
            # Violent mode: filter the raw profile using the reward filter
            import torch
            raw_tensor = torch.from_numpy(self.raw_reference).unsqueeze(0)  # [1, T]
            filtered_tensor = torch.zeros_like(raw_tensor)
            
            # Reset filter state with vehicle's initial speed (not reference speed)
            # This ensures smooth closure of initial speed error
            initial_y = torch.tensor([[initial_speed]], device=torch.device('cpu'), dtype=torch.float32)
            self._reward_filter.reset(initial_y=initial_y)
            
            # Determine look-ahead steps (within preview horizon)
            lookahead_steps = 0
            if self.config.reward_filter_use_lookahead:
                # Convert lookahead time to steps, but don't exceed preview horizon
                lookahead_steps_float = self.config.reward_filter_lookahead_s / self.config.dt
                max_lookahead_steps = min(self.preview_steps, len(self.raw_reference) - 1)
                lookahead_steps = min(int(lookahead_steps_float), max_lookahead_steps)
            
            # Process each timestep through the filter
            for t in range(len(self.raw_reference)):
                # Use look-ahead reference if enabled, otherwise use current-time reference
                if self.config.reward_filter_use_lookahead and t + lookahead_steps < len(self.raw_reference):
                    input_idx = t + lookahead_steps
                else:
                    input_idx = t
                
                u_t = raw_tensor[:, input_idx:input_idx+1]  # [1, 1]
                filtered_y = self._reward_filter.update(u_t)
                filtered_tensor[:, t] = filtered_y.squeeze(0)
            
            self.filtered_reference = filtered_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            self.reference = self.raw_reference  # Use raw profile for observations
        elif self.filtered_reference is None:
            # Fallback: if filtered_reference not set, use reference (for backward compatibility)
            self.filtered_reference = self.reference.copy() if self.reference is not None else None
            self.reference = self.filtered_reference
        else:
            # Normal mode: use filtered profile for both observations and rewards
            self.reference = self.filtered_reference
        
        self.speed = initial_speed  # No clamping - allow negative speeds if sampled
        self.prev_speed = self.speed  # Start with consistent prev_speed
        self._prev_prev_speed = self.speed  # Initialize speed history to initial speed
        self.position = 0.0
        # Initialize previous action: explicit override > randomization > default
        if "initial_action" in options:
            # Explicit override - use provided value
            self._prev_action = float(options["initial_action"])
        elif self.config.randomize_initial_action:
            self._prev_action = self._rng.uniform(self.config.action_low, self.config.action_high)
        else:
            self._prev_action = 0.0
        self._prev_accel = 0.0  # Start with zero acceleration
        self._filtered_accel = 0.0  # Start filtered accel at zero
        self._prev_jerk = 0.0  # Start with zero jerk
        self._prev_target_jerk = 0.0  # Start with zero target jerk
        # Initialize tracking variables for oscillation and overshoot penalties
        self._prev_action_sign = 0.0
        self._prev_error = 0.0
        self._prev_ref_speed = float(self.reference[0]) if len(self.reference) > 0 else 0.0
        self._ref_idx = 0
        
        # Reset max acceleration and jerk tracking
        self._max_abs_acc_exceeding_target = 0.0
        self._max_abs_jerk_exceeding_target = 0.0
        self._prev_target_accel = 0.0
        self._step_count = 0
        
        # Reset bias error buffer
        if self._bias_error_buffer is not None:
            self._bias_error_buffer.clear()
        
        # Compute LQR penalty weight schedules if enabled
        if self._lqr_config is not None:
            from speed_shaper.src.shaper_math import weight_schedule
            episode_length = len(self.reference)
            t_array = np.arange(episode_length) * self.config.dt
            
            # Compute time-varying weights for error, acceleration, and jerk
            self._lqr_error_weights = weight_schedule(
                t_array,
                self._lqr_config.wE_start,
                self._lqr_config.wE_end,
                self._lqr_config.lamE
            )
            self._lqr_accel_weights = weight_schedule(
                t_array,
                self._lqr_config.wA_start,
                self._lqr_config.wA_end,
                self._lqr_config.lamA
            )
            self._lqr_jerk_weights = weight_schedule(
                t_array,
                self._lqr_config.wJ_start,
                self._lqr_config.wJ_end,
                self._lqr_config.lamJ
            )
        
        if self.config.use_extended_plant:
            self.extended = ExtendedPlant(self.extended_params)
            self._last_state = self.extended.reset(speed=self.speed, position=self.position)
        else:
            self.extended = None
            self._last_state = None

        obs = self._build_observation()
        # Use filtered reference for info (for reward tracking)
        filtered_ref = self.filtered_reference[self._ref_idx] if self.filtered_reference is not None else self.reference[self._ref_idx]
        info = {
            "reference_speed": float(filtered_ref),
            "profile_feasible": True,
            "max_profile_adjustment": 0.0,
            "original_reference": self.reference.copy(),
            "feasible_reference": self.filtered_reference.copy() if self.filtered_reference is not None else self.reference.copy(),
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

        # Check if this is the first or last step of the episode
        is_first_step = bool(self._step_count == 0)  # First call to step() has _step_count=0
        is_last_step = bool((self._step_count + 1) >= self.config.max_episode_steps)
        
        # Compute jerk from filtered acceleration
        # At the first step: jerk is an initialization artifact (prev_accel=0), so set to 0
        # At the last step: use the previous step's jerk value for continuity
        if is_first_step:
            jerk = 0.0  # First step jerk is initialization artifact
        elif is_last_step:
            jerk = self._prev_jerk  # Last step uses previous jerk for continuity
        else:
            jerk = (self._filtered_accel - self._prev_accel) / max(self.config.dt, 1e-6)
        
        # Compute target acceleration and jerk from reference profile
        current_ref = float(self.filtered_reference[self._ref_idx])
        prev_ref_idx = max(0, self._ref_idx - 1)
        prev_ref = float(self.filtered_reference[prev_ref_idx])
        target_accel = (current_ref - prev_ref) / max(self.config.dt, 1e-6)
        
        # Compute target jerk from target acceleration change
        # At the first step: target jerk is 0 (initialization)
        # At the last step: use the previous step's target jerk value for continuity
        if is_first_step:
            target_jerk = 0.0  # First step target jerk is initialization
        elif is_last_step:
            target_jerk = self._prev_target_jerk  # Last step uses previous target jerk
        else:
            target_jerk = (target_accel - self._prev_target_accel) / max(self.config.dt, 1e-6)
        self._prev_target_accel = target_accel
        
        # Track maximum absolute acceleration and jerk (only when exceeding target)
        # Skip tracking at first and last steps since they involve initialization/boundary artifacts
        if not is_first_step and not is_last_step:
            abs_acc = abs(self._filtered_accel)
            abs_target_acc = abs(target_accel)
            if abs_acc > abs_target_acc:
                self._max_abs_acc_exceeding_target = max(self._max_abs_acc_exceeding_target, abs_acc)
            
            abs_jerk_val = abs(jerk)
            abs_target_jerk = abs(target_jerk)
            if abs_jerk_val > abs_target_jerk:
                self._max_abs_jerk_exceeding_target = max(self._max_abs_jerk_exceeding_target, abs_jerk_val)

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

        # Current speed tracking penalty - always use filtered reference for reward
        # Normalize by episode length to match scale of single-value penalties (like max_acc_penalty)
        assert self.filtered_reference is not None
        current_filtered_ref = float(self.filtered_reference[self._ref_idx])
        speed_error = self.speed - current_filtered_ref
        
        # Standard tracking error penalty (using track_weight)
        tracking_error_penalty = -self.config.track_weight * (speed_error**2) / self.config.max_episode_steps
        reward = tracking_error_penalty

        # Track reward components for animation/debugging
        reward_components = {
            'tracking_error': tracking_error_penalty,
            'bias_penalty': 0.0,
            'jerk_penalty': 0.0,
            'action_penalty': 0.0,
            'smooth_action_penalty': 0.0,
            'horizon_penalty': 0.0,
            'oscillation_penalty': 0.0,
            'overshoot_penalty': 0.0,
            'max_acc_penalty': 0.0,
            'max_jerk_penalty': 0.0,
            'lqr_error_penalty': 0.0,
            'lqr_accel_penalty': 0.0,
            'lqr_jerk_penalty': 0.0,
            'negative_speed_penalty': 0.0,
            'zero_speed_error_penalty': 0.0,
            'zero_speed_throttle_penalty': 0.0,
        }
        
        # LQR error penalty (additive to standard tracking error penalty)
        if self._lqr_config is not None and self._lqr_error_weights is not None:
            # Ensure we have valid weights for current timestep
            if self._ref_idx < len(self._lqr_error_weights):
                # Get time-varying error weight for current timestep
                w_error_lqr = self._lqr_error_weights[self._ref_idx]
                
                # Error penalty (use speed error squared)
                # Normalize by episode length to match scale of single-value penalties
                lqr_error_penalty = (
                    -self.config.lqr_error_weight
                    * w_error_lqr
                    * (speed_error ** 2)
                    / self.config.max_episode_steps
                )
                reward += lqr_error_penalty
                reward_components['lqr_error_penalty'] = lqr_error_penalty
        
        # Sliding-window mean speed error penalty (bias elimination)
        if self.config.bias_penalty_enabled and self._bias_error_buffer is not None:
            # Add current error to buffer
            self._bias_error_buffer.append(speed_error)
            
            # Compute mean error only when buffer is full
            if len(self._bias_error_buffer) == self._bias_window_steps:
                e_mean = sum(self._bias_error_buffer) / self._bias_window_steps
                
                # Normalize by speed scale
                e_mean_norm = e_mean / self.config.bias_penalty_v_scale
                
                # Gate: only apply penalty during steady-state conditions
                # Check if reference speed is approximately constant
                prev_ref_idx = max(0, self._ref_idx - 1)
                prev_ref = float(self.filtered_reference[prev_ref_idx])
                ref_rate = abs(current_filtered_ref - prev_ref) / max(self.config.dt, 1e-6)
                
                # Check if vehicle acceleration is small
                vehicle_accel = abs(self._filtered_accel)
                
                # Steady-state condition: constant reference and small acceleration
                steady_state = (
                    ref_rate < self.config.bias_penalty_v_ref_dot_threshold and
                    vehicle_accel < self.config.bias_penalty_a_threshold
                )
                
                if steady_state:
                    # Apply bias penalty (normalize by episode length to match scale of single-value penalties)
                    bias_penalty = -self.config.bias_penalty_weight * (e_mean_norm ** 2) / self.config.max_episode_steps
                    reward += bias_penalty
                    reward_components['bias_penalty'] = bias_penalty

        # Horizon penalty: encourage anticipation of future speed changes
        # Penalize deviations from future reference speeds with exponential decay
        if self.config.horizon_penalty_weight > 0.0:
            horizon_penalty = 0.0
            decay_factor = 1.0

            for i in range(1, min(self.preview_steps + 1, len(self.filtered_reference) - self._ref_idx)):
                future_idx = self._ref_idx + i
                future_ref = float(self.filtered_reference[future_idx])  # Use filtered reference for reward

                # Penalize absolute deviation between current speed and future reference
                # This encourages being at the right speed for upcoming changes
                speed_deviation = abs(self.speed - future_ref)
                horizon_penalty -= decay_factor * speed_deviation

                # Apply exponential decay for future timesteps
                decay_factor *= self.config.horizon_penalty_decay

            # Normalize by episode length to match scale of single-value penalties
            horizon_penalty_scaled = self.config.horizon_penalty_weight * horizon_penalty / self.config.max_episode_steps
            reward += horizon_penalty_scaled
            reward_components['horizon_penalty'] = horizon_penalty_scaled

        # Other penalties (comfort penalties use annealing multiplier)
        # Normalize by episode length to match scale of single-value penalties
        # Skip jerk penalty at first and last steps (initialization and boundary artifacts)
        comfort_mult = self.get_comfort_anneal_multiplier()
        if is_first_step or is_last_step:
            jerk_penalty = 0.0  # No jerk penalty at first/last step
        else:
            jerk_penalty = -comfort_mult * self.config.jerk_weight * abs(jerk) / self.config.max_episode_steps
        reward += jerk_penalty
        reward_components['jerk_penalty'] = jerk_penalty
        
        action_penalty = -self.config.action_weight * (abs(action_value)) / self.config.max_episode_steps
        reward += action_penalty
        reward_components['action_penalty'] = action_penalty
        
        # Skip smooth action penalty at the last step since there's no next action to compare to
        if is_last_step:
            smooth_action_penalty = 0.0  # No smooth action penalty at last step (no transition)
        else:
            smooth_action_penalty = -comfort_mult * self.config.smooth_action_weight * abs(action_value - self._prev_action) / self.config.max_episode_steps
        reward += smooth_action_penalty
        reward_components['smooth_action_penalty'] = smooth_action_penalty
        
        # LQR acceleration and jerk penalties (additive to existing penalties)
        if self._lqr_config is not None and self._lqr_accel_weights is not None:
            # Ensure we have valid weights for current timestep
            if self._ref_idx < len(self._lqr_accel_weights):
                # Get time-varying weights for current timestep
                w_accel_lqr = self._lqr_accel_weights[self._ref_idx]
                w_jerk_lqr = self._lqr_jerk_weights[self._ref_idx]
                
                # Acceleration penalty (use filtered acceleration)
                # Normalize by episode length to match scale of single-value penalties
                lqr_accel_penalty = (
                    -self.config.lqr_accel_weight
                    * w_accel_lqr
                    * (self._filtered_accel ** 2)
                    / self.config.max_episode_steps
                )
                reward += lqr_accel_penalty
                reward_components['lqr_accel_penalty'] = lqr_accel_penalty
                
                # Jerk penalty (skip at first/last steps like existing jerk penalty)
                if not is_first_step and not is_last_step:
                    lqr_jerk_penalty = (
                        -self.config.lqr_jerk_weight
                        * w_jerk_lqr
                        * (jerk ** 2)
                        / self.config.max_episode_steps
                    )
                    reward += lqr_jerk_penalty
                    reward_components['lqr_jerk_penalty'] = lqr_jerk_penalty
                # else: lqr_jerk_penalty already initialized to 0.0 in reward_components
        
        if self.config.use_extended_plant and plant_state is not None:
            # Normalize by episode length to match scale of single-value penalties
            reward -= self.config.brake_weight * abs(plant_state.brake_torque) / self.config.max_episode_steps

        # Negative speed penalty (normalize by episode length)
        if self.speed < 0.0:
            negative_speed_penalty = -self.config.negative_speed_weight * abs(self.speed) / self.config.max_episode_steps
            reward += negative_speed_penalty
            reward_components['negative_speed_penalty'] = negative_speed_penalty
        
        # Zero-speed penalties: when target speed is 0 (use filtered reference for reward)
        current_ref = float(self.filtered_reference[self._ref_idx])
        if abs(current_ref) < 1e-6:  # Target speed is 0 (with epsilon for floating point)
            # Penalty for speed error when target is 0 (only if vehicle is moving)
            # Normalize by episode length to match scale of single-value penalties
            if self.speed > 0.0:
                zero_speed_error_penalty = -self.config.zero_speed_error_weight * abs(self.speed) / self.config.max_episode_steps
                reward += zero_speed_error_penalty
                reward_components['zero_speed_error_penalty'] = zero_speed_error_penalty
            # Penalty for throttle usage when target is 0 (constant penalty for any positive action)
            # Normalize by episode length to match scale of single-value penalties
            if action_value > 0.0:
                zero_speed_throttle_penalty = -self.config.zero_speed_throttle_weight / self.config.max_episode_steps
                reward += zero_speed_throttle_penalty
                reward_components['zero_speed_throttle_penalty'] = zero_speed_throttle_penalty
        
        # Oscillation and overshoot penalties (from oscillation_and_overshoot_penalties_for_rl_longitudinal_control.md)
        if self.config.oscillation_weight > 0.0 or self.config.overshoot_weight > 0.0:
            # Compute current tracking error
            current_error = speed_error
            
            # Compute current action sign (needed for both penalties potentially)
            current_action_sign = 0.0
            if self.config.oscillation_weight > 0.0:
                # Step 1: Smooth action sign representation
                epsilon = self.config.oscillation_epsilon
                current_action_sign = np.tanh(action_value / epsilon)
                
                # Step 2: Measure switching energy
                switching_energy = (current_action_sign - self._prev_action_sign) ** 2
                
                # Step 3: Gate by proximity to setpoint
                error_scale = self.config.oscillation_error_scale
                proximity_gate = np.exp(-np.abs(current_error) / error_scale)
                
                # Step 4: Gate by setpoint stationarity
                ref_rate = abs(current_ref - self._prev_ref_speed) / max(self.config.dt, 1e-6)
                ref_scale = self.config.oscillation_ref_scale
                stationarity_gate = np.exp(-ref_rate / ref_scale)
                
                # Final oscillation penalty (normalize by episode length to match scale of single-value penalties)
                oscillation_penalty = (
                    -self.config.oscillation_weight
                    * switching_energy
                    * proximity_gate
                    * stationarity_gate
                    / self.config.max_episode_steps
                )
                reward += oscillation_penalty
                reward_components['oscillation_penalty'] = oscillation_penalty
            
            # Term 2: Overshoot Crossing Penalty
            if self.config.overshoot_weight > 0.0:
                # Step 1: Error time derivative
                error_rate = (current_error - self._prev_error) / max(self.config.dt, 1e-6)
                
                # Step 2: Detect setpoint crossing (smoothly)
                crossing_scale = self.config.overshoot_crossing_scale
                # Use sigmoid: σ(x) = 1 / (1 + exp(-x))
                # For w_c = σ(-e_t * e_{t-1} / c_s), we want high value when crossing occurs
                # Numerically stable computation to avoid overflow
                x = -(-current_error * self._prev_error) / crossing_scale
                # Clip x to prevent overflow in exp
                x_clipped = np.clip(x, -500.0, 500.0)
                crossing_gate = 1.0 / (1.0 + np.exp(-x_clipped))
                
                # Step 3: Gate by proximity to setpoint (reuse same gate as oscillation)
                error_scale = self.config.oscillation_error_scale
                proximity_gate = np.exp(-np.abs(current_error) / error_scale)
                
                # Final overshoot penalty (normalize by episode length to match scale of single-value penalties)
                overshoot_penalty = (
                    -self.config.overshoot_weight
                    * (error_rate ** 2)
                    * crossing_gate
                    * proximity_gate
                    / self.config.max_episode_steps
                )
                reward += overshoot_penalty
                reward_components['overshoot_penalty'] = overshoot_penalty
            
            # Update tracking variables for next step
            if self.config.oscillation_weight > 0.0:
                self._prev_action_sign = current_action_sign
            self._prev_error = current_error
            self._prev_ref_speed = current_ref
        
        # Apply maximum absolute acceleration and jerk penalties at episode end
        terminated = False
        truncated = bool(self._step_count >= self.config.max_episode_steps)
        
        if truncated:
            # Episode ended - apply penalties based on maximum values
            if self.config.max_acc_penalty_weight > 0.0:
                max_acc_penalty = -self.config.max_acc_penalty_weight * (self._max_abs_acc_exceeding_target ** 2)
                reward += max_acc_penalty
                reward_components['max_acc_penalty'] = max_acc_penalty
            
            if self.config.max_jerk_penalty_weight > 0.0:
                max_jerk_penalty = -self.config.max_jerk_penalty_weight * (self._max_abs_jerk_exceeding_target ** 2)
                reward += max_jerk_penalty
                reward_components['max_jerk_penalty'] = max_jerk_penalty
        
        # Check if episode will end after this step
        will_truncate = bool((self._step_count + 1) >= self.config.max_episode_steps)
        
        # Apply maximum absolute acceleration and jerk penalties at episode end
        if will_truncate:
            # Episode ending - apply penalties based on maximum values
            if self.config.max_acc_penalty_weight > 0.0:
                max_acc_penalty = -self.config.max_acc_penalty_weight * (self._max_abs_acc_exceeding_target ** 2)
                reward += max_acc_penalty
                reward_components['max_acc_penalty'] = max_acc_penalty
            
            if self.config.max_jerk_penalty_weight > 0.0:
                max_jerk_penalty = -self.config.max_jerk_penalty_weight * (self._max_abs_jerk_exceeding_target ** 2)
                reward += max_jerk_penalty
                reward_components['max_jerk_penalty'] = max_jerk_penalty
        
        reward = float(np.clip(reward, -self.config.base_reward_clip, self.config.base_reward_clip))

        self._prev_prev_speed = self.prev_speed  # Shift speed history back
        self.prev_speed = self.speed
        self._prev_accel = self._filtered_accel  # Use filtered accel for jerk calculation consistency
        self._prev_jerk = jerk  # Store current jerk for next step (or last step continuity)
        self._prev_target_jerk = target_jerk  # Store current target jerk for next step (or last step continuity)
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
            "reference_speed": float(self.filtered_reference[self._ref_idx]),  # Use filtered reference for info
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
            "grade_rad": current_grade if self.config.use_extended_plant and self.grade_profile is not None else 0.0,  # Current grade for dynamics map
            "reward_components": reward_components,  # Individual reward components for visualization
        }

        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - visualization hook
        return {
            "speed": self.speed,
            "position": self.position,
            "reference_speed": None if self.reference is None else float(self.reference[self._ref_idx]),
            "filtered_reference_speed": None if self.filtered_reference is None else float(self.filtered_reference[self._ref_idx]),
        }

    def close(self):  # pragma: no cover - compatibility hook
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_reference(self, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate reference profiles.
        
        Returns:
            Tuple of (filtered_profile, grade_profile, raw_profile)
        """
        if self._reference_callable is not None:
            profile = self._reference_callable(length, self._rng)
            # For callable references, use same profile for both raw and filtered
            grade_profile = np.zeros(length, dtype=np.float32)
            return profile.astype(np.float32), grade_profile, profile.astype(np.float32)
        else:
            assert self.reference_generator is not None
            # Convert extended_params to VehicleCapabilities if available
            vehicle = None
            if self.extended_params is not None:
                import torch
                device = torch.device('cpu')  # Generator uses CPU by default
                vehicle = extended_params_to_vehicle_capabilities(self.extended_params, device=device)
            
            # Try calling with vehicle parameter, fall back to old interface if not supported
            try:
                result = self.reference_generator.sample(length, self._rng, vehicle=vehicle)
            except TypeError:
                # Old generator interface without vehicle parameter
                result = self.reference_generator.sample(length, self._rng)
            
            if isinstance(result, tuple) and len(result) == 3:
                # New interface: returns (filtered_profile, grade_profile, raw_profile)
                filtered_profile, grade_profile, raw_profile = result
                filtered_profile = np.asarray(filtered_profile, dtype=np.float32)
                grade_profile = np.asarray(grade_profile, dtype=np.float32)
                raw_profile = np.asarray(raw_profile, dtype=np.float32)
                return filtered_profile, grade_profile, raw_profile
            elif isinstance(result, tuple) and len(result) == 2:
                # Legacy interface: returns (speed_profile, grade_profile) - use same for raw and filtered
                profile, grade_profile = result
                profile = np.asarray(profile, dtype=np.float32)
                grade_profile = np.asarray(grade_profile, dtype=np.float32)
                return profile, grade_profile, profile.copy()
            else:
                # Very old legacy: only speed profile
                profile = result
                profile = np.asarray(profile, dtype=np.float32)
                grade_profile = np.zeros(length, dtype=np.float32)
                return profile, grade_profile, profile.copy()

    def _build_observation(self) -> np.ndarray:
        assert self.reference is not None
        assert self.filtered_reference is not None
        state = self._last_state
        speed_meas = (state.speed if state else self.speed) + self._rng.normal(0.0, self._speed_noise_std)

        # Compute speed error with respect to filtered reference (for reward)
        current_filtered_ref = float(self.filtered_reference[self._ref_idx])
        speed_error = speed_meas - current_filtered_ref

        # Include speed history, previous action, speed error, and road grade for observation
        # Observation: [speed, prev_speed, prev_prev_speed, prev_action, speed_error, grade, refs...]
        # This allows the agent to estimate:
        #   - Acceleration: (speed - prev_speed) / dt
        #   - Jerk: ((speed - prev_speed) - (prev_speed - prev_prev_speed)) / dt^2
        #   - Action smoothness: action - prev_action
        #   - Speed error: direct feedback on tracking performance
        #   - Road grade: current road slope (affects vehicle dynamics)
        
        # Get current road grade
        current_grade = float(self.grade_profile[self._ref_idx]) if self.grade_profile is not None else 0.0
        
        base = np.array([
            speed_meas,             # Current speed
            self.prev_speed,        # Previous speed (for acceleration)
            self._prev_prev_speed,  # Speed from 2 steps ago (for jerk)
            self._prev_action,      # Previous action (for action smoothness)
            speed_error,            # Speed error (always included)
            current_grade,          # Current road grade (radians)
        ], dtype=np.float32)

        # Use reference based on mode: raw for violent mode, filtered otherwise
        # Note: self.reference is already set correctly in reset() based on mode
        preview = np.empty(self.preview_steps, dtype=np.float32)
        last_idx = len(self.reference) - 1
        for idx in range(self.preview_steps):
            ref_idx = min(self._ref_idx + idx, last_idx)
            preview[idx] = float(self.reference[ref_idx])
        return np.concatenate([base, preview]).astype(np.float32)


__all__ = ["LongitudinalEnv", "LongitudinalEnvConfig"]
