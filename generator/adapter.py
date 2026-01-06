"""Adapter to use batch generator with single-episode environment interface."""

import numpy as np
import torch

from .generator import BatchTargetGenerator, GeneratorConfig, generate_batch_targets
from .feasibility import VehicleCapabilities


def extended_params_to_vehicle_capabilities(extended_params, device: torch.device = None) -> VehicleCapabilities:
    """Convert ExtendedPlantParams to VehicleCapabilities for feasibility checking.
    
    Args:
        extended_params: ExtendedPlantParams object from dynamics
        device: Torch device (defaults to CPU)
        
    Returns:
        VehicleCapabilities object for feasibility checking
    """
    if device is None:
        device = torch.device('cpu')
    
    # Calculate max current from V_max and R (stall current)
    i_max = extended_params.motor.V_max / extended_params.motor.R
    
    # Get gearbox efficiency (assume it's in the params, or use default)
    # Note: eta_gb is not directly in ExtendedPlantParams, so we'll use a default
    # This should ideally be passed separately or added to ExtendedPlantParams
    eta_gb = 0.9  # Default gearbox efficiency
    
    return VehicleCapabilities(
        m=torch.tensor([extended_params.body.mass], device=device),
        r_w=torch.tensor([extended_params.wheel.radius], device=device),
        gear_ratio=torch.tensor([extended_params.motor.gear_ratio], device=device),
        eta_gb=torch.tensor([eta_gb], device=device),  # TODO: Add to ExtendedPlantParams
        K_t=torch.tensor([extended_params.motor.K_t], device=device),
        K_e=torch.tensor([extended_params.motor.K_e], device=device),
        R=torch.tensor([extended_params.motor.R], device=device),
        i_max=torch.tensor([i_max], device=device),
        V_max=torch.tensor([extended_params.motor.V_max], device=device),
        b=torch.tensor([extended_params.motor.b], device=device),
        T_brake_max=torch.tensor([extended_params.brake.T_br_max], device=device),
        mu=torch.tensor([extended_params.brake.mu], device=device),
        C_dA=torch.tensor([extended_params.body.drag_area], device=device),  # Includes rho
        C_r=torch.tensor([extended_params.body.rolling_coeff], device=device)
    )


class SingleEpisodeGenerator:
    """Adapter to generate single episodes from the batch generator."""

    def __init__(
        self,
        batch_config: GeneratorConfig,
        prediction_horizon: int,
        device: torch.device = None
    ):
        """Initialize with batch generator config.

        Args:
            batch_config: Configuration for the underlying batch generator
            prediction_horizon: Prediction horizon in steps
            device: Torch device
        """
        self.batch_config = batch_config
        self.prediction_horizon = prediction_horizon
        self.device = device or torch.device('cpu')
        self.generator = BatchTargetGenerator(batch_config, batch_size=1, episode_length=1000, device=device)

    def sample(self, length: int, rng: np.random.Generator | None = None, vehicle: VehicleCapabilities | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a single reference profile and grade profile of given length.

        Args:
            length: Length of the profile in steps
            rng: Random number generator (unused, kept for compatibility)
            vehicle: Optional VehicleCapabilities object. If None, uses default vehicle.

        Returns:
            Tuple of (filtered_speed_profile, grade_profile, raw_speed_profile) as numpy arrays [length]
        """
        # Adjust config for single episode
        single_config = GeneratorConfig(
            freq_cutoff=self.batch_config.freq_cutoff,
            zeta=self.batch_config.zeta,
            dt=self.batch_config.dt,
            jerk_max=self.batch_config.jerk_max,
            p_change=self.batch_config.p_change,
            p_zero_stop=self.batch_config.p_zero_stop,
            v_min=self.batch_config.v_min,
            v_max_sample=self.batch_config.v_max_sample,
            t_min=self.batch_config.t_min,
            t_max=self.batch_config.t_max,
            stop_hold_min=self.batch_config.stop_hold_min,
            stop_hold_max=self.batch_config.stop_hold_max,
            # Stochastic parameters
            p_change_acc=self.batch_config.p_change_acc,
            p_change_jerk=self.batch_config.p_change_jerk,
            p_zero_accel=self.batch_config.p_zero_accel,
            accel_beta=self.batch_config.accel_beta,
            # Grade parameters
            ds=self.batch_config.ds,
            l_corr=self.batch_config.l_corr,
            sigma_g_stat=self.batch_config.sigma_g_stat,
            g_min=self.batch_config.g_min,
            g_max=self.batch_config.g_max,
            # Speed limit safety factor (propagate from batch config)
            speed_limit_safety_factor=self.batch_config.speed_limit_safety_factor
        )

        # Use provided vehicle or create default
        if vehicle is None:
            vehicle = self._create_default_vehicle()

        # Generate batch and extract first (only) episode
        generator = BatchTargetGenerator(single_config, batch_size=1, episode_length=length, device=self.device)
        # Override the prediction_horizon in the config
        generator.config.prediction_horizon = min(self.prediction_horizon, length)
        targets, grades, raw_targets = generator.generate_batch(vehicle)

        # Extract the speed profiles (first element of each horizon)
        filtered_speed_profile = targets[0, :, 0].cpu().numpy()  # [length] - filtered profile
        grade_profile = grades[0, :length].cpu().numpy()  # [length]
        raw_speed_profile = raw_targets[0, :, 0].cpu().numpy()  # [length] - raw (unfiltered) profile

        return filtered_speed_profile, grade_profile, raw_speed_profile

    def _create_default_vehicle(self) -> VehicleCapabilities:
        """Create default vehicle parameters for generation.

        These should be overridden with actual vehicle params when available.
        """
        return VehicleCapabilities(
            m=torch.tensor([1500.0], device=self.device),
            r_w=torch.tensor([0.3], device=self.device),
            gear_ratio=torch.tensor([10.0], device=self.device),
            eta_gb=torch.tensor([0.9], device=self.device),
            K_t=torch.tensor([0.5], device=self.device),
            K_e=torch.tensor([0.5], device=self.device),
            R=torch.tensor([0.1], device=self.device),
            i_max=torch.tensor([200.0], device=self.device),
            V_max=torch.tensor([400.0], device=self.device),
            b=torch.tensor([1e-3], device=self.device),  # Motor viscous damping (Nm·s/rad)
            T_brake_max=torch.tensor([2000.0], device=self.device),
            mu=torch.tensor([0.8], device=self.device),
            C_dA=torch.tensor([0.6], device=self.device),  # Includes rho
            C_r=torch.tensor([0.01], device=self.device)
        )


def create_reference_generator(
    dt: float = 0.02,
    batch_size: int = 1,
    prediction_horizon: int = 20,
    generator_config: dict = None,
    device: torch.device = None
) -> SingleEpisodeGenerator:
    """Create a reference generator compatible with the environment interface.

    Args:
        dt: Time step (seconds)
        batch_size: Batch size (for future batch support)
        prediction_horizon: Prediction horizon (steps)
        device: Torch device

    Returns:
        Single episode generator
    """
    # Use provided config or defaults
    if generator_config is not None:
        config = GeneratorConfig(
            dt=dt,
            freq_cutoff=generator_config.get('freq_cutoff', 0.6),
            zeta=generator_config.get('zeta', 0.9),
            jerk_max=generator_config.get('jerk_max', 12.0),
            p_change=generator_config.get('p_change', 0.03),
            p_zero_stop=generator_config.get('p_zero_stop', 0.08),
            v_min=generator_config.get('v_min', 0.0),
            v_max_sample=generator_config.get('v_max_sample', 30.0),
            t_min=generator_config.get('t_min', 2.0),
            t_max=generator_config.get('t_max', 12.0),
            stop_hold_min=generator_config.get('stop_hold_min', 1.0),
            stop_hold_max=generator_config.get('stop_hold_max', 5.0),
            # Stochastic acceleration/jerk parameters
            p_change_acc=generator_config.get('p_change_acc', 0.04),
            p_change_jerk=generator_config.get('p_change_jerk', 0.03),
            p_zero_accel=generator_config.get('p_zero_accel', 0.15),
            accel_beta=generator_config.get('accel_beta', 2.0),
            # Grade parameters
            ds=generator_config.get('ds', 1.0),
            l_corr=generator_config.get('l_corr', 50.0),
            sigma_g_stat=generator_config.get('sigma_g_stat', 0.002),
            g_min=generator_config.get('g_min', -0.08),
            g_max=generator_config.get('g_max', 0.08),
            # Speed limit safety factor
            speed_limit_safety_factor=generator_config.get('speed_limit_safety_factor', 0.75)
        )
    else:
        config = GeneratorConfig(
            dt=dt,
            # Use defaults from the spec
            freq_cutoff=0.6,
            zeta=0.9,
            jerk_max=12.0,
            p_change=0.03,
            p_zero_stop=0.08
        )

    return SingleEpisodeGenerator(config, prediction_horizon, device)
