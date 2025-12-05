"""Adapter to use batch generator with single-episode environment interface."""

import numpy as np
import torch

from .generator import BatchTargetGenerator, GeneratorConfig, generate_batch_targets
from .feasibility import VehicleCapabilities


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

    def sample(self, length: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate a single reference profile and grade profile of given length.

        Args:
            length: Length of the profile in steps
            rng: Random number generator (unused, kept for compatibility)

        Returns:
            Tuple of (speed_profile, grade_profile) as numpy arrays [length]
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
            # Grade parameters
            ds=self.batch_config.ds,
            l_corr=self.batch_config.l_corr,
            sigma_g_stat=self.batch_config.sigma_g_stat,
            g_min=self.batch_config.g_min,
            g_max=self.batch_config.g_max
        )

        # Create a dummy vehicle with reasonable parameters
        # In practice, this should be passed from the environment
        vehicle = self._create_default_vehicle()

        # Generate batch and extract first (only) episode
        generator = BatchTargetGenerator(single_config, batch_size=1, episode_length=length, device=self.device)
        # Override the prediction_horizon in the config
        generator.config.prediction_horizon = min(self.prediction_horizon, length)
        targets, grades = generator.generate_batch(vehicle)

        # Extract the speed profile (first element of each horizon)
        speed_profile = targets[0, :, 0].cpu().numpy()  # [length]
        grade_profile = grades[0, :length].cpu().numpy()  # [length]

        return speed_profile, grade_profile

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
            # Grade parameters
            ds=generator_config.get('ds', 1.0),
            l_corr=generator_config.get('l_corr', 50.0),
            sigma_g_stat=generator_config.get('sigma_g_stat', 0.002),
            g_min=generator_config.get('g_min', -0.08),
            g_max=generator_config.get('g_max', 0.08)
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
