"""High-level context manager for online map reconstruction."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.maps.buffer import ObservationBuffer
from src.maps.encoder import MapEncoder
from src.maps.gravity import GravityCompensator
from src.maps.grid import MapGrid
from src.maps.sampler_diffusion import GuidedDiffusionSampler
from src.maps.warmstart import warmstart_sample
from src.models.diffusion_prior import DiffusionPrior


class DynamicsMapContext:
    """High-level interface for online dynamics map reconstruction.
    
    Encapsulates:
        - Grid discretization
        - Observation buffer with time decay
        - Gravity compensation
        - Guided diffusion sampler
        - Map encoder for RL context
    
    Usage:
        context = DynamicsMapContext(config, diffusion_prior, encoder, device)
        context.reset()  # Start new episode
        
        # In control loop:
        context.add_observation(u, v, a_meas, theta, t)
        if step % M_update == 0:
            context.update_map()
        c = context.get_context()  # Get context vector for policy
    """
    
    def __init__(
        self,
        grid: MapGrid,
        diffusion_prior: DiffusionPrior,
        encoder: MapEncoder,
        norm_mean: float,
        norm_std: float,
        buffer_capacity: int = 10000,
        lambda_decay: float = 0.3,
        w_min: float = 0.2,
        guidance_scale: float = 1.0,
        sigma_meas: float = 0.5,
        num_inference_steps: int = 50,
        warmstart_rho: float = 0.8,
        gradient_smoothing_sigma: float = 10.0,
        patch_filtering_enabled: bool = False,
        patch_size_u: int = 10,
        patch_size_v: int = 10,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize dynamics map context.
        
        Args:
            grid: Map grid
            diffusion_prior: Pretrained diffusion prior
            encoder: Map encoder
            norm_mean: Normalization mean
            norm_std: Normalization std
            buffer_capacity: Observation buffer capacity
            lambda_decay: Weight decay rate (1/time_units)
            w_min: Minimum weight floor
            guidance_scale: Guidance strength (eta)
            sigma_meas: Measurement noise std (m/s²)
            num_inference_steps: Number of diffusion steps
            warmstart_rho: Warm-start mixing coefficient
            gradient_smoothing_sigma: Gaussian smoothing sigma for guidance gradient
            patch_filtering_enabled: Enable patch-based uniform sampling
            patch_size_u: Patch size in u direction
            patch_size_v: Patch size in v direction
            device: Device for computation
        """
        self.grid = grid
        self.diffusion_prior = diffusion_prior.to(device)
        self.encoder = encoder.to(device)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.warmstart_rho = warmstart_rho
        self.device = device
        
        # Create components
        self.gravity_comp = GravityCompensator()
        self.obs_buffer = ObservationBuffer(
            capacity=buffer_capacity,
            lambda_decay=lambda_decay,
            w_min=w_min,
        )
        self.sampler = GuidedDiffusionSampler(
            diffusion_prior=diffusion_prior,
            guidance_scale=guidance_scale,
            sigma_meas=sigma_meas,
            num_inference_steps=num_inference_steps,
            gradient_smoothing_sigma=gradient_smoothing_sigma,
            patch_filtering_enabled=patch_filtering_enabled,
            patch_size_u=patch_size_u,
            patch_size_v=patch_size_v,
        )
        
        # State
        self.current_map: Optional[torch.Tensor] = None  # Current map estimate (normalized)
        self.current_time: float = 0.0
    
    def reset(self) -> None:
        """Reset for new episode."""
        self.obs_buffer.reset()
        self.current_map = None
        self.current_time = 0.0
    
    def add_observation(
        self,
        u: float,
        v: float,
        a_meas: float,
        theta: float,
        t: float,
    ) -> None:
        """Add observation to buffer.
        
        Args:
            u: Command (signed, in [-1, 1])
            v: Speed (m/s)
            a_meas: Measured longitudinal acceleration (m/s²)
            theta: Pitch angle (radians, positive = uphill)
            t: Timestamp (seconds or steps)
        """
        # Compensate for gravity
        a_dyn = self.gravity_comp.compensate(a_meas, theta)
        
        # Bin to grid
        i_u = self.grid.bin_u(u)
        i_v = self.grid.bin_v(v)
        
        # Add to buffer
        self.obs_buffer.add(i_u, i_v, a_dyn, t)
        
        # Update current time
        self.current_time = t
    
    def update_map(self) -> None:
        """Run guided reconstruction to update map estimate."""
        # Create warm-start initial sample
        if self.current_map is None or self.warmstart_rho <= 0:
            x_init = None  # Pure noise
        else:
            # Mix previous map with noise
            noise = torch.randn_like(self.current_map)
            x_init = warmstart_sample(self.current_map, self.warmstart_rho, noise)
        
        # Sample posterior
        self.current_map = self.sampler.sample(
            obs_buffer=self.obs_buffer,
            t_now=self.current_time,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            batch_size=1,
            device=self.device,
            x_init=x_init,
        )
    
    def get_context(self) -> torch.Tensor:
        """Get context vector for RL policy.
        
        Returns:
            Context vector, shape (context_dim,)
        """
        if self.current_map is None:
            # No map yet, sample from prior
            self.current_map = self.diffusion_prior.sample(
                batch_size=1,
                num_inference_steps=self.sampler.num_inference_steps,
                device=self.device,
            )
        
        # Encode map to context vector
        with torch.no_grad():
            context = self.encoder(self.current_map)  # (1, context_dim)
        
        return context.squeeze(0)  # (context_dim,)
    
    def get_map(self) -> Optional[np.ndarray]:
        """Get current map estimate (denormalized).
        
        Returns:
            Map array (N_u, N_v) or None if no map yet
        """
        if self.current_map is None:
            return None
        
        # Denormalize
        map_norm = self.current_map.squeeze(0).squeeze(0).cpu().numpy()  # (N_u, N_v)
        map_denorm = map_norm * self.norm_std + self.norm_mean
        
        return map_denorm

