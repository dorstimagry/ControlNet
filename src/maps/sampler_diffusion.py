"""Guided diffusion sampler with measurement constraints."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.maps.buffer import ObservationBuffer
from src.maps.energy import compute_energy_gradient
from src.models.diffusion_prior import DiffusionPrior


class GuidedDiffusionSampler(nn.Module):
    """Sample from posterior using diffusion prior + measurement guidance.
    
    At each reverse timestep:
        1. Get prior score from diffusion model
        2. Compute measurement gradient from observations
        3. Modify with guidance: x ← step(x, noise_pred - eta * grad_E)
    """
    
    def __init__(
        self,
        diffusion_prior: DiffusionPrior,
        guidance_scale: float = 1.0,
        sigma_meas: float = 0.3,
        num_inference_steps: int = 20,
        gradient_smoothing_sigma: float = 0.0,
        patch_filtering_enabled: bool = False,
        patch_size_u: int = 10,
        patch_size_v: int = 10,
    ):
        """Initialize guided sampler.
        
        Args:
            diffusion_prior: Pretrained diffusion prior model
            guidance_scale: Guidance strength (eta)
            sigma_meas: Measurement noise std (m/s²)
            num_inference_steps: Number of DDIM steps
            gradient_smoothing_sigma: Gaussian smoothing sigma for gradient (0=no smoothing)
            patch_filtering_enabled: Enable patch-based uniform sampling
            patch_size_u: Patch size in u direction
            patch_size_v: Patch size in v direction
        """
        super().__init__()
        
        self.diffusion_prior = diffusion_prior
        self.guidance_scale = guidance_scale
        self.sigma_meas = sigma_meas
        self.num_inference_steps = num_inference_steps
        self.gradient_smoothing_sigma = gradient_smoothing_sigma
        self.patch_filtering_enabled = patch_filtering_enabled
        self.patch_size_u = patch_size_u
        self.patch_size_v = patch_size_v
    
    @torch.no_grad()
    def sample(
        self,
        obs_buffer: ObservationBuffer,
        t_now: float,
        norm_mean: float,
        norm_std: float,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        x_init: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        progress_desc: Optional[str] = None,
    ) -> torch.Tensor:
        """Sample posterior map given observations.
        
        Args:
            obs_buffer: Buffer of observations with constraints
            t_now: Current time (for weight decay)
            norm_mean: Normalization mean
            norm_std: Normalization std
            batch_size: Batch size
            device: Device
            x_init: Optional initial sample (e.g., from warmstart)
            generator: Random generator
        
        Returns:
            Reconstructed map (normalized), shape (B, C, H, W)
        """
        # Set inference scheduler
        self.diffusion_prior.inference_scheduler.set_timesteps(
            self.num_inference_steps, device=device
        )
        
        # Initialize from noise or provided initial sample
        if x_init is None:
            x = torch.randn(
                (batch_size, self.diffusion_prior.in_channels, *self.diffusion_prior.sample_size),
                generator=generator,
                device=device,
            )
        else:
            x = x_init.to(device)
        
        # Get observations and weights (with optional patch filtering)
        if self.patch_filtering_enabled:
            # Use patch-based uniform sampling
            grid_shape = self.diffusion_prior.sample_size  # (N_u, N_v)
            # Seed is None - filtering uses deterministic patch_id-based seeds for reproducibility
            observations, weights = obs_buffer.get_filtered_observations(
                t_now=t_now,
                patch_size_u=self.patch_size_u,
                patch_size_v=self.patch_size_v,
                grid_shape=grid_shape,
                seed=None,  # Uses deterministic patch_id-based seeding internally
            )
        else:
            # Use all observations
            observations = obs_buffer.get_observations()
            weights = obs_buffer.get_weights(t_now)
        
        if len(observations) == 0:
            # No constraints, just sample from prior
            return self.diffusion_prior.sample(
                batch_size=batch_size,
                num_inference_steps=self.num_inference_steps,
                device=device,
                generator=generator,
            )
        
        # Normalize observations to match the model's output space
        # This is critical: diffusion model outputs are normalized!
        from .buffer import Observation
        normalized_observations = [
            Observation(
                i_u=obs.i_u,
                i_v=obs.i_v,
                a_dyn=(obs.a_dyn - norm_mean) / norm_std,  # Normalize
                timestamp=obs.timestamp,
            )
            for obs in observations
        ]
        
        # Adjust sigma_meas to normalized space
        sigma_meas_normalized = self.sigma_meas / norm_std
        
        # Reverse diffusion with guidance on predicted x0
        timesteps = self.diffusion_prior.inference_scheduler.timesteps
        timesteps_iter = tqdm(timesteps, desc=progress_desc, unit="step") if progress_desc else timesteps
        
        for i, t in enumerate(timesteps_iter):
            # Predict noise from prior
            noise_pred = self.diffusion_prior(x, t.unsqueeze(0).expand(batch_size).to(device))
            
            # ===================================================================
            # GUIDANCE: Apply measurement guidance to the predicted clean sample (x0)
            # ===================================================================
            
            if self.guidance_scale > 0 and len(observations) > 0:
                # Get alpha schedule parameters
                alpha_prod_t = self.diffusion_prior.inference_scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                
                # Predict x0 (the clean sample estimate at this step)
                # x0 = (x_t - sqrt(1-alpha_t) * epsilon) / sqrt(alpha_t)
                x0_pred = (x - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                
                # Compute measurement gradient on x0_pred
                x0_pred_np = x0_pred.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
                
                grad_E_np = compute_energy_gradient(
                    x0_pred_np,
                    normalized_observations,
                    weights,
                    sigma_meas_normalized,
                )
                
                # Apply spatial smoothing to gradient if requested
                # This spreads the influence of sparse observations across the map
                if self.gradient_smoothing_sigma > 0:
                    from scipy.ndimage import gaussian_filter
                    grad_E_np = gaussian_filter(grad_E_np, sigma=self.gradient_smoothing_sigma)
                
                # Convert gradient back to torch and clip to prevent instability
                grad_E = torch.from_numpy(grad_E_np).unsqueeze(0).unsqueeze(0).to(device)
                grad_E = torch.clamp(grad_E, -10.0, 10.0)
                
                # Apply guidance: move x0 toward observations
                x0_guided = x0_pred - self.guidance_scale * grad_E
                
                # Compute the implied "corrected noise" from the guided x0
                # epsilon_guided = (x_t - sqrt(alpha_t) * x0_guided) / sqrt(1-alpha_t)
                noise_pred_guided = (x - alpha_prod_t ** 0.5 * x0_guided) / beta_prod_t ** 0.5
                
                # Take diffusion step with the guided noise prediction
                x = self.diffusion_prior.inference_scheduler.step(
                    noise_pred_guided, t, x
                ).prev_sample
            else:
                # No guidance: standard diffusion step
                x = self.diffusion_prior.inference_scheduler.step(
                    noise_pred, t, x
                ).prev_sample
        
        return x

