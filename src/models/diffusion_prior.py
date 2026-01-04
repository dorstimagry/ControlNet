"""Diffusion prior model for acceleration maps using HuggingFace Diffusers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel


class DiffusionPrior(nn.Module):
    """Diffusion prior for acceleration maps.
    
    Wraps a UNet2DModel from HuggingFace Diffusers with DDPM/DDIM schedulers.
    Trained to denoise maps and can sample new maps or perform guided reconstruction.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        sample_size: Tuple[int, int] = (100, 100),
        block_out_channels: Tuple[int, ...] = (32, 64),
        down_block_types: Optional[Tuple[str, ...]] = None,
        up_block_types: Optional[Tuple[str, ...]] = None,
        num_train_timesteps: int = 1000,
    ):
        """Initialize diffusion prior.
        
        Args:
            in_channels: Input channels (1 for grayscale maps)
            out_channels: Output channels (1 for grayscale maps)
            sample_size: (height, width) of maps (N_u, N_v)
            block_out_channels: Channel dimensions for UNet blocks
            down_block_types: Types of down blocks (auto-generated if None)
            up_block_types: Types of up blocks (auto-generated if None)
            num_train_timesteps: Number of diffusion timesteps
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_size = sample_size
        self.num_train_timesteps = num_train_timesteps
        
        # Auto-generate block types if not provided
        num_blocks = len(block_out_channels)
        if down_block_types is None:
            down_block_types = ("DownBlock2D",) * num_blocks
        if up_block_types is None:
            up_block_types = ("UpBlock2D",) * num_blocks
        
        # Create UNet model
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=2,
        )
        
        # Create schedulers
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        self.inference_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass through UNet (noise prediction).
        
        Args:
            x: Noisy samples, shape (B, C, H, W)
            timesteps: Timestep indices, shape (B,)
        
        Returns:
            Predicted noise, shape (B, C, H, W)
        """
        return self.unet(x, timesteps).sample
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 50,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample maps from prior using DDIM.
        
        Args:
            batch_size: Number of samples to generate
            num_inference_steps: Number of DDIM steps
            device: Device to sample on
            generator: Random generator for reproducibility
        
        Returns:
            Sampled maps, shape (B, C, H, W)
        """
        # Set inference scheduler
        self.inference_scheduler.set_timesteps(num_inference_steps, device=device)
        
        # Start from pure noise
        x = torch.randn(
            (batch_size, self.in_channels, *self.sample_size),
            generator=generator,
            device=device,
        )
        
        # Reverse diffusion process
        for t in self.inference_scheduler.timesteps:
            # Predict noise
            noise_pred = self(x, t.unsqueeze(0).expand(batch_size).to(device))
            
            # Compute previous sample
            x = self.inference_scheduler.step(noise_pred, t, x).prev_sample
        
        return x
    
    def compute_loss(
        self,
        x_start: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Compute DDPM training loss (noise prediction MSE).
        
        Args:
            x_start: Clean samples, shape (B, C, H, W)
            device: Device for computation
        
        Returns:
            MSE loss scalar
        """
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        
        # Sample noise
        noise = torch.randn_like(x_start, device=device)
        
        # Add noise to clean samples
        noisy_x = self.noise_scheduler.add_noise(x_start, noise, timesteps)
        
        # Predict noise
        noise_pred = self(noisy_x, timesteps)
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return loss
    
    def save(self, path: Path) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "unet": self.unet.state_dict(),
            "config": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "sample_size": self.sample_size,
                "num_train_timesteps": self.num_train_timesteps,
                "block_out_channels": self.unet.config.block_out_channels,
                "down_block_types": self.unet.config.down_block_types,
                "up_block_types": self.unet.config.up_block_types,
            },
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: Path, device: torch.device = torch.device("cpu")) -> DiffusionPrior:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load to
        
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        
        model = cls(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            sample_size=tuple(config["sample_size"]),
            block_out_channels=tuple(config["block_out_channels"]),
            down_block_types=tuple(config["down_block_types"]),
            up_block_types=tuple(config["up_block_types"]),
            num_train_timesteps=config["num_train_timesteps"],
        )
        
        model.unet.load_state_dict(checkpoint["unet"])
        model.to(device)
        
        return model

