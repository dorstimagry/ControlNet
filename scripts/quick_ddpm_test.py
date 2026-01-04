#!/usr/bin/env python3
"""Evaluate diffusion with DDPM scheduler for better quality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Run a quick test with DDPM
import torch
import numpy as np
import json
from src.models.diffusion_prior import DiffusionPrior
from diffusers import DDPMScheduler

print("Testing DDPM scheduler...")

# Load
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prior = DiffusionPrior.load('training/diffusion_prior/best.pt', device=device)

# Replace inference scheduler with DDPM
prior.inference_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear",  # Same as training
    prediction_type="epsilon",
    clip_sample=False,  # IMPORTANT: Don't clip!
)

# Load norm stats
with open('data/maps/norm_stats.json') as f:
    norm_stats = json.load(f)

# Sample
print("Sampling with DDPM, 200 steps...")
sample_norm = prior.sample(batch_size=1, num_inference_steps=200, device=device)
sample_np = sample_norm.squeeze().cpu().numpy()
sample = sample_np * norm_stats['std'] + norm_stats['mean']

print(f"Sample range: [{sample.min():.2f}, {sample.max():.2f}]")
print(f"Sample mean: {sample.mean():.2f}, std: {sample.std():.2f}")

# Check smoothness
grad_y, grad_x = np.gradient(sample)
smoothness = np.sqrt(grad_y**2 + grad_x**2).mean()
print(f"Smoothness (gradient mag): {smoothness:.3f}")

print("Done! Now update evaluate_diffusion.py to use DDPM")

