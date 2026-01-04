#!/usr/bin/env python3
"""Quick visual test: DDPM vs DDIM sampling quality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from src.models.diffusion_prior import DiffusionPrior
from src.data.gt_map_generator import generate_map_from_params
from utils.dynamics import ExtendedPlantRandomization, sample_extended_params
from diffusers import DDPMScheduler, DDIMScheduler

print("Comparing DDPM vs DDIM sampling quality...")

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prior = DiffusionPrior.load('training/diffusion_prior/best.pt', device=device)

with open('data/maps/norm_stats.json') as f:
    norm_stats = json.load(f)
norm_mean, norm_std = norm_stats['mean'], norm_stats['std']

# Generate GT
rand = ExtendedPlantRandomization(
    mass_range=(1200.0, 1800.0),
    wheel_radius_range=(0.3, 0.35),
    brake_accel_range=(8.0, 11.0),
)
rng = np.random.default_rng(42)
params = sample_extended_params(rng, rand)
gt_map = generate_map_from_params(params, N_u=100, N_v=100, v_max=30.0)

# Test configurations
tests = [
    ("DDIM 50 steps", DDIMScheduler(num_train_timesteps=1000), 50),
    ("DDIM 200 steps", DDIMScheduler(num_train_timesteps=1000), 200),
    ("DDPM 200 steps", DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", 
                                      prediction_type="epsilon", clip_sample=False), 200),
    ("DDPM 500 steps", DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                                      prediction_type="epsilon", clip_sample=False), 500),
]

samples = []
for name, scheduler, steps in tests:
    print(f"\n{name}...")
    prior.inference_scheduler = scheduler
    
    sample_norm = prior.sample(batch_size=1, num_inference_steps=steps, device=device)
    sample_np = sample_norm.squeeze().cpu().numpy()
    sample = sample_np * norm_std + norm_mean
    
    # Metrics
    mae = np.mean(np.abs(sample - gt_map))
    grad_y, grad_x = np.gradient(sample)
    smoothness = np.sqrt(grad_y**2 + grad_x**2).mean()
    
    print(f"  MAE: {mae:.3f}, Smoothness: {smoothness:.3f}")
    samples.append((name, sample, mae, smoothness))

# Plot
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

# GT
ax = axes[0]
im = ax.imshow(gt_map, aspect='auto', origin='lower', cmap='RdYlGn')
ax.set_title(f'Ground Truth\n(smooth)', fontsize=12, fontweight='bold')
ax.set_xlabel('Speed Index')
ax.set_ylabel('Actuation Index')
plt.colorbar(im, ax=ax, label='Accel (m/s²)')

# Samples
for idx, (name, sample, mae, smoothness) in enumerate(samples):
    ax = axes[idx + 1]
    im = ax.imshow(sample, aspect='auto', origin='lower', cmap='RdYlGn',
                   vmin=gt_map.min(), vmax=gt_map.max())
    ax.set_title(f'{name}\nMAE={mae:.2f}, Smooth={smoothness:.2f}', fontsize=11)
    ax.set_xlabel('Speed Index')
    ax.set_ylabel('Actuation Index')
    plt.colorbar(im, ax=ax, label='Accel (m/s²)')

plt.suptitle('Sampling Quality: DDPM vs DDIM', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path('evaluation/ddpm_vs_ddim_comparison.png')
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved comparison to {output_path}")
print("\nConclusion: DDPM with 200+ steps should produce smooth samples!")

