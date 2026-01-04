#!/usr/bin/env python3
"""Test different schedulers and step counts for sampling quality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from src.models.diffusion_prior import DiffusionPrior
from src.data.gt_map_generator import generate_map_from_params
from utils.dynamics import ExtendedPlantRandomization, sample_extended_params
from diffusers import DDPMScheduler

print("="*70)
print("SCHEDULER COMPARISON TEST")
print("="*70)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prior = DiffusionPrior.load('training/diffusion_prior/best.pt', device=device)
print(f"Model loaded on {device}")

# Load normalization
with open('data/maps/norm_stats.json') as f:
    norm_stats = json.load(f)
norm_mean = norm_stats['mean']
norm_std = norm_stats['std']

# Generate GT map
rand = ExtendedPlantRandomization(
    mass_range=(1200.0, 1800.0),
    wheel_radius_range=(0.3, 0.35),
    brake_accel_range=(8.0, 11.0),
)
rng = np.random.default_rng(42)
params = sample_extended_params(rng, rand)
gt_map = generate_map_from_params(params, N_u=100, N_v=100, v_max=30.0)

print(f"\nGT map range: [{gt_map.min():.2f}, {gt_map.max():.2f}]")

# Test configurations
configs = [
    ('DDIM', 50),
    ('DDIM', 200),
    ('DDIM', 1000),
    ('DDPM', 200),
    ('DDPM', 500),
    ('DDPM', 1000),
]

results = []
for scheduler_name, num_steps in configs:
    print(f"\nTesting {scheduler_name} with {num_steps} steps...")
    
    # Set up scheduler
    if scheduler_name == 'DDPM':
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        prior.inference_scheduler = scheduler
    else:
        # Use existing DDIM scheduler
        pass
    
    # Sample
    prior.inference_scheduler.set_timesteps(num_steps, device=device)
    
    x = torch.randn((1, 1, 100, 100), device=device)
    for t in prior.inference_scheduler.timesteps:
        noise_pred = prior(x, t.unsqueeze(0).to(device))
        x = prior.inference_scheduler.step(noise_pred, t, x).prev_sample
    
    # Denormalize
    sample_np = x.squeeze().cpu().numpy()
    sample = sample_np * norm_std + norm_mean
    
    # Compute metrics
    mse = float(np.mean((sample - gt_map) ** 2))
    mae = float(np.mean(np.abs(sample - gt_map)))
    
    # Smoothness
    grad_y, grad_x = np.gradient(sample)
    smoothness = np.sqrt(grad_y**2 + grad_x**2).mean()
    
    print(f"  MAE: {mae:.3f}, Smoothness: {smoothness:.3f}")
    
    results.append({
        'scheduler': scheduler_name,
        'steps': num_steps,
        'sample': sample,
        'mae': mae,
        'smoothness': smoothness,
    })

# Plot
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# GT
ax = axes[0]
im = ax.imshow(gt_map, aspect='auto', origin='lower', cmap='RdYlGn')
ax.set_title('Ground Truth', fontsize=12)
plt.colorbar(im, ax=ax)

# Samples
for idx, result in enumerate(results):
    ax = axes[idx + 1]
    im = ax.imshow(result['sample'], aspect='auto', origin='lower', cmap='RdYlGn',
                   vmin=gt_map.min(), vmax=gt_map.max())
    ax.set_title(
        f"{result['scheduler']} {result['steps']} steps\n"
        f"MAE={result['mae']:.2f}, Smooth={result['smoothness']:.2f}",
        fontsize=10
    )
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('evaluation/scheduler_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved to evaluation/scheduler_comparison.png")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Scheduler':<10} {'Steps':<8} {'MAE':<8} {'Smoothness':<12}")
print("-" * 40)
for r in results:
    print(f"{r['scheduler']:<10} {r['steps']:<8} {r['mae']:<8.3f} {r['smoothness']:<12.3f}")

