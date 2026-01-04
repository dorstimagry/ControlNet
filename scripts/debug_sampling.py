#!/usr/bin/env python3
"""Debug sampling quality by testing different number of inference steps."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.models.diffusion_prior import DiffusionPrior
from src.data.gt_map_generator import generate_map_from_params
from utils.dynamics import ExtendedPlantRandomization, sample_extended_params

output_file = open('/tmp/sampling_debug.log', 'w')

def log(msg):
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()

log("="*70)
log("SAMPLING QUALITY DEBUG")
log("="*70)

# Load model
log("\nLoading diffusion prior...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prior = DiffusionPrior.load('training/diffusion_prior/best.pt', device=device)
log(f"Model loaded on {device}")
log(f"Sample size: {prior.sample_size}")

# Load normalization
import json
with open('data/maps/norm_stats.json') as f:
    norm_stats = json.load(f)
norm_mean = norm_stats['mean']
norm_std = norm_stats['std']
log(f"Normalization: mean={norm_mean:.3f}, std={norm_std:.3f}")

# Generate a GT map for reference
log("\nGenerating GT map...")
rand = ExtendedPlantRandomization(
    mass_range=(1200.0, 1800.0),
    wheel_radius_range=(0.3, 0.35),
    brake_accel_range=(8.0, 11.0),
)
rng = np.random.default_rng(42)
params = sample_extended_params(rng, rand)
gt_map = generate_map_from_params(params, N_u=100, N_v=100, v_max=30.0)
gt_map_norm = (gt_map - norm_mean) / norm_std

log(f"GT map: shape={gt_map.shape}, range=[{gt_map.min():.2f}, {gt_map.max():.2f}]")
log(f"GT map normalized: mean={gt_map_norm.mean():.3f}, std={gt_map_norm.std():.3f}")

# Test different numbers of inference steps
test_steps = [20, 50, 100, 200, 500, 1000]
log("\n" + "="*70)
log("TESTING DIFFERENT INFERENCE STEPS")
log("="*70)

results = []
for num_steps in test_steps:
    log(f"\n### Testing {num_steps} inference steps ###")
    
    # Sample from prior
    sample_norm = prior.sample(
        batch_size=1,
        num_inference_steps=num_steps,
        device=device,
    )
    
    # Denormalize
    sample_np = sample_norm.squeeze().cpu().numpy()
    sample = sample_np * norm_std + norm_mean
    
    # Compute statistics
    mse = float(np.mean((sample - gt_map) ** 2))
    mae = float(np.mean(np.abs(sample - gt_map)))
    
    # Compute smoothness (gradient magnitude)
    grad_y, grad_x = np.gradient(sample)
    smoothness = np.sqrt(grad_y**2 + grad_x**2).mean()
    
    log(f"  Sample range: [{sample.min():.2f}, {sample.max():.2f}]")
    log(f"  Sample mean: {sample.mean():.2f}, std: {sample.std():.2f}")
    log(f"  MSE vs GT: {mse:.3f}")
    log(f"  MAE vs GT: {mae:.3f}")
    log(f"  Smoothness (avg gradient): {smoothness:.3f}")
    
    results.append({
        'steps': num_steps,
        'sample': sample,
        'mse': mse,
        'mae': mae,
        'smoothness': smoothness,
    })

# Create visualization
log("\n" + "="*70)
log("Creating visualization...")
log("="*70)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Plot GT
ax = axes[0]
im = ax.imshow(gt_map, aspect='auto', origin='lower', cmap='RdYlGn', vmin=gt_map.min(), vmax=gt_map.max())
ax.set_title(f'Ground Truth\nRange=[{gt_map.min():.1f}, {gt_map.max():.1f}]', fontsize=10)
plt.colorbar(im, ax=ax)

# Plot samples
for idx, result in enumerate(results[:7]):
    ax = axes[idx + 1]
    sample = result['sample']
    im = ax.imshow(sample, aspect='auto', origin='lower', cmap='RdYlGn', vmin=gt_map.min(), vmax=gt_map.max())
    ax.set_title(
        f"{result['steps']} steps\n"
        f"MAE={result['mae']:.2f}, Smooth={result['smoothness']:.3f}",
        fontsize=10
    )
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('/tmp/sampling_steps_comparison.png', dpi=150, bbox_inches='tight')
log("Saved visualization to /tmp/sampling_steps_comparison.png")

# Summary
log("\n" + "="*70)
log("SUMMARY")
log("="*70)
log(f"\n{'Steps':<8} {'MAE':<8} {'Smoothness':<12}")
log("-" * 30)
for r in results:
    log(f"{r['steps']:<8} {r['mae']:<8.3f} {r['smoothness']:<12.3f}")

log("\n" + "="*70)
log("ANALYSIS")
log("="*70)

# Check if smoothness improves with more steps
smoothness_values = [r['smoothness'] for r in results]
if smoothness_values[-1] < smoothness_values[0]:
    log("✓ Smoothness IMPROVES with more steps")
else:
    log("✗ Smoothness does NOT improve with more steps")
    log("  -> This suggests the model itself produces noisy samples")

# Check if MAE improves
mae_values = [r['mae'] for r in results]
if mae_values[-1] < mae_values[0]:
    log("✓ MAE IMPROVES with more steps")
else:
    log("✗ MAE does NOT improve significantly with more steps")

log("\nDone!")
output_file.close()

