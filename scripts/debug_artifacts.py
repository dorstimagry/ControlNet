#!/usr/bin/env python3
"""Debug artifacts in guided diffusion reconstruction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.ndimage import gaussian_filter

from src.data.gt_map_generator import generate_map_from_params
from src.maps.buffer import Observation, ObservationBuffer
from src.maps.grid import MapGrid
from src.maps.energy import compute_energy_gradient
from src.models.diffusion_prior import DiffusionPrior
from src.maps.sampler_diffusion import GuidedDiffusionSampler
from utils.dynamics import ExtendedPlantRandomization, sample_extended_params
from diffusers import DDPMScheduler

sns.set_style("whitegrid")

print("="*80)
print("DEBUGGING GUIDANCE ARTIFACTS")
print("="*80)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
print(f"\nDevice: {device}")
print(f"Grid: {grid.N_u}x{grid.N_v}")

# Load model
print("\nLoading diffusion prior...")
prior = DiffusionPrior.load('training/diffusion_prior/best.pt', device=device)
prior.inference_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear",
    prediction_type="epsilon",
    clip_sample=False,
)

# Load normalization
with open('data/maps/norm_stats.json') as f:
    norm_stats = json.load(f)
norm_mean, norm_std = norm_stats['mean'], norm_stats['std']
print(f"✓ Normalization: mean={norm_mean:.3f}, std={norm_std:.3f}")

# Generate GT map
print("\nGenerating test map...")
rand = ExtendedPlantRandomization(
    mass_range=(1200.0, 1800.0),
    wheel_radius_range=(0.3, 0.35),
    brake_accel_range=(8.0, 11.0),
)
rng = np.random.default_rng(42)
params = sample_extended_params(rng, rand)
gt_map = generate_map_from_params(params, N_u=100, N_v=100, v_max=30.0)
gt_map_norm = (gt_map - norm_mean) / norm_std

print(f"✓ GT map range: [{gt_map.min():.2f}, {gt_map.max():.2f}] m/s²")

# ==============================================================================
# TEST 1.1: Check observation quality
# ==============================================================================
print("\n" + "="*80)
print("TEST 1.1: OBSERVATION QUALITY ANALYSIS")
print("="*80)

n_obs = 200
sigma_noise = 0.3  # Same as in evaluate_diffusion.py

print(f"\nGenerating {n_obs} observations with σ_noise={sigma_noise} m/s²")

rng_obs = np.random.RandomState(123)
observations_noisy = []
obs_errors = []

for i in range(n_obs):
    i_u = rng_obs.randint(0, grid.N_u)
    i_v = rng_obs.randint(0, grid.N_v)
    
    a_true = gt_map[i_u, i_v]
    a_noisy = a_true + rng_obs.normal(0, sigma_noise)
    error = abs(a_noisy - a_true)
    obs_errors.append(error)
    
    observations_noisy.append(Observation(
        i_u=i_u, i_v=i_v,
        a_dyn=float(a_noisy),
        timestamp=float(i),
    ))

print(f"Observation errors - Mean: {np.mean(obs_errors):.3f}, Std: {np.std(obs_errors):.3f}, Max: {np.max(obs_errors):.3f}")

# Also create perfect observations for comparison
observations_perfect = []
for i in range(n_obs):
    i_u = observations_noisy[i].i_u
    i_v = observations_noisy[i].i_v
    a_true = gt_map[i_u, i_v]
    
    observations_perfect.append(Observation(
        i_u=i_u, i_v=i_v,
        a_dyn=float(a_true),
        timestamp=float(i),
    ))

# ==============================================================================
# TEST 1.2: Visualize gradient behavior
# ==============================================================================
print("\n" + "="*80)
print("TEST 1.2: GRADIENT VISUALIZATION")
print("="*80)

# Create buffer with noisy observations (normalized)
obs_buffer = ObservationBuffer(capacity=n_obs*2, lambda_decay=0.0, w_min=1.0)

observations_norm = []
for obs in observations_noisy:
    a_dyn_norm = (obs.a_dyn - norm_mean) / norm_std
    observations_norm.append(Observation(
        i_u=obs.i_u, i_v=obs.i_v,
        a_dyn=float(a_dyn_norm),
        timestamp=obs.timestamp,
    ))
    obs_buffer.add(obs.i_u, obs.i_v, a_dyn_norm, obs.timestamp)

weights = obs_buffer.get_weights(float(n_obs))
sigma_meas_norm = 0.3 / norm_std

print(f"\nComputing gradient on GT map (simulating one diffusion step)...")
# Use normalized GT as x0_pred to see what gradient looks like
grad_E = compute_energy_gradient(
    gt_map_norm,
    observations_norm,
    weights,
    sigma_meas_norm,
)

print(f"Gradient stats:")
print(f"  Range: [{grad_E.min():.3f}, {grad_E.max():.3f}]")
print(f"  Mean abs: {np.abs(grad_E).mean():.3f}")
print(f"  Std: {grad_E.std():.3f}")
print(f"  Non-zero pixels: {np.count_nonzero(np.abs(grad_E) > 1e-6)}/{grad_E.size}")

# Apply smoothing
grad_E_smooth = gaussian_filter(grad_E, sigma=2.0)
print(f"\nAfter smoothing (σ=2.0):")
print(f"  Range: [{grad_E_smooth.min():.3f}, {grad_E_smooth.max():.3f}]")
print(f"  Mean abs: {np.abs(grad_E_smooth).mean():.3f}")
print(f"  Std: {grad_E_smooth.std():.3f}")
print(f"  Non-zero pixels: {np.count_nonzero(np.abs(grad_E_smooth) > 1e-6)}/{grad_E_smooth.size}")

# ==============================================================================
# TEST 1.3: Reconstruct with perfect vs noisy observations
# ==============================================================================
print("\n" + "="*80)
print("TEST 1.3: PERFECT VS NOISY OBSERVATIONS")
print("="*80)

def reconstruct(observations, label, sigma_meas, guidance_scale, gradient_smoothing_sigma):
    """Helper to reconstruct with given parameters."""
    # Normalize observations
    obs_norm = [
        Observation(
            i_u=obs.i_u, i_v=obs.i_v,
            a_dyn=(obs.a_dyn - norm_mean) / norm_std,
            timestamp=obs.timestamp,
        )
        for obs in observations
    ]
    
    # Create buffer
    obs_buffer_local = ObservationBuffer(capacity=len(observations)*2, lambda_decay=0.0, w_min=1.0)
    for obs in obs_norm:
        obs_buffer_local.add(obs.i_u, obs.i_v, obs.a_dyn, obs.timestamp)
    
    # Create sampler
    sampler = GuidedDiffusionSampler(
        diffusion_prior=prior,
        guidance_scale=guidance_scale,
        sigma_meas=sigma_meas / norm_std,
        num_inference_steps=50,
        gradient_smoothing_sigma=gradient_smoothing_sigma,
    )
    
    # Sample
    recon_norm = sampler.sample(
        obs_buffer=obs_buffer_local,
        t_now=float(len(observations)),
        norm_mean=0.0,
        norm_std=1.0,
        batch_size=1,
        device=device,
        x_init=None,
        generator=torch.Generator(device=device).manual_seed(42),
    )
    
    # Denormalize
    recon = recon_norm.squeeze().cpu().numpy() * norm_std + norm_mean
    
    # Metrics
    mae = float(np.mean(np.abs(recon - gt_map)))
    mse = float(np.mean((recon - gt_map) ** 2))
    
    # Smoothness
    grad_y, grad_x = np.gradient(recon)
    smoothness = np.sqrt(grad_y**2 + grad_x**2).mean()
    
    print(f"\n{label}:")
    print(f"  MAE: {mae:.3f} m/s²")
    print(f"  MSE: {mse:.3f}")
    print(f"  Smoothness: {smoothness:.3f}")
    
    return recon, mae, smoothness

# Test with perfect observations
print("\n### Current parameters (σ_meas=0.3, guidance=1.0, smoothing=2.0) ###")
recon_perfect, mae_perfect, smooth_perfect = reconstruct(
    observations_perfect, "Perfect observations", 
    sigma_meas=0.3, guidance_scale=1.0, gradient_smoothing_sigma=2.0
)

# Test with noisy observations
recon_noisy, mae_noisy, smooth_noisy = reconstruct(
    observations_noisy, "Noisy observations",
    sigma_meas=0.3, guidance_scale=1.0, gradient_smoothing_sigma=2.0
)

# ==============================================================================
# TEST 2: Parameter sweep
# ==============================================================================
print("\n" + "="*80)
print("TEST 2: PARAMETER SWEEP (with noisy observations)")
print("="*80)

results = []

# Test different sigma_meas values
print("\n### Varying sigma_meas (guidance=1.0, smoothing=2.0) ###")
for sigma_meas in [0.3, 0.5, 1.0, 2.0]:
    recon, mae, smooth = reconstruct(
        observations_noisy, f"σ_meas={sigma_meas}",
        sigma_meas=sigma_meas, guidance_scale=1.0, gradient_smoothing_sigma=2.0
    )
    results.append({
        'type': 'sigma_meas',
        'value': sigma_meas,
        'mae': mae,
        'smoothness': smooth,
        'recon': recon,
    })

# Test different guidance_scale values
print("\n### Varying guidance_scale (σ_meas=0.3, smoothing=2.0) ###")
for guidance_scale in [0.1, 0.3, 0.5, 1.0]:
    recon, mae, smooth = reconstruct(
        observations_noisy, f"guidance={guidance_scale}",
        sigma_meas=0.3, guidance_scale=guidance_scale, gradient_smoothing_sigma=2.0
    )
    results.append({
        'type': 'guidance_scale',
        'value': guidance_scale,
        'mae': mae,
        'smoothness': smooth,
        'recon': recon,
    })

# Test different smoothing values
print("\n### Varying gradient_smoothing_sigma (σ_meas=0.3, guidance=1.0) ###")
for smoothing_sigma in [2.0, 5.0, 10.0, 20.0]:
    recon, mae, smooth = reconstruct(
        observations_noisy, f"smoothing={smoothing_sigma}",
        sigma_meas=0.3, guidance_scale=1.0, gradient_smoothing_sigma=smoothing_sigma
    )
    results.append({
        'type': 'smoothing',
        'value': smoothing_sigma,
        'mae': mae,
        'smoothness': smooth,
        'recon': recon,
    })

# ==============================================================================
# VISUALIZATION
# ==============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

output_dir = Path('evaluation/artifact_debug')
output_dir.mkdir(exist_ok=True, parents=True)

# Plot 1: Perfect vs Noisy observations
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
im = ax.imshow(gt_map, aspect='auto', origin='lower', cmap='RdYlGn')
ax.set_title('Ground Truth', fontweight='bold')
plt.colorbar(im, ax=ax)

ax = axes[1]
im = ax.imshow(recon_perfect, aspect='auto', origin='lower', cmap='RdYlGn',
               vmin=gt_map.min(), vmax=gt_map.max())
ax.set_title(f'Perfect Obs\nMAE={mae_perfect:.2f}, S={smooth_perfect:.2f}')
plt.colorbar(im, ax=ax)

ax = axes[2]
im = ax.imshow(recon_noisy, aspect='auto', origin='lower', cmap='RdYlGn',
               vmin=gt_map.min(), vmax=gt_map.max())
ax.set_title(f'Noisy Obs\nMAE={mae_noisy:.2f}, S={smooth_noisy:.2f}')
plt.colorbar(im, ax=ax)

plt.suptitle(f'Perfect vs Noisy Observations ({n_obs} obs, σ_noise=0.3 m/s²)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'perfect_vs_noisy.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved {output_dir / 'perfect_vs_noisy.png'}")
plt.close()

# Plot 2: Gradient visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
# Create observation mask
obs_mask = np.zeros_like(gt_map)
for obs in observations_noisy:
    obs_mask[obs.i_u, obs.i_v] += 1
im = ax.imshow(obs_mask, aspect='auto', origin='lower', cmap='hot')
ax.set_title(f'Observation Density\n(max={obs_mask.max():.0f} at one pixel)')
plt.colorbar(im, ax=ax)

ax = axes[1]
im = ax.imshow(grad_E, aspect='auto', origin='lower', cmap='RdBu_r', 
               vmin=-abs(grad_E).max(), vmax=abs(grad_E).max())
ax.set_title(f'Raw Gradient\nRange=[{grad_E.min():.2f}, {grad_E.max():.2f}]')
plt.colorbar(im, ax=ax)

ax = axes[2]
im = ax.imshow(grad_E_smooth, aspect='auto', origin='lower', cmap='RdBu_r',
               vmin=-abs(grad_E_smooth).max(), vmax=abs(grad_E_smooth).max())
ax.set_title(f'Smoothed (σ=2.0)\nRange=[{grad_E_smooth.min():.2f}, {grad_E_smooth.max():.2f}]')
plt.colorbar(im, ax=ax)

plt.suptitle('Gradient Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'gradient_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved {output_dir / 'gradient_analysis.png'}")
plt.close()

# Plot 3: Parameter sweep results
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# sigma_meas sweep
sigma_results = [r for r in results if r['type'] == 'sigma_meas']
for idx, result in enumerate(sigma_results):
    ax = axes[0, idx]
    im = ax.imshow(result['recon'], aspect='auto', origin='lower', cmap='RdYlGn',
                   vmin=gt_map.min(), vmax=gt_map.max())
    ax.set_title(f"σ_meas={result['value']:.1f}\nMAE={result['mae']:.2f}, S={result['smoothness']:.2f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

# guidance_scale sweep
guidance_results = [r for r in results if r['type'] == 'guidance_scale']
for idx, result in enumerate(guidance_results):
    ax = axes[1, idx]
    im = ax.imshow(result['recon'], aspect='auto', origin='lower', cmap='RdYlGn',
                   vmin=gt_map.min(), vmax=gt_map.max())
    ax.set_title(f"guidance={result['value']:.1f}\nMAE={result['mae']:.2f}, S={result['smoothness']:.2f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

# smoothing sweep
smoothing_results = [r for r in results if r['type'] == 'smoothing']
for idx, result in enumerate(smoothing_results):
    ax = axes[2, idx]
    im = ax.imshow(result['recon'], aspect='auto', origin='lower', cmap='RdYlGn',
                   vmin=gt_map.min(), vmax=gt_map.max())
    ax.set_title(f"smooth={result['value']:.1f}\nMAE={result['mae']:.2f}, S={result['smoothness']:.2f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

axes[0, 0].set_ylabel('σ_meas sweep', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('guidance sweep', fontsize=12, fontweight='bold')
axes[2, 0].set_ylabel('smoothing sweep', fontsize=12, fontweight='bold')

plt.suptitle(f'Parameter Sweep ({n_obs} noisy observations)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'parameter_sweep.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved {output_dir / 'parameter_sweep.png'}")
plt.close()

# Save numerical results
results_json = {
    'perfect_obs': {'mae': float(mae_perfect), 'smoothness': float(smooth_perfect)},
    'noisy_obs': {'mae': float(mae_noisy), 'smoothness': float(smooth_noisy)},
    'parameter_sweep': [
        {k: v for k, v in r.items() if k != 'recon'}
        for r in results
    ]
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"✓ Saved {output_dir / 'results.json'}")

# ==============================================================================
# ANALYSIS & RECOMMENDATIONS
# ==============================================================================
print("\n" + "="*80)
print("ANALYSIS & RECOMMENDATIONS")
print("="*80)

print(f"\n1. PERFECT VS NOISY OBSERVATIONS:")
print(f"   Perfect: MAE={mae_perfect:.3f}, Smoothness={smooth_perfect:.3f}")
print(f"   Noisy:   MAE={mae_noisy:.3f}, Smoothness={smooth_noisy:.3f}")

if mae_perfect < 0.5 and mae_noisy > 1.5:
    print("   → ISSUE: Noisy observations cause significantly higher MAE")
    print("   → Need to increase σ_meas or reduce guidance_scale")

if smooth_noisy > smooth_perfect * 1.5:
    print("   → ISSUE: Noisy observations increase roughness")
    print("   → Artifacts are likely from over-confident guidance")

print(f"\n2. BEST σ_meas:")
best_sigma = min(sigma_results, key=lambda r: r['mae'])
print(f"   σ_meas={best_sigma['value']}: MAE={best_sigma['mae']:.3f}, Smoothness={best_sigma['smoothness']:.3f}")

print(f"\n3. BEST guidance_scale:")
best_guidance = min(guidance_results, key=lambda r: r['mae'])
print(f"   guidance={best_guidance['value']}: MAE={best_guidance['mae']:.3f}, Smoothness={best_guidance['smoothness']:.3f}")

print(f"\n4. BEST smoothing_sigma:")
best_smoothing = min(smoothing_results, key=lambda r: r['mae'])
print(f"   smoothing={best_smoothing['value']}: MAE={best_smoothing['mae']:.3f}, Smoothness={best_smoothing['smoothness']:.3f}")

print("\n" + "="*80)
print("COMPLETE - Check evaluation/artifact_debug/ for visualizations")
print("="*80)

