#!/usr/bin/env python3
"""Debug guidance pixelation - Phase 1: Perfect GT observations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from src.data.gt_map_generator import generate_map_from_params
from src.maps.buffer import Observation, ObservationBuffer
from src.maps.grid import MapGrid
from src.models.diffusion_prior import DiffusionPrior
from src.maps.sampler_diffusion import GuidedDiffusionSampler
from utils.dynamics import ExtendedPlantRandomization, sample_extended_params
from diffusers import DDPMScheduler

print("="*70)
print("PHASE 1: TESTING GUIDANCE WITH PERFECT GT OBSERVATIONS")
print("="*70)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
grid = MapGrid(N_u=100, N_v=100, v_max=30.0)

# Load model with DDPM
print("\nLoading diffusion prior with DDPM...")
prior = DiffusionPrior.load('training/diffusion_prior/best.pt', device=device)
prior.inference_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear",
    prediction_type="epsilon",
    clip_sample=False,
)
print(f"✓ Model loaded on {device}")

# Load normalization
with open('data/maps/norm_stats.json') as f:
    norm_stats = json.load(f)
norm_mean, norm_std = norm_stats['mean'], norm_stats['std']
print(f"✓ Normalization: mean={norm_mean:.3f}, std={norm_std:.3f}")

# Generate GT map
print("\nGenerating GT map...")
rand = ExtendedPlantRandomization(
    mass_range=(1200.0, 1800.0),
    wheel_radius_range=(0.3, 0.35),
    brake_accel_range=(8.0, 11.0),
)
rng = np.random.default_rng(42)
params = sample_extended_params(rng, rand)
gt_map = generate_map_from_params(params, N_u=100, N_v=100, v_max=30.0)
gt_map_norm = (gt_map - norm_mean) / norm_std

print(f"✓ GT map: range=[{gt_map.min():.2f}, {gt_map.max():.2f}] m/s²")

# ============================================================================
# TEST 1.1: Perfect observations from GT (100 points)
# ============================================================================
print("\n" + "="*70)
print("TEST 1.1: 100 PERFECT GT observations (no noise, no gravity)")
print("="*70)

n_obs = 100
obs_indices = []
observations_norm = []

# Sample random grid locations
rng_obs = np.random.default_rng(123)
for _ in range(n_obs):
    i_u = rng_obs.integers(0, grid.N_u)
    i_v = rng_obs.integers(0, grid.N_v)
    
    # PERFECT observation: exact GT value, normalized
    a_dyn_norm = gt_map_norm[i_u, i_v]
    
    obs_indices.append((i_u, i_v))
    observations_norm.append(Observation(
        i_u=i_u,
        i_v=i_v,
        a_dyn=float(a_dyn_norm),  # Already normalized!
        timestamp=float(_),
    ))

print(f"✓ Created {len(observations_norm)} perfect GT observations")

# Create observation buffer
obs_buffer = ObservationBuffer(capacity=n_obs*2, lambda_decay=0.0, w_min=1.0)
for obs in observations_norm:
    obs_buffer.add(obs.i_u, obs.i_v, obs.a_dyn, obs.timestamp)

# Test with different guidance scales
test_configs = [
    (0.0, "No guidance (prior only)"),
    (0.5, "Weak guidance"),
    (1.0, "Medium guidance"),
    (2.0, "Strong guidance"),
]

results = []
for guidance_scale, desc in test_configs:
    print(f"\n### Testing: {desc} (scale={guidance_scale}) ###")
    
    # Create guided sampler
    sampler = GuidedDiffusionSampler(
        diffusion_prior=prior,
        guidance_scale=guidance_scale,
        sigma_meas=0.3 / norm_std,  # Normalized sigma
        num_inference_steps=50,  # DDPM with 50 steps
    )
    
    # Sample
    recon_norm = sampler.sample(
        obs_buffer=obs_buffer,
        t_now=float(n_obs),
        norm_mean=0.0,  # Already normalized
        norm_std=1.0,    # Already normalized
        batch_size=1,
        device=device,
        x_init=None,
        generator=torch.Generator(device=device).manual_seed(42),
    )
    
    # Denormalize
    recon_np = recon_norm.squeeze().cpu().numpy()
    recon = recon_np * norm_std + norm_mean
    
    # Compute metrics
    mse = float(np.mean((recon - gt_map) ** 2))
    mae = float(np.mean(np.abs(recon - gt_map)))
    
    # Error at observation locations
    obs_errors = []
    for i_u, i_v in obs_indices:
        error = abs(recon[i_u, i_v] - gt_map[i_u, i_v])
        obs_errors.append(error)
    mae_at_obs = np.mean(obs_errors)
    
    # Smoothness
    grad_y, grad_x = np.gradient(recon)
    smoothness = np.sqrt(grad_y**2 + grad_x**2).mean()
    
    print(f"  Overall MAE: {mae:.3f} m/s²")
    print(f"  MAE at observations: {mae_at_obs:.3f} m/s²")
    print(f"  Smoothness (gradient): {smoothness:.3f}")
    
    results.append({
        'guidance_scale': guidance_scale,
        'desc': desc,
        'recon': recon,
        'mae': mae,
        'mae_at_obs': mae_at_obs,
        'smoothness': smoothness,
    })

# ============================================================================
# TEST 1.2: Dense observations (1000 points)
# ============================================================================
print("\n" + "="*70)
print("TEST 1.2: 1000 DENSE PERFECT GT observations")
print("="*70)

n_obs_dense = 1000
observations_dense = []
rng_obs2 = np.random.default_rng(456)

for _ in range(n_obs_dense):
    i_u = rng_obs2.integers(0, grid.N_u)
    i_v = rng_obs2.integers(0, grid.N_v)
    a_dyn_norm = gt_map_norm[i_u, i_v]
    observations_dense.append(Observation(
        i_u=i_u, i_v=i_v,
        a_dyn=float(a_dyn_norm),
        timestamp=float(_),
    ))

obs_buffer_dense = ObservationBuffer(capacity=n_obs_dense*2, lambda_decay=0.0, w_min=1.0)
for obs in observations_dense:
    obs_buffer_dense.add(obs.i_u, obs.i_v, obs.a_dyn, obs.timestamp)

print(f"✓ Created {n_obs_dense} dense observations")
print("Testing with guidance_scale=1.0...")

sampler_dense = GuidedDiffusionSampler(
    diffusion_prior=prior,
    guidance_scale=1.0,
    sigma_meas=0.3 / norm_std,
    num_inference_steps=50,
)

recon_dense_norm = sampler_dense.sample(
    obs_buffer=obs_buffer_dense,
    t_now=float(n_obs_dense),
    norm_mean=0.0,
    norm_std=1.0,
    batch_size=1,
    device=device,
    x_init=None,
    generator=torch.Generator(device=device).manual_seed(42),
)

recon_dense = recon_dense_norm.squeeze().cpu().numpy() * norm_std + norm_mean
mae_dense = float(np.mean(np.abs(recon_dense - gt_map)))
grad_y, grad_x = np.gradient(recon_dense)
smoothness_dense = np.sqrt(grad_y**2 + grad_x**2).mean()

print(f"  Overall MAE: {mae_dense:.3f} m/s²")
print(f"  Smoothness: {smoothness_dense:.3f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("Creating visualizations...")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Different guidance scales (100 obs)
ax = axes[0, 0]
im = ax.imshow(gt_map, aspect='auto', origin='lower', cmap='RdYlGn')
ax.set_title('Ground Truth\n(Target)', fontsize=12, fontweight='bold')
# Mark observation locations
obs_i_u = [idx[0] for idx in obs_indices[:50]]  # Show first 50
obs_i_v = [idx[1] for idx in obs_indices[:50]]
ax.scatter(obs_i_v, obs_i_u, c='black', s=10, alpha=0.5, marker='x')
plt.colorbar(im, ax=ax)

for idx, result in enumerate([results[0], results[2]]):  # prior, medium guidance
    ax = axes[0, idx + 1]
    im = ax.imshow(result['recon'], aspect='auto', origin='lower', cmap='RdYlGn',
                   vmin=gt_map.min(), vmax=gt_map.max())
    ax.set_title(
        f"{result['desc']}\n"
        f"MAE={result['mae']:.2f}, Smooth={result['smoothness']:.2f}",
        fontsize=11
    )
    plt.colorbar(im, ax=ax)

# Row 2: Dense observations
ax = axes[1, 0]
im = ax.imshow(gt_map, aspect='auto', origin='lower', cmap='RdYlGn')
ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax)

ax = axes[1, 1]
im = ax.imshow(recon_dense, aspect='auto', origin='lower', cmap='RdYlGn',
               vmin=gt_map.min(), vmax=gt_map.max())
ax.set_title(
    f"1000 observations (guidance=1.0)\n"
    f"MAE={mae_dense:.2f}, Smooth={smoothness_dense:.2f}",
    fontsize=11
)
plt.colorbar(im, ax=ax)

# Error map
ax = axes[1, 2]
error_map = np.abs(recon_dense - gt_map)
im = ax.imshow(error_map, aspect='auto', origin='lower', cmap='hot')
ax.set_title(f"Absolute Error\nMax={error_map.max():.2f}", fontsize=11)
plt.colorbar(im, ax=ax)

plt.suptitle('Phase 1: Perfect GT Observations Test', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path('evaluation/phase1_perfect_observations.png')
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to {output_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PHASE 1 SUMMARY")
print("="*70)

print("\nWith 100 perfect GT observations:")
print(f"{'Guidance Scale':<15} {'MAE':<10} {'MAE@Obs':<12} {'Smoothness':<12}")
print("-" * 50)
for r in results:
    print(f"{r['guidance_scale']:<15} {r['mae']:<10.3f} {r['mae_at_obs']:<12.3f} {r['smoothness']:<12.3f}")

print(f"\nWith 1000 perfect GT observations:")
print(f"  MAE: {mae_dense:.3f}")
print(f"  Smoothness: {smoothness_dense:.3f}")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

if results[2]['mae'] < results[0]['mae']:
    print("✓ Guidance HELPS (MAE decreases with guidance)")
else:
    print("✗ Guidance HURTS (MAE increases with guidance)")

if results[2]['smoothness'] > results[0]['smoothness'] * 1.5:
    print("✗ Guidance creates PIXELATION (smoothness >>)")
    print("  → Need to smooth the guidance gradient!")
else:
    print("✓ Guidance preserves smoothness")

if mae_dense < results[2]['mae'] * 0.7:
    print("✓ More observations help significantly")
else:
    print("~ More observations help marginally")

print("\nNext step: Phase 2 - Apply gradient smoothing to fix pixelation")

