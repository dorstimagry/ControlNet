#!/usr/bin/env python3
"""Simplified Phase 1 test - writes to file."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import json

# Redirect output
output_file = Path('evaluation/phase1_log.txt')
output_file.parent.mkdir(exist_ok=True)
log = open(output_file, 'w')

def write_log(msg):
    print(msg)
    log.write(msg + '\n')
    log.flush()

try:
    write_log("="*70)
    write_log("PHASE 1: PERFECT GT OBSERVATIONS TEST")
    write_log("="*70)
    
    from src.data.gt_map_generator import generate_map_from_params
    from src.maps.buffer import Observation, ObservationBuffer
    from src.maps.grid import MapGrid
    from src.models.diffusion_prior import DiffusionPrior
    from src.maps.sampler_diffusion import GuidedDiffusionSampler
    from utils.dynamics import ExtendedPlantRandomization, sample_extended_params
    from diffusers import DDPMScheduler
    
    write_log("\n✓ Imports successful")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write_log(f"✓ Device: {device}")
    
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    write_log(f"✓ Grid created: {grid.N_u}x{grid.N_v}")
    
    # Load model
    write_log("\nLoading diffusion prior...")
    prior = DiffusionPrior.load('training/diffusion_prior/best.pt', device=device)
    prior.inference_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    write_log("✓ Model loaded")
    
    # Load normalization
    with open('data/maps/norm_stats.json') as f:
        norm_stats = json.load(f)
    norm_mean, norm_std = norm_stats['mean'], norm_stats['std']
    write_log(f"✓ Normalization: mean={norm_mean:.3f}, std={norm_std:.3f}")
    
    # Generate GT map
    write_log("\nGenerating GT map...")
    rand = ExtendedPlantRandomization(
        mass_range=(1200.0, 1800.0),
        wheel_radius_range=(0.3, 0.35),
        brake_accel_range=(8.0, 11.0),
    )
    rng = np.random.default_rng(42)
    params = sample_extended_params(rng, rand)
    gt_map = generate_map_from_params(params, N_u=100, N_v=100, v_max=30.0)
    gt_map_norm = (gt_map - norm_mean) / norm_std
    write_log(f"✓ GT map: range=[{gt_map.min():.2f}, {gt_map.max():.2f}] m/s²")
    
    # Create 100 perfect observations
    write_log("\nCreating 100 perfect GT observations...")
    n_obs = 100
    observations_norm = []
    rng_obs = np.random.default_rng(123)
    
    for i in range(n_obs):
        i_u = rng_obs.integers(0, grid.N_u)
        i_v = rng_obs.integers(0, grid.N_v)
        a_dyn_norm = gt_map_norm[i_u, i_v]
        observations_norm.append(Observation(
            i_u=i_u, i_v=i_v,
            a_dyn=float(a_dyn_norm),
            timestamp=float(i),
        ))
    
    obs_buffer = ObservationBuffer(capacity=n_obs*2, lambda_decay=0.0, w_min=1.0)
    for obs in observations_norm:
        obs_buffer.add(obs.i_u, obs.i_v, obs.a_dyn, obs.timestamp)
    
    write_log(f"✓ Created {len(observations_norm)} observations")
    
    # Test with no guidance (prior only)
    write_log("\n### TEST 1: No guidance (prior only) ###")
    sampler_prior = GuidedDiffusionSampler(
        diffusion_prior=prior,
        guidance_scale=0.0,
        sigma_meas=0.3 / norm_std,
        num_inference_steps=50,
    )
    
    recon_prior_norm = sampler_prior.sample(
        obs_buffer=obs_buffer,
        t_now=float(n_obs),
        norm_mean=0.0,
        norm_std=1.0,
        batch_size=1,
        device=device,
        x_init=None,
        generator=torch.Generator(device=device).manual_seed(42),
    )
    
    recon_prior = recon_prior_norm.squeeze().cpu().numpy() * norm_std + norm_mean
    mae_prior = float(np.mean(np.abs(recon_prior - gt_map)))
    grad_y, grad_x = np.gradient(recon_prior)
    smooth_prior = np.sqrt(grad_y**2 + grad_x**2).mean()
    
    write_log(f"  MAE: {mae_prior:.3f} m/s²")
    write_log(f"  Smoothness: {smooth_prior:.3f}")
    
    # Test with guidance
    write_log("\n### TEST 2: With guidance (scale=1.0) ###")
    sampler_guided = GuidedDiffusionSampler(
        diffusion_prior=prior,
        guidance_scale=1.0,
        sigma_meas=0.3 / norm_std,
        num_inference_steps=50,
    )
    
    recon_guided_norm = sampler_guided.sample(
        obs_buffer=obs_buffer,
        t_now=float(n_obs),
        norm_mean=0.0,
        norm_std=1.0,
        batch_size=1,
        device=device,
        x_init=None,
        generator=torch.Generator(device=device).manual_seed(42),
    )
    
    recon_guided = recon_guided_norm.squeeze().cpu().numpy() * norm_std + norm_mean
    mae_guided = float(np.mean(np.abs(recon_guided - gt_map)))
    grad_y, grad_x = np.gradient(recon_guided)
    smooth_guided = np.sqrt(grad_y**2 + grad_x**2).mean()
    
    write_log(f"  MAE: {mae_guided:.3f} m/s²")
    write_log(f"  Smoothness: {smooth_guided:.3f}")
    
    # Analysis
    write_log("\n" + "="*70)
    write_log("SUMMARY")
    write_log("="*70)
    write_log(f"\nNo guidance:   MAE={mae_prior:.3f}, Smoothness={smooth_prior:.3f}")
    write_log(f"With guidance: MAE={mae_guided:.3f}, Smoothness={smooth_guided:.3f}")
    
    if mae_guided < mae_prior:
        write_log("\n✓ Guidance HELPS (lower MAE)")
    else:
        write_log("\n✗ Guidance HURTS (higher MAE)")
    
    if smooth_guided > smooth_prior * 1.5:
        write_log("✗ Guidance creates PIXELATION (smoothness increased >50%)")
        write_log("  → PROCEED TO PHASE 2: Apply gradient smoothing")
    else:
        write_log("✓ Guidance preserves smoothness reasonably well")
    
    # Save reconstructions for visualization
    np.save('evaluation/phase1_gt.npy', gt_map)
    np.save('evaluation/phase1_prior.npy', recon_prior)
    np.save('evaluation/phase1_guided.npy', recon_guided)
    write_log("\n✓ Saved reconstructions to evaluation/phase1_*.npy")
    
    write_log("\n" + "="*70)
    write_log("TEST COMPLETE")
    write_log("="*70)

except Exception as e:
    write_log(f"\n✗ ERROR: {e}")
    import traceback
    write_log(traceback.format_exc())

finally:
    log.close()

print(f"\nLog written to {output_file}")

