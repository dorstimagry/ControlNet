#!/usr/bin/env python3
"""Test guidance with perfect GT observations and gradient smoothing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.gt_map_generator import generate_map_from_params
from src.maps.buffer import Observation, ObservationBuffer
from src.maps.grid import MapGrid
from src.models.diffusion_prior import DiffusionPrior
from src.maps.sampler_diffusion import GuidedDiffusionSampler
from utils.dynamics import ExtendedPlantRandomization, sample_extended_params
from diffusers import DDPMScheduler

sns.set_style("whitegrid")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_obs', type=int, default=100, help='Number of observations')
    parser.add_argument('--n_test_maps', type=int, default=3, help='Number of test maps')
    parser.add_argument('--guidance_scales', nargs='+', type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument('--smoothing_sigmas', nargs='+', type=float, default=[0.0, 1.0, 2.0, 5.0])
    parser.add_argument('--output_dir', type=str, default='evaluation/guidance_debug')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("TESTING GUIDED DIFFUSION WITH PERFECT GT OBSERVATIONS")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
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
    print("✓ Model loaded with DDPM scheduler (50 steps)")
    
    # Load normalization
    with open('data/maps/norm_stats.json') as f:
        norm_stats = json.load(f)
    norm_mean, norm_std = norm_stats['mean'], norm_stats['std']
    print(f"✓ Normalization: mean={norm_mean:.3f}, std={norm_std:.3f}")
    
    # =========================================================================
    # PHASE 1: Test with perfect GT observations
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: PERFECT GT OBSERVATIONS (no noise, no gravity)")
    print("="*80)
    
    results_phase1 = []
    
    for map_idx in range(args.n_test_maps):
        print(f"\n--- Test Map {map_idx + 1}/{args.n_test_maps} ---")
        
        # Generate GT map
        rand = ExtendedPlantRandomization(
            mass_range=(1200.0, 1800.0),
            wheel_radius_range=(0.3, 0.35),
            brake_accel_range=(8.0, 11.0),
        )
        rng = np.random.default_rng(42 + map_idx)
        params = sample_extended_params(rng, rand)
        gt_map = generate_map_from_params(params, N_u=100, N_v=100, v_max=30.0)
        gt_map_norm = (gt_map - norm_mean) / norm_std
        print(f"GT map range: [{gt_map.min():.2f}, {gt_map.max():.2f}] m/s²")
        
        # Sample perfect observations
        rng_obs = np.random.default_rng(123 + map_idx)
        observations_norm = []
        obs_indices = []
        
        for i in range(args.n_obs):
            i_u = rng_obs.integers(0, grid.N_u)
            i_v = rng_obs.integers(0, grid.N_v)
            a_dyn_norm = gt_map_norm[i_u, i_v]
            
            observations_norm.append(Observation(
                i_u=i_u, i_v=i_v,
                a_dyn=float(a_dyn_norm),
                timestamp=float(i),
            ))
            obs_indices.append((i_u, i_v))
        
        obs_buffer = ObservationBuffer(capacity=args.n_obs*2, lambda_decay=0.0, w_min=1.0)
        for obs in observations_norm:
            obs_buffer.add(obs.i_u, obs.i_v, obs.a_dyn, obs.timestamp)
        
        print(f"Created {args.n_obs} perfect observations")
        
        # Test different guidance scales
        for guidance_scale in args.guidance_scales:
            print(f"  Testing guidance_scale={guidance_scale:.1f}...", end=" ")
            
            sampler = GuidedDiffusionSampler(
                diffusion_prior=prior,
                guidance_scale=guidance_scale,
                sigma_meas=0.3 / norm_std,
                num_inference_steps=50,
                gradient_smoothing_sigma=0.0,  # No smoothing in Phase 1
            )
            
            recon_norm = sampler.sample(
                obs_buffer=obs_buffer,
                t_now=float(args.n_obs),
                norm_mean=0.0,
                norm_std=1.0,
                batch_size=1,
                device=device,
                x_init=None,
                generator=torch.Generator(device=device).manual_seed(42),
            )
            
            recon = recon_norm.squeeze().cpu().numpy() * norm_std + norm_mean
            
            # Metrics
            mae = float(np.mean(np.abs(recon - gt_map)))
            mse = float(np.mean((recon - gt_map) ** 2))
            
            # Smoothness
            grad_y, grad_x = np.gradient(recon)
            smoothness = np.sqrt(grad_y**2 + grad_x**2).mean()
            
            # Error at observations
            obs_errors = [abs(recon[i_u, i_v] - gt_map[i_u, i_v]) for i_u, i_v in obs_indices]
            mae_at_obs = np.mean(obs_errors)
            
            print(f"MAE={mae:.3f}, Smooth={smoothness:.3f}")
            
            results_phase1.append({
                'map_idx': map_idx,
                'guidance_scale': guidance_scale,
                'mae': mae,
                'mse': mse,
                'smoothness': smoothness,
                'mae_at_obs': mae_at_obs,
                'recon': recon,
                'gt_map': gt_map,
                'obs_indices': obs_indices[:50],  # Save first 50 for plotting
            })
    
    # =========================================================================
    # PHASE 2: Test gradient smoothing
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: GRADIENT SMOOTHING")
    print("="*80)
    
    results_phase2 = []
    
    # Use the first test map for Phase 2
    rng = np.random.default_rng(42)
    rand = ExtendedPlantRandomization(
        mass_range=(1200.0, 1800.0),
        wheel_radius_range=(0.3, 0.35),
        brake_accel_range=(8.0, 11.0),
    )
    params = sample_extended_params(rng, rand)
    gt_map = generate_map_from_params(params, N_u=100, N_v=100, v_max=30.0)
    gt_map_norm = (gt_map - norm_mean) / norm_std
    
    # Create observations
    rng_obs = np.random.default_rng(123)
    observations_norm = []
    for i in range(args.n_obs):
        i_u = rng_obs.integers(0, grid.N_u)
        i_v = rng_obs.integers(0, grid.N_v)
        a_dyn_norm = gt_map_norm[i_u, i_v]
        observations_norm.append(Observation(
            i_u=i_u, i_v=i_v,
            a_dyn=float(a_dyn_norm),
            timestamp=float(i),
        ))
    
    obs_buffer = ObservationBuffer(capacity=args.n_obs*2, lambda_decay=0.0, w_min=1.0)
    for obs in observations_norm:
        obs_buffer.add(obs.i_u, obs.i_v, obs.a_dyn, obs.timestamp)
    
    print(f"\nTesting different smoothing scales (with guidance_scale=1.0):")
    
    for smoothing_sigma in args.smoothing_sigmas:
        print(f"  Smoothing sigma={smoothing_sigma:.1f}...", end=" ")
        
        sampler = GuidedDiffusionSampler(
            diffusion_prior=prior,
            guidance_scale=1.0,
            sigma_meas=0.3 / norm_std,
            num_inference_steps=50,
            gradient_smoothing_sigma=smoothing_sigma,
        )
        
        recon_norm = sampler.sample(
            obs_buffer=obs_buffer,
            t_now=float(args.n_obs),
            norm_mean=0.0,
            norm_std=1.0,
            batch_size=1,
            device=device,
            x_init=None,
            generator=torch.Generator(device=device).manual_seed(42),
        )
        
        recon = recon_norm.squeeze().cpu().numpy() * norm_std + norm_mean
        
        mae = float(np.mean(np.abs(recon - gt_map)))
        grad_y, grad_x = np.gradient(recon)
        smoothness = np.sqrt(grad_y**2 + grad_x**2).mean()
        
        print(f"MAE={mae:.3f}, Smooth={smoothness:.3f}")
        
        results_phase2.append({
            'smoothing_sigma': smoothing_sigma,
            'mae': mae,
            'smoothness': smoothness,
            'recon': recon,
        })
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Plot Phase 1: Different guidance scales
    fig, axes = plt.subplots(len(args.guidance_scales), args.n_test_maps + 1, 
                              figsize=(4*(args.n_test_maps+1), 4*len(args.guidance_scales)))
    
    if len(args.guidance_scales) == 1:
        axes = axes[np.newaxis, :]
    
    for scale_idx, scale in enumerate(args.guidance_scales):
        # Plot GT in first column
        ax = axes[scale_idx, 0]
        results_map0 = [r for r in results_phase1 if r['guidance_scale'] == scale and r['map_idx'] == 0][0]
        im = ax.imshow(results_map0['gt_map'], aspect='auto', origin='lower', cmap='RdYlGn')
        ax.set_title(f'Ground Truth\n(guidance={scale:.1f})', fontsize=10, fontweight='bold')
        if scale_idx == 0:
            # Mark observations
            obs_i_u = [idx[0] for idx in results_map0['obs_indices']]
            obs_i_v = [idx[1] for idx in results_map0['obs_indices']]
            ax.scatter(obs_i_v, obs_i_u, c='black', s=5, alpha=0.5, marker='x')
        plt.colorbar(im, ax=ax)
        
        # Plot reconstructions
        for map_idx in range(args.n_test_maps):
            ax = axes[scale_idx, map_idx + 1]
            result = [r for r in results_phase1 if r['guidance_scale'] == scale and r['map_idx'] == map_idx][0]
            vmin, vmax = result['gt_map'].min(), result['gt_map'].max()
            im = ax.imshow(result['recon'], aspect='auto', origin='lower', cmap='RdYlGn',
                          vmin=vmin, vmax=vmax)
            ax.set_title(
                f"Map {map_idx+1}\nMAE={result['mae']:.2f}, S={result['smoothness']:.2f}",
                fontsize=10
            )
            plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'Phase 1: Perfect GT Observations (n_obs={args.n_obs})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'phase1_guidance_scales.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_dir / 'phase1_guidance_scales.png'}")
    plt.close()
    
    # Plot Phase 2: Different smoothing scales
    fig, axes = plt.subplots(2, len(args.smoothing_sigmas), figsize=(4*len(args.smoothing_sigmas), 8))
    
    if len(args.smoothing_sigmas) == 1:
        axes = axes[:, np.newaxis]
    
    for sigma_idx, sigma in enumerate(args.smoothing_sigmas):
        result = results_phase2[sigma_idx]
        
        # Reconstruction
        ax = axes[0, sigma_idx]
        im = ax.imshow(result['recon'], aspect='auto', origin='lower', cmap='RdYlGn')
        ax.set_title(f"Smoothing σ={sigma:.1f}\nMAE={result['mae']:.2f}", fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax)
        
        # Error map
        ax = axes[1, sigma_idx]
        error_map = np.abs(result['recon'] - gt_map)
        im = ax.imshow(error_map, aspect='auto', origin='lower', cmap='hot')
        ax.set_title(f"Error (max={error_map.max():.2f})", fontsize=10)
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'Phase 2: Gradient Smoothing (guidance=1.0, n_obs={args.n_obs})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_gradient_smoothing.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_dir / 'phase2_gradient_smoothing.png'}")
    plt.close()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nPhase 1: Effect of Guidance Scale (no smoothing)")
    print(f"{'Guidance':<12} {'MAE':<12} {'Smoothness':<12}")
    print("-" * 40)
    
    for scale in args.guidance_scales:
        results_scale = [r for r in results_phase1 if r['guidance_scale'] == scale]
        avg_mae = np.mean([r['mae'] for r in results_scale])
        avg_smooth = np.mean([r['smoothness'] for r in results_scale])
        print(f"{scale:<12.1f} {avg_mae:<12.3f} {avg_smooth:<12.3f}")
    
    print("\nPhase 2: Effect of Gradient Smoothing (guidance=1.0)")
    print(f"{'Sigma':<12} {'MAE':<12} {'Smoothness':<12}")
    print("-" * 40)
    
    for result in results_phase2:
        print(f"{result['smoothing_sigma']:<12.1f} {result['mae']:<12.3f} {result['smoothness']:<12.3f}")
    
    # Find best configuration
    best_phase2 = min(results_phase2, key=lambda r: r['mae'])
    print(f"\n✓ BEST: Smoothing sigma={best_phase2['smoothing_sigma']:.1f}, MAE={best_phase2['mae']:.3f}")
    
    # Save results
    results_json = {
        'phase1': [
            {k: v for k, v in r.items() if k not in ['recon', 'gt_map', 'obs_indices']}
            for r in results_phase1
        ],
        'phase2': [
            {k: v for k, v in r.items() if k != 'recon'}
            for r in results_phase2
        ],
        'best_smoothing_sigma': float(best_phase2['smoothing_sigma']),
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'results.json'}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

