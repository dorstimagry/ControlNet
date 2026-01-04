#!/usr/bin/env python3
"""
Compare reconstruction performance with different minimal weight (w_min) values.

This tests how "memory" of old observations affects reconstruction quality.
Higher w_min = more weight on old observations = more memory.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm.auto import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.gt_map_generator import generate_map_from_params
from src.maps.buffer import ObservationBuffer
from src.maps.grid import MapGrid
from src.maps.sampler_diffusion import GuidedDiffusionSampler
from src.models.diffusion_prior import DiffusionPrior
from utils.dynamics import ExtendedPlantRandomization, sample_extended_params


class TripSimulator:
    """Simplified trip simulator for comparison study."""
    
    def __init__(self, gt_map, grid, trip_duration=2000, dt=0.1, noise_std=0.3, 
                 sampling_mode='random_walk', seed=None):
        self.gt_map = gt_map
        self.grid = grid
        self.trip_duration = trip_duration
        self.dt = dt
        self.noise_std = noise_std
        self.sampling_mode = sampling_mode  # 'random_walk' or 'uniform'
        self.rng = np.random.default_rng(seed)
        
        # Initial state (for random walk)
        self.u = 0.0
        self.v = 10.0
        self.t = 0
        
    def step(self):
        """Simulate one timestep."""
        if self.sampling_mode == 'uniform':
            # Uniform sampling: pick random grid cell
            i_u = self.rng.integers(0, self.grid.N_u)
            i_v = self.rng.integers(0, self.grid.N_v)
            self.u = self.grid.u_values[i_u]
            self.v = self.grid.v_values[i_v]
        else:
            # Random walk: update state based on current position
            i_u = self.grid.bin_u(self.u)
            i_v = self.grid.bin_v(self.v)
            
            # Update state with random walk
            a_true = self.gt_map[i_u, i_v]
            self.u = np.clip(self.u + self.rng.normal(0, 0.1), -1.0, 1.0)
            v_new = self.v + a_true * self.dt + self.rng.normal(0, 0.5)
            self.v = np.clip(v_new, 0.0, self.grid.v_max)
            
            # Re-compute indices after update
            i_u = self.grid.bin_u(self.u)
            i_v = self.grid.bin_v(self.v)
        
        # Get GT value and add noise
        a_true = self.gt_map[i_u, i_v]
        a_meas = a_true + self.rng.normal(0, self.noise_std)
        
        obs = (self.u, self.v, a_meas)
        self.t += 1
        
        return obs


def run_comparison_scenario(
    w_min: float,
    diffusion_prior: DiffusionPrior,
    sampler_base_config: Dict,
    gt_map: np.ndarray,
    grid: MapGrid,
    norm_mean: float,
    norm_std: float,
    trip_duration: int,
    reconstruct_every_k: int,
    buffer_size: int,
    sampling_mode: str,
    device: torch.device,
    seed: int,
) -> Dict:
    """Run one scenario with a specific w_min value."""
    
    # Create simulator
    simulator = TripSimulator(
        gt_map, grid, trip_duration=trip_duration, 
        noise_std=0.3, sampling_mode=sampling_mode, seed=seed
    )
    
    # Create buffer with specific w_min
    buffer = ObservationBuffer(
        capacity=buffer_size,
        lambda_decay=0.3,
        w_min=w_min,
    )
    
    # Create sampler
    sampler = GuidedDiffusionSampler(
        diffusion_prior=diffusion_prior,
        guidance_scale=sampler_base_config['guidance_scale'],
        sigma_meas=sampler_base_config['sigma_meas'],
        num_inference_steps=sampler_base_config['num_inference_steps'],
        gradient_smoothing_sigma=sampler_base_config['gradient_smoothing_sigma'],
    )
    
    # Run simulation
    results = {
        'w_min': w_min,
        'timesteps': [],
        'maes': [],
        'n_obs': [],
    }
    
    for t in tqdm(range(trip_duration), desc=f"w_min={w_min:.2f}", leave=False):
        # Step simulator
        u, v, a_meas = simulator.step()
        
        # Add observation
        i_u = grid.bin_u(u)
        i_v = grid.bin_v(v)
        buffer.add(i_u, i_v, a_meas, t * 0.1)
        
        # Reconstruct every k steps
        if t % reconstruct_every_k == 0 and t > 0:
            observations = buffer.get_observations()
            n_obs = len(observations)
            
            if n_obs > 0:
                # Guided reconstruction
                t_now = t * 0.1
                map_tensor = sampler.sample(
                    obs_buffer=buffer,
                    t_now=t_now,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    device=device,
                )
                recon_map = map_tensor.squeeze().cpu().numpy()
                recon_map = recon_map * norm_std + norm_mean
                
                # Compute MAE
                mae = np.mean(np.abs(recon_map - gt_map))
                
                results['timesteps'].append(t)
                results['maes'].append(mae)
                results['n_obs'].append(n_obs)
    
    return results


def plot_comparison(all_results: List[Dict], output_dir: Path):
    """Create comparison plots."""
    
    # Plot 1: MAE vs timestep for different w_min values
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Panel 1: MAE vs timestep
    ax = axes[0, 0]
    for results in all_results:
        ax.plot(
            results['timesteps'], 
            results['maes'], 
            label=f"w_min={results['w_min']:.2f}",
            linewidth=2,
            alpha=0.8
        )
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("MAE (m/s²)", fontsize=12)
    ax.set_title("Reconstruction Error vs Time", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: MAE vs observations
    ax = axes[0, 1]
    for results in all_results:
        ax.plot(
            results['n_obs'], 
            results['maes'], 
            label=f"w_min={results['w_min']:.2f}",
            linewidth=2,
            alpha=0.8
        )
    ax.set_xlabel("Number of Observations", fontsize=12)
    ax.set_ylabel("MAE (m/s²)", fontsize=12)
    ax.set_title("Reconstruction Error vs Observations", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Final MAE comparison (bar plot)
    ax = axes[1, 0]
    w_mins = [r['w_min'] for r in all_results]
    final_maes = [r['maes'][-1] if r['maes'] else np.nan for r in all_results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(w_mins)))
    bars = ax.bar(range(len(w_mins)), final_maes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(w_mins)))
    ax.set_xticklabels([f"{w:.2f}" for w in w_mins], fontsize=11)
    ax.set_xlabel("Minimal Weight (w_min)", fontsize=12)
    ax.set_ylabel("Final MAE (m/s²)", fontsize=12)
    ax.set_title("Final Reconstruction Error by w_min", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mae) in enumerate(zip(bars, final_maes)):
        if not np.isnan(mae):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{mae:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel 4: Average MAE over last 50% of trip
    ax = axes[1, 1]
    w_mins = [r['w_min'] for r in all_results]
    avg_maes = []
    for r in all_results:
        if r['maes']:
            # Average over last 50% of reconstructions
            halfway = len(r['maes']) // 2
            avg_maes.append(np.mean(r['maes'][halfway:]))
        else:
            avg_maes.append(np.nan)
    
    bars = ax.bar(range(len(w_mins)), avg_maes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(w_mins)))
    ax.set_xticklabels([f"{w:.2f}" for w in w_mins], fontsize=11)
    ax.set_xlabel("Minimal Weight (w_min)", fontsize=12)
    ax.set_ylabel("Average MAE (m/s²)", fontsize=12)
    ax.set_title("Average MAE (Last 50% of Trip)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, mae) in enumerate(zip(bars, avg_maes)):
        if not np.isnan(mae):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{mae:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(
        "Minimal Weight (w_min) Comparison: Effect of Observation Memory",
        fontsize=16, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    plt.savefig(output_dir / "wmin_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_dir / 'wmin_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare reconstruction with different w_min values"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained diffusion prior checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/wmin_comparison",
        help="Output directory",
    )
    parser.add_argument(
        "--w-min-values",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        help="List of w_min values to test",
    )
    parser.add_argument(
        "--trip-duration",
        type=int,
        default=2000,
        help="Trip duration in timesteps (longer = more state space coverage)",
    )
    parser.add_argument(
        "--reconstruct-every-k",
        type=int,
        default=10,
        help="Reconstruct every k timesteps",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=200,
        help="Observation buffer size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="random_walk",
        choices=["random_walk", "uniform"],
        help="Observation sampling strategy: random_walk (trajectory) or uniform (random)",
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("W_MIN COMPARISON STUDY")
    print("="*70)
    print(f"Testing w_min values: {args.w_min_values}")
    print(f"Sampling mode: {args.sampling_mode}")
    print(f"Trip duration: {args.trip_duration} timesteps ({args.trip_duration * 0.1:.1f} seconds)")
    print(f"Buffer size: {args.buffer_size} observations")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    # Load diffusion prior
    print("Loading diffusion prior...")
    diffusion_prior = DiffusionPrior.load(args.checkpoint, device=device)
    
    # Configure DDPM scheduler for high-quality samples (like offline evaluation)
    from diffusers import DDPMScheduler
    diffusion_prior.inference_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        clip_sample=False,
    )
    diffusion_prior.eval()
    print(f"Configured DDPM scheduler for high-quality reconstruction")
    
    # Load normalization stats
    norm_stats_path = Path("data/maps/norm_stats.json")
    if norm_stats_path.exists():
        with open(norm_stats_path, "r") as f:
            norm_stats = json.load(f)
        norm_mean = norm_stats["mean"]
        norm_std = norm_stats["std"]
        print(f"Loaded normalization: mean={norm_mean:.3f}, std={norm_std:.3f}")
    else:
        norm_mean = 0.0
        norm_std = 1.0
        print("Warning: Using default normalization")
    
    # Create grid
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    
    # Generate GT map
    print("Generating ground truth map...")
    rng = np.random.default_rng(args.seed)
    randomization = ExtendedPlantRandomization()
    vehicle_params = sample_extended_params(rng, randomization)
    gt_map = generate_map_from_params(vehicle_params, N_u=100, N_v=100, v_max=30.0)
    print(f"GT map range: [{gt_map.min():.2f}, {gt_map.max():.2f}] m/s²\n")
    
    # Sampler base configuration
    sampler_config = {
        'guidance_scale': 1.0,
        'sigma_meas': 0.5,
        'num_inference_steps': 50,
        'gradient_smoothing_sigma': 10.0,
    }
    
    # Run comparisons
    all_results = []
    
    for w_min in args.w_min_values:
        print(f"\n{'='*70}")
        print(f"Testing w_min = {w_min:.2f}")
        print(f"{'='*70}")
        
        results = run_comparison_scenario(
            w_min=w_min,
            diffusion_prior=diffusion_prior,
            sampler_base_config=sampler_config,
            gt_map=gt_map,
            grid=grid,
            norm_mean=norm_mean,
            norm_std=norm_std,
            trip_duration=args.trip_duration,
            reconstruct_every_k=args.reconstruct_every_k,
            buffer_size=args.buffer_size,
            sampling_mode=args.sampling_mode,
            device=device,
            seed=args.seed,
        )
        
        all_results.append(results)
        
        # Print summary
        if results['maes']:
            final_mae = results['maes'][-1]
            min_mae = min(results['maes'])
            avg_mae = np.mean(results['maes'])
            print(f"  Final MAE: {final_mae:.4f} m/s²")
            print(f"  Min MAE:   {min_mae:.4f} m/s²")
            print(f"  Avg MAE:   {avg_mae:.4f} m/s²")
    
    # Create comparison plots
    print(f"\n{'='*70}")
    print("Creating comparison plots...")
    print(f"{'='*70}\n")
    plot_comparison(all_results, output_dir)
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'w_min':<10} {'Final MAE':<15} {'Avg MAE':<15} {'Min MAE':<15}")
    print("-"*70)
    for r in all_results:
        if r['maes']:
            final = r['maes'][-1]
            avg = np.mean(r['maes'])
            minimum = min(r['maes'])
            print(f"{r['w_min']:<10.2f} {final:<15.4f} {avg:<15.4f} {minimum:<15.4f}")
    print(f"{'='*70}\n")
    
    # Find best w_min
    if all_results and all_results[0]['maes']:
        best_idx = np.argmin([np.mean(r['maes']) for r in all_results])
        best_wmin = all_results[best_idx]['w_min']
        best_avg_mae = np.mean(all_results[best_idx]['maes'])
        print(f"✓ Best w_min: {best_wmin:.2f} (Avg MAE: {best_avg_mae:.4f} m/s²)\n")


if __name__ == "__main__":
    main()

