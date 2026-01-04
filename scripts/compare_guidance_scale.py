"""Compare different guidance scale values for online reconstruction."""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.gt_map_generator import generate_map_from_params
from src.maps.buffer import Observation, ObservationBuffer
from src.maps.grid import MapGrid
from src.maps.sampler_diffusion import GuidedDiffusionSampler
from src.models.diffusion_prior import DiffusionPrior
from utils.dynamics import ExtendedPlantRandomization, sample_extended_params


@dataclass
class TripSimulator:
    """Simulate vehicle movement as a random walk."""
    
    grid: MapGrid
    gt_map: np.ndarray
    noise_std: float
    grade_rad: float
    dt: float
    rng: np.random.Generator
    sampling_mode: str = "random_walk"
    
    # Current state
    u: float = 0.0
    v: float = 10.0
    
    # Random walk parameters
    du_std: float = 0.05
    dv_std: float = 1.0
    
    def step(self) -> Tuple[float, float, float, float]:
        """Take one simulation step and return (u, v, a_meas, grade)."""
        if self.sampling_mode == "random_walk":
            # Random walk
            self.u += self.rng.normal(0, self.du_std)
            self.v += self.rng.normal(0, self.dv_std)
            
            # Clamp to valid range (u: [-1, 1], v: [0, v_max])
            self.u = np.clip(self.u, -1.0, 1.0)
            self.v = np.clip(self.v, 0.0, self.grid.v_max)
        elif self.sampling_mode == "uniform":
            # Uniform random sampling
            i_u = self.rng.integers(0, self.grid.N_u)
            i_v = self.rng.integers(0, self.grid.N_v)
            self.u = self.grid.u_values[i_u]
            self.v = self.grid.v_values[i_v]
        
        # Get ground truth acceleration
        i_u = self.grid.bin_u(self.u)
        i_v = self.grid.bin_v(self.v)
        a_true = self.gt_map[i_u, i_v]
        
        # Add measurement noise
        a_meas = a_true + self.rng.normal(0, self.noise_std)
        
        return self.u, self.v, a_meas, self.grade_rad


def run_guidance_scale_scenario(
    guidance_scale: float,
    diffusion_prior: DiffusionPrior,
    sampler_base_config: Dict,
    gt_map: np.ndarray,
    grid: MapGrid,
    norm_mean: float,
    norm_std: float,
    trip_duration: int,
    reconstruct_every_k: int,
    buffer_capacity: int,
    lambda_decay: float,
    w_min: float,
    output_dir: Path,
    seed: int,
    sampling_mode: str,
) -> Dict[str, Any]:
    """Run a single scenario with a specific guidance_scale value.
    
    Returns:
        Dictionary with results including MAE history.
    """
    rng = np.random.default_rng(seed)
    device = diffusion_prior.unet.device
    
    # Create simulator
    simulator = TripSimulator(
        grid=grid,
        gt_map=gt_map,
        noise_std=0.3,
        grade_rad=0.0,
        dt=0.1,
        rng=rng,
        sampling_mode=sampling_mode,
    )
    
    # Create buffer
    buffer = ObservationBuffer(
        capacity=buffer_capacity,
        lambda_decay=lambda_decay,
        w_min=w_min,
    )
    
    # Create sampler with specific guidance_scale
    sampler = GuidedDiffusionSampler(
        diffusion_prior=diffusion_prior,
        guidance_scale=guidance_scale,
        sigma_meas=sampler_base_config['sigma_meas'],
        num_inference_steps=sampler_base_config['num_inference_steps'],
        gradient_smoothing_sigma=sampler_base_config['gradient_smoothing_sigma'],
    )
    
    # Run simulation
    mae_history = []
    n_obs_history = []
    
    for t_step in tqdm(range(trip_duration), desc=f"guidance_scale={guidance_scale:.2e}"):
        # Get observation
        u, v, a_meas, theta = simulator.step()
        
        # Add to buffer (gravity compensation)
        i_u = grid.bin_u(u)
        i_v = grid.bin_v(v)
        g = 9.81
        a_dyn = a_meas + g * np.sin(theta)
        buffer.add(i_u, i_v, a_dyn, float(t_step))
        
        # Reconstruct every k steps
        if (t_step + 1) % reconstruct_every_k == 0:
            t_now = float(t_step)
            weights = buffer.get_weights(t_now)
            observations = buffer.get_observations()
            
            # Sample from diffusion with guidance
            recon_norm = sampler.sample(
                obs_buffer=buffer,
                t_now=t_now,
                norm_mean=norm_mean,
                norm_std=norm_std,
                batch_size=1,
                device=device,
                x_init=None,
                generator=None,
            )
            
            # Denormalize
            recon = recon_norm.squeeze().cpu().numpy() * norm_std + norm_mean
            
            # Compute MAE
            mae = float(np.mean(np.abs(recon - gt_map)))
            mae_history.append(mae)
            n_obs_history.append(len(observations))
    
    final_mae = mae_history[-1] if mae_history else float('nan')
    min_mae = float(np.min(mae_history)) if mae_history else float('nan')
    avg_mae_second_half = float(np.mean(mae_history[len(mae_history)//2:])) if len(mae_history) > 1 else float('nan')
    
    return {
        "guidance_scale": float(guidance_scale),
        "final_mae": final_mae,
        "min_mae": min_mae,
        "avg_mae_second_half": avg_mae_second_half,
        "mae_history": [float(m) for m in mae_history],
        "n_obs_history": [int(n) for n in n_obs_history],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare guidance_scale values for online reconstruction.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to diffusion prior checkpoint.")
    parser.add_argument("--output-dir", type=Path, default="evaluation/guidance_scale_comparison", help="Output directory.")
    parser.add_argument(
        "--guidance-scale-values",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        help="List of guidance_scale values to test.",
    )
    parser.add_argument("--trip-duration", type=int, default=2000, help="Total timesteps for each simulated trip.")
    parser.add_argument("--reconstruct-every-k", type=int, default=10, help="Reconstruct map every k timesteps.")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Capacity of the observation buffer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="random_walk",
        choices=["random_walk", "uniform"],
        help="Sampling mode for trip simulation (random_walk or uniform)",
    )
    parser.add_argument("--lambda-decay", type=float, default=0.0, help="Lambda decay for observation buffer.")
    parser.add_argument("--w-min", type=float, default=1.0, help="Minimum weight for observation buffer.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu).")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Load normalization stats
    norm_stats_path = Path("data/maps/norm_stats.json")
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    norm_mean = norm_stats["mean"]
    norm_std = norm_stats["std"]
    print(f"Loaded normalization: mean={norm_mean:.3f}, std={norm_std:.3f}")
    
    # Load diffusion prior
    print("Loading diffusion prior...")
    diffusion_prior = DiffusionPrior.load(args.checkpoint, device=device)
    # Configure DDPM scheduler for high-quality reconstruction
    diffusion_prior.inference_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        clip_sample=False,
    )
    print("Configured DDPM scheduler for high-quality reconstruction")
    diffusion_prior.eval()
    
    # Generate GT map
    print("Generating ground truth map...")
    rng_map = np.random.default_rng(args.seed)
    randomization = ExtendedPlantRandomization()
    vehicle_params = sample_extended_params(rng_map, randomization)
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    gt_map = generate_map_from_params(
        vehicle_params, N_u=grid.N_u, N_v=grid.N_v, v_max=grid.v_max
    )
    
    # Base sampler config (guidance_scale will vary)
    sampler_base_config = {
        'sigma_meas': 0.5,
        'num_inference_steps': 50,
        'gradient_smoothing_sigma': 10.0,
    }
    
    # Run scenarios for each guidance_scale value
    print(f"\n{'='*70}")
    print("GUIDANCE SCALE COMPARISON")
    print(f"{'='*70}\n")
    
    all_results = []
    for gs_val in args.guidance_scale_values:
        print(f"\n{'='*70}\nTesting guidance_scale = {gs_val:.2e}\n{'='*70}")
        result = run_guidance_scale_scenario(
            guidance_scale=gs_val,
            diffusion_prior=diffusion_prior,
            sampler_base_config=sampler_base_config,
            gt_map=gt_map,
            grid=grid,
            norm_mean=norm_mean,
            norm_std=norm_std,
            trip_duration=args.trip_duration,
            reconstruct_every_k=args.reconstruct_every_k,
            buffer_capacity=args.buffer_size,
            lambda_decay=args.lambda_decay,
            w_min=args.w_min,
            output_dir=args.output_dir,
            seed=args.seed,
            sampling_mode=args.sampling_mode,
        )
        all_results.append(result)
        print(f"  Final MAE: {result['final_mae']:.4f} m/s²")
        print(f"  Min MAE:   {result['min_mae']:.4f} m/s²")
        print(f"  Avg MAE (2nd half): {result['avg_mae_second_half']:.4f} m/s²")
    
    # Save all results to JSON
    print(f"\n{'='*70}\nSaving results...\n{'='*70}")
    with open(args.output_dir / "guidance_scale_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Guidance Scale':<20} {'Final MAE':>12} {'Min MAE':>12} {'Avg MAE':>12}")
    print("-" * 70)
    for result in all_results:
        print(
            f"{result['guidance_scale']:<20.2e} "
            f"{result['final_mae']:>12.4f} "
            f"{result['min_mae']:>12.4f} "
            f"{result['avg_mae_second_half']:>12.4f}"
        )
    print(f"{'='*70}\n")
    
    # Plotting
    print(f"\n{'='*70}\nCreating comparison plots...\n{'='*70}")
    
    # Plot 1: MAE over time for all guidance scales
    plt.figure(figsize=(14, 8))
    for result in all_results:
        plt.plot(
            result["mae_history"],
            label=f"gs={result['guidance_scale']:.2e} (Final: {result['final_mae']:.2f})",
            linewidth=2,
        )
    plt.xlabel("Reconstruction Step", fontsize=12)
    plt.ylabel("MAE (m/s²)", fontsize=12)
    plt.title(
        f"MAE over Time for Different Guidance Scale Values\n"
        f"(Trip: {args.trip_duration} steps, Sampling: {args.sampling_mode})",
        fontsize=14,
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = args.output_dir / "guidance_scale_mae_over_time.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Time series plot saved to {plot_path}")
    
    # Plot 2: Final MAE vs guidance_scale (log scale)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    guidance_scales = [r["guidance_scale"] for r in all_results]
    final_maes = [r["final_mae"] for r in all_results]
    min_maes = [r["min_mae"] for r in all_results]
    avg_maes = [r["avg_mae_second_half"] for r in all_results]
    
    # Final MAE
    ax1.plot(guidance_scales, final_maes, 'o-', linewidth=2, markersize=8)
    ax1.set_xscale('log')
    ax1.set_xlabel('Guidance Scale', fontsize=12)
    ax1.set_ylabel('Final MAE (m/s²)', fontsize=12)
    ax1.set_title('Final MAE vs Guidance Scale', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Min MAE
    ax2.plot(guidance_scales, min_maes, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xscale('log')
    ax2.set_xlabel('Guidance Scale', fontsize=12)
    ax2.set_ylabel('Min MAE (m/s²)', fontsize=12)
    ax2.set_title('Min MAE vs Guidance Scale', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    # Avg MAE (2nd half)
    ax3.plot(guidance_scales, avg_maes, 'o-', linewidth=2, markersize=8, color='green')
    ax3.set_xscale('log')
    ax3.set_xlabel('Guidance Scale', fontsize=12)
    ax3.set_ylabel('Avg MAE (2nd half) (m/s²)', fontsize=12)
    ax3.set_title('Avg MAE (2nd half) vs Guidance Scale', fontsize=13)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = args.output_dir / "guidance_scale_summary.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Summary plot saved to {plot_path}")
    
    # Find optimal guidance_scale
    optimal_idx = int(np.argmin(final_maes))
    optimal_gs = all_results[optimal_idx]["guidance_scale"]
    optimal_mae = all_results[optimal_idx]["final_mae"]
    
    print(f"\n{'='*70}")
    print(f"✓ Comparison complete!")
    print(f"{'='*70}")
    print(f"Optimal guidance_scale: {optimal_gs:.2e} (Final MAE: {optimal_mae:.4f} m/s²)")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

