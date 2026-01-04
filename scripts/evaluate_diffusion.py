#!/usr/bin/env python3
"""Evaluate diffusion prior model with guided sampling.

This script comprehensively evaluates the trained diffusion prior by:
1. Generating fresh test maps with known GT parameters
2. Synthetically sampling observations from these maps
3. Reconstructing maps using guided diffusion with varying observation counts
4. Computing quantitative metrics (MSE, MAE) and creating visual comparisons
5. Measuring guidance effectiveness by comparing guided vs unguided reconstruction
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from tqdm.auto import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.gt_map_generator import generate_map_from_params
from src.maps.buffer import Observation, ObservationBuffer
from src.maps.grid import MapGrid
from src.maps.sampler_diffusion import GuidedDiffusionSampler
from src.models.diffusion_prior import DiffusionPrior
from utils.dynamics import (
    ExtendedPlantRandomization,
    sample_extended_params,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion prior with guided sampling"
    )
    
    # Model and config
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained diffusion prior checkpoint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/map_prior.yaml"),
        help="Path to diffusion prior config YAML",
    )
    parser.add_argument(
        "--randomization-config",
        type=Path,
        default=Path("training/config_dynamics_map.yaml"),
        help="Path to vehicle randomization config",
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--n-test-maps",
        type=int,
        default=20,
        help="Number of test maps to generate",
    )
    parser.add_argument(
        "--obs-counts",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100, 200, 500],
        help="List of observation counts to evaluate",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=100.0,  # Updated after gradient normalization tuning
        help="Guidance strength (eta)",
    )
    parser.add_argument(
        "--sigma-meas",
        type=float,
        default=0.5,
        help="Measurement noise std (m/s²)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of diffusion steps (DDPM: use 50-500, DDIM: use 20-50)",
    )
    parser.add_argument(
        "--gradient-smoothing-sigma",
        type=float,
        default=10.0,
        help="Gaussian smoothing sigma for guidance gradient (0=no smoothing)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="DDPM",
        choices=["DDPM", "DDIM"],
        help="Scheduler to use for sampling (DDPM gives smoother results)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/diffusion"),
        help="Directory for results and plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def sample_observations(
    gt_map: np.ndarray,
    grid: MapGrid,
    n_obs: int,
    sigma_noise: float = 0.3,
    seed: int | None = None,
) -> List[Observation]:
    """Randomly sample observations from GT map with noise.
    
    Args:
        gt_map: Ground truth acceleration map, shape (N_u, N_v)
        grid: Map grid
        n_obs: Number of observations to sample
        sigma_noise: Gaussian noise std to add to observations (m/s²)
        seed: Random seed
    
    Returns:
        List of Observation objects
    """
    rng = np.random.RandomState(seed)
    
    observations = []
    for _ in range(n_obs):
        # Sample random grid indices
        i_u = rng.randint(0, grid.N_u)
        i_v = rng.randint(0, grid.N_v)
        
        # Get GT value and add noise
        a_true = gt_map[i_u, i_v]
        a_noisy = a_true + rng.normal(0, sigma_noise)
        
        # Create observation with dummy timestamp
        obs = Observation(
            i_u=i_u,
            i_v=i_v,
            a_dyn=float(a_noisy),
            timestamp=float(_),  # Use index as timestamp
        )
        observations.append(obs)
    
    return observations


def evaluate_reconstruction(
    diffusion_prior: DiffusionPrior,
    gt_map: np.ndarray,
    grid: MapGrid,
    observations: List[Observation],
    norm_mean: float,
    norm_std: float,
    guidance_scale: float,
    sigma_meas: float,
    num_inference_steps: int,
    gradient_smoothing_sigma: float,
    device: torch.device,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Evaluate guided reconstruction on one map.
    
    Args:
        diffusion_prior: Trained diffusion prior model
        gt_map: Ground truth map, shape (N_u, N_v)
        grid: Map grid
        observations: List of observations
        norm_mean: Normalization mean
        norm_std: Normalization std
        guidance_scale: Guidance strength (eta)
        sigma_meas: Measurement noise std (m/s²)
        num_inference_steps: Number of diffusion steps
        gradient_smoothing_sigma: Gaussian smoothing sigma for guidance gradient
        device: Device
        seed: Random seed
    
    Returns:
        Dictionary with metrics and reconstructed maps
    """
    # Normalize GT map for comparison
    gt_map_norm = (gt_map - norm_mean) / norm_std
    
    # Create observation buffer
    obs_buffer = ObservationBuffer(
        capacity=len(observations) * 2,  # Extra capacity
        lambda_decay=0.0,  # No decay for evaluation
        w_min=1.0,
    )
    
    # Add observations to buffer
    for obs in observations:
        obs_buffer.add(obs.i_u, obs.i_v, obs.a_dyn, obs.timestamp)
    
    # Create guided sampler
    guided_sampler = GuidedDiffusionSampler(
        diffusion_prior=diffusion_prior,
        guidance_scale=guidance_scale,
        sigma_meas=sigma_meas,
        num_inference_steps=num_inference_steps,
        gradient_smoothing_sigma=gradient_smoothing_sigma,
    )
    
    # Set up generator
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    
    # Guided reconstruction
    t_now = float(len(observations))  # Use observation count as time
    recon_guided_norm = guided_sampler.sample(
        obs_buffer=obs_buffer,
        t_now=t_now,
        norm_mean=norm_mean,
        norm_std=norm_std,
        batch_size=1,
        device=device,
        x_init=None,
        generator=generator,
    )
    
    # Convert to numpy and denormalize
    recon_guided_norm_np = recon_guided_norm.squeeze().cpu().numpy()
    recon_guided = recon_guided_norm_np * norm_std + norm_mean
    
    # Unguided (prior-only) sample - just sample from prior
    if seed is not None:
        generator.manual_seed(seed + 1)  # Different seed for unguided
    
    prior_sample_norm = diffusion_prior.sample(
        batch_size=1,
        num_inference_steps=num_inference_steps,
        device=device,
        generator=generator,
    )
    
    # Convert to numpy and denormalize
    prior_sample_norm_np = prior_sample_norm.squeeze().cpu().numpy()
    prior_sample = prior_sample_norm_np * norm_std + norm_mean
    
    # Compute metrics
    mse_guided = float(np.mean((recon_guided - gt_map) ** 2))
    mse_unguided = float(np.mean((prior_sample - gt_map) ** 2))
    mae_guided = float(np.mean(np.abs(recon_guided - gt_map)))
    mae_unguided = float(np.mean(np.abs(prior_sample - gt_map)))
    
    return {
        "mse_guided": mse_guided,
        "mse_unguided": mse_unguided,
        "mae_guided": mae_guided,
        "mae_unguided": mae_unguided,
        "recon_guided": recon_guided,
        "prior_sample": prior_sample,
    }


def aggregate_results(
    all_results: Dict[int, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Aggregate results across test maps and observation counts.
    
    Args:
        all_results: Dict mapping n_obs -> list of per-map results
    
    Returns:
        Dictionary with aggregated statistics
    """
    aggregated = {
        "n_obs": [],
        "mse_guided_mean": [],
        "mse_guided_std": [],
        "mse_unguided_mean": [],
        "mse_unguided_std": [],
        "mae_guided_mean": [],
        "mae_guided_std": [],
        "mae_unguided_mean": [],
        "mae_unguided_std": [],
    }
    
    for n_obs in sorted(all_results.keys()):
        results = all_results[n_obs]
        
        mse_guided = [r["mse_guided"] for r in results]
        mse_unguided = [r["mse_unguided"] for r in results]
        mae_guided = [r["mae_guided"] for r in results]
        mae_unguided = [r["mae_unguided"] for r in results]
        
        aggregated["n_obs"].append(n_obs)
        aggregated["mse_guided_mean"].append(float(np.mean(mse_guided)))
        aggregated["mse_guided_std"].append(float(np.std(mse_guided)))
        aggregated["mse_unguided_mean"].append(float(np.mean(mse_unguided)))
        aggregated["mse_unguided_std"].append(float(np.std(mse_unguided)))
        aggregated["mae_guided_mean"].append(float(np.mean(mae_guided)))
        aggregated["mae_guided_std"].append(float(np.std(mae_guided)))
        aggregated["mae_unguided_mean"].append(float(np.mean(mae_unguided)))
        aggregated["mae_unguided_std"].append(float(np.std(mae_unguided)))
    
    return aggregated


def plot_accuracy_curves(
    aggregated: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot reconstruction accuracy vs observation count.
    
    Args:
        aggregated: Aggregated results
        output_path: Path to save plot
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_obs = aggregated["n_obs"]
    
    # MSE plot
    ax = axes[0]
    ax.errorbar(
        n_obs,
        aggregated["mse_guided_mean"],
        yerr=aggregated["mse_guided_std"],
        label="Guided",
        marker="o",
        linewidth=2,
        capsize=5,
    )
    ax.errorbar(
        n_obs,
        aggregated["mse_unguided_mean"],
        yerr=aggregated["mse_unguided_std"],
        label="Unguided (Prior)",
        marker="s",
        linewidth=2,
        capsize=5,
    )
    ax.set_xlabel("Number of Observations", fontsize=12)
    ax.set_ylabel("MSE (m²/s⁴)", fontsize=12)
    ax.set_title("Reconstruction Accuracy: MSE vs Observations", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    
    # MAE plot
    ax = axes[1]
    ax.errorbar(
        n_obs,
        aggregated["mae_guided_mean"],
        yerr=aggregated["mae_guided_std"],
        label="Guided",
        marker="o",
        linewidth=2,
        capsize=5,
    )
    ax.errorbar(
        n_obs,
        aggregated["mae_unguided_mean"],
        yerr=aggregated["mae_unguided_std"],
        label="Unguided (Prior)",
        marker="s",
        linewidth=2,
        capsize=5,
    )
    ax.set_xlabel("Number of Observations", fontsize=12)
    ax.set_ylabel("MAE (m/s²)", fontsize=12)
    ax.set_title("Reconstruction Accuracy: MAE vs Observations", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy curves to {output_path}")


def plot_guidance_effectiveness(
    aggregated: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot guidance effectiveness (relative improvement).
    
    Args:
        aggregated: Aggregated results
        output_path: Path to save plot
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_obs = aggregated["n_obs"]
    
    # Compute relative improvement: (MSE_unguided - MSE_guided) / MSE_unguided
    rel_improvement = [
        (unguided - guided) / unguided * 100
        for guided, unguided in zip(
            aggregated["mse_guided_mean"],
            aggregated["mse_unguided_mean"],
        )
    ]
    
    ax.bar(range(len(n_obs)), rel_improvement, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(n_obs)))
    ax.set_xticklabels([str(n) for n in n_obs])
    ax.set_xlabel("Number of Observations", fontsize=12)
    ax.set_ylabel("Relative Improvement (%)", fontsize=12)
    ax.set_title(
        "Guidance Effectiveness\n(MSE_unguided - MSE_guided) / MSE_unguided × 100%",
        fontsize=14,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved guidance effectiveness plot to {output_path}")


def plot_reconstruction_examples(
    gt_maps: List[np.ndarray],
    results: List[Dict[str, Any]],
    grid: MapGrid,
    n_obs: int,
    output_path: Path,
    n_examples: int = 3,
) -> None:
    """Plot example reconstructions for a specific observation count.
    
    Args:
        gt_maps: List of ground truth maps
        results: List of reconstruction results
        grid: Map grid
        n_obs: Number of observations
        output_path: Path to save plot
        n_examples: Number of examples to plot
    """
    sns.set_style("white")
    n_examples = min(n_examples, len(gt_maps))
    
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 5 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_examples):
        gt_map = gt_maps[i]
        result = results[i]
        recon_guided = result["recon_guided"]
        prior_sample = result["prior_sample"]
        
        # Shared colorbar limits
        vmin = min(gt_map.min(), recon_guided.min(), prior_sample.min())
        vmax = max(gt_map.max(), recon_guided.max(), prior_sample.max())
        
        # Ground truth
        ax = axes[i, 0]
        im = ax.imshow(
            gt_map,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Ground Truth (Map {i+1})", fontsize=12)
        ax.set_xlabel("Speed Index", fontsize=10)
        ax.set_ylabel("Actuation Index", fontsize=10)
        plt.colorbar(im, ax=ax, label="Acceleration (m/s²)")
        
        # Guided reconstruction
        ax = axes[i, 1]
        im = ax.imshow(
            recon_guided,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
        )
        mse = result["mse_guided"]
        mae = result["mae_guided"]
        ax.set_title(
            f"Guided Reconstruction\nMSE={mse:.3f}, MAE={mae:.3f}",
            fontsize=12,
        )
        ax.set_xlabel("Speed Index", fontsize=10)
        ax.set_ylabel("Actuation Index", fontsize=10)
        plt.colorbar(im, ax=ax, label="Acceleration (m/s²)")
        
        # Unguided (prior) sample
        ax = axes[i, 2]
        im = ax.imshow(
            prior_sample,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
        )
        mse = result["mse_unguided"]
        mae = result["mae_unguided"]
        ax.set_title(
            f"Prior Sample (No Guidance)\nMSE={mse:.3f}, MAE={mae:.3f}",
            fontsize=12,
        )
        ax.set_xlabel("Speed Index", fontsize=10)
        ax.set_ylabel("Actuation Index", fontsize=10)
        plt.colorbar(im, ax=ax, label="Acceleration (m/s²)")
    
    plt.suptitle(f"Reconstructions with {n_obs} Observations", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction examples to {output_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load configs
    print(f"\nLoading configurations...")
    with open(args.config, "r") as f:
        prior_config = yaml.safe_load(f)
    
    with open(args.randomization_config, "r") as f:
        rand_config = yaml.safe_load(f)
    
    # Create grid
    map_config = rand_config["dynamics_map"]
    grid = MapGrid(
        N_u=map_config["N_u"],
        N_v=map_config["N_v"],
        v_max=map_config["v_max"],
    )
    print(f"Grid: {grid.N_u}x{grid.N_v}, v_max={grid.v_max} m/s")
    
    # Load normalization stats
    norm_stats_path = Path("data/maps/norm_stats.json")
    if norm_stats_path.exists():
        with open(norm_stats_path, "r") as f:
            norm_stats = json.load(f)
        norm_mean = norm_stats["mean"]
        norm_std = norm_stats["std"]
        print(f"Loaded normalization: mean={norm_mean:.3f}, std={norm_std:.3f}")
    else:
        print("Warning: norm_stats.json not found, using default normalization")
        norm_mean = 0.0
        norm_std = 1.0
    
    # Load diffusion prior
    print(f"\nLoading diffusion prior from {args.checkpoint}...")
    device = torch.device(args.device)
    
    diffusion_prior = DiffusionPrior.load(args.checkpoint, device=device)
    
    # Set the scheduler based on args
    if args.scheduler == "DDPM":
        from diffusers import DDPMScheduler
        diffusion_prior.inference_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=False,  # Don't clip for smoother samples
        )
        print(f"Using DDPM scheduler (recommended for quality)")
    else:
        print(f"Using DDIM scheduler (faster but may be pixelated)")
    
    diffusion_prior.eval()
    print(f"Loaded model successfully")
    
    # Generate test maps
    print(f"\nGenerating {args.n_test_maps} test maps...")
    rand = ExtendedPlantRandomization.from_config(rand_config["vehicle_randomization"])
    
    gt_maps = []
    test_params = []
    for i in tqdm(range(args.n_test_maps), desc="Generating test maps"):
        rng = np.random.default_rng(args.seed + i)
        params = sample_extended_params(rng, rand)
        test_params.append(params)
        
        gt_map = generate_map_from_params(
            params=params,
            N_u=grid.N_u,
            N_v=grid.N_v,
            v_max=grid.v_max,
            dt_eval=0.02,
            n_eval_steps=5,
            smoothing_sigma=None,
        )
        gt_maps.append(gt_map)
    
    print(f"Generated {len(gt_maps)} test maps")
    
    # Evaluate for each observation count
    print(f"\nEvaluating for observation counts: {args.obs_counts}")
    
    all_results = {}  # n_obs -> list of results
    
    for n_obs in args.obs_counts:
        print(f"\n{'='*60}")
        print(f"Evaluating with {n_obs} observations...")
        print(f"{'='*60}")
        
        results = []
        for i, gt_map in enumerate(tqdm(gt_maps, desc=f"n_obs={n_obs}")):
            # Sample observations
            observations = sample_observations(
                gt_map=gt_map,
                grid=grid,
                n_obs=n_obs,
                sigma_noise=args.sigma_meas,
                seed=args.seed + i + n_obs * 1000,
            )
            
            # Evaluate reconstruction
            result = evaluate_reconstruction(
                diffusion_prior=diffusion_prior,
                gt_map=gt_map,
                grid=grid,
                observations=observations,
                norm_mean=norm_mean,
                norm_std=norm_std,
                guidance_scale=args.guidance_scale,
                sigma_meas=args.sigma_meas,
                num_inference_steps=args.num_inference_steps,
                gradient_smoothing_sigma=args.gradient_smoothing_sigma,
                device=device,
                seed=args.seed + i + n_obs * 1000,
            )
            results.append(result)
        
        all_results[n_obs] = results
        
        # Print summary for this n_obs
        mse_guided_mean = np.mean([r["mse_guided"] for r in results])
        mse_unguided_mean = np.mean([r["mse_unguided"] for r in results])
        mae_guided_mean = np.mean([r["mae_guided"] for r in results])
        mae_unguided_mean = np.mean([r["mae_unguided"] for r in results])
        
        print(f"\nResults for {n_obs} observations:")
        print(f"  MSE  - Guided: {mse_guided_mean:.4f}, Unguided: {mse_unguided_mean:.4f}")
        print(f"  MAE  - Guided: {mae_guided_mean:.4f}, Unguided: {mae_unguided_mean:.4f}")
        improvement = (mse_unguided_mean - mse_guided_mean) / mse_unguided_mean * 100
        print(f"  Improvement: {improvement:.2f}%")
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("Aggregating results...")
    print(f"{'='*60}")
    aggregated = aggregate_results(all_results)
    
    # Save metrics
    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Save evaluation config
    config_path = args.output_dir / "eval_config.yaml"
    eval_config = {
        "checkpoint": str(args.checkpoint),
        "n_test_maps": args.n_test_maps,
        "obs_counts": args.obs_counts,
        "guidance_scale": args.guidance_scale,
        "sigma_meas": args.sigma_meas,
        "num_inference_steps": args.num_inference_steps,
        "gradient_smoothing_sigma": args.gradient_smoothing_sigma,
        "scheduler": args.scheduler,
        "seed": args.seed,
        "grid": {"N_u": grid.N_u, "N_v": grid.N_v, "v_max": grid.v_max},
    }
    with open(config_path, "w") as f:
        yaml.dump(eval_config, f, default_flow_style=False)
    print(f"Saved evaluation config to {config_path}")
    
    # Create plots
    print(f"\n{'='*60}")
    print("Creating visualizations...")
    print(f"{'='*60}\n")
    
    # Accuracy curves
    plot_accuracy_curves(
        aggregated=aggregated,
        output_path=args.output_dir / "accuracy_vs_observations.png",
    )
    
    # Guidance effectiveness
    plot_guidance_effectiveness(
        aggregated=aggregated,
        output_path=args.output_dir / "guidance_effectiveness.png",
    )
    
    # Reconstruction examples for each n_obs
    for n_obs in args.obs_counts:
        plot_reconstruction_examples(
            gt_maps=gt_maps,
            results=all_results[n_obs],
            grid=grid,
            n_obs=n_obs,
            output_path=args.output_dir / f"reconstructions_nobs_{n_obs}.png",
            n_examples=3,
        )
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")
    print(f"\nAll results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  - metrics.json")
    print(f"  - eval_config.yaml")
    print(f"  - accuracy_vs_observations.png")
    print(f"  - guidance_effectiveness.png")
    for n_obs in args.obs_counts:
        print(f"  - reconstructions_nobs_{n_obs}.png")


if __name__ == "__main__":
    main()

