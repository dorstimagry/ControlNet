#!/usr/bin/env python3
"""Generate ground truth acceleration maps from randomized vehicles.

This script generates a dataset of GT acceleration maps by:
1. Sampling random vehicles from parameter ranges
2. Computing acceleration maps over (u, v) grid
3. Saving maps to disk with metadata
4. Computing and saving normalization statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import compute_normalization_stats, save_normalization_stats
from src.data.gt_map_generator import generate_map_from_params
from src.maps.grid import MapGrid
from utils.dynamics import (
    ExtendedPlantRandomization,
    sample_extended_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GT acceleration maps")
    parser.add_argument(
        "--n-maps",
        type=int,
        default=1000,
        help="Total number of maps to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/maps"),
        help="Output directory for maps",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config file with vehicle randomization parameters",
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.9,
        help="Fraction of maps for training (rest for validation)",
    )
    parser.add_argument(
        "--N-u",
        type=int,
        default=100,
        help="Number of u bins",
    )
    parser.add_argument(
        "--N-v",
        type=int,
        default=100,
        help="Number of v bins",
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=30.0,
        help="Maximum speed (m/s)",
    )
    parser.add_argument(
        "--dt-eval",
        type=float,
        default=0.02,
        help="Evaluation timestep (seconds)",
    )
    parser.add_argument(
        "--n-eval-steps",
        type=int,
        default=5,
        help="Number of evaluation steps per bin",
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=None,
        help="Optional Gaussian smoothing sigma",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=5,
        help="Number of sample maps to plot (0 = no plots)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    
    # Load randomization config
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        rand_config = ExtendedPlantRandomization.from_config(config)
    else:
        rand_config = ExtendedPlantRandomization()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine split
    n_train = int(args.n_maps * args.train_val_split)
    n_val = args.n_maps - n_train
    
    print(f"Generating {args.n_maps} maps ({n_train} train, {n_val} val)")
    print(f"Grid: {args.N_u} x {args.N_v}, v_max={args.v_max} m/s")
    print(f"Output: {output_dir}")
    
    # Create grid (for metadata)
    grid = MapGrid(N_u=args.N_u, N_v=args.N_v, v_max=args.v_max)
    
    # Generate maps
    for i in tqdm(range(args.n_maps), desc="Generating maps"):
        # Sample vehicle parameters
        params = sample_extended_params(rng, rand_config)
        
        # Generate map
        seed_i = args.seed + i
        X = generate_map_from_params(
            params,
            N_u=args.N_u,
            N_v=args.N_v,
            v_max=args.v_max,
            dt_eval=args.dt_eval,
            n_eval_steps=args.n_eval_steps,
            smoothing_sigma=args.smoothing_sigma,
            seed=seed_i,
        )
        
        # Determine split
        if i < n_train:
            split_dir = train_dir
            split_name = "train"
        else:
            split_dir = val_dir
            split_name = "val"
        
        # Save map with metadata
        output_path = split_dir / f"map_{i:05d}.npz"
        
        # Extract key parameters for metadata
        metadata = {
            "map": X,
            "seed": seed_i,
            "N_u": args.N_u,
            "N_v": args.N_v,
            "v_max": args.v_max,
            "mass": params.body.mass,
            "drag_area": params.body.drag_area,
            "rolling_coeff": params.body.rolling_coeff,
            "motor_V_max": params.motor.V_max,
            "motor_R": params.motor.R,
            "motor_K_e": params.motor.K_e,
            "motor_K_t": params.motor.K_t,
            "gear_ratio": params.motor.gear_ratio,
            "wheel_radius": params.wheel.radius,
            "split": split_name,
        }
        
        np.savez_compressed(output_path, **metadata)
    
    print("\nComputing normalization statistics...")
    norm_stats = compute_normalization_stats(output_dir, split="train")
    print(f"  Mean: {norm_stats['mean']:.4f}")
    print(f"  Std:  {norm_stats['std']:.4f}")
    
    # Save normalization stats
    save_normalization_stats(norm_stats, output_dir / "norm_stats.json")
    
    print(f"\nDone! Maps saved to {output_dir}")
    print(f"  Train: {n_train} maps in {train_dir}")
    print(f"  Val:   {n_val} maps in {val_dir}")
    
    # Plot sample maps
    if args.plot_samples > 0:
        print(f"\nGenerating plots for {args.plot_samples} sample maps...")
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Plot from training set
        n_to_plot = min(args.plot_samples, n_train)
        for i in range(n_to_plot):
            # Load map
            map_file = train_dir / f"map_{i:05d}.npz"
            data = np.load(map_file)
            X = data["map"]
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Full map
            im1 = axes[0].imshow(
                X.T,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                extent=[-1, 1, 0, args.v_max],
            )
            axes[0].set_xlabel("Command u")
            axes[0].set_ylabel("Speed v (m/s)")
            axes[0].set_title(f"Acceleration Map (map_{i:05d})")
            axes[0].grid(True, alpha=0.3)
            cbar1 = plt.colorbar(im1, ax=axes[0])
            cbar1.set_label("Acceleration (m/s²)")
            
            # Plot 2: Cross-sections
            mid_v = args.N_v // 2
            mid_u = args.N_u // 2
            
            u_centers = np.linspace(-1, 1, args.N_u)
            v_centers = np.linspace(0, args.v_max, args.N_v)
            
            axes[1].plot(u_centers, X[:, mid_v], 'b-', linewidth=2, label=f'v={v_centers[mid_v]:.1f} m/s')
            axes[1].plot(v_centers, X[mid_u, :], 'r-', linewidth=2, label=f'u={u_centers[mid_u]:.2f}')
            axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1].set_xlabel("Command u or Speed v")
            axes[1].set_ylabel("Acceleration (m/s²)")
            axes[1].set_title("Cross-Sections")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Add metadata
            mass = float(data["mass"])
            gear_ratio = float(data["gear_ratio"])
            fig.suptitle(
                f"Vehicle: mass={mass:.0f}kg, gear_ratio={gear_ratio:.1f}",
                fontsize=10,
                y=0.98,
            )
            
            plt.tight_layout()
            
            # Save
            plot_path = plot_dir / f"map_{i:05d}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
        
        print(f"  Plots saved to {plot_dir}")
        
        # Create summary plot with multiple maps
        if n_to_plot >= 4:
            print("\nGenerating summary plot...")
            n_summary = min(9, n_to_plot)  # 3x3 grid
            rows = int(np.ceil(np.sqrt(n_summary)))
            cols = int(np.ceil(n_summary / rows))
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if n_summary == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for idx in range(n_summary):
                map_file = train_dir / f"map_{idx:05d}.npz"
                data = np.load(map_file)
                X = data["map"]
                
                ax = axes[idx]
                im = ax.imshow(
                    X.T,
                    origin="lower",
                    aspect="auto",
                    cmap="RdBu_r",
                    extent=[-1, 1, 0, args.v_max],
                    vmin=-60,
                    vmax=10,
                )
                ax.set_xlabel("Command u", fontsize=8)
                ax.set_ylabel("Speed v (m/s)", fontsize=8)
                ax.set_title(f"Map {idx} (m={float(data['mass']):.0f}kg)", fontsize=9)
                ax.grid(True, alpha=0.2)
            
            # Hide unused subplots
            for idx in range(n_summary, len(axes)):
                axes[idx].axis('off')
            
            # Add colorbar
            fig.colorbar(im, ax=axes, label="Acceleration (m/s²)", shrink=0.8)
            
            fig.suptitle("GT Acceleration Maps - Sample Overview", fontsize=14, y=0.995)
            plt.tight_layout()
            
            summary_path = plot_dir / "summary_overview.png"
            plt.savefig(summary_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            print(f"  Summary plot saved to {summary_path}")


if __name__ == "__main__":
    main()

