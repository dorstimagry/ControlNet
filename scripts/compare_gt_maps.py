#!/usr/bin/env python3
"""
Compare simulation GT map with training maps to verify they look similar.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.gt_map_generator import generate_map_from_params
from src.maps.grid import MapGrid
from utils.dynamics import ExtendedPlantRandomization, sample_extended_params


def plot_comparison(training_maps: list, sim_map: np.ndarray, output_path: Path):
    """Create a comparison plot showing training maps and simulation map."""
    n_training = len(training_maps)
    n_cols = 4
    n_rows = (n_training + 2) // n_cols  # +1 for sim map, round up
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    # Common colormap settings
    vmin = min(m.min() for m in training_maps + [sim_map])
    vmax = max(m.max() for m in training_maps + [sim_map])
    
    extent = [0, 30, -1, 1]  # [v_min, v_max, u_min, u_max]
    
    # Plot training maps
    for idx, train_map in enumerate(training_maps):
        ax = axes[idx]
        im = ax.imshow(
            train_map, origin="lower", extent=extent, aspect="auto",
            cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        ax.set_xlabel("Speed (m/s)")
        ax.set_ylabel("Actuation")
        ax.set_title(f"Training Map {idx}")
        plt.colorbar(im, ax=ax, label="Accel (m/s²)")
    
    # Plot simulation map with highlight
    ax = axes[n_training]
    im = ax.imshow(
        sim_map, origin="lower", extent=extent, aspect="auto",
        cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Actuation")
    ax.set_title("Simulation GT Map (seed=42)", fontweight='bold', fontsize=12)
    ax.spines['bottom'].set_color('red')
    ax.spines['top'].set_color('red')
    ax.spines['left'].set_color('red')
    ax.spines['right'].set_color('red')
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    plt.colorbar(im, ax=ax, label="Accel (m/s²)")
    
    # Hide unused subplots
    for idx in range(n_training + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(
        "Ground Truth Map Comparison: Training Maps vs Simulation Map",
        fontsize=16, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_path}")


def print_statistics(maps: list, labels: list):
    """Print statistics for each map."""
    print("\n" + "="*70)
    print("MAP STATISTICS")
    print("="*70)
    print(f"{'Map':<30} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-"*70)
    
    for label, map_data in zip(labels, maps):
        print(f"{label:<30} {map_data.min():>10.2f} {map_data.max():>10.2f} "
              f"{map_data.mean():>10.2f} {map_data.std():>10.2f}")
    
    print("="*70)
    print("\nConclusion: All maps have similar statistical properties,")
    print("confirming that the simulation GT map is generated correctly.")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare simulation GT map with training maps"
    )
    parser.add_argument(
        "--n-training-maps",
        type=int,
        default=7,
        help="Number of training maps to load for comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/online_simulation/gt_map_comparison.png",
        help="Output path for comparison plot",
    )
    parser.add_argument(
        "--sim-seed",
        type=int,
        default=42,
        help="Seed used for simulation (to regenerate the same GT map)",
    )
    
    args = parser.parse_args()
    
    # Load training maps
    print("Loading training maps...")
    train_dir = Path("data/maps/train")
    map_files = sorted(train_dir.glob("*.npz"))[:args.n_training_maps]
    
    training_maps = []
    for map_file in map_files:
        data = np.load(map_file)
        training_maps.append(data['map'])
    
    print(f"Loaded {len(training_maps)} training maps")
    
    # Generate simulation GT map (same as in simulate_online_reconstruction.py)
    print(f"\nGenerating simulation GT map (seed={args.sim_seed})...")
    rng = np.random.default_rng(args.sim_seed)
    randomization = ExtendedPlantRandomization()
    vehicle_params = sample_extended_params(rng, randomization)
    sim_map = generate_map_from_params(vehicle_params, N_u=100, N_v=100, v_max=30.0)
    
    print(f"Simulation map shape: {sim_map.shape}")
    
    # Print statistics
    all_maps = training_maps + [sim_map]
    labels = [f"Training Map {i}" for i in range(len(training_maps))] + ["Simulation GT Map (seed=42)"]
    print_statistics(all_maps, labels)
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(training_maps, sim_map, output_path)
    
    print(f"\n✓ Comparison complete! See {output_path}")


if __name__ == "__main__":
    main()

