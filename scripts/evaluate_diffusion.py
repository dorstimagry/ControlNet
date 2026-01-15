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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
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

GRAVITY = 9.80665  # m/s^2


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
    
    # Evaluation mode
    parser.add_argument(
        "--mode",
        type=str,
        default="synthetic",
        choices=["synthetic", "offline", "online"],
        help="Evaluation mode: synthetic (generate GT maps), offline (load all observations), online (process incrementally)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to .pt file with trip data (required for offline/online modes)",
    )
    parser.add_argument(
        "--max-trips",
        type=int,
        default=None,
        help="Limit number of trips to process (None = all trips)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Timestep duration in seconds (default: 0.1s)",
    )
    parser.add_argument(
        "--reconstruct-every-k",
        type=int,
        default=10,
        help="For online mode: reconstruct every k timesteps",
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--n-test-maps",
        type=int,
        default=20,
        help="Number of test maps to generate (synthetic mode only)",
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


def load_trip_data(
    data_path: Path,
    max_trips: int | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load trip data from .pt file.
    
    Args:
        data_path: Path to all_trips_data.pt file
        max_trips: Maximum number of trips to load (None = all trips)
    
    Returns:
        Dictionary mapping trip_id -> trip_data dict with keys:
        - speed: np.ndarray (m/s)
        - acceleration: np.ndarray (m/s²)
        - throttle: np.ndarray (0-100)
        - brake: np.ndarray (0-100)
        - angle: np.ndarray (radians, road grade)
        - time: np.ndarray (optional, seconds)
    """
    print(f"Loading trip data from {data_path}...")
    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    
    trips = {}
    trip_keys = [k for k in raw.keys() if k != "metadata" and isinstance(raw[k], dict)]
    
    if max_trips is not None:
        trip_keys = trip_keys[:max_trips]
    
    for key in trip_keys:
        value = raw[key]
        if not isinstance(value, dict):
            continue
        
        # Extract required fields
        try:
            trip_data = {
                "speed": np.asarray(value["speed"], dtype=np.float64),
                "acceleration": np.asarray(value["acceleration"], dtype=np.float64),
                "throttle": np.asarray(value["throttle"], dtype=np.float64),
                "brake": np.asarray(value["brake"], dtype=np.float64),
                "angle": np.asarray(value.get("angle", np.zeros_like(value["speed"])), dtype=np.float64),
            }
            
            # Try to get time or estimate dt
            if "time" in value:
                trip_data["time"] = np.asarray(value["time"], dtype=np.float64)
            
            # Validate throttle/brake are in [0, 100] range
            throttle_min, throttle_max = trip_data["throttle"].min(), trip_data["throttle"].max()
            brake_min, brake_max = trip_data["brake"].min(), trip_data["brake"].max()
            
            if throttle_min < 0 or throttle_max > 100:
                print(f"Warning: Trip {key} throttle range [{throttle_min:.2f}, {throttle_max:.2f}] outside [0, 100], clamping")
                trip_data["throttle"] = np.clip(trip_data["throttle"], 0, 100)
            
            if brake_min < 0 or brake_max > 100:
                print(f"Warning: Trip {key} brake range [{brake_min:.2f}, {brake_max:.2f}] outside [0, 100], clamping")
                trip_data["brake"] = np.clip(trip_data["brake"], 0, 100)
            
            trips[key] = trip_data
        except KeyError as e:
            print(f"Warning: Trip {key} missing field {e}, skipping")
            continue
    
    print(f"Loaded {len(trips)} trips from {data_path}")
    return trips


def convert_trip_to_observations(
    trip_data: Dict[str, np.ndarray],
    grid: MapGrid,
    dt: float | None = None,
) -> List[Observation]:
    """Convert trip data to Observation objects.
    
    Args:
        trip_data: Dictionary with speed, acceleration, throttle, brake, angle arrays
        grid: Map grid for binning
        dt: Timestep duration (if None, use index as timestamp)
    
    Returns:
        List of Observation objects
    """
    speed = trip_data["speed"]
    acceleration = trip_data["acceleration"]
    throttle = trip_data["throttle"]
    brake = trip_data["brake"]
    angle = trip_data["angle"]
    
    n_points = len(speed)
    
    # Convert throttle/brake to actuation u = (throttle - brake) / 100.0
    u = (throttle - brake) / 100.0
    u = np.clip(u, -1.0, 1.0)
    
    # Use speed as v
    v = speed
    
    # Compute dynamics acceleration: a_dyn = a_meas - g * sin(grade)
    a_dyn = acceleration - GRAVITY * np.sin(angle)
    
    # Convert (u, v) to grid indices
    i_u = grid.bin_u(u)
    i_v = grid.bin_v(v)
    
    # Create timestamps
    if dt is not None:
        timestamps = np.arange(n_points) * dt
    else:
        timestamps = np.arange(n_points, dtype=np.float64)
    
    # Create observations
    observations = []
    for i in range(n_points):
        obs = Observation(
            i_u=int(i_u[i]),
            i_v=int(i_v[i]),
            a_dyn=float(a_dyn[i]),
            timestamp=float(timestamps[i]),
        )
        observations.append(obs)
    
    return observations


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


def evaluate_reconstruction_real_data(
    diffusion_prior: DiffusionPrior,
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
    """Evaluate guided reconstruction on real data (no GT comparison).
    
    Args:
        diffusion_prior: Trained diffusion prior model
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
        Dictionary with reconstructed map and statistics
    """
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
        progress_desc=f"Diffusion (n_obs={len(observations)})",
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
    
    # Compute statistics (no GT comparison)
    recon_mean = float(np.mean(recon_guided))
    recon_std = float(np.std(recon_guided))
    recon_min = float(np.min(recon_guided))
    recon_max = float(np.max(recon_guided))
    
    prior_mean = float(np.mean(prior_sample))
    prior_std = float(np.std(prior_sample))
    
    # Compute spatial smoothness (gradient magnitude)
    grad_u = np.gradient(recon_guided, axis=0)
    grad_v = np.gradient(recon_guided, axis=1)
    smoothness = float(np.mean(np.sqrt(grad_u**2 + grad_v**2)))
    
    # Compute bin-wise errors
    bin_errors = compute_bin_wise_errors(recon_guided, observations, grid)
    
    return {
        "recon_guided": recon_guided,
        "prior_sample": prior_sample,
        "recon_mean": recon_mean,
        "recon_std": recon_std,
        "recon_min": recon_min,
        "recon_max": recon_max,
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "smoothness": smoothness,
        "n_obs": len(observations),
        "bin_errors": bin_errors,
    }


def compute_bin_wise_errors(
    recon_map: np.ndarray,
    observations: List[Observation],
    grid: MapGrid,
) -> Dict[str, Any]:
    """Compute average error per bin between reconstruction and observations.
    
    For each bin (i_u, i_v), computes the average error:
    error = recon_map[i_u, i_v] - observed_acceleration
    
    Args:
        recon_map: Reconstructed map, shape (N_u, N_v)
        observations: List of observations
        grid: Map grid
    
    Returns:
        Dictionary with:
        - error_map: Average error per bin, shape (N_u, N_v), NaN for unvisited bins
        - error_std: Std dev of errors per bin, shape (N_u, N_v)
        - n_obs_per_bin: Number of observations per bin, shape (N_u, N_v)
    """
    error_map = np.full((grid.N_u, grid.N_v), np.nan, dtype=np.float32)
    error_std_map = np.full((grid.N_u, grid.N_v), np.nan, dtype=np.float32)
    n_obs_per_bin = np.zeros((grid.N_u, grid.N_v), dtype=np.int32)
    
    # Accumulate errors per bin
    bin_errors = {}  # (i_u, i_v) -> list of errors
    
    for obs in observations:
        i_u = obs.i_u
        i_v = obs.i_v
        
        # Get reconstructed value at this bin
        recon_value = recon_map[i_u, i_v]
        
        # Compute error: reconstruction - observation
        error = recon_value - obs.a_dyn
        
        # Accumulate
        if (i_u, i_v) not in bin_errors:
            bin_errors[(i_u, i_v)] = []
        bin_errors[(i_u, i_v)].append(error)
        n_obs_per_bin[i_u, i_v] += 1
    
    # Compute mean and std per bin
    for (i_u, i_v), errors in bin_errors.items():
        errors_array = np.array(errors)
        error_map[i_u, i_v] = float(np.mean(errors_array))
        if len(errors_array) > 1:
            error_std_map[i_u, i_v] = float(np.std(errors_array))
        else:
            error_std_map[i_u, i_v] = 0.0
    
    return {
        "error_map": error_map,
        "error_std": error_std_map,
        "n_obs_per_bin": n_obs_per_bin,
    }


def compute_coverage_metrics(
    observations: List[Observation],
    grid: MapGrid,
) -> Dict[str, Any]:
    """Compute observation coverage metrics.
    
    Args:
        observations: List of observations
        grid: Map grid
    
    Returns:
        Dictionary with coverage metrics
    """
    # Create visit count map
    visit_counts = np.zeros((grid.N_u, grid.N_v), dtype=np.int32)
    for obs in observations:
        visit_counts[obs.i_u, obs.i_v] += 1
    
    # Coverage: fraction of cells visited at least once
    n_visited = np.sum(visit_counts > 0)
    n_total = grid.N_u * grid.N_v
    coverage_fraction = n_visited / n_total
    
    # Average visits per visited cell
    visited_cells = visit_counts[visit_counts > 0]
    avg_visits = float(np.mean(visited_cells)) if len(visited_cells) > 0 else 0.0
    
    # Max visits
    max_visits = int(np.max(visit_counts))
    
    return {
        "coverage_fraction": float(coverage_fraction),
        "n_visited_cells": int(n_visited),
        "n_total_cells": int(n_total),
        "avg_visits_per_cell": float(avg_visits),
        "max_visits": max_visits,
        "visit_counts": visit_counts,
    }


def aggregate_results(
    all_results: Dict[int, List[Dict[str, Any]]],
    has_gt: bool = True,
) -> Dict[str, Any]:
    """Aggregate results across test maps and observation counts.
    
    Args:
        all_results: Dict mapping n_obs -> list of per-map results
        has_gt: Whether results include ground truth comparison metrics
    
    Returns:
        Dictionary with aggregated statistics
    """
    if has_gt:
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
    else:
        # Real data mode: aggregate reconstruction statistics
        aggregated = {
            "n_obs": [],
            "recon_mean_mean": [],
            "recon_mean_std": [],
            "recon_std_mean": [],
            "recon_std_std": [],
            "smoothness_mean": [],
            "smoothness_std": [],
            "coverage_mean": [],
            "coverage_std": [],
        }
        
        for n_obs in sorted(all_results.keys()):
            results = all_results[n_obs]
            
            recon_means = [r["recon_mean"] for r in results]
            recon_stds = [r["recon_std"] for r in results]
            smoothnesses = [r["smoothness"] for r in results]
            coverages = [r.get("coverage_fraction", 0.0) for r in results]
            
            aggregated["n_obs"].append(n_obs)
            aggregated["recon_mean_mean"].append(float(np.mean(recon_means)))
            aggregated["recon_mean_std"].append(float(np.std(recon_means)))
            aggregated["recon_std_mean"].append(float(np.mean(recon_stds)))
            aggregated["recon_std_std"].append(float(np.std(recon_stds)))
            aggregated["smoothness_mean"].append(float(np.mean(smoothnesses)))
            aggregated["smoothness_std"].append(float(np.std(smoothnesses)))
            aggregated["coverage_mean"].append(float(np.mean(coverages)))
            aggregated["coverage_std"].append(float(np.std(coverages)))
    
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


def plot_reconstruction_real_data(
    results: List[Dict[str, Any]],
    grid: MapGrid,
    n_obs: int,
    output_path: Path,
    n_examples: int = 3,
) -> None:
    """Plot example reconstructions for real data (no GT comparison).
    
    Args:
        results: List of reconstruction results
        grid: Map grid
        n_obs: Number of observations
        output_path: Path to save plot
        n_examples: Number of examples to plot
    """
    sns.set_style("white")
    n_examples = min(n_examples, len(results))
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(8, 5 * n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for i in range(n_examples):
        result = results[i]
        recon_guided = result["recon_guided"]
        
        # Colorbar limits from guided reconstruction only
        vmin = recon_guided.min()
        vmax = recon_guided.max()
        
        # Guided reconstruction
        ax = axes[i]
        im = ax.imshow(
            recon_guided,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
        )
        mean_val = result["recon_mean"]
        std_val = result["recon_std"]
        smoothness = result["smoothness"]
        ax.set_title(
            f"Guided Reconstruction\nMean={mean_val:.3f}, Std={std_val:.3f}, Smoothness={smoothness:.3f}",
            fontsize=12,
        )
        ax.set_xlabel("Speed Index", fontsize=10)
        ax.set_ylabel("Actuation Index", fontsize=10)
        plt.colorbar(im, ax=ax, label="Acceleration (m/s²)")
    
    plt.suptitle(f"Reconstructions with {n_obs} Observations (Real Data)", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction examples to {output_path}")


def plot_coverage_heatmap(
    visit_counts: np.ndarray,
    grid: MapGrid,
    output_path: Path,
    title: str = "Observation Coverage",
) -> None:
    """Plot observation coverage heatmap.
    
    Args:
        visit_counts: Visit count matrix, shape (N_u, N_v)
        grid: Map grid
        output_path: Path to save plot
        title: Plot title
    """
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        visit_counts,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
    )
    ax.set_xlabel("Speed Index", fontsize=12)
    ax.set_ylabel("Actuation Index", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, label="Visit Count")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved coverage heatmap to {output_path}")


def plot_bin_wise_errors(
    error_map: np.ndarray,
    grid: MapGrid,
    output_path: Path,
    n_obs: int,
    title: str = "Bin-wise Reconstruction Error",
) -> None:
    """Plot average error per bin between reconstruction and observations.
    
    Args:
        error_map: Average error per bin, shape (N_u, N_v), NaN for unvisited bins
        grid: Map grid
        output_path: Path to save plot
        n_obs: Number of observations used
        title: Plot title
    """
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a diverging colormap centered at zero
    # Mask NaN values (unvisited bins)
    masked_errors = np.ma.masked_invalid(error_map)
    
    # Determine symmetric colorbar limits
    valid_errors = error_map[~np.isnan(error_map)]
    if len(valid_errors) > 0:
        vmax = max(abs(valid_errors.min()), abs(valid_errors.max()))
        vmin = -vmax
    else:
        vmin, vmax = -1.0, 1.0
    
    im = ax.imshow(
        masked_errors,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",  # Red-Blue reversed: red for positive error (overestimate), blue for negative (underestimate)
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Speed Index", fontsize=12)
    ax.set_ylabel("Actuation Index", fontsize=12)
    ax.set_title(f"{title}\n(n_obs={n_obs})", fontsize=14)
    cbar = plt.colorbar(im, ax=ax, label="Average Error (m/s²)")
    
    # Add text annotation for statistics
    n_visited = np.sum(~np.isnan(error_map))
    n_total = grid.N_u * grid.N_v
    mean_error = float(np.nanmean(error_map))
    std_error = float(np.nanstd(error_map))
    textstr = f"Visited bins: {n_visited}/{n_total}\nMean error: {mean_error:.3f} m/s²\nStd error: {std_error:.3f} m/s²"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bin-wise error heatmap to {output_path}")


def create_online_animation(
    trip_result: Dict[str, Any],
    grid: MapGrid,
    output_path: Path,
    fps: int = 10,
) -> None:
    """Create animation for online reconstruction from trip results.
    
    Panels:
    1. Reconstruction (updating)
    2. Trajectory visualization (heatmap + current position)
    3. Observation density map
    4. Reconstruction statistics over time
    
    Args:
        trip_result: Dictionary with reconstruction_history, trajectory_u, trajectory_v
        grid: Map grid
        output_path: Path to save animation
        fps: Frames per second
    """
    reconstruction_history = trip_result["reconstruction_history"]
    trajectory_u = trip_result["trajectory_u"]
    trajectory_v = trip_result["trajectory_v"]
    trip_id = trip_result["trip_id"]
    
    if len(reconstruction_history) == 0:
        print(f"Warning: No reconstruction history for trip {trip_id}, skipping animation")
        return
    
    print(f"Creating animation for trip {trip_id} with {len(reconstruction_history)} frames...")
    
    # Setup figure with 2x2 grid
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Reconstruction
    ax2 = fig.add_subplot(gs[0, 1])  # Trajectory
    ax3 = fig.add_subplot(gs[1, 0])  # Observations
    ax4 = fig.add_subplot(gs[1, 1])  # Statistics
    
    # Common settings
    extent = [0, grid.v_max, -1, 1]  # [v_min, v_max, u_min, u_max]
    
    # Determine colorbar limits from all reconstructions
    recon_maps = []
    for snapshot in reconstruction_history:
        if "recon_map" in snapshot:
            recon_maps.append(snapshot["recon_map"])
    
    if len(recon_maps) > 0:
        vmin = min([np.min(m) for m in recon_maps])
        vmax = max([np.max(m) for m in recon_maps])
        map_shape = recon_maps[0].shape
    else:
        # Fallback: use final reconstruction
        final_recon = trip_result.get("final_reconstruction")
        if final_recon is not None:
            map_shape = final_recon.shape
            vmin = float(np.min(final_recon))
            vmax = float(np.max(final_recon))
        else:
            map_shape = (grid.N_u, grid.N_v)
            vmin, vmax = -3.0, 3.0
    
    # Panel 1: Reconstruction (will update)
    im1 = ax1.imshow(
        np.zeros(map_shape), origin="lower", extent=extent, aspect="auto",
        cmap="RdYlGn", vmin=vmin, vmax=vmax
    )
    ax1.set_xlabel("Speed (m/s)", fontsize=12)
    ax1.set_ylabel("Actuation", fontsize=12)
    title1 = ax1.set_title("Reconstruction (t=0, n_obs=0)")
    plt.colorbar(im1, ax=ax1, label="Acceleration (m/s²)")
    
    # Panel 2: Trajectory (will update)
    im2 = ax2.imshow(
        np.zeros((grid.N_u, grid.N_v)), origin="lower", extent=extent,
        aspect="auto", cmap="YlOrRd", vmin=0, vmax=1
    )
    ax2.set_xlabel("Speed (m/s)", fontsize=12)
    ax2.set_ylabel("Actuation", fontsize=12)
    ax2.set_title("State Space Exploration")
    plt.colorbar(im2, ax=ax2, label="Visit Count")
    
    # Trail and current position markers
    trail_line, = ax2.plot([], [], 'b-', alpha=0.5, linewidth=1, label="Recent trail")
    current_pos, = ax2.plot([], [], 'ro', markersize=10, label="Current position")
    ax2.legend(loc="upper right")
    
    # Panel 3: Observation density (will update)
    scatter = ax3.scatter([], [], c='blue', alpha=0.3, s=10)
    ax3.set_xlabel("Speed (m/s)", fontsize=12)
    ax3.set_ylabel("Actuation", fontsize=12)
    ax3.set_title("Observation Locations")
    ax3.set_xlim(0, grid.v_max)
    ax3.set_ylim(-1, 1)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Statistics over time
    # Pre-compute all statistics for efficiency
    time_axis = [s["timestep"] for s in reconstruction_history]
    n_obs_history = [s["n_obs"] for s in reconstruction_history]
    recon_mean_history = [s.get("recon_mean", 0.0) for s in reconstruction_history]
    smoothness_history = [s.get("smoothness", 0.0) for s in reconstruction_history]
    
    line_nobs, = ax4.plot([], [], 'b-', label='n_obs', linewidth=2)
    ax4_twin = ax4.twinx()
    line_mean, = ax4_twin.plot([], [], 'r-', label='Mean (m/s²)', linewidth=2)
    line_smooth, = ax4_twin.plot([], [], 'g-', label='Smoothness', linewidth=2)
    
    ax4.set_xlabel("Timestep", fontsize=12)
    ax4.set_ylabel("Number of Observations", fontsize=12, color='b')
    ax4_twin.set_ylabel("Reconstruction Stats", fontsize=12)
    ax4.set_title("Reconstruction Statistics")
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lines = [line_nobs, line_mean, line_smooth]
    labels = ['n_obs', 'Mean (m/s²)', 'Smoothness']
    ax4.legend(lines, labels, loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # Set axis limits based on all data
    if len(time_axis) > 0:
        ax4.set_xlim(0, max(time_axis))
        ax4.set_ylim(0, max(n_obs_history) if n_obs_history else 1)
        if recon_mean_history:
            stats_min = min(min(recon_mean_history), min(smoothness_history))
            stats_max = max(max(recon_mean_history), max(smoothness_history))
            ax4_twin.set_ylim(stats_min, stats_max)
    
    # Animation update function
    def update(frame_idx):
        if frame_idx >= len(reconstruction_history):
            return im1, title1, im2, trail_line, current_pos, scatter, line_nobs, line_mean, line_smooth
        
        snapshot = reconstruction_history[frame_idx]
        timestep = snapshot["timestep"]
        n_obs = snapshot["n_obs"]
        recon_mean_val = snapshot.get("recon_mean", 0.0)
        smoothness_val = snapshot.get("smoothness", 0.0)
        
        # Get reconstruction map
        recon_map = snapshot.get("recon_map")
        if recon_map is None:
            # Fallback: use final reconstruction for last frame
            if frame_idx == len(reconstruction_history) - 1:
                recon_map = trip_result.get("final_reconstruction")
            if recon_map is None:
                # Last resort: create empty map
                recon_map = np.zeros(map_shape)
        
        # Update Panel 1: Reconstruction
        im1.set_data(recon_map)
        title1.set_text(f"Reconstruction (t={timestep}, n_obs={n_obs}, mean={recon_mean_val:.3f} m/s²)")
        
        # Update Panel 2: Trajectory
        traj_end_idx = min(timestep, len(trajectory_u))
        u_traj = trajectory_u[:traj_end_idx]
        v_traj = trajectory_v[:traj_end_idx]
        
        # Update visit counts
        visit_counts_frame = np.zeros((grid.N_u, grid.N_v))
        for u_val, v_val in zip(u_traj, v_traj):
            i_u = grid.bin_u(u_val)
            i_v = grid.bin_v(v_val)
            visit_counts_frame[i_u, i_v] += 1
        
        # Normalize for visualization
        visit_max = visit_counts_frame.max()
        if visit_max > 0:
            visit_counts_norm = visit_counts_frame / visit_max
        else:
            visit_counts_norm = visit_counts_frame
        
        im2.set_data(visit_counts_norm)
        im2.set_clim(0, 1)
        
        # Update trail (last 100 steps)
        trail_start = max(0, traj_end_idx - 100)
        trail_u = u_traj[trail_start:traj_end_idx]
        trail_v = v_traj[trail_start:traj_end_idx]
        if len(trail_u) > 0:
            trail_line.set_data(trail_v, trail_u)
        
        # Update current position
        if traj_end_idx > 0:
            current_pos.set_data([v_traj[-1]], [u_traj[-1]])
        
        # Update Panel 3: Observations (approximate from trajectory)
        if len(u_traj) > 0:
            scatter.set_offsets(np.c_[v_traj, u_traj])
        
        # Update Panel 4: Statistics (show up to current frame)
        frame_end = frame_idx + 1
        line_nobs.set_data(time_axis[:frame_end], n_obs_history[:frame_end])
        line_mean.set_data(time_axis[:frame_end], recon_mean_history[:frame_end])
        line_smooth.set_data(time_axis[:frame_end], smoothness_history[:frame_end])
        
        return im1, title1, im2, trail_line, current_pos, scatter, line_nobs, line_mean, line_smooth
    
    # Create animation
    n_frames = len(reconstruction_history)
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000//fps, blit=False
    )
    
    # Save (try MP4, fall back to GIF if ffmpeg unavailable)
    print(f"Saving animation to {output_path}...")
    dpi = 150
    try:
        anim.save(str(output_path), writer='ffmpeg', fps=fps, dpi=dpi)
        print(f"Saved animation to {output_path}")
    except (ValueError, RuntimeError):
        # ffmpeg not available, save as GIF
        gif_path = output_path.with_suffix('.gif')
        print(f"ffmpeg unavailable, saving as GIF: {gif_path}")
        anim.save(str(gif_path), writer='pillow', fps=fps, dpi=dpi)
        print(f"Saved animation to {gif_path}")
    finally:
        plt.close()


def plot_self_consistency_curves(
    aggregated: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot self-consistency metrics vs observation count.
    
    Args:
        aggregated: Aggregated results
        output_path: Path to save plot
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    n_obs = aggregated["n_obs"]
    
    # Reconstruction statistics
    ax = axes[0, 0]
    ax.errorbar(
        n_obs,
        aggregated["recon_mean_mean"],
        yerr=aggregated["recon_mean_std"],
        label="Mean",
        marker="o",
        linewidth=2,
        capsize=5,
    )
    ax.set_xlabel("Number of Observations", fontsize=12)
    ax.set_ylabel("Mean Acceleration (m/s²)", fontsize=12)
    ax.set_title("Reconstruction Mean vs Observations", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    
    # Smoothness
    ax = axes[0, 1]
    ax.errorbar(
        n_obs,
        aggregated["smoothness_mean"],
        yerr=aggregated["smoothness_std"],
        label="Smoothness",
        marker="s",
        linewidth=2,
        capsize=5,
        color="green",
    )
    ax.set_xlabel("Number of Observations", fontsize=12)
    ax.set_ylabel("Spatial Smoothness", fontsize=12)
    ax.set_title("Reconstruction Smoothness vs Observations", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    
    # Coverage
    ax = axes[1, 0]
    ax.errorbar(
        n_obs,
        aggregated["coverage_mean"],
        yerr=aggregated["coverage_std"],
        label="Coverage",
        marker="^",
        linewidth=2,
        capsize=5,
        color="orange",
    )
    ax.set_xlabel("Number of Observations", fontsize=12)
    ax.set_ylabel("Coverage Fraction", fontsize=12)
    ax.set_title("Observation Coverage vs Observations", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
    
    # Standard deviation
    ax = axes[1, 1]
    ax.errorbar(
        n_obs,
        aggregated["recon_std_mean"],
        yerr=aggregated["recon_std_std"],
        label="Std Dev",
        marker="d",
        linewidth=2,
        capsize=5,
        color="red",
    )
    ax.set_xlabel("Number of Observations", fontsize=12)
    ax.set_ylabel("Reconstruction Std Dev (m/s²)", fontsize=12)
    ax.set_title("Reconstruction Variance vs Observations", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved self-consistency curves to {output_path}")


@dataclass
class ReconstructionConfig:
    """Configuration for online reconstruction."""
    reconstruct_every_k: int = 10
    buffer_size: int = 200
    num_inference_steps: int = 50
    guidance_scale: float = 100.0
    sigma_meas: float = 0.5
    gradient_smoothing_sigma: float = 10.0
    lambda_decay: float = 0.3
    w_min: float = 0.2
    patch_filtering_enabled: bool = False
    patch_size_u: int = 10
    patch_size_v: int = 10


class ReconstructionTracker:
    """Tracks observations and performs periodic reconstructions."""
    
    def __init__(
        self,
        diffusion_prior: DiffusionPrior,
        grid: MapGrid,
        config: ReconstructionConfig,
        norm_mean: float,
        norm_std: float,
        device: torch.device,
    ):
        self.diffusion_prior = diffusion_prior
        self.grid = grid
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.device = device
        
        # Observation buffer
        self.buffer = ObservationBuffer(
            capacity=config.buffer_size,
            lambda_decay=config.lambda_decay,
            w_min=config.w_min,
        )
        
        # Guided sampler
        self.sampler = GuidedDiffusionSampler(
            diffusion_prior=diffusion_prior,
            guidance_scale=config.guidance_scale,
            sigma_meas=config.sigma_meas,
            num_inference_steps=config.num_inference_steps,
            gradient_smoothing_sigma=config.gradient_smoothing_sigma,
        )
        
        # Reconstruction history
        self.reconstruction_history = []
        self.current_reconstruction = None
        
        # Trajectory tracking
        self.trajectory_u = []
        self.trajectory_v = []
        self.visit_counts = np.zeros((grid.N_u, grid.N_v))
        
    def add_observation(
        self,
        u: float,
        v: float,
        a_meas: float,
        grade: float,
        t: float,
    ) -> None:
        """Add a new observation to the buffer."""
        i_u = self.grid.bin_u(u)
        i_v = self.grid.bin_v(v)
        
        # Gravity compensation: a_dyn = a_meas - g*sin(grade)
        a_dyn = a_meas - GRAVITY * np.sin(grade)
        
        self.buffer.add(i_u, i_v, a_dyn, t)
        self.trajectory_u.append(u)
        self.trajectory_v.append(v)
        self.visit_counts[i_u, i_v] += 1
    
    def reconstruct(self, t_now: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform guided reconstruction."""
        observations = self.buffer.get_observations()
        n_obs = len(observations)
        
        if n_obs == 0:
            map_tensor = self.diffusion_prior.sample(
                batch_size=1,
                num_inference_steps=self.config.num_inference_steps,
                device=self.device,
            )
            recon_map = map_tensor.squeeze().cpu().numpy()
        else:
            map_tensor = self.sampler.sample(
                obs_buffer=self.buffer,
                t_now=t_now,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
                device=self.device,
                x_init=None,
            )
            recon_map = map_tensor.squeeze().cpu().numpy()
        
        # Denormalize
        recon_map = recon_map * self.norm_std + self.norm_mean
        
        # Compute statistics
        stats = {
            "n_obs": n_obs,
            "recon_mean": float(np.mean(recon_map)),
            "recon_std": float(np.std(recon_map)),
            "recon_min": float(np.min(recon_map)),
            "recon_max": float(np.max(recon_map)),
        }
        
        # Compute smoothness
        grad_u = np.gradient(recon_map, axis=0)
        grad_v = np.gradient(recon_map, axis=1)
        stats["smoothness"] = float(np.mean(np.sqrt(grad_u**2 + grad_v**2)))
        
        self.current_reconstruction = recon_map
        return recon_map, stats


def evaluate_offline_real_data(
    diffusion_prior: DiffusionPrior,
    grid: MapGrid,
    trips: Dict[str, Dict[str, np.ndarray]],
    obs_counts: List[int],
    norm_mean: float,
    norm_std: float,
    guidance_scale: float,
    sigma_meas: float,
    num_inference_steps: int,
    gradient_smoothing_sigma: float,
    device: torch.device,
    dt: float,
    seed: int,
) -> Dict[int, List[Dict[str, Any]]]:
    """Evaluate offline mode on real trip data.
    
    Concatenates all trips into one sequence, then subsamples observations
    from the concatenated sequence for each observation count.
    
    Args:
        diffusion_prior: Trained diffusion prior model
        grid: Map grid
        trips: Dictionary of trip data
        obs_counts: List of observation counts to evaluate (total counts in concatenated sequence)
        norm_mean: Normalization mean
        norm_std: Normalization std
        guidance_scale: Guidance strength
        sigma_meas: Measurement noise std
        num_inference_steps: Number of diffusion steps
        gradient_smoothing_sigma: Gradient smoothing sigma
        device: Device
        dt: Timestep duration
        seed: Random seed
    
    Returns:
        Dictionary mapping n_obs -> list of results (single result per n_obs)
    """
    print(f"\nConcatenating {len(trips)} trips...")
    
    # Concatenate all trips into one observation list
    concatenated_observations = []
    current_time = 0.0
    
    for trip_id, trip_data in tqdm(trips.items(), desc="Concatenating trips"):
        # Convert trip to observations
        trip_observations = convert_trip_to_observations(trip_data, grid, dt)
        
        # Update timestamps to continue from previous trip
        for obs in trip_observations:
            obs.timestamp = current_time + obs.timestamp
            concatenated_observations.append(obs)
        
        # Update current time for next trip
        if len(trip_observations) > 0:
            current_time = concatenated_observations[-1].timestamp + dt
    
    total_obs = len(concatenated_observations)
    print(f"Concatenated {total_obs} total observations from {len(trips)} trips")
    
    all_results = {}
    rng = np.random.default_rng(seed)
    
    # Filter out observation counts that exceed available observations
    valid_obs_counts = [n for n in obs_counts if n <= total_obs]
    if len(valid_obs_counts) < len(obs_counts):
        skipped = [n for n in obs_counts if n > total_obs]
        print(f"Warning: Skipping observation counts {skipped} (exceed available {total_obs} observations)")
    
    for n_obs in valid_obs_counts:
        print(f"\n{'='*60}")
        print(f"Evaluating with {n_obs} observations (from {total_obs} total)...")
        print(f"{'='*60}")
        # Subsample observations (random) from concatenated sequence
        indices = rng.choice(total_obs, size=n_obs, replace=False)
        observations = [concatenated_observations[i] for i in sorted(indices)]
        
        # Evaluate reconstruction
        result = evaluate_reconstruction_real_data(
            diffusion_prior=diffusion_prior,
            grid=grid,
            observations=observations,
            norm_mean=norm_mean,
            norm_std=norm_std,
            guidance_scale=guidance_scale,
            sigma_meas=sigma_meas,
            num_inference_steps=num_inference_steps,
            gradient_smoothing_sigma=gradient_smoothing_sigma,
            device=device,
            seed=seed + n_obs,
        )
        
        # Add coverage metrics
        coverage = compute_coverage_metrics(observations, grid)
        result.update(coverage)
        result["total_observations"] = total_obs
        result["n_trips"] = len(trips)
        
        all_results[n_obs] = [result]  # Single result per n_obs
        
        # Print summary
        print(f"\nResults for {n_obs} observations:")
        print(f"  Reconstruction mean: {result['recon_mean']:.4f}")
        print(f"  Smoothness: {result['smoothness']:.4f}")
        print(f"  Coverage: {result['coverage_fraction']:.4f}")
    
    return all_results


def evaluate_online_real_data(
    diffusion_prior: DiffusionPrior,
    grid: MapGrid,
    trips: Dict[str, Dict[str, np.ndarray]],
    norm_mean: float,
    norm_std: float,
    guidance_scale: float,
    sigma_meas: float,
    num_inference_steps: int,
    gradient_smoothing_sigma: float,
    device: torch.device,
    dt: float,
    reconstruct_every_k: int,
    buffer_size: int,
    lambda_decay: float,
    w_min: float,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate online mode on real trip data.
    
    Args:
        diffusion_prior: Trained diffusion prior model
        grid: Map grid
        trips: Dictionary of trip data
        norm_mean: Normalization mean
        norm_std: Normalization std
        guidance_scale: Guidance strength
        sigma_meas: Measurement noise std
        num_inference_steps: Number of diffusion steps
        gradient_smoothing_sigma: Gradient smoothing sigma
        device: Device
        dt: Timestep duration
        reconstruct_every_k: Reconstruct every k timesteps
        buffer_size: Observation buffer size
        lambda_decay: Time decay rate
        w_min: Minimum weight floor
        seed: Random seed
    
    Returns:
        Dictionary with online evaluation results
    """
    config = ReconstructionConfig(
        reconstruct_every_k=reconstruct_every_k,
        buffer_size=buffer_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        sigma_meas=sigma_meas,
        gradient_smoothing_sigma=gradient_smoothing_sigma,
        lambda_decay=lambda_decay,
        w_min=w_min,
    )
    
    all_trip_results = []
    
    for trip_id, trip_data in tqdm(trips.items(), desc="Processing trips"):
        speed = trip_data["speed"]
        acceleration = trip_data["acceleration"]
        throttle = trip_data["throttle"]
        brake = trip_data["brake"]
        angle = trip_data.get("angle", np.zeros_like(speed))
        
        n_points = len(speed)
        if n_points == 0:
            print(f"Warning: Trip {trip_id} has no data points, skipping")
            continue
        
        # Initialize tracker
        tracker = ReconstructionTracker(
            diffusion_prior=diffusion_prior,
            grid=grid,
            config=config,
            norm_mean=norm_mean,
            norm_std=norm_std,
            device=device,
        )
        
        # Process observations incrementally
        reconstruction_history = []
        
        for i in range(n_points):
            # Convert throttle/brake to actuation
            u = (throttle[i] - brake[i]) / 100.0
            u = np.clip(u, -1.0, 1.0)
            v = speed[i]
            a_meas = acceleration[i]
            grade = angle[i]
            
            # Add observation (tracker will handle gravity compensation)
            t = i * dt
            tracker.add_observation(u, v, a_meas, grade, t)
            
            # Reconstruct periodically
            if (i + 1) % reconstruct_every_k == 0 or i == n_points - 1:
                recon_map, stats = tracker.reconstruct(t_now=t)
                reconstruction_history.append({
                    "timestep": i + 1,
                    "timestamp": t,
                    "recon_map": recon_map,  # Store the actual map
                    **stats,
                })
        
        # Compute final coverage from tracker's visit counts
        # Create observations list for coverage computation
        final_observations = []
        for i in range(n_points):
            u = (throttle[i] - brake[i]) / 100.0
            u = np.clip(u, -1.0, 1.0)
            v = speed[i]
            i_u = grid.bin_u(u)
            i_v = grid.bin_v(v)
            a_dyn = acceleration[i] - GRAVITY * np.sin(angle[i])
            obs = Observation(i_u=int(i_u), i_v=int(i_v), a_dyn=float(a_dyn), timestamp=i*dt)
            final_observations.append(obs)
        
        coverage = compute_coverage_metrics(final_observations, grid)
        
        trip_result = {
            "trip_id": trip_id,
            "n_total_obs": n_points,
            "reconstruction_history": reconstruction_history,
            "coverage": coverage,
            "final_reconstruction": tracker.current_reconstruction,
            "trajectory_u": tracker.trajectory_u,
            "trajectory_v": tracker.trajectory_v,
        }
        all_trip_results.append(trip_result)
    
    return {
        "trips": all_trip_results,
        "n_trips": len(all_trip_results),
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Validate arguments based on mode
    if args.mode in ["offline", "online"]:
        if args.data_path is None:
            raise ValueError(f"--data-path is required for {args.mode} mode")
        if not args.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    
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
    
    # Branch based on mode
    if args.mode == "synthetic":
        # Original synthetic evaluation
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
        
        all_results = {}
        
        for n_obs in args.obs_counts:
            print(f"\n{'='*60}")
            print(f"Evaluating with {n_obs} observations...")
            print(f"{'='*60}")
            
            results = []
            for i, gt_map in enumerate(tqdm(gt_maps, desc=f"n_obs={n_obs}")):
                observations = sample_observations(
                    gt_map=gt_map,
                    grid=grid,
                    n_obs=n_obs,
                    sigma_noise=args.sigma_meas,
                    seed=args.seed + i + n_obs * 1000,
                )
                
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
            
            # Print summary
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
        aggregated = aggregate_results(all_results, has_gt=True)
        
        # Save metrics
        metrics_path = args.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        
        # Save config
        config_path = args.output_dir / "eval_config.yaml"
        eval_config = {
            "mode": args.mode,
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
        
        # Create plots
        plot_accuracy_curves(
            aggregated=aggregated,
            output_path=args.output_dir / "accuracy_vs_observations.png",
        )
        plot_guidance_effectiveness(
            aggregated=aggregated,
            output_path=args.output_dir / "guidance_effectiveness.png",
        )
        for n_obs in args.obs_counts:
            plot_reconstruction_examples(
                gt_maps=gt_maps,
                results=all_results[n_obs],
                grid=grid,
                n_obs=n_obs,
                output_path=args.output_dir / f"reconstructions_nobs_{n_obs}.png",
                n_examples=3,
            )
    
    elif args.mode == "offline":
        # Offline evaluation on real data
        trips = load_trip_data(args.data_path, args.max_trips)
        
        all_results = evaluate_offline_real_data(
            diffusion_prior=diffusion_prior,
            grid=grid,
            trips=trips,
            obs_counts=args.obs_counts,
            norm_mean=norm_mean,
            norm_std=norm_std,
            guidance_scale=args.guidance_scale,
            sigma_meas=args.sigma_meas,
            num_inference_steps=args.num_inference_steps,
            gradient_smoothing_sigma=args.gradient_smoothing_sigma,
            device=device,
            dt=args.dt,
            seed=args.seed,
        )
        
        # Aggregate results
        aggregated = aggregate_results(all_results, has_gt=False)
        
        # Save metrics
        metrics_path = args.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        
        # Save detailed results
        detailed_path = args.output_dir / "detailed_results.json"
        # Convert numpy arrays to lists for JSON serialization
        detailed_results = {}
        for n_obs, results in all_results.items():
            detailed_results[n_obs] = []
            for r in results:
                r_copy = {k: v for k, v in r.items() if k not in ["visit_counts", "recon_guided", "prior_sample", "bin_errors"]}
                if "visit_counts" in r:
                    r_copy["visit_counts"] = r["visit_counts"].tolist()
                if "bin_errors" in r:
                    bin_errors = r["bin_errors"]
                    r_copy["bin_errors"] = {
                        "error_map": bin_errors["error_map"].tolist(),
                        "error_std": bin_errors["error_std"].tolist(),
                        "n_obs_per_bin": bin_errors["n_obs_per_bin"].tolist(),
                    }
                detailed_results[n_obs].append(r_copy)
        with open(detailed_path, "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save config
        config_path = args.output_dir / "eval_config.yaml"
        eval_config = {
            "mode": args.mode,
            "checkpoint": str(args.checkpoint),
            "data_path": str(args.data_path),
            "max_trips": args.max_trips,
            "obs_counts": args.obs_counts,
            "dt": args.dt,
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
        
        # Create plots
        plot_self_consistency_curves(
            aggregated=aggregated,
            output_path=args.output_dir / "self_consistency_curves.png",
        )
        for n_obs in args.obs_counts:
            if n_obs in all_results and len(all_results[n_obs]) > 0:
                plot_reconstruction_real_data(
                    results=all_results[n_obs],
                    grid=grid,
                    n_obs=n_obs,
                    output_path=args.output_dir / f"reconstructions_nobs_{n_obs}.png",
                    n_examples=min(3, len(all_results[n_obs])),
                )
                # Plot coverage heatmap for first result
                if "visit_counts" in all_results[n_obs][0]:
                    plot_coverage_heatmap(
                        visit_counts=all_results[n_obs][0]["visit_counts"],
                        grid=grid,
                        output_path=args.output_dir / f"coverage_nobs_{n_obs}.png",
                        title=f"Observation Coverage (n_obs={n_obs})",
                    )
                
                # Plot bin-wise errors
                if "bin_errors" in all_results[n_obs][0]:
                    bin_errors = all_results[n_obs][0]["bin_errors"]
                    plot_bin_wise_errors(
                        error_map=bin_errors["error_map"],
                        grid=grid,
                        output_path=args.output_dir / f"bin_errors_nobs_{n_obs}.png",
                        n_obs=n_obs,
                        title=f"Bin-wise Reconstruction Error",
                    )
    
    elif args.mode == "online":
        # Online evaluation on real data
        trips = load_trip_data(args.data_path, args.max_trips)
        
        online_results = evaluate_online_real_data(
            diffusion_prior=diffusion_prior,
            grid=grid,
            trips=trips,
            norm_mean=norm_mean,
            norm_std=norm_std,
            guidance_scale=args.guidance_scale,
            sigma_meas=args.sigma_meas,
            num_inference_steps=args.num_inference_steps,
            gradient_smoothing_sigma=args.gradient_smoothing_sigma,
            device=device,
            dt=args.dt,
            reconstruct_every_k=args.reconstruct_every_k,
            buffer_size=200,
            lambda_decay=0.3,
            w_min=0.2,
            seed=args.seed,
        )
        
        # Save results
        results_path = args.output_dir / "online_results.json"
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            "n_trips": online_results["n_trips"],
            "trips": [],
        }
        for trip_result in online_results["trips"]:
            # Serialize reconstruction history (exclude recon_map numpy arrays)
            serialized_history = []
            for snapshot in trip_result["reconstruction_history"]:
                snapshot_copy = {k: v for k, v in snapshot.items() if k != "recon_map"}
                serialized_history.append(snapshot_copy)
            
            trip_copy = {
                "trip_id": trip_result["trip_id"],
                "n_total_obs": trip_result["n_total_obs"],
                "reconstruction_history": serialized_history,
                "coverage": {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                            for k, v in trip_result["coverage"].items()},
                "trajectory_u": trip_result.get("trajectory_u", []),
                "trajectory_v": trip_result.get("trajectory_v", []),
            }
            serializable_results["trips"].append(trip_copy)
        
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        # Create animations for each trip
        print(f"\n{'='*60}")
        print("Creating animations...")
        print(f"{'='*60}")
        for trip_result in tqdm(online_results["trips"], desc="Creating animations"):
            trip_id = trip_result["trip_id"]
            # Sanitize trip_id for filename
            safe_trip_id = trip_id.replace("/", "_").replace(":", "_")
            animation_path = args.output_dir / f"animation_{safe_trip_id}.mp4"
            create_online_animation(
                trip_result=trip_result,
                grid=grid,
                output_path=animation_path,
                fps=10,
            )
        
        # Save config
        config_path = args.output_dir / "eval_config.yaml"
        eval_config = {
            "mode": args.mode,
            "checkpoint": str(args.checkpoint),
            "data_path": str(args.data_path),
            "max_trips": args.max_trips,
            "dt": args.dt,
            "reconstruct_every_k": args.reconstruct_every_k,
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
        
        print(f"\nOnline evaluation complete!")
        print(f"Processed {online_results['n_trips']} trips")
        print(f"Results saved to {results_path}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

