#!/usr/bin/env python3
"""
Online reconstruction simulation for dynamics map estimation.

Simulates a vehicle trip with random walk over (u, v) state space,
performing online map reconstruction at regular intervals and visualizing
the gradual improvement.
"""

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
from tqdm.auto import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.gt_map_generator import generate_map_from_params
from src.maps.buffer import Observation, ObservationBuffer
from src.maps.grid import MapGrid
from src.maps.sampler_diffusion import GuidedDiffusionSampler
from src.models.diffusion_prior import DiffusionPrior
from utils.dynamics import ExtendedPlantRandomization, sample_extended_params


@dataclass
class TripConfig:
    """Configuration for trip simulation."""
    trip_duration: int = 1000  # Total timesteps
    dt: float = 0.1  # Timestep duration (seconds)
    noise_std: float = 0.3  # Measurement noise std (m/s²)
    grade_type: str = "flat"  # "flat", "sinusoidal", "random_walk"
    grade_amplitude: float = 0.05  # For sinusoidal (±5%)
    grade_period: int = 500  # For sinusoidal
    grade_walk_std: float = 0.001  # For random walk
    
    # State space sampling
    sampling_mode: str = "random_walk"  # "random_walk" or "uniform"
    
    # Random walk parameters
    u_step_std: float = 0.1  # Actuation change std per step
    v_step_std: float = 0.5  # Speed change std per step (m/s)
    
    # Initial state
    u_init: float = 0.0
    v_init: float = 10.0


@dataclass
class ReconstructionConfig:
    """Configuration for online reconstruction."""
    reconstruct_every_k: int = 10  # Reconstruct every k timesteps
    buffer_size: int = 200  # Keep last N observations
    num_inference_steps: int = 50  # Diffusion steps
    guidance_scale: float = 100.0  # Updated after gradient normalization tuning
    sigma_meas: float = 0.5  # Measurement noise for energy function
    gradient_smoothing_sigma: float = 10.0  # Gradient smoothing
    
    # Buffer parameters (default to trajectory-optimized values)
    lambda_decay: float = 0.3
    w_min: float = 0.2
    
    # Patch filtering parameters
    patch_filtering_enabled: bool = False
    patch_size_u: int = 10
    patch_size_v: int = 10


class TripSimulator:
    """Simulates a vehicle trip through (u, v) state space."""
    
    def __init__(
        self,
        gt_map: np.ndarray,
        grid: MapGrid,
        config: TripConfig,
        seed: Optional[int] = None,
    ):
        self.gt_map = gt_map
        self.grid = grid
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Current state
        self.u = config.u_init
        self.v = config.v_init
        self.t = 0
        
        # Grade profile
        self.grade_profile = self._generate_grade_profile()
        
        # History
        self.u_history = [self.u]
        self.v_history = [self.v]
        self.grade_history = [self.grade_profile[0]]
        
    def _generate_grade_profile(self) -> np.ndarray:
        """Generate grade profile for the trip."""
        n_steps = self.config.trip_duration
        
        if self.config.grade_type == "flat":
            return np.zeros(n_steps)
        
        elif self.config.grade_type == "sinusoidal":
            t = np.arange(n_steps)
            return self.config.grade_amplitude * np.sin(
                2 * np.pi * t / self.config.grade_period
            )
        
        elif self.config.grade_type == "random_walk":
            grades = np.zeros(n_steps)
            grade = 0.0
            for i in range(n_steps):
                grade += self.rng.normal(0, self.config.grade_walk_std)
                grade = np.clip(grade, -0.08, 0.08)  # ±8% max
                grades[i] = grade
            return grades
        
        else:
            raise ValueError(f"Unknown grade_type: {self.config.grade_type}")
    
    def step(self) -> Tuple[float, float, float, float]:
        """
        Simulate one timestep.
        
        Returns:
            (u, v, a_meas, grade) - actuation, speed, measured acceleration, grade
        """
        # Get current indices
        i_u = self.grid.bin_u(self.u)
        i_v = self.grid.bin_v(self.v)
        
        # Query ground truth acceleration (dynamics-only, on flat ground)
        a_true = self.gt_map[i_u, i_v]
        
        # Add measurement noise
        a_noise = self.rng.normal(0, self.config.noise_std)
        
        # Get current grade
        grade = self.grade_profile[self.t]
        
        # Measured acceleration = true + noise + gravity
        g = 9.81
        a_meas = a_true + a_noise + g * np.sin(grade)
        
        # Update state based on sampling mode
        if self.config.sampling_mode == "uniform":
            # Uniform random sampling of state space
            i_u_new = self.rng.integers(0, self.grid.N_u)
            i_v_new = self.rng.integers(0, self.grid.N_v)
            u_new = self.grid.u_values[i_u_new]
            v_new = self.grid.v_values[i_v_new]
        else:  # random_walk
            # New actuation
            u_new = self.u + self.rng.normal(0, self.config.u_step_std)
            u_new = np.clip(u_new, -1.0, 1.0)
            
            # New speed (with physics-based update)
            v_new = self.v + a_true * self.config.dt + self.rng.normal(0, self.config.v_step_std)
            v_new = np.clip(v_new, 0.0, self.grid.v_max)
        
        # Store current observation before updating state
        obs = (self.u, self.v, a_meas, grade)
        
        # Update state
        self.u = u_new
        self.v = v_new
        self.t += 1
        
        # Record history
        self.u_history.append(self.u)
        self.v_history.append(self.v)
        self.grade_history.append(grade)
        
        return obs


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
            patch_filtering_enabled=config.patch_filtering_enabled,
            patch_size_u=config.patch_size_u,
            patch_size_v=config.patch_size_v,
        )
        
        # Reconstruction history
        self.reconstruction_history = []  # (timestep, map, mae, n_obs)
        self.current_reconstruction = None
        
        # Trajectory tracking
        self.trajectory_u = []  # All u values
        self.trajectory_v = []  # All v values
        self.visit_counts = np.zeros((grid.N_u, grid.N_v))  # Visit count heatmap
        
    def add_observation(
        self,
        u: float,
        v: float,
        a_meas: float,
        grade: float,
        t: float,
    ) -> None:
        """Add a new observation to the buffer."""
        # Convert to grid indices
        i_u = self.grid.bin_u(u)
        i_v = self.grid.bin_v(v)
        
        # Gravity compensation: a_dyn = a_meas - g*sin(grade)
        g = 9.81
        a_dyn = a_meas - g * np.sin(grade)
        
        # Add to buffer
        self.buffer.add(i_u, i_v, a_dyn, t)
    
    def add_trajectory_point(self, u: float, v: float) -> None:
        """Record a trajectory point."""
        self.trajectory_u.append(u)
        self.trajectory_v.append(v)
        
        # Update visit counts
        i_u = self.grid.bin_u(u)
        i_v = self.grid.bin_v(v)
        self.visit_counts[i_u, i_v] += 1
    
    def reconstruct(self, gt_map: np.ndarray, t_now: float) -> Tuple[np.ndarray, float, int]:
        """
        Perform guided reconstruction.
        
        Returns:
            (reconstructed_map, mae, n_obs)
        """
        # Get observations count
        observations = self.buffer.get_observations()
        n_obs = len(observations)
        
        if n_obs == 0:
            # No observations yet, sample from prior
            map_tensor = self.diffusion_prior.sample(
                batch_size=1,
                num_inference_steps=self.config.num_inference_steps,
                device=self.device,
            )
            recon_map = map_tensor.squeeze().cpu().numpy()
        else:
            # Guided reconstruction
            map_tensor = self.sampler.sample(
                obs_buffer=self.buffer,
                t_now=t_now,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
                device=self.device,
                x_init=None,  # No warm-start for now
            )
            recon_map = map_tensor.squeeze().cpu().numpy()
        
        # Denormalize
        recon_map = recon_map * self.norm_std + self.norm_mean
        
        # Compute MAE vs ground truth
        mae = np.mean(np.abs(recon_map - gt_map))
        
        # Store current reconstruction
        self.current_reconstruction = recon_map
        
        return recon_map, mae, n_obs
    
    def save_snapshot(
        self,
        timestep: int,
        recon_map: np.ndarray,
        mae: float,
        n_obs: int,
    ) -> None:
        """Save a reconstruction snapshot for animation."""
        self.reconstruction_history.append({
            "timestep": timestep,
            "map": recon_map.copy(),
            "mae": mae,
            "n_obs": n_obs,
        })


def create_animation(
    gt_map: np.ndarray,
    grid: MapGrid,
    tracker: ReconstructionTracker,
    trip_simulator: TripSimulator,
    output_path: Path,
    fps: int = 10,
) -> None:
    """
    Create 4-panel animation showing reconstruction progress.
    
    Panels:
    1. Ground truth (static)
    2. Current reconstruction (updating)
    3. Trajectory visualization (heatmap + current position)
    4. Observation density map
    """
    print(f"Creating animation with {len(tracker.reconstruction_history)} frames...")
    
    # Setup figure with 2x2 grid
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # GT
    ax2 = fig.add_subplot(gs[0, 1])  # Reconstruction
    ax3 = fig.add_subplot(gs[1, 0])  # Trajectory
    ax4 = fig.add_subplot(gs[1, 1])  # Observations
    
    # Common settings
    extent = [0, grid.v_max, -1, 1]  # [v_min, v_max, u_min, u_max]
    vmin, vmax = gt_map.min(), gt_map.max()
    
    # Panel 1: Ground truth (static)
    im1 = ax1.imshow(
        gt_map, origin="lower", extent=extent, aspect="auto",
        cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    ax1.set_xlabel("Speed (m/s)")
    ax1.set_ylabel("Actuation")
    ax1.set_title("Ground Truth Map")
    plt.colorbar(im1, ax=ax1, label="Acceleration (m/s²)")
    
    # Panel 2: Reconstruction (will update)
    im2 = ax2.imshow(
        np.zeros_like(gt_map), origin="lower", extent=extent, aspect="auto",
        cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    ax2.set_xlabel("Speed (m/s)")
    ax2.set_ylabel("Actuation")
    title2 = ax2.set_title("Reconstruction (t=0, n_obs=0, MAE=N/A)")
    plt.colorbar(im2, ax=ax2, label="Acceleration (m/s²)")
    
    # Panel 3: Trajectory (will update)
    # Start with empty heatmap
    im3 = ax3.imshow(
        np.zeros((grid.N_u, grid.N_v)), origin="lower", extent=extent,
        aspect="auto", cmap="YlOrRd", vmin=0, vmax=1
    )
    ax3.set_xlabel("Speed (m/s)")
    ax3.set_ylabel("Actuation")
    ax3.set_title("State Space Exploration")
    plt.colorbar(im3, ax=ax3, label="Visit Count")
    
    # Trail and current position markers
    trail_line, = ax3.plot([], [], 'b-', alpha=0.5, linewidth=1, label="Recent trail")
    current_pos, = ax3.plot([], [], 'ro', markersize=10, label="Current position")
    ax3.legend(loc="upper right")
    
    # Panel 4: Observation density (will update)
    scatter = ax4.scatter([], [], c='blue', alpha=0.3, s=10)
    ax4.set_xlabel("Speed (m/s)")
    ax4.set_ylabel("Actuation")
    ax4.set_title("Observation Locations")
    ax4.set_xlim(0, grid.v_max)
    ax4.set_ylim(-1, 1)
    ax4.grid(True, alpha=0.3)
    
    # Animation update function
    def update(frame_idx):
        if frame_idx >= len(tracker.reconstruction_history):
            return im2, title2, im3, trail_line, current_pos, scatter
        
        snapshot = tracker.reconstruction_history[frame_idx]
        t = snapshot["timestep"]
        recon_map = snapshot["map"]
        mae = snapshot["mae"]
        n_obs = snapshot["n_obs"]
        
        # Update Panel 2: Reconstruction
        im2.set_data(recon_map)
        title2.set_text(f"Reconstruction (t={t}, n_obs={n_obs}, MAE={mae:.3f} m/s²)")
        
        # Update Panel 3: Trajectory
        # Get trajectory up to this point
        traj_end_idx = min(t + 1, len(trip_simulator.u_history))
        u_traj = trip_simulator.u_history[:traj_end_idx]
        v_traj = trip_simulator.v_history[:traj_end_idx]
        
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
        
        im3.set_data(visit_counts_norm)
        im3.set_clim(0, 1)
        
        # Update trail (last 100 steps)
        trail_start = max(0, traj_end_idx - 100)
        trail_u = u_traj[trail_start:traj_end_idx]
        trail_v = v_traj[trail_start:traj_end_idx]
        trail_line.set_data(trail_v, trail_u)
        
        # Update current position
        if traj_end_idx > 0:
            current_pos.set_data([v_traj[-1]], [u_traj[-1]])
        
        # Update Panel 4: Observations
        observations = tracker.buffer.get_observations()
        if len(observations) > 0:
            obs_v = [grid.center_v(obs.i_v) for obs in observations]
            obs_u = [grid.center_u(obs.i_u) for obs in observations]
            scatter.set_offsets(np.c_[obs_v, obs_u])
        
        return im2, title2, im3, trail_line, current_pos, scatter
    
    # Create animation
    n_frames = len(tracker.reconstruction_history)
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000//fps, blit=False
    )
    
    # Save (try MP4, fall back to GIF if ffmpeg unavailable)
    print(f"Saving animation to {output_path}...")
    dpi = 150  # Higher DPI for better visual quality
    try:
        anim.save(str(output_path), writer='ffmpeg', fps=fps, dpi=dpi)
    except (ValueError, RuntimeError):
        # ffmpeg not available, save as GIF
        gif_path = output_path.with_suffix('.gif')
        print(f"ffmpeg unavailable, saving as GIF: {gif_path}")
        anim.save(str(gif_path), writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)
    print("Animation saved successfully!")


def create_metrics_plot(
    tracker: ReconstructionTracker,
    output_path: Path,
) -> None:
    """Create static plots showing convergence metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    timesteps = [s["timestep"] for s in tracker.reconstruction_history]
    maes = [s["mae"] for s in tracker.reconstruction_history]
    n_obs_list = [s["n_obs"] for s in tracker.reconstruction_history]
    
    # Coverage (% of grid with at least one visit)
    coverage = [(tracker.visit_counts[:t+1] > 0).sum() / tracker.visit_counts.size * 100
                for t in range(len(timesteps))]
    
    # Plot 1: MAE vs timestep
    axes[0, 0].plot(timesteps, maes, 'b-', linewidth=2)
    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].set_ylabel("MAE (m/s²)")
    axes[0, 0].set_title("Reconstruction Error vs Time")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: MAE vs n_observations
    axes[0, 1].plot(n_obs_list, maes, 'g-', linewidth=2)
    axes[0, 1].set_xlabel("Number of Observations")
    axes[0, 1].set_ylabel("MAE (m/s²)")
    axes[0, 1].set_title("Reconstruction Error vs Observations")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Coverage vs timestep
    axes[1, 0].plot(timesteps, coverage, 'r-', linewidth=2)
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("Coverage (%)")
    axes[1, 0].set_title("State Space Exploration")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Final visit heatmap
    im = axes[1, 1].imshow(
        tracker.visit_counts, origin="lower",
        extent=[0, 30, -1, 1], aspect="auto", cmap="YlOrRd"
    )
    axes[1, 1].set_xlabel("Speed (m/s)")
    axes[1, 1].set_ylabel("Actuation")
    axes[1, 1].set_title("Final Visit Distribution")
    plt.colorbar(im, ax=axes[1, 1], label="Visit Count")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics plot saved to {output_path}")


def run_scenario(
    scenario_name: str,
    trip_config: TripConfig,
    recon_config: ReconstructionConfig,
    diffusion_prior: DiffusionPrior,
    gt_map: np.ndarray,
    grid: MapGrid,
    norm_mean: float,
    norm_std: float,
    output_dir: Path,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Run a single simulation scenario."""
    print(f"\n{'='*60}")
    print(f"Running Scenario: {scenario_name}")
    print(f"{'='*60}")
    
    # Create output directory
    scenario_dir = output_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize simulator and tracker
    trip_simulator = TripSimulator(gt_map, grid, trip_config, seed=seed)
    tracker = ReconstructionTracker(
        diffusion_prior, grid, recon_config, norm_mean, norm_std, device
    )
    
    # Run simulation
    print(f"Simulating trip for {trip_config.trip_duration} timesteps...")
    for t in tqdm(range(trip_config.trip_duration), desc="Trip simulation"):
        # Step simulator
        u, v, a_meas, grade = trip_simulator.step()
        
        # Add observation and trajectory point
        tracker.add_observation(u, v, a_meas, grade, t * trip_config.dt)
        tracker.add_trajectory_point(u, v)
        
        # Reconstruct every k steps
        if t % recon_config.reconstruct_every_k == 0 and t > 0:
            t_now = t * trip_config.dt
            recon_map, mae, n_obs = tracker.reconstruct(gt_map, t_now)
            tracker.save_snapshot(t, recon_map, mae, n_obs)
            
            if t % 100 == 0:
                print(f"  t={t}: n_obs={n_obs}, MAE={mae:.4f} m/s²")
    
    # Final reconstruction
    t_now = trip_config.trip_duration * trip_config.dt
    recon_map, mae, n_obs = tracker.reconstruct(gt_map, t_now)
    tracker.save_snapshot(trip_config.trip_duration, recon_map, mae, n_obs)
    print(f"Final: n_obs={n_obs}, MAE={mae:.4f} m/s²")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_animation(
        gt_map, grid, tracker, trip_simulator,
        scenario_dir / "animation.mp4", fps=10
    )
    create_metrics_plot(tracker, scenario_dir / "metrics.png")
    
    # Save metrics
    metrics = {
        "scenario": scenario_name,
        "trip_config": vars(trip_config),
        "recon_config": vars(recon_config),
        "final_mae": float(mae),
        "final_n_obs": int(n_obs),
        "reconstruction_history": [
            {
                "timestep": s["timestep"],
                "mae": float(s["mae"]),
                "n_obs": int(s["n_obs"]),
            }
            for s in tracker.reconstruction_history
        ]
    }
    
    with open(scenario_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {scenario_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Online reconstruction simulation"
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
        default="evaluation/online_simulation",
        help="Output directory",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["baseline", "low_noise", "varying_grade", "high_noise"],
        choices=["baseline", "low_noise", "varying_grade", "high_noise"],
        help="Scenarios to run",
    )
    parser.add_argument(
        "--trip-duration",
        type=int,
        default=1000,
        help="Trip duration in timesteps",
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
        "--lambda-decay",
        type=float,
        default=0.3,
        help="Time decay rate for observation weights (0.0 = no decay)",
    )
    parser.add_argument(
        "--w-min",
        type=float,
        default=0.2,
        help="Minimum weight floor for observations (1.0 = all equal weight)",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="random_walk",
        choices=["random_walk", "uniform"],
        help="State space sampling mode: 'random_walk' (correlated) or 'uniform' (IID)",
    )
    parser.add_argument(
        "--patch-filtering-enabled",
        action="store_true",
        help="Enable patch-based uniform sampling for observations",
    )
    parser.add_argument(
        "--patch-size-u",
        type=int,
        default=10,
        help="Patch size in u direction (default: 10)",
    )
    parser.add_argument(
        "--patch-size-v",
        type=int,
        default=10,
        help="Patch size in v direction (default: 10)",
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load diffusion prior
    print("Loading diffusion prior...")
    diffusion_prior = DiffusionPrior.load(args.checkpoint, device=device)
    
    # Configure DDPM scheduler for high-quality reconstruction
    from diffusers import DDPMScheduler
    diffusion_prior.inference_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        clip_sample=False,
    )
    diffusion_prior.eval()
    print("Configured DDPM scheduler for high-quality reconstruction")
    
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
    
    # Create grid
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    
    # Generate GT map
    print("Generating ground truth map...")
    rng = np.random.default_rng(args.seed)
    randomization = ExtendedPlantRandomization()
    vehicle_params = sample_extended_params(rng, randomization)
    gt_map = generate_map_from_params(
        vehicle_params, N_u=100, N_v=100, v_max=30.0
    )
    
    # Define scenarios
    scenario_configs = {
        "baseline": TripConfig(
            trip_duration=args.trip_duration,
            noise_std=0.0,
            grade_type="flat",
            sampling_mode=args.sampling_mode,
        ),
        "low_noise": TripConfig(
            trip_duration=args.trip_duration,
            noise_std=0.3,
            grade_type="flat",
            sampling_mode=args.sampling_mode,
        ),
        "varying_grade": TripConfig(
            trip_duration=args.trip_duration,
            noise_std=0.3,
            grade_type="sinusoidal",
            grade_amplitude=0.05,
            grade_period=500,
            sampling_mode=args.sampling_mode,
        ),
        "high_noise": TripConfig(
            trip_duration=args.trip_duration,
            noise_std=0.5,
            grade_type="random_walk",
            sampling_mode=args.sampling_mode,
        ),
    }
    
    # Reconstruction config (same for all scenarios, using command-line args)
    recon_config = ReconstructionConfig(
        reconstruct_every_k=args.reconstruct_every_k,
        buffer_size=args.buffer_size,
        num_inference_steps=50,
        guidance_scale=100.0,  # Optimal value after gradient normalization tuning
        sigma_meas=0.5,
        gradient_smoothing_sigma=10.0,
        lambda_decay=args.lambda_decay,  # Use command-line arg
        w_min=args.w_min,  # Use command-line arg
        patch_filtering_enabled=args.patch_filtering_enabled,
        patch_size_u=args.patch_size_u,
        patch_size_v=args.patch_size_v,
    )
    
    # Run scenarios
    results = {}
    for scenario_name in args.scenarios:
        trip_config = scenario_configs[scenario_name]
        metrics = run_scenario(
            scenario_name,
            trip_config,
            recon_config,
            diffusion_prior,
            gt_map,
            grid,
            norm_mean,
            norm_std,
            output_dir,
            device,
            args.seed,
        )
        results[scenario_name] = metrics
    
    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("All scenarios completed!")
    print(f"Results saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

