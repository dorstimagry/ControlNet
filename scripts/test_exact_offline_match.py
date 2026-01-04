"""
Test online setting with EXACT offline parameters to match offline performance.

This script identifies and eliminates ALL differences between online and offline evaluation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.gt_map_generator import generate_map_from_params
from src.maps.buffer import Observation, ObservationBuffer
from src.maps.grid import MapGrid
from src.maps.sampler_diffusion import GuidedDiffusionSampler
from src.models.diffusion_prior import DiffusionPrior
from utils.dynamics import ExtendedPlantRandomization, sample_extended_params


def sample_observations_uniform(
    gt_map: np.ndarray,
    grid: MapGrid,
    n_obs: int,
    sigma_noise: float,
    seed: int,
) -> list:
    """Sample observations exactly like offline evaluation."""
    rng = np.random.RandomState(seed)
    
    observations = []
    for idx in range(n_obs):
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
            timestamp=float(idx),  # Use index as timestamp
        )
        observations.append(obs)
    
    return observations


def reconstruct_exactly_like_offline(
    diffusion_prior: DiffusionPrior,
    gt_map: np.ndarray,
    observations: list,
    norm_mean: float,
    norm_std: float,
    guidance_scale: float,
    sigma_meas: float,
    num_inference_steps: int,
    gradient_smoothing_sigma: float,
    device: torch.device,
    seed: int,
) -> tuple:
    """Reconstruct exactly like offline evaluation."""
    
    # Create observation buffer with OFFLINE parameters
    obs_buffer = ObservationBuffer(
        capacity=len(observations) * 2,  # Extra capacity
        lambda_decay=0.0,  # NO DECAY - this is critical!
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
    
    # Compute metrics
    mae = float(np.mean(np.abs(recon_guided - gt_map)))
    mse = float(np.mean((recon_guided - gt_map) ** 2))
    
    return mae, mse, recon_guided


def main():
    parser = argparse.ArgumentParser(
        description="Test online setting with exact offline parameters"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default="training/diffusion_prior/best.pt",
        help="Path to diffusion prior checkpoint",
    )
    parser.add_argument(
        "--n-obs",
        type=int,
        default=200,
        help="Number of observations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("EXACT OFFLINE PARAMETER MATCH TEST")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of observations: {args.n_obs}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    # Load model
    print("Loading diffusion prior...")
    diffusion_prior = DiffusionPrior.load(args.checkpoint, device=device)
    diffusion_prior.eval()
    
    # Load normalization stats
    norm_stats_path = Path("data/maps/norm_stats.json")
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    norm_mean = norm_stats["mean"]
    norm_std = norm_stats["std"]
    print(f"Loaded normalization: mean={norm_mean:.3f}, std={norm_std:.3f}")
    
    # Create grid
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    
    # Generate GT map EXACTLY like offline evaluation (first map, i=0)
    print("Generating ground truth map (exactly like offline evaluation)...")
    map_index = 0  # First test map
    rng = np.random.default_rng(args.seed + map_index)  # Offline uses seed + i
    randomization = ExtendedPlantRandomization()  # Default randomization
    vehicle_params = sample_extended_params(rng, randomization)
    gt_map = generate_map_from_params(
        params=vehicle_params,
        N_u=grid.N_u,
        N_v=grid.N_v,
        v_max=grid.v_max,
        dt_eval=0.02,  # Offline default
        n_eval_steps=5,  # Offline default
        smoothing_sigma=None,  # Offline default
    )
    print(f"GT map range: [{gt_map.min():.2f}, {gt_map.max():.2f}] m/s²\n")
    
    # OFFLINE PARAMETERS (from eval_config.yaml and evaluate_diffusion.py)
    offline_params = {
        'guidance_scale': 1.0,
        'sigma_meas': 0.5,
        'num_inference_steps': 50,
        'gradient_smoothing_sigma': 10.0,
        'sigma_noise': 0.5,  # CRITICAL: Uses sigma_meas as observation noise (line 653 of evaluate_diffusion.py)
    }
    
    print("Sampling observations (EXACTLY like offline evaluation)...")
    # Offline uses: seed = args.seed + i + n_obs * 1000
    obs_seed = args.seed + map_index + args.n_obs * 1000
    print(f"  Observation seed: {obs_seed} (offline formula: seed + map_idx + n_obs*1000)")
    observations = sample_observations_uniform(
        gt_map=gt_map,
        grid=grid,
        n_obs=args.n_obs,
        sigma_noise=offline_params['sigma_noise'],
        seed=obs_seed,  # Use offline seed formula
    )
    
    print(f"Reconstructing with EXACT offline parameters...")
    print(f"  - lambda_decay: 0.0 (NO DECAY)")
    print(f"  - w_min: 1.0")
    print(f"  - guidance_scale: {offline_params['guidance_scale']}")
    print(f"  - sigma_meas: {offline_params['sigma_meas']}")
    print(f"  - num_inference_steps: {offline_params['num_inference_steps']}")
    print(f"  - gradient_smoothing_sigma: {offline_params['gradient_smoothing_sigma']}")
    print(f"  - sigma_noise: {offline_params['sigma_noise']}")
    print(f"  - reconstruction_seed: {obs_seed} (same as observation seed)\n")
    
    mae, mse, recon = reconstruct_exactly_like_offline(
        diffusion_prior=diffusion_prior,
        gt_map=gt_map,
        observations=observations,
        norm_mean=norm_mean,
        norm_std=norm_std,
        guidance_scale=offline_params['guidance_scale'],
        sigma_meas=offline_params['sigma_meas'],
        num_inference_steps=offline_params['num_inference_steps'],
        gradient_smoothing_sigma=offline_params['gradient_smoothing_sigma'],
        device=device,
        seed=obs_seed,  # Offline uses same seed for reconstruction as observations
    )
    
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"MAE: {mae:.4f} m/s²")
    print(f"MSE: {mse:.4f} (m/s²)²")
    print("="*70)
    print("\nExpected offline MAE: ~0.08-0.13 m/s² (from evaluation/final_fixed)")
    print(f"Achieved MAE: {mae:.4f} m/s²")
    
    if mae < 0.15:
        print("\n✅ SUCCESS! Online matches offline performance!")
    elif mae < 0.5:
        print(f"\n⚠️  Close but not exact match. Difference: {mae - 0.10:.3f} m/s²")
    else:
        print(f"\n❌ FAILED! Large discrepancy: {mae - 0.10:.3f} m/s²")
    print("="*70)


if __name__ == "__main__":
    main()

