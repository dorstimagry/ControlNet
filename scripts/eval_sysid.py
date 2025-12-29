#!/usr/bin/env python3
"""Evaluate SysID quality and vehicle-conditioned SAC performance.

This script evaluates:
1. SysID quality: prediction accuracy on held-out vehicles
2. Control performance: SAC+z vs baseline SAC
3. Ablations: z=0, different horizons, etc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from env.longitudinal_env import LongitudinalEnv, LongitudinalEnvConfig
from utils.dynamics import RandomizationConfig
from training.train_sac import (
    SACTrainer,
    ConfigBundle,
    TrainingParams,
    SysIDParams,
    OutputConfig,
    load_config,
    build_config_bundle
)
from src.sysid import (
    ContextEncoder,
    DynamicsPredictor,
    FeatureBuilder,
    RunningNorm
)


def evaluate_sysid_prediction(
    checkpoint_path: Path,
    config: ConfigBundle,
    num_episodes: int = 10,
    horizons: List[int] = [10, 20, 40],
    device: torch.device | None = None
) -> Dict[str, float]:
    """Evaluate SysID prediction accuracy on held-out vehicles.
    
    Args:
        checkpoint_path: Path to SAC+SysID checkpoint
        config: Configuration bundle
        num_episodes: Number of episodes to evaluate
        horizons: Rollout horizons to test
        device: Device for computation
    
    Returns:
        Dictionary of metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if not checkpoint.get("meta", {}).get("sysid_enabled", False):
        print("[warning] Checkpoint does not have SysID enabled")
        return {}
    
    # Create encoder and predictor
    encoder = ContextEncoder(
        input_dim=4,
        hidden_dim=config.sysid.gru_hidden,
        z_dim=config.sysid.dz
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    
    predictor = DynamicsPredictor(
        z_dim=config.sysid.dz,
        hidden_dim=config.sysid.predictor_hidden,
        num_layers=config.sysid.predictor_layers
    ).to(device)
    predictor.load_state_dict(checkpoint["predictor"])
    predictor.eval()
    
    # Load normalizers
    encoder_norm = RunningNorm(dim=4, eps=config.sysid.norm_eps, clip=config.sysid.norm_clip).to(device)
    predictor_v_norm = RunningNorm(dim=1, eps=config.sysid.norm_eps, clip=config.sysid.norm_clip).to(device)
    predictor_u_norm = RunningNorm(dim=1, eps=config.sysid.norm_eps, clip=config.sysid.norm_clip).to(device)
    
    encoder_norm.load_state_dict(checkpoint["encoder_norm"])
    predictor_v_norm.load_state_dict(checkpoint["predictor_v_norm"])
    predictor_u_norm.load_state_dict(checkpoint["predictor_u_norm"])
    
    # Create evaluation environment with different seed (held-out vehicles)
    env = LongitudinalEnv(
        config.env,
        randomization=RandomizationConfig(),
        generator_config=config.generator_config,
        seed=config.seed + 1000  # Held-out seed
    )
    
    feature_builder = FeatureBuilder(dt=config.env.dt)
    
    # Collect prediction errors for each horizon
    errors = {h: [] for h in horizons}
    errors_ablation_z0 = {h: [] for h in horizons}  # Ablation with z=0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        feature_builder.reset()
        h = encoder.reset(batch_size=1, device=device).squeeze(0)
        done = False
        episode_speeds = []
        episode_actions = []
        
        # Collect episode data
        while not done:
            v_t = obs[0]
            # Random action for exploration
            u_t = float(env.action_space.sample()[0])
            
            episode_speeds.append(v_t)
            episode_actions.append(u_t)
            
            obs, _, terminated, truncated, _ = env.step(u_t)
            done = terminated or truncated
        
        # Evaluate prediction at multiple points in the episode
        episode_speeds = np.array(episode_speeds)
        episode_actions = np.array(episode_actions)
        
        max_horizon = max(horizons)
        burn_in = config.sysid.burn_in
        
        for anchor in range(burn_in, len(episode_speeds) - max_horizon, 10):
            # Burn-in phase
            v_burnin = episode_speeds[anchor - burn_in:anchor + 1]
            u_burnin = episode_actions[anchor - burn_in:anchor]
            
            # Build features and encode
            features = feature_builder.build_features_batch(
                torch.from_numpy(v_burnin).unsqueeze(0).float().to(device),
                torch.from_numpy(u_burnin).unsqueeze(0).float().to(device)
            )
            
            features_norm = encoder_norm(features.reshape(-1, 4), update_stats=False).reshape(1, -1, 4)
            
            with torch.no_grad():
                _, z_seq, _ = encoder(features_norm)
                z_t = z_seq[:, -1, :]
            
            # Rollout for each horizon
            for H in horizons:
                v_hat = episode_speeds[anchor]
                v_true = episode_speeds[anchor + 1:anchor + H + 1]
                u_rollout = episode_actions[anchor:anchor + H]
                
                pred_errors = []
                for k in range(H):
                    v_hat_tensor = torch.tensor([[v_hat]], device=device, dtype=torch.float32)
                    u_k_tensor = torch.tensor([[u_rollout[k]]], device=device, dtype=torch.float32)
                    
                    v_hat_norm = predictor_v_norm(v_hat_tensor, update_stats=False)
                    u_k_norm = predictor_u_norm(u_k_tensor, update_stats=False)
                    
                    with torch.no_grad():
                        dv_hat = predictor(v_hat_norm, u_k_norm, z_t).squeeze().item()
                    
                    v_hat = v_hat + dv_hat
                    pred_errors.append((v_hat - v_true[k]) ** 2)
                
                errors[H].append(np.mean(pred_errors))
                
                # Ablation: z=0
                v_hat_z0 = episode_speeds[anchor]
                z_zero = torch.zeros_like(z_t)
                pred_errors_z0 = []
                for k in range(H):
                    v_hat_tensor = torch.tensor([[v_hat_z0]], device=device, dtype=torch.float32)
                    u_k_tensor = torch.tensor([[u_rollout[k]]], device=device, dtype=torch.float32)
                    
                    v_hat_norm = predictor_v_norm(v_hat_tensor, update_stats=False)
                    u_k_norm = predictor_u_norm(u_k_tensor, update_stats=False)
                    
                    with torch.no_grad():
                        dv_hat = predictor(v_hat_norm, u_k_norm, z_zero).squeeze().item()
                    
                    v_hat_z0 = v_hat_z0 + dv_hat
                    pred_errors_z0.append((v_hat_z0 - v_true[k]) ** 2)
                
                errors_ablation_z0[H].append(np.mean(pred_errors_z0))
    
    # Compute metrics
    metrics = {}
    for H in horizons:
        rmse = np.sqrt(np.mean(errors[H]))
        rmse_z0 = np.sqrt(np.mean(errors_ablation_z0[H]))
        metrics[f"sysid_eval/rmse_h{H}"] = rmse
        metrics[f"sysid_eval/rmse_h{H}_z0"] = rmse_z0
        metrics[f"sysid_eval/improvement_h{H}"] = (rmse_z0 - rmse) / rmse_z0 * 100
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SysID and vehicle-conditioned SAC")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=Path, default=None, help="Path to config (if not in checkpoint)")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 20, 40], help="Rollout horizons to test")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file for metrics")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        raw = load_config(args.config)
        config = build_config_bundle(raw, argparse.Namespace())
    else:
        # Try to load from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "config" not in checkpoint:
            raise ValueError("No config in checkpoint and --config not provided")
        # Reconstruct config from checkpoint
        config = ConfigBundle(
            seed=checkpoint["config"]["seed"],
            env=LongitudinalEnvConfig(**checkpoint["config"]["env"]),
            training=TrainingParams(**checkpoint["config"]["training"]),
            sysid=SysIDParams(**checkpoint["config"].get("sysid", {})),
            output=OutputConfig(),
            reference_dataset=None,
            generator_config=None
        )
    
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"[eval] Evaluating SysID on {args.num_episodes} episodes")
    print(f"[eval] Horizons: {args.horizons}")
    print(f"[eval] Device: {device}")
    
    # Evaluate SysID prediction
    metrics = evaluate_sysid_prediction(
        checkpoint_path=args.checkpoint,
        config=config,
        num_episodes=args.num_episodes,
        horizons=args.horizons,
        device=device
    )
    
    # Print metrics
    print("\n" + "="*60)
    print("SysID Evaluation Metrics")
    print("="*60)
    for key, value in sorted(metrics.items()):
        print(f"{key:40s}: {value:10.4f}")
    print("="*60)
    
    # Save metrics
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[eval] Metrics saved to {args.output}")


if __name__ == "__main__":
    main()

