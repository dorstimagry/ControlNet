#!/usr/bin/env python3
"""Pretrain SysID encoder and predictor on vehicle dynamics.

This script trains the SysID module in isolation before SAC training.
The encoder learns to capture vehicle-specific dynamics through multi-step
rollout prediction using random exploration trajectories.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from env.longitudinal_env import LongitudinalEnv, LongitudinalEnvConfig
from utils.dynamics import RandomizationConfig
from training.train_sac import (
    ConfigBundle,
    SysIDParams,
    build_config_bundle,
    load_config,
)
from src.sysid import (
    ContextEncoder,
    DynamicsPredictor,
    FeatureBuilder,
    RunningNorm,
    SysIDTrainer,
    sample_sequences,
)


class SysIDReplayBuffer:
    """Simplified replay buffer for SysID pretraining (stores only speed and action)."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.speed_buf = np.zeros((capacity,), dtype=np.float32)
        self.action_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.episode_id_buf = np.zeros((capacity,), dtype=np.int32)
        self.step_in_episode_buf = np.zeros((capacity,), dtype=np.int32)
        self.ptr = 0
        self.size = 0
        self._current_episode_id = 0
        self._current_episode_step = 0
    
    def add(self, speed: float, action: float, done: bool) -> None:
        self.speed_buf[self.ptr] = speed
        self.action_buf[self.ptr, 0] = action
        self.episode_id_buf[self.ptr] = self._current_episode_id
        self.step_in_episode_buf[self.ptr] = self._current_episode_step
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._current_episode_step += 1
        
        if done:
            self._current_episode_id += 1
            self._current_episode_step = 0
    
    def sample_sequences(self, batch_size: int, burn_in: int, horizon: int, rng=None):
        """Sample sequences for SysID training."""
        return sample_sequences(
            speed_buf=self.speed_buf,
            action_buf=self.action_buf,
            episode_id_buf=self.episode_id_buf,
            step_in_episode_buf=self.step_in_episode_buf,
            buffer_size=self.size,
            batch_size=batch_size,
            burn_in=burn_in,
            horizon=horizon,
            rng=rng
        )


def collect_exploration_data(
    env: LongitudinalEnv,
    buffer: SysIDReplayBuffer,
    num_steps: int,
    rng: np.random.Generator
) -> Dict[str, float]:
    """Collect exploration data with random actions.
    
    Args:
        env: Environment
        buffer: Replay buffer
        num_steps: Number of steps to collect
        rng: Random number generator
    
    Returns:
        Statistics about data collection
    """
    obs, _ = env.reset()
    episode_lengths = []
    episode_count = 0
    current_episode_length = 0
    
    for _ in range(num_steps):
        # Random action
        action = float(env.action_space.sample()[0])
        speed = obs[0]
        
        # Step environment
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        current_episode_length += 1
        
        # Store transition
        buffer.add(speed, action, done)
        
        if done:
            obs, _ = env.reset()
            episode_lengths.append(current_episode_length)
            episode_count += 1
            current_episode_length = 0
    
    return {
        "episodes_collected": episode_count,
        "avg_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
        "total_steps": num_steps,
    }


def pretrain_sysid(
    config: ConfigBundle,
    num_pretrain_steps: int,
    eval_interval: int = 1000,
    checkpoint_interval: int = 10000,
    output_dir: Path | None = None,
    device: torch.device | None = None,
    min_buffer_size: int = 10000,
) -> None:
    """Pretrain SysID encoder and predictor with on-the-fly data generation.
    
    Args:
        config: Configuration bundle
        num_pretrain_steps: Number of pretraining steps
        eval_interval: Steps between evaluation
        checkpoint_interval: Steps between checkpoints
        output_dir: Output directory for checkpoints
        device: Device for training
        min_buffer_size: Minimum buffer size before starting training
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if output_dir is None:
        output_dir = Path("training/sysid_pretrained")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[pretrain] Device: {device}")
    print(f"[pretrain] Output: {output_dir}")
    print(f"[pretrain] Pretraining for {num_pretrain_steps} steps")
    print(f"[pretrain] Generating episodes on-the-fly (min buffer: {min_buffer_size})")
    
    # Create environment for data collection
    generator_config = config.generator_config or {}
    
    if config.fitted_params_path:
        from utils.dynamics import ExtendedPlantRandomization
        from fitting.randomization_config import CenteredRandomizationConfig
        from fitting import FittedVehicleParams
        
        print(f"[pretrain] Using fitted params from {config.fitted_params_path}")
        fitted = FittedVehicleParams.load(Path(config.fitted_params_path))
        centered_config = CenteredRandomizationConfig.from_fitted_params(
            fitted, spread_pct=config.fitted_spread_pct
        )
        generator_config.update(centered_config.to_extended_randomization_dict())
    
    env = LongitudinalEnv(
        config.env,
        randomization=RandomizationConfig(),
        generator_config=generator_config,
        seed=config.seed,
    )
    env.reset(seed=config.seed)
    
    # Create replay buffer (smaller, refilled on-the-fly)
    buffer_size = max(config.training.replay_size, 100_000)
    buffer = SysIDReplayBuffer(capacity=buffer_size)
    
    # Create SysID components
    encoder = ContextEncoder(
        input_dim=4,
        hidden_dim=config.sysid.gru_hidden,
        z_dim=config.sysid.dz
    ).to(device)
    
    predictor = DynamicsPredictor(
        z_dim=config.sysid.dz,
        hidden_dim=config.sysid.predictor_hidden,
        num_layers=config.sysid.predictor_layers
    ).to(device)
    
    # Create normalizers
    encoder_norm = RunningNorm(
        dim=4,
        eps=config.sysid.norm_eps,
        clip=config.sysid.norm_clip
    ).to(device)
    
    predictor_v_norm = RunningNorm(
        dim=1,
        eps=config.sysid.norm_eps,
        clip=config.sysid.norm_clip
    ).to(device)
    
    predictor_u_norm = RunningNorm(
        dim=1,
        eps=config.sysid.norm_eps,
        clip=config.sysid.norm_clip
    ).to(device)
    
    # Create SysID trainer
    sysid_trainer = SysIDTrainer(
        encoder=encoder,
        predictor=predictor,
        encoder_norm=encoder_norm,
        predictor_v_norm=predictor_v_norm,
        predictor_u_norm=predictor_u_norm,
        learning_rate=config.sysid.learning_rate,
        lambda_slow=config.sysid.lambda_slow,
        lambda_z=config.sysid.lambda_z,
        dt=config.env.dt,
        device=device
    )
    
    # Collect initial data to fill buffer
    print(f"\n[pretrain] Collecting initial {min_buffer_size} exploration steps...")
    rng = np.random.default_rng(config.seed)
    stats = collect_exploration_data(env, buffer, min_buffer_size, rng)
    print(f"[pretrain] Collected {stats['episodes_collected']} episodes, "
          f"avg length {stats['avg_episode_length']:.1f}")
    
    # Training loop with on-the-fly data collection
    print(f"\n[pretrain] Training SysID for {num_pretrain_steps} steps (collecting data on-the-fly)...")
    
    best_loss = float('inf')
    metrics_history = []
    start_time = time.time()
    last_log = start_time
    
    # Initialize environment state for on-the-fly collection
    obs, _ = env.reset()
    episode_steps = 0
    collect_every = 10  # Collect new data every N training steps
    
    progress = tqdm(range(num_pretrain_steps), desc="SysID Pretraining")
    
    for step in progress:
        # Periodically collect more data (on-the-fly)
        if step % collect_every == 0:
            # Collect a few steps
            for _ in range(collect_every * 2):  # Collect 2x training steps
                action = float(env.action_space.sample()[0])
                speed = obs[0]
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.add(speed, action, done)
                episode_steps += 1
                
                if done:
                    obs, _ = env.reset()
                    episode_steps = 0
        # Sample sequences
        batch = buffer.sample_sequences(
            batch_size=config.training.batch_size,
            burn_in=config.sysid.burn_in,
            horizon=config.sysid.horizon,
            rng=rng
        )
        
        if batch is None:
            print(f"[pretrain] Warning: Could not sample sequences at step {step}")
            continue
        
        # Train step
        metrics = sysid_trainer.train_step(batch)
        
        # Log metrics
        if step % eval_interval == 0:
            metrics["step"] = step
            metrics["time_elapsed"] = time.time() - start_time
            metrics_history.append(metrics)
            
            pred_loss = metrics["sysid/pred_loss"]
            slow_loss = metrics["sysid/slow_loss"]
            z_norm = metrics["sysid/z_norm"]
            
            progress.set_postfix(
                pred_loss=f"{pred_loss:.4f}",
                z_norm=f"{z_norm:.3f}",
                refresh=False
            )
            
            print(f"\n[pretrain] Step {step}: "
                  f"pred_loss={pred_loss:.4f}, "
                  f"slow_loss={slow_loss:.6f}, "
                  f"z_norm={z_norm:.3f}")
            
            # Save best model
            if pred_loss < best_loss:
                best_loss = pred_loss
                checkpoint = {
                    "step": step,
                    "encoder": encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "encoder_norm": encoder_norm.state_dict(),
                    "predictor_v_norm": predictor_v_norm.state_dict(),
                    "predictor_u_norm": predictor_u_norm.state_dict(),
                    "sysid_optimizer": sysid_trainer.optimizer.state_dict(),
                    "config": {
                        "sysid": dataclasses.asdict(config.sysid),
                        "env": dataclasses.asdict(config.env),
                    },
                    "metrics": {
                        "pred_loss": pred_loss,
                        "best_loss": best_loss,
                    }
                }
                torch.save(checkpoint, output_dir / "sysid_best.pt")
                print(f"[pretrain] Saved best checkpoint (loss={best_loss:.4f})")
        
        # Checkpoint
        if step > 0 and step % checkpoint_interval == 0:
            checkpoint = {
                "step": step,
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "encoder_norm": encoder_norm.state_dict(),
                "predictor_v_norm": predictor_v_norm.state_dict(),
                "predictor_u_norm": predictor_u_norm.state_dict(),
                "sysid_optimizer": sysid_trainer.optimizer.state_dict(),
                "config": {
                    "sysid": dataclasses.asdict(config.sysid),
                    "env": dataclasses.asdict(config.env),
                },
                "metrics": metrics,
            }
            torch.save(checkpoint, output_dir / f"sysid_step_{step}.pt")
            print(f"[pretrain] Saved checkpoint at step {step}")
    
    # Save final checkpoint
    final_checkpoint = {
        "step": num_pretrain_steps,
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "encoder_norm": encoder_norm.state_dict(),
        "predictor_v_norm": predictor_v_norm.state_dict(),
        "predictor_u_norm": predictor_u_norm.state_dict(),
        "sysid_optimizer": sysid_trainer.optimizer.state_dict(),
        "config": {
            "sysid": dataclasses.asdict(config.sysid),
            "env": dataclasses.asdict(config.env),
        },
        "metrics": metrics,
    }
    torch.save(final_checkpoint, output_dir / "sysid_final.pt")
    
    # Save metrics history
    with open(output_dir / "metrics_history.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"\n[pretrain] Pretraining complete!")
    print(f"[pretrain] Best loss: {best_loss:.4f}")
    print(f"[pretrain] Final loss: {metrics['sysid/pred_loss']:.4f}")
    print(f"[pretrain] Checkpoints saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pretrain SysID encoder")
    parser.add_argument("--config", type=Path, default=Path("training/config_sysid.yaml"))
    parser.add_argument("--num-steps", type=int, default=50000, help="Number of pretraining steps")
    parser.add_argument("--eval-interval", type=int, default=1000, help="Evaluation interval")
    parser.add_argument("--checkpoint-interval", type=int, default=10000, help="Checkpoint interval")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--min-buffer-size", type=int, default=10000, help="Minimum buffer size before training")
    
    args = parser.parse_args()
    
    # Load config
    raw = load_config(args.config)
    
    # Override seed if provided
    if args.seed is not None:
        raw["seed"] = args.seed
    
    # Create minimal argparse namespace for build_config_bundle
    config_overrides = argparse.Namespace(
        seed=args.seed,
        num_train_timesteps=None,
        output_dir=None,
        reference_dataset=None,
        fitted_params=None,
        fitted_spread=0.1,
        resume_from=None
    )
    
    config = build_config_bundle(raw, config_overrides)
    
    if not config.sysid.enabled:
        raise ValueError("SysID must be enabled in config for pretraining")
    
    # Set device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Set output dir
    output_dir = args.output_dir if args.output_dir else Path("training/sysid_pretrained")
    
    # Pretrain
    pretrain_sysid(
        config=config,
        num_pretrain_steps=args.num_steps,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=output_dir,
        device=device,
        min_buffer_size=args.min_buffer_size,
    )


if __name__ == "__main__":
    main()

