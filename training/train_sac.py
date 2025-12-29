#!/usr/bin/env python3
"""Train the SAC longitudinal controller with Hugging Face Accelerate."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from tqdm.auto import tqdm

from env.longitudinal_env import LongitudinalEnv, LongitudinalEnvConfig
from utils.dynamics import RandomizationConfig


@dataclass(slots=True)
class TrainingParams:
    num_train_timesteps: int = 500_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    replay_size: int = 1_000_000
    warmup_steps: int = 10_000
    eval_interval: int = 5_000
    eval_episodes: int = 5
    log_interval: int = 1_000
    checkpoint_interval: int = 10_000
    max_grad_norm: float = 5.0
    target_entropy_scale: float = 1.0
    action_horizon_steps: int = 1


@dataclass(slots=True)
class SysIDParams:
    """SysID configuration for vehicle-conditioned SAC."""
    enabled: bool = False
    pretrained_path: str | None = None  # Path to pretrained SysID checkpoint
    freeze_encoder: bool = False  # Freeze encoder during SAC training
    dz: int = 12
    gru_hidden: int = 64
    predictor_hidden: int = 128
    predictor_layers: int = 2
    burn_in: int = 20
    horizon: int = 40
    lambda_slow: float = 5e-3
    lambda_z: float = 5e-4
    learning_rate: float = 1e-3
    update_every: int = 1
    updates_per_step: int = 1
    pretrain_steps: int = 0  # Deprecated: use pretrain_sysid.py instead
    norm_clip: float = 10.0
    norm_eps: float = 1e-6


@dataclass(slots=True)
class OutputConfig:
    dir: Path = Path("training/checkpoints")
    save_latest: bool = True


@dataclass(slots=True)
class ConfigBundle:
    seed: int
    env: LongitudinalEnvConfig
    training: TrainingParams
    output: OutputConfig
    sysid: SysIDParams
    reference_dataset: str | None = None
    # Fitted vehicle randomization config
    fitted_params_path: str | None = None
    fitted_spread_pct: float = 0.1
    generator_config: Dict[str, Any] | None = None
    resume_from_checkpoint: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC for ego-only longitudinal control.")
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument("--num-train-timesteps", type=int, default=None, help="Override training.num_train_timesteps.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-dataset", type=str, default=None)
    parser.add_argument("--fitted-params", type=str, default=None,
                        help="Path to fitted_params.json for centered vehicle randomization")
    parser.add_argument("--fitted-spread", type=float, default=0.1,
                        help="Spread percentage around fitted params (default: 0.1 = ±10%%)")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config file: {path}")
    return raw


def build_config_bundle(raw: Dict[str, Any], overrides: argparse.Namespace) -> ConfigBundle:
    seed = overrides.seed if overrides.seed is not None else raw.get("seed", 0)
    env_cfg = LongitudinalEnvConfig(**raw.get("env", {}))
    training_cfg = TrainingParams(**raw.get("training", {}))
    sysid_cfg = SysIDParams(**raw.get("sysid", {}))
    output_block = raw.get("output", {})
    output_dir = overrides.output_dir if overrides.output_dir else Path(output_block.get("dir", "training/checkpoints"))
    output = OutputConfig(dir=Path(output_dir), save_latest=output_block.get("save_latest", True))

    if overrides.num_train_timesteps is not None:
        training_cfg.num_train_timesteps = overrides.num_train_timesteps
    reference_dataset = overrides.reference_dataset or raw.get("reference_dataset")
    
    # Fitted vehicle randomization config
    vr_config = raw.get("vehicle_randomization", {})
    fitted_params_path = overrides.fitted_params or vr_config.get("fitted_params_path")
    fitted_spread_pct = overrides.fitted_spread if overrides.fitted_params else vr_config.get("spread_pct", 0.1)
    
    # Build generator config from raw config
    generator_config = {}
    if "generator" in raw:
        generator_config["generator"] = raw["generator"]
    if "vehicle_randomization" in raw:
        generator_config["vehicle_randomization"] = raw["vehicle_randomization"]

    resume_from_checkpoint = overrides.resume_from if hasattr(overrides, 'resume_from') else None
    
    return ConfigBundle(
        seed=seed,
        env=env_cfg,
        training=training_cfg,
        sysid=sysid_cfg,
        output=output,
        reference_dataset=reference_dataset,
        fitted_params_path=fitted_params_path,
        fitted_spread_pct=fitted_spread_pct,
        generator_config=generator_config,
        resume_from_checkpoint=resume_from_checkpoint,
    )


class ReplayBuffer:
    """Unified replay buffer for SAC and SysID training.
    
    Stores both single-step transitions (for SAC) and sequence data (for SysID).
    Supports:
        - Single-step sampling with z_t and z_{t+1} for SAC
        - Sequence sampling with burn-in and rollout windows for SysID
    """

    def __init__(self, obs_dim: int, action_dim: int, capacity: int, z_dim: int = 0):
        """Initialize replay buffer.
        
        Args:
            obs_dim: Observation dimension (without z)
            action_dim: Action dimension
            capacity: Buffer capacity
            z_dim: Dynamics latent dimension (0 = disabled, for backward compatibility)
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        
        # SAC buffers
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        
        # SysID buffers (only allocated if z_dim > 0)
        if z_dim > 0:
            self.speed_buf = np.zeros((capacity,), dtype=np.float32)
            self.z_buf = np.zeros((capacity, z_dim), dtype=np.float32)
            self.z_next_buf = np.zeros((capacity, z_dim), dtype=np.float32)
            self.episode_id_buf = np.zeros((capacity,), dtype=np.int32)
            self.step_in_episode_buf = np.zeros((capacity,), dtype=np.int32)
        else:
            self.speed_buf = None
            self.z_buf = None
            self.z_next_buf = None
            self.episode_id_buf = None
            self.step_in_episode_buf = None
        
        self.ptr = 0
        self.size = 0
        self._current_episode_id = 0
        self._current_episode_step = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        speed: float | None = None,
        z: np.ndarray | None = None,
        z_next: np.ndarray | None = None,
        episode_id: int | None = None,
        step_in_episode: int | None = None
    ) -> None:
        """Add transition to buffer.
        
        Args:
            obs: Observation (without z)
            action: Action
            reward: Reward
            next_obs: Next observation (without z)
            done: Done flag
            speed: Raw speed v_t (for SysID, optional)
            z: Dynamics latent z_t (for SAC, optional)
            z_next: Dynamics latent z_{t+1} (for SAC, optional)
            episode_id: Episode ID (for SysID sequence sampling, optional)
            step_in_episode: Step index in episode (for SysID, optional)
        """
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)
        
        # Store SysID data if available
        if self.z_dim > 0:
            if speed is not None:
                self.speed_buf[self.ptr] = speed
            if z is not None:
                self.z_buf[self.ptr] = z
            if z_next is not None:
                self.z_next_buf[self.ptr] = z_next
            if episode_id is not None:
                self.episode_id_buf[self.ptr] = episode_id
            if step_in_episode is not None:
                self.step_in_episode_buf[self.ptr] = step_in_episode
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Sample single-step transitions for SAC training.
        
        Returns:
            If z_dim == 0: (obs, action, reward, next_obs, done)
            If z_dim > 0: (obs_aug, action, reward, next_obs_aug, done)
                where obs_aug = [obs, z] and next_obs_aug = [next_obs, z_next]
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        if self.z_dim == 0:
            # Backward compatibility: no z augmentation
            batch = (
                torch.as_tensor(self.obs_buf[idxs], device=device),
                torch.as_tensor(self.action_buf[idxs], device=device),
                torch.as_tensor(self.reward_buf[idxs], device=device),
                torch.as_tensor(self.next_obs_buf[idxs], device=device),
                torch.as_tensor(self.done_buf[idxs], device=device),
            )
        else:
            # Augment observations with z
            obs = self.obs_buf[idxs]
            next_obs = self.next_obs_buf[idxs]
            z = self.z_buf[idxs]
            z_next = self.z_next_buf[idxs]
            
            obs_aug = np.concatenate([obs, z], axis=1)
            next_obs_aug = np.concatenate([next_obs, z_next], axis=1)
            
            batch = (
                torch.as_tensor(obs_aug, device=device),
                torch.as_tensor(self.action_buf[idxs], device=device),
                torch.as_tensor(self.reward_buf[idxs], device=device),
                torch.as_tensor(next_obs_aug, device=device),
                torch.as_tensor(self.done_buf[idxs], device=device),
            )
        
        return batch
    
    def sample_sequences(
        self,
        batch_size: int,
        burn_in: int,
        horizon: int,
        rng: np.random.Generator | None = None
    ):
        """Sample sequences for SysID training.
        
        Args:
            batch_size: Number of sequences to sample
            burn_in: Number of burn-in steps
            horizon: Number of rollout steps
            rng: Random number generator
        
        Returns:
            SequenceBatch or None if not enough valid sequences
        """
        if self.z_dim == 0:
            raise ValueError("Cannot sample sequences: SysID is disabled (z_dim=0)")
        
        from src.sysid.dataset import sample_sequences
        
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
    
    def start_new_episode(self) -> None:
        """Start a new episode (increments episode ID, resets step counter)."""
        if self.z_dim > 0:
            self._current_episode_id += 1
            self._current_episode_step = 0
    
    def get_current_episode_info(self) -> Tuple[int, int]:
        """Get current episode ID and step index.
        
        Returns:
            Tuple of (episode_id, step_in_episode)
        """
        if self.z_dim == 0:
            return 0, 0
        return self._current_episode_id, self._current_episode_step
    
    def increment_episode_step(self) -> None:
        """Increment step counter in current episode."""
        if self.z_dim > 0:
            self._current_episode_step += 1


def mlp(input_dim: int, output_dim: int, hidden_dim: int = 256, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Gaussian policy with Tanh squashing and action-range scaling.
    
    Supports optional vehicle dynamics latent z:
        - If z_dim > 0: expects obs of shape (batch, obs_dim + z_dim)
        - If z_dim == 0: backward compatible with obs of shape (batch, obs_dim)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int = 256,
        z_dim: int = 0,
    ):
        """Initialize policy.
        
        Args:
            obs_dim: Base observation dimension (without z)
            action_dim: Action dimension
            action_low: Action lower bounds
            action_high: Action upper bounds
            hidden_dim: Hidden layer dimension
            z_dim: Dynamics latent dimension (0 = disabled, backward compatible)
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        
        # Network input is obs_dim + z_dim
        input_dim = obs_dim + z_dim
        self.net = mlp(input_dim, hidden_dim, hidden_dim=hidden_dim, depth=2)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)
        if action_low.shape[0] != action_dim or action_high.shape[0] != action_dim:
            raise ValueError("action_low/high must match action_dim")
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.from_numpy(action_scale))
        self.register_buffer("action_bias", torch.from_numpy(action_bias))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(obs)
        mu = self.mu_head(features)
        log_std = torch.clamp(self.log_std_head(features), -20.0, 2.0)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mu_action = torch.tanh(mu) * self.action_scale + self.action_bias
        return action, log_prob, mu_action


class QNetwork(nn.Module):
    """Q-network for SAC critic.
    
    Supports optional vehicle dynamics latent z:
        - If z_dim > 0: expects obs of shape (batch, obs_dim + z_dim)
        - If z_dim == 0: backward compatible with obs of shape (batch, obs_dim)
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, z_dim: int = 0):
        """Initialize Q-network.
        
        Args:
            obs_dim: Base observation dimension (without z)
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            z_dim: Dynamics latent dimension (0 = disabled, backward compatible)
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        
        # Network input is obs_dim + z_dim + action_dim
        input_dim = obs_dim + z_dim + action_dim
        self.net = mlp(input_dim, 1, hidden_dim=hidden_dim, depth=2)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class SACTrainer:
    def __init__(
        self,
        env: LongitudinalEnv,
        eval_env: LongitudinalEnv,
        cfg: ConfigBundle,
        accelerator: Accelerator,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.cfg = cfg
        self.accelerator = accelerator
        self.device = accelerator.device

        obs_dim = env.observation_space.shape[0]
        self.env_action_dim = env.action_space.shape[0]
        self.env_action_low = env.action_space.low
        self.env_action_high = env.action_space.high
        self.action_horizon = max(1, cfg.training.action_horizon_steps)
        self.policy_action_dim = self.env_action_dim * self.action_horizon

        action_low = np.tile(self.env_action_low, self.action_horizon)
        action_high = np.tile(self.env_action_high, self.action_horizon)

        # SysID setup
        self.sysid_enabled = cfg.sysid.enabled
        z_dim = cfg.sysid.dz if self.sysid_enabled else 0
        
        # Initialize SysID components if enabled
        if self.sysid_enabled:
            from src.sysid import (
                ContextEncoder,
                DynamicsPredictor,
                FeatureBuilder,
                RunningNorm,
                SysIDTrainer as SysIDTrainerClass,
            )
            
            # Create encoder and predictor
            self.encoder = ContextEncoder(
                input_dim=4,
                hidden_dim=cfg.sysid.gru_hidden,
                z_dim=cfg.sysid.dz
            ).to(self.device)
            
            self.predictor = DynamicsPredictor(
                z_dim=cfg.sysid.dz,
                hidden_dim=cfg.sysid.predictor_hidden,
                num_layers=cfg.sysid.predictor_layers
            ).to(self.device)
            
            # Create normalizers
            self.encoder_norm = RunningNorm(
                dim=4,
                eps=cfg.sysid.norm_eps,
                clip=cfg.sysid.norm_clip
            ).to(self.device)
            
            self.predictor_v_norm = RunningNorm(
                dim=1,
                eps=cfg.sysid.norm_eps,
                clip=cfg.sysid.norm_clip
            ).to(self.device)
            
            self.predictor_u_norm = RunningNorm(
                dim=1,
                eps=cfg.sysid.norm_eps,
                clip=cfg.sysid.norm_clip
            ).to(self.device)
            
            # Load pretrained SysID if provided
            if cfg.sysid.pretrained_path:
                pretrained_path = Path(cfg.sysid.pretrained_path)
                if pretrained_path.exists():
                    accelerator.print(f"[sysid] Loading pretrained SysID from {pretrained_path}")
                    checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
                    
                    self.encoder.load_state_dict(checkpoint["encoder"])
                    self.predictor.load_state_dict(checkpoint["predictor"])
                    self.encoder_norm.load_state_dict(checkpoint["encoder_norm"])
                    self.predictor_v_norm.load_state_dict(checkpoint["predictor_v_norm"])
                    self.predictor_u_norm.load_state_dict(checkpoint["predictor_u_norm"])
                    
                    accelerator.print(f"[sysid] Loaded pretrained SysID (step {checkpoint.get('step', 'unknown')})")
                    
                    # Optionally freeze encoder
                    if cfg.sysid.freeze_encoder:
                        for param in self.encoder.parameters():
                            param.requires_grad = False
                        for param in self.predictor.parameters():
                            param.requires_grad = False
                        accelerator.print("[sysid] Froze encoder and predictor parameters")
                else:
                    accelerator.print(f"[sysid] Warning: Pretrained path {pretrained_path} not found, training from scratch")
            
            # Create SysID trainer (even if frozen, for potential fine-tuning)
            self.sysid_trainer = SysIDTrainerClass(
                encoder=self.encoder,
                predictor=self.predictor,
                encoder_norm=self.encoder_norm,
                predictor_v_norm=self.predictor_v_norm,
                predictor_u_norm=self.predictor_u_norm,
                learning_rate=cfg.sysid.learning_rate,
                lambda_slow=cfg.sysid.lambda_slow,
                lambda_z=cfg.sysid.lambda_z,
                dt=cfg.env.dt,
                device=self.device
            )
            
            # Feature builder for online inference
            self.feature_builder = FeatureBuilder(dt=cfg.env.dt)
            
            # Encoder hidden state (single env, so shape is (hidden_dim,))
            self.encoder_hidden = self.encoder.reset(batch_size=1, device=self.device).squeeze(0)
            
            # Speed and action history for features
            self.prev_speed = 0.0
            self.prev_action = 0.0
            
            # Track if encoder is frozen
            self.encoder_frozen = cfg.sysid.freeze_encoder and cfg.sysid.pretrained_path is not None
        else:
            self.encoder = None
            self.predictor = None
            self.sysid_trainer = None
            self.feature_builder = None
            self.encoder_hidden = None

        # Replay buffer with SysID support
        self.replay_buffer = ReplayBuffer(obs_dim, self.policy_action_dim, cfg.training.replay_size, z_dim=z_dim)
        
        # SAC networks (with z_dim)
        self.policy = GaussianPolicy(obs_dim, self.policy_action_dim, action_low, action_high, z_dim=z_dim)
        self.q1 = QNetwork(obs_dim, self.policy_action_dim, z_dim=z_dim)
        self.q2 = QNetwork(obs_dim, self.policy_action_dim, z_dim=z_dim)
        self.target_q1 = QNetwork(obs_dim, self.policy_action_dim, z_dim=z_dim)
        self.target_q2 = QNetwork(obs_dim, self.policy_action_dim, z_dim=z_dim)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.training.learning_rate)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=cfg.training.learning_rate,
        )
        self.log_alpha = torch.nn.Parameter(torch.zeros(1))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.training.learning_rate)

        (
            self.policy,
            self.q1,
            self.q2,
            self.policy_optimizer,
            self.q_optimizer,
            self.alpha_optimizer,
        ) = accelerator.prepare(
            self.policy,
            self.q1,
            self.q2,
            self.policy_optimizer,
            self.q_optimizer,
            self.alpha_optimizer,
        )

        self.target_q1.to(self.device)
        self.target_q2.to(self.device)
        self.log_alpha = self.log_alpha.to(self.device)

        self.target_entropy = -cfg.training.target_entropy_scale * float(self.policy_action_dim)
        self._global_step = 0
        self._episode_reward = 0.0
        self._episode_length = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mu_action = self.policy.sample(obs_tensor)
            chosen = mu_action if deterministic else action
        action_vec = chosen.squeeze(0).cpu().numpy()
        env_action = action_vec[: self.env_action_dim]
        env_action = np.clip(env_action, self.env_action_low, self.env_action_high)
        return action_vec.astype(np.float32), float(env_action[0])

    def collect_step(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Collect a single environment step.
        
        If SysID is enabled:
            1. Extract speed from observation
            2. Compute z_t from encoder
            3. Augment observation with z_t
            4. Sample action from policy(obs_aug)
            5. Step environment
            6. Compute z_{t+1}
            7. Store transition with z_t, z_{t+1}, speed, etc.
        """
        # Get current speed from observation (first element)
        current_speed = obs[0]
        
        # Compute z_t if SysID enabled
        if self.sysid_enabled:
            from src.sysid.integration import compute_z_online
            
            # Compute z_t (this updates feature_builder history internally)
            self.encoder_hidden, z_t = compute_z_online(
                encoder=self.encoder,
                feature_builder=self.feature_builder,
                encoder_norm=self.encoder_norm,
                h_prev=self.encoder_hidden,
                v_t=current_speed,
                u_t=self.prev_action,
                device=self.device
            )
            
            z_t_np = z_t.cpu().numpy()
            
            # Augment observation with z_t
            obs_aug = np.concatenate([obs, z_t_np])
        else:
            obs_aug = obs
            z_t_np = None
        
        # Sample action
        if self._global_step < self.cfg.training.warmup_steps:
            env_action = self.env.action_space.sample()
            action_vec = np.tile(env_action, self.action_horizon).astype(np.float32)
            action_scalar = float(env_action[0])
        else:
            action_vec, action_scalar = self.act(obs_aug)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = self.env.step(action_scalar)
        done = terminated or truncated
        
        # Get next speed
        next_speed = next_obs[0]
        
        # Compute z_{t+1} if SysID enabled
        if self.sysid_enabled:
            from src.sysid.integration import compute_z_online
            
            # Compute z_{t+1} (updates hidden state)
            self.encoder_hidden, z_next = compute_z_online(
                encoder=self.encoder,
                feature_builder=self.feature_builder,
                encoder_norm=self.encoder_norm,
                h_prev=self.encoder_hidden,
                v_t=next_speed,
                u_t=action_scalar,
                device=self.device
            )
            
            z_next_np = z_next.cpu().numpy()
        else:
            z_next_np = None
        
        # Get episode info
        episode_id, step_in_episode = self.replay_buffer.get_current_episode_info()
        
        # Store transition (with SysID data if enabled)
        self.replay_buffer.add(
            obs=obs,
            action=action_vec,
            reward=reward,
            next_obs=next_obs,
            done=done,
            speed=current_speed if self.sysid_enabled else None,
            z=z_t_np,
            z_next=z_next_np,
            episode_id=episode_id,
            step_in_episode=step_in_episode
        )
        
        # Update replay buffer episode tracking
        self.replay_buffer.increment_episode_step()
        
        # Update history
        if self.sysid_enabled:
            self.prev_speed = current_speed
            self.prev_action = action_scalar
        
        # Episode bookkeeping
        self._episode_reward += reward
        self._episode_length += 1
        self._global_step += 1
        
        if done:
            obs, _ = self.env.reset()
            
            # Reset encoder state for new episode
            if self.sysid_enabled:
                self.encoder_hidden = self.encoder.reset(batch_size=1, device=self.device).squeeze(0)
                self.feature_builder.reset()
                self.prev_speed = obs[0]
                self.prev_action = 0.0
            
            # Start new episode in replay buffer
            self.replay_buffer.start_new_episode()
            
            self.accelerator.log(
                {
                    "train/episode_reward": self._episode_reward,
                    "train/episode_length": self._episode_length,
                },
                step=self._global_step,
            )
            self._episode_reward = 0.0
            self._episode_length = 0
        
        return (next_obs if not done else obs), reward

    def update(self) -> Dict[str, float]:
        batch = self.replay_buffer.sample(self.cfg.training.batch_size, self.device)
        obs, actions, rewards, next_obs, dones = batch

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_obs)
            target_q1 = self.target_q1(next_obs, next_action)
            target_q2 = self.target_q2(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = rewards + (1.0 - dones) * self.cfg.training.gamma * target_q

        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        q_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        self.q_optimizer.zero_grad()
        self.accelerator.backward(q_loss)
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.cfg.training.max_grad_norm)
        self.q_optimizer.step()

        new_actions, log_prob, _ = self.policy.sample(obs)
        q_new_actions = torch.min(self.q1(obs, new_actions), self.q2(obs, new_actions))
        policy_loss = (self.alpha * log_prob - q_new_actions).mean()
        self.policy_optimizer.zero_grad()
        self.accelerator.backward(policy_loss)
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.training.max_grad_norm)
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        self.accelerator.backward(alpha_loss)
        self.alpha_optimizer.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        metrics = {
            "train/q_loss": float(q_loss.detach().cpu().item()),
            "train/policy_loss": float(policy_loss.detach().cpu().item()),
            "train/alpha": float(self.alpha.detach().cpu().item()),
        }
        return metrics

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.cfg.training.tau
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)
    
    def update_sysid(self) -> Dict[str, float]:
        """Update SysID encoder and predictor.
        
        Returns:
            Dictionary of SysID metrics for logging
        """
        if not self.sysid_enabled:
            return {}
        
        # Sample sequences from replay buffer
        batch = self.replay_buffer.sample_sequences(
            batch_size=self.cfg.training.batch_size,
            burn_in=self.cfg.sysid.burn_in,
            horizon=self.cfg.sysid.horizon
        )
        
        if batch is None:
            # Not enough valid sequences yet
            return {"sysid/skipped": 1.0}
        
        # Train SysID
        metrics = self.sysid_trainer.train_step(batch)
        
        return metrics

    def evaluate(self) -> Dict[str, float]:
        rewards = []
        speed_errors = []
        
        # For SysID: maintain separate eval encoder state
        if self.sysid_enabled:
            from src.sysid.encoder import FeatureBuilder
            eval_encoder_hidden = self.encoder.reset(batch_size=1, device=self.device).squeeze(0)
            eval_feature_builder = FeatureBuilder(dt=self.cfg.env.dt)
            eval_prev_speed = 0.0
            eval_prev_action = 0.0
        
        for _ in range(self.cfg.training.eval_episodes):
            obs, _ = self.eval_env.reset()
            
            # Reset encoder state for new eval episode
            if self.sysid_enabled:
                eval_encoder_hidden = self.encoder.reset(batch_size=1, device=self.device).squeeze(0)
                eval_feature_builder.reset()
                eval_prev_speed = obs[0]
                eval_prev_action = 0.0
            
            done = False
            episode_reward = 0.0
            while not done:
                # Compute z_t if SysID enabled
                if self.sysid_enabled:
                    from src.sysid.integration import compute_z_online
                    
                    current_speed = obs[0]
                    eval_encoder_hidden, z_t = compute_z_online(
                        encoder=self.encoder,
                        feature_builder=eval_feature_builder,
                        encoder_norm=self.encoder_norm,
                        h_prev=eval_encoder_hidden,
                        v_t=current_speed,
                        u_t=eval_prev_action,
                        device=self.device
                    )
                    z_t_np = z_t.cpu().numpy()
                    obs_aug = np.concatenate([obs, z_t_np])
                else:
                    obs_aug = obs
                
                _, action_scalar = self.act(obs_aug, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action_scalar)
                
                # Update eval history for next step
                if self.sysid_enabled:
                    eval_prev_speed = obs[0]
                    eval_prev_action = action_scalar
                
                episode_reward += reward
                speed_errors.append(abs(info.get("speed_error", 0.0)))
                done = terminated or truncated
            rewards.append(episode_reward)
        return {
            "eval/avg_reward": float(np.mean(rewards)),
            "eval/avg_abs_speed_error": float(np.mean(speed_errors)) if speed_errors else 0.0,
        }

    def save_checkpoint(self, step: int) -> None:
        if not self.accelerator.is_main_process:
            return
        state = {
            "step": step,
            "policy": self.accelerator.get_state_dict(self.policy),
            "q1": self.accelerator.get_state_dict(self.q1),
            "q2": self.accelerator.get_state_dict(self.q2),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "config": {
                "env": asdict(self.cfg.env),
                "training": asdict(self.cfg.training),
                "sysid": asdict(self.cfg.sysid),
                "seed": self.cfg.seed,
                "num_train_timesteps": self.cfg.training.num_train_timesteps,
            },
            "meta": {
                "obs_dim": self.env.observation_space.shape[0],
                "env_action_dim": self.env_action_dim,
                "policy_action_dim": self.policy_action_dim,
                "sysid_enabled": self.sysid_enabled,
            },
        }
        
        # Save SysID components if enabled
        if self.sysid_enabled:
            state["encoder"] = self.encoder.state_dict()
            state["predictor"] = self.predictor.state_dict()
            state["encoder_norm"] = self.encoder_norm.state_dict()
            state["predictor_v_norm"] = self.predictor_v_norm.state_dict()
            state["predictor_u_norm"] = self.predictor_u_norm.state_dict()
            state["sysid_optimizer"] = self.sysid_trainer.optimizer.state_dict()
        
        self.cfg.output.dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.cfg.output.dir / f"sac_step_{step}.pt"
        torch.save(state, ckpt_path)
        if self.cfg.output.save_latest:
            torch.save(state, self.cfg.output.dir / "latest.pt")

        self.accelerator.print(f"[checkpoint] saved {ckpt_path}")
        self.accelerator.print(f"[checkpoint] saved {ckpt_path}")


def train(config: ConfigBundle) -> None:
    accelerator = Accelerator()
    accelerator.init_trackers(
        project_name="sac_longitudinal",
        config={
            "seed": config.seed,
            "num_train_timesteps": config.training.num_train_timesteps,
        },
    )

    # Build generator config with vehicle randomization
    generator_config = config.generator_config or {}
    
    # If fitted params specified, create centered randomization
    if config.fitted_params_path:
        from utils.dynamics import ExtendedPlantRandomization
        accelerator.print(f"[setup] Using fitted params from {config.fitted_params_path}")
        accelerator.print(f"[setup] Spread: ±{config.fitted_spread_pct*100:.0f}%")
        
        # Create centered randomization and get its config dict
        centered_rand = ExtendedPlantRandomization.from_fitted_params(
            config.fitted_params_path,
            spread_pct=config.fitted_spread_pct,
        )
        # Update generator_config with fitted ranges
        from fitting.randomization_config import CenteredRandomizationConfig
        from fitting import FittedVehicleParams
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
    eval_env = LongitudinalEnv(
        config.env,
        randomization=RandomizationConfig(),
        generator_config=generator_config,
        seed=config.seed + 1,
    )
    env.reset(seed=config.seed)
    eval_env.reset(seed=config.seed + 1)

    trainer = SACTrainer(env, eval_env, config, accelerator)

    # Load checkpoint if specified
    start_step = 0
    if config.resume_from_checkpoint and config.resume_from_checkpoint.exists():
        accelerator.print(f"[resume] Loading checkpoint from {config.resume_from_checkpoint}")
        checkpoint = torch.load(config.resume_from_checkpoint, map_location="cpu", weights_only=False)
        start_step = checkpoint.get("step", 0)
        
        # Load network weights (unwrap from accelerator if needed)
        unwrapped_policy = accelerator.unwrap_model(trainer.policy)
        unwrapped_q1 = accelerator.unwrap_model(trainer.q1)
        unwrapped_q2 = accelerator.unwrap_model(trainer.q2)
        
        unwrapped_policy.load_state_dict(checkpoint["policy"])
        unwrapped_q1.load_state_dict(checkpoint["q1"])
        unwrapped_q2.load_state_dict(checkpoint["q2"])
        trainer.target_q1.load_state_dict(checkpoint["target_q1"])
        trainer.target_q2.load_state_dict(checkpoint["target_q2"])
        
        # Load optimizer states
        trainer.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        trainer.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        trainer.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        
        # Load log_alpha
        trainer.log_alpha.data = checkpoint["log_alpha"].to(trainer.device)
        
        # Load SysID components if available
        if config.sysid.enabled and "encoder" in checkpoint:
            accelerator.print("[resume] Loading SysID components")
            trainer.encoder.load_state_dict(checkpoint["encoder"])
            trainer.predictor.load_state_dict(checkpoint["predictor"])
            trainer.encoder_norm.load_state_dict(checkpoint["encoder_norm"])
            trainer.predictor_v_norm.load_state_dict(checkpoint["predictor_v_norm"])
            trainer.predictor_u_norm.load_state_dict(checkpoint["predictor_u_norm"])
            trainer.sysid_trainer.optimizer.load_state_dict(checkpoint["sysid_optimizer"])
        
        # Update trainer's global step
        trainer._global_step = start_step
        
        accelerator.print(f"[resume] Resuming from step {start_step}")

    obs, _ = env.reset(seed=config.seed)
    last_log = time.time()

    accelerator.print(f"[setup] num_train_timesteps={config.training.num_train_timesteps}")
    progress = tqdm(
        range(start_step + 1, config.training.num_train_timesteps + 1),
        disable=not accelerator.is_main_process,
        desc="Training",
    )
    for step in progress:
        obs, reward = trainer.collect_step(obs)

        if trainer.replay_buffer.size >= config.training.batch_size:
            # SAC update
            metrics = trainer.update()
            
            # SysID update (if enabled, scheduled, and not frozen)
            if config.sysid.enabled and step % config.sysid.update_every == 0:
                if hasattr(trainer, 'encoder_frozen') and trainer.encoder_frozen:
                    # Skip SysID updates if encoder is frozen
                    pass
                else:
                    for _ in range(config.sysid.updates_per_step):
                        sysid_metrics = trainer.update_sysid()
                        metrics.update(sysid_metrics)
            
            if step % config.training.log_interval == 0:
                accelerator.log(metrics, step=step)

        if step % config.training.eval_interval == 0:
            eval_metrics = trainer.evaluate()
            accelerator.log(eval_metrics, step=step)
            accelerator.print(f"[eval] step={step} metrics={json.dumps(eval_metrics)}")

        if step % config.training.checkpoint_interval == 0:
            trainer.save_checkpoint(step)

        if accelerator.is_main_process and step % config.training.log_interval == 0:
            now = time.time()
            fps = config.training.log_interval / max(now - last_log, 1e-6)
            
            # Build log message
            log_msg = f"[train] step={step} reward={reward:.3f} buffer={trainer.replay_buffer.size} fps={fps:.1f}"
            if config.sysid.enabled and "sysid/pred_loss" in metrics:
                log_msg += f" sysid_loss={metrics['sysid/pred_loss']:.4f}"
            
            accelerator.print(log_msg)
            progress.set_postfix(
                reward=f"{reward:.2f}",
                buffer=trainer.replay_buffer.size,
                fps=f"{fps:.1f}",
                refresh=False,
            )
            last_log = now

    trainer.save_checkpoint(config.training.num_train_timesteps)
    progress.close()
    accelerator.end_training()
    accelerator.print("[done] training complete")


def main() -> None:
    args = parse_args()
    raw = load_config(args.config)
    bundle = build_config_bundle(raw, args)
    train(bundle)


if __name__ == "__main__":
    main()


