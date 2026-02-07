#!/usr/bin/env python3
"""Train the SAC longitudinal controller with Hugging Face Accelerate."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from tqdm.auto import tqdm

from env.longitudinal_env import LongitudinalEnv, LongitudinalEnvConfig
from utils.dynamics import RandomizationConfig

# ClearML integration (optional)
try:
    import clearml
    from utils.clearml_logger import (
        init_clearml_task,
        log_metrics,
        log_config,
        upload_plot,
        upload_artifact,
        close_task,
    )
    CLEARML_LOGGER_AVAILABLE = True
except ImportError:
    CLEARML_LOGGER_AVAILABLE = False
    clearml = None
    init_clearml_task = None
    log_metrics = None
    log_config = None
    upload_plot = None
    upload_artifact = None
    close_task = None


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
    reward_scale: float = 1.0  # Scale rewards to prevent exploding Q-values
    # Animation options
    enable_animation: bool = False  # Enable online animation during training
    animation_interval: int = 5_000  # Steps between animation updates (default: same as eval_interval)
    num_animation_episodes: int = 3  # Number of example episodes to show (each with different initial speed error)


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
class DynamicsMapParams:
    """Dynamics map configuration for vehicle-conditioned SAC."""
    enabled: bool = False
    pretrained_prior_path: str | None = None
    pretrained_encoder_path: str | None = None
    freeze_encoder: bool = False
    
    # Grid
    N_u: int = 100
    N_v: int = 100
    v_max: float = 30.0
    
    # Reconstruction
    update_every: int = 10
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    sigma_meas: float = 0.5  # Increased from 0.3 to handle observation noise better
    warmstart_rho: float = 0.8
    gradient_smoothing_sigma: float = 10.0  # Increased from 2.0 to reduce artifacts
    
    # Buffer
    buffer_capacity: int = 10000
    lambda_decay: float = 0.3
    w_min: float = 0.2
    
    # Encoder
    context_dim: int = 32
    hidden_channels: list[int] | None = None
    encoder_learning_rate: float = 3e-4
    
    # Patch filtering (for uniform spatial distribution)
    patch_filtering_enabled: bool = False
    patch_size_u: int = 10
    patch_size_v: int = 10
    
    def __post_init__(self):
        if self.hidden_channels is None:
            self.hidden_channels = [16, 32]


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
    context_mode: str = "none"  # "none" | "sysid" | "dynamics_map"
    sysid: SysIDParams = SysIDParams()
    dynamics_map: DynamicsMapParams = DynamicsMapParams()
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
    parser.add_argument(
        "--disable-clearml",
        action="store_true",
        help="Disable ClearML logging even if ClearML is installed",
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
    
    # Context mode
    context_mode = raw.get("context_mode", "none")
    
    # SysID config
    sysid_cfg = SysIDParams(**raw.get("sysid", {}))
    if context_mode == "sysid":
        sysid_cfg.enabled = True
    
    # Dynamics map config
    dynamics_map_raw = raw.get("dynamics_map", {})
    dynamics_map_cfg = DynamicsMapParams(**dynamics_map_raw)
    if context_mode == "dynamics_map":
        dynamics_map_cfg.enabled = True
    
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
        context_mode=context_mode,
        sysid=sysid_cfg,
        dynamics_map=dynamics_map_cfg,
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
        # Check input for NaN/Inf
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(f"[ERROR] Invalid observation input to policy:")
            print(f"  NaN: {torch.isnan(obs).any()}, Inf: {torch.isinf(obs).any()}")
            print(f"  Stats: min={obs.min():.4f}, max={obs.max():.4f}, mean={obs.mean():.4f}")
            # Replace invalid values with zeros as emergency fallback
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        features = self.net(obs)
        
        # Check features for NaN/Inf
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"[ERROR] Invalid features in policy network:")
            print(f"  NaN: {torch.isnan(features).any()}, Inf: {torch.isinf(features).any()}")
            print(f"  Input obs stats: min={obs.min():.4f}, max={obs.max():.4f}, mean={obs.mean():.4f}")
        
        mu = self.mu_head(features)
        log_std = torch.clamp(self.log_std_head(features), -20.0, 2.0)
        
        # Check outputs for NaN/Inf
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print(f"[ERROR] Invalid mu output from policy:")
            print(f"  NaN: {torch.isnan(mu).any()}, Inf: {torch.isinf(mu).any()}")
            print(f"  Features stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
        
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

        # Context mode setup
        self.context_mode = cfg.context_mode
        z_dim = 0
        
        # Initialize SysID components if enabled
        if self.context_mode == "sysid" and cfg.sysid.enabled:
            z_dim = cfg.sysid.dz
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
            
        elif self.context_mode == "dynamics_map" and cfg.dynamics_map.enabled:
            z_dim = cfg.dynamics_map.context_dim
            accelerator.print("[dynamics_map] Initializing dynamics map context...")
            
            from src.maps.grid import MapGrid
            from src.maps.context_manager import DynamicsMapContext
            from src.models.diffusion_prior import DiffusionPrior
            from src.maps.encoder import MapEncoder
            
            # Load diffusion prior
            prior_path = Path(cfg.dynamics_map.pretrained_prior_path)
            accelerator.print(f"[dynamics_map] Loading diffusion prior from {prior_path}")
            self.diffusion_prior = DiffusionPrior.load(prior_path, device=self.device)
            self.diffusion_prior.eval()
            
            # Load normalization stats
            norm_path = prior_path.parent / "norm_stats.json"
            with open(norm_path, "r") as f:
                norm_stats = json.load(f)
            accelerator.print(f"[dynamics_map] Loaded norm stats: mean={norm_stats['mean']:.3f}, std={norm_stats['std']:.3f}")
            
            # Create or load encoder
            if cfg.dynamics_map.pretrained_encoder_path:
                encoder_path = Path(cfg.dynamics_map.pretrained_encoder_path)
                accelerator.print(f"[dynamics_map] Loading pretrained encoder from {encoder_path}")
                self.map_encoder = MapEncoder.load(encoder_path, device=self.device)
                if cfg.dynamics_map.freeze_encoder:
                    for param in self.map_encoder.parameters():
                        param.requires_grad = False
                    accelerator.print("[dynamics_map] Froze encoder parameters")
            else:
                accelerator.print("[dynamics_map] Creating new encoder (training jointly)")
                self.map_encoder = MapEncoder(
                    input_size=(cfg.dynamics_map.N_u, cfg.dynamics_map.N_v),
                    context_dim=cfg.dynamics_map.context_dim,
                    hidden_channels=tuple(cfg.dynamics_map.hidden_channels),
                ).to(self.device)
            
            # Create grid
            grid = MapGrid(
                N_u=cfg.dynamics_map.N_u,
                N_v=cfg.dynamics_map.N_v,
                v_max=cfg.dynamics_map.v_max,
            )
            
            # Create context manager
            self.dynamics_map_context = DynamicsMapContext(
                grid=grid,
                diffusion_prior=self.diffusion_prior,
                encoder=self.map_encoder,
                norm_mean=norm_stats["mean"],
                norm_std=norm_stats["std"],
                buffer_capacity=cfg.dynamics_map.buffer_capacity,
                lambda_decay=cfg.dynamics_map.lambda_decay,
                w_min=cfg.dynamics_map.w_min,
                guidance_scale=cfg.dynamics_map.guidance_scale,
                sigma_meas=cfg.dynamics_map.sigma_meas,
                num_inference_steps=cfg.dynamics_map.num_inference_steps,
                warmstart_rho=cfg.dynamics_map.warmstart_rho,
                gradient_smoothing_sigma=cfg.dynamics_map.gradient_smoothing_sigma,
                patch_filtering_enabled=cfg.dynamics_map.patch_filtering_enabled,
                patch_size_u=cfg.dynamics_map.patch_size_u,
                patch_size_v=cfg.dynamics_map.patch_size_v,
                device=self.device,
            )
            
            # Create optimizer for encoder (if not frozen)
            if not cfg.dynamics_map.freeze_encoder:
                self.encoder_optimizer = torch.optim.Adam(
                    self.map_encoder.parameters(),
                    lr=cfg.dynamics_map.encoder_learning_rate,
                )
                self.encoder_optimizer = accelerator.prepare(self.encoder_optimizer)
                accelerator.print(f"[dynamics_map] Encoder optimizer created (lr={cfg.dynamics_map.encoder_learning_rate})")
            
            accelerator.print(f"[dynamics_map] Context manager initialized (context_dim={z_dim})")
            
            # Set dummy attributes for compatibility
            self.encoder = None
            self.predictor = None
            self.sysid_trainer = None
            self.feature_builder = None
            self.encoder_hidden = None
            
        else:
            # Baseline mode (no context)
            self.context_mode = "none"
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
            
        If dynamics_map is enabled:
            1. Get context z_t from dynamics map
            2. Augment observation with z_t
            3. Sample action from policy(obs_aug)
            4. Step environment
            5. Add observation to buffer
            6. Update map periodically
            7. Get next context z_{t+1}
        """
        # Get current speed from observation (first element)
        current_speed = obs[0]
        
        # Compute z_t based on context mode
        if self.context_mode == "sysid":
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
            
        elif self.context_mode == "dynamics_map":
            # Get context from dynamics map
            z_t = self.dynamics_map_context.get_context()  # Returns (context_dim,) torch tensor
            z_t_np = z_t.cpu().numpy()
            
            # Validate context for NaN/Inf
            if np.isnan(z_t_np).any() or np.isinf(z_t_np).any():
                self.accelerator.print(f"[WARNING] Invalid dynamics map context at step {self._global_step}:")
                self.accelerator.print(f"  NaN: {np.isnan(z_t_np).any()}, Inf: {np.isinf(z_t_np).any()}")
                self.accelerator.print(f"  Stats: min={z_t_np.min():.4f}, max={z_t_np.max():.4f}, mean={z_t_np.mean():.4f}")
                # Use zero context as fallback
                z_t_np = np.zeros_like(z_t_np)
                self.accelerator.print(f"  Replaced with zero context")
            
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
        
        # Apply reward scaling to prevent exploding Q-values
        reward = reward * self.cfg.training.reward_scale
        
        # Get next speed
        next_speed = next_obs[0]
        
        # Compute z_{t+1} based on context mode
        if self.context_mode == "sysid":
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
            
        elif self.context_mode == "dynamics_map":
            # Extract grade from info
            theta = info.get("grade_rad", 0.0)
            
            # Get measured acceleration from info (raw_acceleration is the measured value)
            a_meas = info.get("raw_acceleration", 0.0)
            
            # Add observation to dynamics map buffer
            self.dynamics_map_context.add_observation(
                u=action_scalar,
                v=current_speed,
                a_meas=a_meas,
                theta=theta,
                t=float(self._global_step * self.cfg.env.dt),
            )
            
            # Update map periodically
            if self._global_step % self.cfg.dynamics_map.update_every == 0:
                self.dynamics_map_context.update_map()
            
            # Get next context
            z_next = self.dynamics_map_context.get_context()
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
            speed=current_speed if self.context_mode == "sysid" else None,
            z=z_t_np,
            z_next=z_next_np,
            episode_id=episode_id,
            step_in_episode=step_in_episode
        )
        
        # Update replay buffer episode tracking
        self.replay_buffer.increment_episode_step()
        
        # Update history
        if self.context_mode == "sysid":
            self.prev_speed = current_speed
            self.prev_action = action_scalar
        
        # Episode bookkeeping
        self._episode_reward += reward
        self._episode_length += 1
        self._global_step += 1
        
        if done:
            obs, _ = self.env.reset()
            
            # Reset encoder state for new episode
            if self.context_mode == "sysid":
                self.encoder_hidden = self.encoder.reset(batch_size=1, device=self.device).squeeze(0)
                self.feature_builder.reset()
                self.prev_speed = obs[0]
                self.prev_action = 0.0
            elif self.context_mode == "dynamics_map":
                self.dynamics_map_context.reset()
            
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
        q_grad_norm = nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.cfg.training.max_grad_norm)
        self.q_optimizer.step()

        new_actions, log_prob, _ = self.policy.sample(obs)
        q_new_actions = torch.min(self.q1(obs, new_actions), self.q2(obs, new_actions))
        policy_loss = (self.alpha * log_prob - q_new_actions).mean()
        self.policy_optimizer.zero_grad()
        self.accelerator.backward(policy_loss)
        policy_grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.training.max_grad_norm)
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
            "train/policy_grad_norm": float(policy_grad_norm),
            "train/q_grad_norm": float(q_grad_norm),
            "train/q_mean": float(current_q1.mean().detach().cpu().item()),
            "train/q_std": float(current_q1.std().detach().cpu().item()),
        }
        return metrics

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.cfg.training.tau
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)
    
    def _check_for_nans(self, step: int) -> bool:
        """Check if any network has NaN parameters.
        
        Args:
            step: Current training step
            
        Returns:
            True if NaN detected, False otherwise
        """
        has_nan = False
        
        for name, param in self.policy.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                self.accelerator.print(f"[ERROR] NaN/Inf detected in policy.{name} at step {step}")
                self.accelerator.print(f"  Stats: min={param.min():.4f}, max={param.max():.4f}, mean={param.mean():.4f}")
                has_nan = True
                
        for name, param in self.q1.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                self.accelerator.print(f"[ERROR] NaN/Inf detected in q1.{name} at step {step}")
                has_nan = True
                
        for name, param in self.q2.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                self.accelerator.print(f"[ERROR] NaN/Inf detected in q2.{name} at step {step}")
                has_nan = True
                
        return has_nan
    
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
        if self.context_mode == "sysid":
            from src.sysid.encoder import FeatureBuilder
            eval_encoder_hidden = self.encoder.reset(batch_size=1, device=self.device).squeeze(0)
            eval_feature_builder = FeatureBuilder(dt=self.cfg.env.dt)
            eval_prev_speed = 0.0
            eval_prev_action = 0.0
        elif self.context_mode == "dynamics_map":
            # For dynamics map: create separate context manager for eval
            # (reuse the same pretrained models but fresh state)
            eval_dynamics_map_context = None  # Will use training context for simplicity
        
        for _ in range(self.cfg.training.eval_episodes):
            obs, _ = self.eval_env.reset()
            
            # Reset encoder state for new eval episode
            if self.context_mode == "sysid":
                eval_encoder_hidden = self.encoder.reset(batch_size=1, device=self.device).squeeze(0)
                eval_feature_builder.reset()
                eval_prev_speed = obs[0]
                eval_prev_action = 0.0
            elif self.context_mode == "dynamics_map":
                # Reset dynamics map context for new episode
                self.dynamics_map_context.reset()
            
            done = False
            episode_reward = 0.0
            while not done:
                # Compute z_t based on context mode
                if self.context_mode == "sysid":
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
                elif self.context_mode == "dynamics_map":
                    # Get context from dynamics map (frozen encoder)
                    z_t = self.dynamics_map_context.get_context()
                    z_t_np = z_t.cpu().numpy()
                    obs_aug = np.concatenate([obs, z_t_np])
                else:
                    obs_aug = obs
                
                _, action_scalar = self.act(obs_aug, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action_scalar)
                
                # Update eval history for next step
                if self.context_mode == "sysid":
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

    def save_checkpoint(self, step: int, checkpoint_path: Path | None = None) -> None:
        """Save a checkpoint.
        
        Args:
            step: Current training step
            checkpoint_path: Optional custom path for checkpoint. If None, uses default naming.
        """
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
                "dynamics_map": asdict(self.cfg.dynamics_map),
                "context_mode": self.cfg.context_mode,
                "seed": self.cfg.seed,
                "num_train_timesteps": self.cfg.training.num_train_timesteps,
            },
            "meta": {
                "obs_dim": self.env.observation_space.shape[0],
                "env_action_dim": self.env_action_dim,
                "policy_action_dim": self.policy_action_dim,
                "context_mode": self.context_mode,
            },
        }
        
        # Save SysID components if enabled
        if self.context_mode == "sysid":
            state["encoder"] = self.encoder.state_dict()
            state["predictor"] = self.predictor.state_dict()
            state["encoder_norm"] = self.encoder_norm.state_dict()
            state["predictor_v_norm"] = self.predictor_v_norm.state_dict()
            state["predictor_u_norm"] = self.predictor_u_norm.state_dict()
            state["sysid_optimizer"] = self.sysid_trainer.optimizer.state_dict()
        
        # Save dynamics map encoder if enabled and not frozen
        if self.context_mode == "dynamics_map":
            state["map_encoder"] = self.map_encoder.state_dict()
            if not self.cfg.dynamics_map.freeze_encoder:
                state["encoder_optimizer"] = self.encoder_optimizer.state_dict()
        
        self.cfg.output.dir.mkdir(parents=True, exist_ok=True)
        
        # Use custom path if provided, otherwise use default naming
        if checkpoint_path is not None:
            ckpt_path = checkpoint_path
        else:
            ckpt_path = self.cfg.output.dir / f"sac_step_{step}.pt"
        
        torch.save(state, ckpt_path)
        if self.cfg.output.save_latest:
            torch.save(state, self.cfg.output.dir / "latest.pt")

        # Checkpoint saved silently (no print to avoid interrupting progress bar)


def train(config: ConfigBundle, disable_clearml: bool = False) -> None:
    accelerator = Accelerator()
    accelerator.init_trackers(
        project_name="sac_longitudinal",
        config={
            "seed": config.seed,
            "num_train_timesteps": config.training.num_train_timesteps,
        },
    )
    
    # Initialize ClearML task
    clearml_task = None
    if not disable_clearml and CLEARML_LOGGER_AVAILABLE and init_clearml_task is not None and clearml is not None:
        clearml_task = init_clearml_task(
            task_name="SAC_Training",
            project="RL_Longitudinal",
            tags=["training", "sac"],
            task_type=clearml.Task.TaskTypes.training,
        )
        if clearml_task:
            # Log full config as hyperparameters
            config_dict = {
                "seed": config.seed,
                "context_mode": config.context_mode,
                "env": asdict(config.env),
                "training": asdict(config.training),
                "sysid": asdict(config.sysid) if config.sysid else None,
                "dynamics_map": asdict(config.dynamics_map) if config.dynamics_map else None,
                "output_dir": str(config.output.dir),
                "fitted_params_path": config.fitted_params_path,
                "fitted_spread_pct": config.fitted_spread_pct,
            }
            log_config(clearml_task, config_dict)

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

    # Generate example plot for violent profile mode
    if config.env.violent_profile_mode:
        accelerator.print("[violent_profile] Generating example plot...")
        try:
            from generator.lpf import SecondOrderLPF
            import matplotlib
            # Try to use interactive backend if available, otherwise fall back to Agg
            # Note: matplotlib.use() must be called before importing pyplot
            backend_set = False
            interactive_backend = None
            for backend in ['TkAgg', 'Qt5Agg', 'Qt4Agg']:
                try:
                    matplotlib.use(backend)
                    backend_set = True
                    interactive_backend = backend
                    accelerator.print(f"[violent_profile] Successfully set interactive backend: {backend}")
                    break
                except Exception as e:
                    accelerator.print(f"[violent_profile] Failed to set backend {backend}: {e}")
                    continue
            if not backend_set:
                matplotlib.use('Agg')  # Fall back to non-interactive
                accelerator.print("[violent_profile] WARNING: Using non-interactive backend (Agg) - plot will be saved but not displayed")
                accelerator.print("[violent_profile] To display plots, install: sudo apt-get install python3-tk (for TkAgg)")
            import matplotlib.pyplot as plt
            
            # Enable interactive mode if we have an interactive backend
            if backend_set:
                plt.ion()  # Turn on interactive mode
            
            # Generate a sample profile using the environment's generator
            profile_length = 800  # 4x longer for better visualization
            filtered_profile, grade_profile, raw_profile = env._generate_reference(profile_length)
            
            # Apply reward filter to raw profile (as done in reset)
            if env._reward_filter is not None:
                # torch already imported at module level
                raw_tensor = torch.from_numpy(raw_profile).unsqueeze(0)  # [1, T]
                filtered_by_reward_filter = torch.zeros_like(raw_tensor)
                
                # Reset filter state with initial value from raw profile
                initial_y = torch.tensor([[raw_profile[0]]], device=torch.device('cpu'), dtype=torch.float32)
                env._reward_filter.reset(initial_y=initial_y)
                
                # Process each timestep through the filter
                for t in range(len(raw_profile)):
                    u_t = raw_tensor[:, t:t+1]  # [1, 1]
                    filtered_y = env._reward_filter.update(u_t)
                    filtered_by_reward_filter[:, t] = filtered_y.squeeze(0)
                
                reward_filtered_profile = filtered_by_reward_filter.squeeze(0).cpu().numpy().astype(np.float32)
            else:
                reward_filtered_profile = raw_profile.copy()
            
            # Create time axis
            dt = config.env.dt
            time_axis = np.arange(profile_length) * dt
            
            # Clamp filtered profile to non-negative for visualization
            reward_filtered_profile_clamped = np.maximum(reward_filtered_profile, 0.0)
            
            # Create plot
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Top plot: Raw profile vs Reward filter profile
            axes[0].plot(time_axis, raw_profile, label="Raw Profile (RL observations in violent mode)", 
                         color="#d62728", linewidth=2.5, linestyle="-", alpha=0.9)
            axes[0].plot(time_axis, reward_filtered_profile_clamped, label="Reward Filtered Profile (for reward computation)", 
                         color="#2ca02c", linewidth=2, linestyle="--", alpha=0.9)
            axes[0].plot(time_axis, filtered_profile, label="Generator Filtered Profile (normal mode)", 
                         color="#1f77b4", linewidth=1.5, linestyle=":", alpha=0.7)
            axes[0].set_ylabel("Speed (m/s)", fontsize=12)
            axes[0].set_title("Training Initialization: Profiles Generated During Reset (Violent Mode)", fontsize=14, fontweight="bold")
            axes[0].legend(loc="upper right", fontsize=10)
            axes[0].grid(alpha=0.3, linestyle=":")
            axes[0].set_ylim([-1, max(25, raw_profile.max() * 1.1)])
            
            # Bottom plot: Show the difference/smoothing effect
            speed_diff = reward_filtered_profile_clamped - raw_profile
            axes[1].plot(time_axis, speed_diff, label="Reward Filtered - Raw (smoothing effect)", 
                         color="#9467bd", linewidth=2, linestyle="-")
            axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
            axes[1].set_xlabel("Time (s)", fontsize=12)
            axes[1].set_ylabel("Speed Difference (m/s)", fontsize=12)
            axes[1].set_title("Smoothing Effect: How the Reward Filter Smooths the Raw Profile", fontsize=14, fontweight="bold")
            axes[1].legend(loc="upper right", fontsize=10)
            axes[1].grid(alpha=0.3, linestyle=":")
            
            # Add text annotation explaining what happens during training
            fig.text(0.5, 0.02, 
                     "During training initialization: Generator creates raw_profile (discontinuous). "
                     "In violent mode: RL model receives raw_profile in observations, "
                     "but reward is computed against reward-filtered (smooth) version of raw_profile.",
                     ha="center", fontsize=10, style="italic")
            
            fig.suptitle("Violent Profile Mode: Training Initialization Example", fontsize=16, fontweight="bold", y=0.98)
            fig.tight_layout(rect=(0, 0.05, 1, 0.97))
            
            # Save plot to output directory
            output_path = config.output.dir / "violent_profile_example.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            
            # Show the plot
            if backend_set:
                # Interactive backend available - display the plot
                accelerator.print(f"[violent_profile] Displaying plot window (close window to continue training)...")
                plt.show(block=True)  # Block until window is closed
                accelerator.print(f"[violent_profile] Plot window closed, continuing training...")
            else:
                # Non-interactive backend - can't display, just inform user
                accelerator.print(f"[violent_profile] Non-interactive backend - plot saved to {output_path}")
                accelerator.print(f"[violent_profile] To view: open {output_path}")
                plt.close(fig)
            
            accelerator.print(f"[violent_profile] Example plot saved to {output_path}")
            accelerator.print(f"[violent_profile] Raw profile: min={raw_profile.min():.2f} m/s, max={raw_profile.max():.2f} m/s, std={raw_profile.std():.2f} m/s")
            accelerator.print(f"[violent_profile] Reward filtered: min={reward_filtered_profile.min():.2f} m/s, max={reward_filtered_profile.max():.2f} m/s, std={reward_filtered_profile.std():.2f} m/s")
        except Exception as e:
            accelerator.print(f"[violent_profile] Warning: Failed to generate example plot: {e}")
            import traceback
            accelerator.print(traceback.format_exc())

    trainer = SACTrainer(env, eval_env, config, accelerator)

    # Initialize animation if enabled
    animation_history = []
    fixed_reference_for_animation = None
    if config.training.enable_animation:
        from training.training_animation import load_animation_history, create_training_animation
        animation_output_dir = config.output.dir / "animations"
        
        # Check if checkpoint folder exists (indicates previous training run)
        checkpoint_dir_exists = False
        if config.output.dir.exists():
            # Check for checkpoint files (e.g., sac_step_*.pt)
            checkpoint_files = list(config.output.dir.glob("sac_step_*.pt")) + list(config.output.dir.glob("checkpoint_*.pt"))
            checkpoint_dir_exists = len(checkpoint_files) > 0
        
        if checkpoint_dir_exists:
            # Start with clean animation history if checkpoint folder exists
            accelerator.print("[animation] Checkpoint folder detected - starting with clean animation history")
            animation_history = []
        else:
            # Load history only if no checkpoint folder exists (fresh start)
            animation_history = load_animation_history(animation_output_dir)
        
        # Generate a fixed reference profile for consistent evaluation
        # Use the same generator as training to ensure consistency and respect vehicle capabilities
        accelerator.print("[animation] Generating fixed reference profile for animation...")
        profile_length = config.env.max_episode_steps
        
        # Use eval_env's reference generator (same as training) with a fixed seed for reproducibility
        # This ensures the profile respects vehicle capabilities and uses the same filtering/feasibility constraints
        animation_seed = config.seed + 9999  # Use a different seed from training but fixed for consistency
        eval_env._rng = np.random.default_rng(animation_seed)
        fixed_reference_for_animation, _, _ = eval_env._generate_reference(profile_length)
        
        reference_initial_speed = float(fixed_reference_for_animation[0])
        reference_max_speed = float(np.max(fixed_reference_for_animation))
        reference_min_speed = float(np.min(fixed_reference_for_animation))
        accelerator.print(f"[animation] Fixed reference profile generated: length={len(fixed_reference_for_animation)}")
        accelerator.print(f"[animation] Speed range: {reference_min_speed:.2f} - {reference_max_speed:.2f} m/s")
        accelerator.print(f"[animation] Reference initial speed: {reference_initial_speed:.2f} m/s")
        
        # Generate initial speed errors for episodes (configurable via training.num_animation_episodes)
        num_animation_episodes = getattr(config.training, 'num_animation_episodes', 3)
        # Default to -2, 0, 2 m/s errors for 3 episodes
        if num_animation_episodes == 1:
            initial_speed_errors_for_animation = [0.0]
        elif num_animation_episodes == 2:
            initial_speed_errors_for_animation = [-1.0, 1.0]
        elif num_animation_episodes == 3:
            initial_speed_errors_for_animation = [-2.0, 0.0, 2.0]
        else:
            # For more episodes, distribute evenly from -2 to 2
            initial_speed_errors_for_animation = np.linspace(-2.0, 2.0, num_animation_episodes).tolist()
        
        # Generate fixed initial actions for each episode (constant across training)
        # Use a fixed seed to ensure reproducibility
        action_rng = np.random.default_rng(animation_seed + 1000)
        initial_actions_for_animation = []
        for _ in range(num_animation_episodes):
            # Sample initial actions uniformly from action space
            initial_action = action_rng.uniform(config.env.action_low, config.env.action_high)
            initial_actions_for_animation.append(float(initial_action))
        accelerator.print(f"[animation] Initial actions: {initial_actions_for_animation}")

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
    
    # Track last good checkpoint for NaN recovery
    last_good_checkpoint_step = start_step
    
    for step in progress:
        obs, reward = trainer.collect_step(obs)

        if trainer.replay_buffer.size >= config.training.batch_size:
            # SAC update
            metrics = trainer.update()
            
            # Check for NaN in networks after update
            if trainer._check_for_nans(step):
                accelerator.print(f"[ERROR] NaN detected at step {step}! Training cannot continue safely.")
                accelerator.print(f"[ERROR] Last good checkpoint was at step {last_good_checkpoint_step}")
                accelerator.print(f"[ERROR] Consider:")
                accelerator.print(f"  1. Resuming from checkpoint: training/checkpoints_dynamics_map/sac_step_{last_good_checkpoint_step}.pt")
                accelerator.print(f"  2. Reducing learning rate (current: {config.training.learning_rate})")
                accelerator.print(f"  3. Increasing reward scaling (current: {config.training.reward_scale})")
                accelerator.print(f"  4. Checking dynamics map context for extreme values")
                
                # Save emergency checkpoint for debugging
                emergency_path = config.output.dir / f"sac_step_{step}_NaN_ERROR.pt"
                try:
                    trainer.save_checkpoint(step, checkpoint_path=emergency_path)
                    accelerator.print(f"[ERROR] Saved emergency checkpoint to {emergency_path} for debugging")
                except Exception as e:
                    accelerator.print(f"[ERROR] Failed to save emergency checkpoint: {e}")
                
                # Exit training
                progress.close()
                accelerator.end_training()
                raise RuntimeError(f"NaN detected in network parameters at step {step}")
            
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
                # Also log to ClearML
                if clearml_task:
                    log_metrics(clearml_task, metrics, step)

        eval_metrics = None
        if step % config.training.eval_interval == 0:
            eval_metrics = trainer.evaluate()
            accelerator.log(eval_metrics, step=step)
            # Also log eval metrics to ClearML
            if clearml_task:
                log_metrics(clearml_task, eval_metrics, step)
        
        # Create animation if enabled (independent of evaluation interval)
        if config.training.enable_animation and fixed_reference_for_animation is not None:
            if step % config.training.animation_interval == 0:
                num_animation_episodes = getattr(config.training, 'num_animation_episodes', 3)
                create_training_animation(
                    trainer=trainer,
                    eval_env=eval_env,
                    fixed_reference=fixed_reference_for_animation,
                    output_dir=config.output.dir / "animations",
                    step=step,
                    animation_history=animation_history,
                    max_history=50,
                    initial_speed_errors=initial_speed_errors_for_animation,
                    initial_actions=initial_actions_for_animation,
                    num_episodes=num_animation_episodes
                )
                # Note: Animation plots are not uploaded to ClearML during training to avoid slowdown

        if step % config.training.checkpoint_interval == 0:
            trainer.save_checkpoint(step)
            # Update last good checkpoint tracker after successful save
            last_good_checkpoint_step = step

        if accelerator.is_main_process and step % config.training.log_interval == 0:
            now = time.time()
            fps = config.training.log_interval / max(now - last_log, 1e-6)
            
            # Update progress bar with all relevant info
            postfix_dict = {
                'reward': f"{reward:.2f}",
                'buffer': trainer.replay_buffer.size,
                'fps': f"{fps:.1f}",
            }
            
            # Add eval metrics if available
            if eval_metrics:
                if 'eval/avg_reward' in eval_metrics:
                    postfix_dict['eval_reward'] = f"{eval_metrics['eval/avg_reward']:.2f}"
                if 'eval/avg_abs_speed_error' in eval_metrics:
                    postfix_dict['eval_error'] = f"{eval_metrics['eval/avg_abs_speed_error']:.3f}"
            
            # Add sysid loss if available
            if config.sysid.enabled and "sysid/pred_loss" in metrics:
                postfix_dict['sysid_loss'] = f"{metrics['sysid/pred_loss']:.4f}"
            
            progress.set_postfix(**postfix_dict, refresh=True)
            last_log = now

    trainer.save_checkpoint(config.training.num_train_timesteps)
    
    # Upload final checkpoint to ClearML
    if clearml_task:
        final_checkpoint_path = config.output.dir / f"sac_step_{config.training.num_train_timesteps}.pt"
        if final_checkpoint_path.exists():
            upload_artifact(
                clearml_task,
                final_checkpoint_path,
                artifact_name="final_checkpoint",
                description=f"Final checkpoint after {config.training.num_train_timesteps} training steps",
            )
        close_task(clearml_task)
    
    progress.close()
    accelerator.end_training()
    accelerator.print("[done] training complete")


def main() -> None:
    args = parse_args()
    raw = load_config(args.config)
    bundle = build_config_bundle(raw, args)
    train(bundle, disable_clearml=args.disable_clearml)


if __name__ == "__main__":
    main()


