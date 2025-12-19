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
class OutputConfig:
    dir: Path = Path("training/checkpoints")
    save_latest: bool = True


@dataclass(slots=True)
class ConfigBundle:
    seed: int
    env: LongitudinalEnvConfig
    training: TrainingParams
    output: OutputConfig
    reference_dataset: str | None = None
    # Fitted vehicle randomization config
    fitted_params_path: str | None = None
    fitted_spread_pct: float = 0.1
    generator_config: Dict[str, Any] | None = None


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

    return ConfigBundle(
        seed=seed,
        env=env_cfg,
        training=training_cfg,
        output=output,
        reference_dataset=reference_dataset,
        fitted_params_path=fitted_params_path,
        fitted_spread_pct=fitted_spread_pct,
        generator_config=generator_config,
    )


class ReplayBuffer:
    """Simple numpy-backed replay buffer."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = (
            torch.as_tensor(self.obs_buf[idxs], device=device),
            torch.as_tensor(self.action_buf[idxs], device=device),
            torch.as_tensor(self.reward_buf[idxs], device=device),
            torch.as_tensor(self.next_obs_buf[idxs], device=device),
            torch.as_tensor(self.done_buf[idxs], device=device),
        )
        return batch


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
    """Gaussian policy with Tanh squashing and action-range scaling."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.net = mlp(obs_dim, hidden_dim, hidden_dim=hidden_dim, depth=2)
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
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, 1, hidden_dim=hidden_dim, depth=2)

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

        self.replay_buffer = ReplayBuffer(obs_dim, self.policy_action_dim, cfg.training.replay_size)
        self.policy = GaussianPolicy(obs_dim, self.policy_action_dim, action_low, action_high)
        self.q1 = QNetwork(obs_dim, self.policy_action_dim)
        self.q2 = QNetwork(obs_dim, self.policy_action_dim)
        self.target_q1 = QNetwork(obs_dim, self.policy_action_dim)
        self.target_q2 = QNetwork(obs_dim, self.policy_action_dim)
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
        if self._global_step < self.cfg.training.warmup_steps:
            env_action = self.env.action_space.sample()
            action_vec = np.tile(env_action, self.action_horizon).astype(np.float32)
            action_scalar = float(env_action[0])
        else:
            action_vec, action_scalar = self.act(obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action_scalar)
        done = terminated or truncated
        self.replay_buffer.add(obs, action_vec, reward, next_obs, done)
        self._episode_reward += reward
        self._episode_length += 1
        self._global_step += 1
        if done:
            obs, _ = self.env.reset()
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

    def evaluate(self) -> Dict[str, float]:
        rewards = []
        speed_errors = []
        for _ in range(self.cfg.training.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                _, action_scalar = self.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action_scalar)
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
                "seed": self.cfg.seed,
                "num_train_timesteps": self.cfg.training.num_train_timesteps,
            },
            "meta": {
                "obs_dim": self.env.observation_space.shape[0],
                "env_action_dim": self.env_action_dim,
                "policy_action_dim": self.policy_action_dim,
            },
        }
        self.cfg.output.dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.cfg.output.dir / f"sac_step_{step}.pt"
        torch.save(state, ckpt_path)
        if self.cfg.output.save_latest:
            torch.save(state, self.cfg.output.dir / "latest.pt")
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

    obs, _ = env.reset(seed=config.seed)
    last_log = time.time()

    accelerator.print(f"[setup] num_train_timesteps={config.training.num_train_timesteps}")
    progress = tqdm(
        range(1, config.training.num_train_timesteps + 1),
        disable=not accelerator.is_main_process,
        desc="Training",
    )
    for step in progress:
        obs, reward = trainer.collect_step(obs)

        if trainer.replay_buffer.size >= config.training.batch_size:
            metrics = trainer.update()
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
            accelerator.print(
                f"[train] step={step} reward={reward:.3f} buffer={trainer.replay_buffer.size} fps={fps:.1f}"
            )
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


