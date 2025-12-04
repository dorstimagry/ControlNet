"""Helpers for loading SAC policies for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from env.longitudinal_env import LongitudinalEnvConfig
from training.train_sac import GaussianPolicy


def load_policy_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> Tuple[GaussianPolicy, LongitudinalEnvConfig, int]:
    """Load a trained GaussianPolicy and corresponding env config."""

    device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_block = checkpoint["config"]["env"]
    env_cfg = LongitudinalEnvConfig(**config_block)
    meta = checkpoint.get("meta", {})
    obs_dim = int(meta.get("obs_dim", 4))
    env_action_dim = int(meta.get("env_action_dim", 1))
    policy_action_dim = int(meta.get("policy_action_dim", env_action_dim))
    horizon = max(1, policy_action_dim // max(1, env_action_dim))

    action_low = np.tile(np.array([env_cfg.action_low], dtype=np.float32), horizon)
    action_high = np.tile(np.array([env_cfg.action_high], dtype=np.float32), horizon)

    policy = GaussianPolicy(
        obs_dim=obs_dim,
        action_dim=policy_action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    policy.load_state_dict(checkpoint["policy"])
    policy.to(device)
    policy.eval()
    return policy, env_cfg, horizon


def select_action(
    policy: GaussianPolicy,
    obs: torch.Tensor,
    deterministic: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the action vector along with policy statistics."""

    with torch.no_grad():
        mu, log_std = policy(obs)
        if deterministic:
            action = torch.tanh(mu) * policy.action_scale + policy.action_bias
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mu, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * policy.action_scale + policy.action_bias

    plan = action.squeeze(0).detach().cpu().numpy()
    mu_np = mu.squeeze().detach().cpu().numpy()
    log_std_np = log_std.squeeze().detach().cpu().numpy()
    first_mean = float(mu_np if np.isscalar(mu_np) else mu_np.flat[0])
    first_log_std = float(log_std_np if np.isscalar(log_std_np) else log_std_np.flat[0])
    stats = {
        "pre_tanh_mean": first_mean,
        "log_std": first_log_std,
        "plan": plan.tolist(),
    }
    return plan, stats


__all__ = ["load_policy_from_checkpoint", "select_action"]


