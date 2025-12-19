"""Unit tests for the SAC training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from env.longitudinal_env import LongitudinalEnvConfig
from training.train_sac import (
    ConfigBundle,
    OutputConfig,
    TrainingParams,
    build_config_bundle,
    load_config,
    train,
)


def test_build_config_bundle_respects_overrides(tmp_path: Path) -> None:
    """CLI overrides should take precedence over YAML config values."""

    config_path = tmp_path / "config.yaml"
    config_payload = {
        "seed": 1,
        "env": {"dt": 0.1},
        "training": {"num_train_timesteps": 100},
        "output": {"dir": "foo", "save_latest": False},
    }
    config_path.write_text(yaml.safe_dump(config_payload))

    raw = load_config(config_path)
    args = argparse.Namespace(
        config=config_path,
        num_train_timesteps=123,
        seed=99,
        output_dir=tmp_path / "out",
        reference_dataset="hf://dataset",
        fitted_params=None,
        fitted_spread=0.1,
    )
    bundle = build_config_bundle(raw, args)

    assert bundle.seed == 99
    assert bundle.training.num_train_timesteps == 123
    assert bundle.output.dir == args.output_dir
    assert bundle.reference_dataset == "hf://dataset"


def test_train_executes_short_run(tmp_path: Path) -> None:
    """Running the trainer with a tiny budget should produce a checkpoint."""

    env_cfg = LongitudinalEnvConfig(max_episode_steps=32)
    training_cfg = TrainingParams(
        num_train_timesteps=20,
        learning_rate=3e-4,
        batch_size=8,
        gamma=0.99,
        tau=0.01,
        replay_size=256,
        warmup_steps=5,
        eval_interval=10,
        eval_episodes=1,
        log_interval=10,
        checkpoint_interval=50,
        max_grad_norm=5.0,
        target_entropy_scale=0.5,
    )
    output_cfg = OutputConfig(dir=tmp_path / "ckpts", save_latest=True)
    bundle = ConfigBundle(
        seed=0,
        env=env_cfg,
        training=training_cfg,
        output=output_cfg,
        reference_dataset=None,
    )

    train(bundle)

    latest_ckpt = output_cfg.dir / "latest.pt"
    assert latest_ckpt.exists()


def test_train_with_action_horizon(tmp_path: Path) -> None:
    """Trainer should handle multi-step action horizons."""

    env_cfg = LongitudinalEnvConfig(max_episode_steps=24)
    training_cfg = TrainingParams(
        num_train_timesteps=15,
        learning_rate=3e-4,
        batch_size=8,
        gamma=0.99,
        tau=0.02,
        replay_size=128,
        warmup_steps=4,
        eval_interval=10,
        eval_episodes=1,
        log_interval=10,
        checkpoint_interval=50,
        max_grad_norm=5.0,
        target_entropy_scale=0.5,
        action_horizon_steps=3,
    )
    output_cfg = OutputConfig(dir=tmp_path / "ckpts_multi", save_latest=True)
    bundle = ConfigBundle(
        seed=5,
        env=env_cfg,
        training=training_cfg,
        output=output_cfg,
        reference_dataset=None,
    )

    train(bundle)

    latest_ckpt = output_cfg.dir / "latest.pt"
    assert latest_ckpt.exists()

