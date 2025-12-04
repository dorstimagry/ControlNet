"""Tests for offline and closed-loop evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import pytest

from env.longitudinal_env import LongitudinalEnvConfig
from evaluation.eval_closed_loop import run_closed_loop_evaluation
from evaluation.eval_offline import run_offline_evaluation
from training.train_sac import ConfigBundle, OutputConfig, TrainingParams, train
from utils.data_utils import SyntheticTrajectoryConfig, build_synthetic_reference_dataset


@pytest.fixture(scope="module")
def small_checkpoint(tmp_path_factory) -> Path:
    workdir = tmp_path_factory.mktemp("ckpt")
    env_cfg = LongitudinalEnvConfig(max_episode_steps=24)
    training_cfg = TrainingParams(
        num_train_timesteps=15,
        learning_rate=3e-4,
        batch_size=8,
        gamma=0.99,
        tau=0.02,
        replay_size=128,
        warmup_steps=5,
        eval_interval=10,
        eval_episodes=1,
        log_interval=10,
        checkpoint_interval=50,
        max_grad_norm=5.0,
        target_entropy_scale=0.5,
    )
    output_cfg = OutputConfig(dir=workdir / "ckpts", save_latest=True)
    bundle = ConfigBundle(
        seed=123,
        env=env_cfg,
        training=training_cfg,
        output=output_cfg,
        reference_dataset=None,
    )
    train(bundle)
    return output_cfg.dir / "latest.pt"


def test_run_offline_evaluation_outputs_metrics(tmp_path: Path, small_checkpoint: Path) -> None:
    dataset = build_synthetic_reference_dataset(SyntheticTrajectoryConfig(num_sequences=3, sequence_length=64))
    dataset_dir = tmp_path / "hf_dataset"
    dataset.save_to_disk(dataset_dir)

    output_path = tmp_path / "offline.json"
    plot_dir = tmp_path / "offline_plots"
    summary = run_offline_evaluation(
        checkpoint=small_checkpoint,
        dataset_path=str(dataset_dir),
        output_path=output_path,
        max_samples=2,
        plot_dir=plot_dir,
        device_str="cpu",
    )
    assert output_path.exists()
    assert "rmse_mean" in summary
    assert summary["count"] == 2
    assert (plot_dir / "offline_000.png").exists()
    assert (plot_dir / "offline_summary.png").exists()


def test_run_closed_loop_evaluation_outputs_metrics(tmp_path: Path, small_checkpoint: Path) -> None:
    output_path = tmp_path / "closed_loop.json"
    plot_dir = tmp_path / "closed_plots"
    summary = run_closed_loop_evaluation(
        checkpoint=small_checkpoint,
        episodes=2,
        output_path=output_path,
        plot_dir=plot_dir,
        device_str="cpu",
    )
    assert output_path.exists()
    assert "reward_mean" in summary
    assert summary["episodes"] == 2
    assert (plot_dir / "episode_000.png").exists()
    assert (plot_dir / "closed_loop_summary.png").exists()


