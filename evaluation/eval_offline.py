#!/usr/bin/env python3
"""Offline evaluation for SAC longitudinal policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_from_disk

from env.longitudinal_env import LongitudinalEnv
from evaluation.policy_loader import load_policy_from_checkpoint, select_action
from evaluation.plotting import plot_sequence_diagnostics, plot_summary
from utils.data_utils import SyntheticTrajectoryConfig, build_synthetic_reference_dataset
from utils.dynamics import RandomizationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation using logged reference-speed sequences.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to SAC checkpoint (.pt).")
    parser.add_argument("--dataset", type=str, default=None, help="Path to HuggingFace dataset saved via save_to_disk.")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/offline_metrics.json"))
    parser.add_argument("--max-samples", type=int, default=16, help="Number of sequences to evaluate.")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional directory to save per-sequence and summary plots.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override (cpu or cuda).")
    return parser.parse_args()


def _prepare_dataset(dataset_path: Optional[str], max_samples: Optional[int]) -> Dataset:
    if dataset_path:
        dataset = load_from_disk(dataset_path)
    else:
        config = SyntheticTrajectoryConfig(num_sequences=max_samples or 16, sequence_length=256)
        dataset = build_synthetic_reference_dataset(config)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    if "reference_speed" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'reference_speed' column with speed sequences.")
    return dataset


def _evaluate_sequence(
    env: LongitudinalEnv,
    policy,
    reference: np.ndarray,
    device: torch.device,
    sequence_id: str,
) -> tuple[dict[str, float], dict[str, object]]:
    options = {"reference_profile": reference.tolist(), "initial_speed": float(reference[0])}
    obs, _ = env.reset(options=options)
    done = False
    speeds: list[float] = []
    references: list[float] = []
    actions: list[float] = []
    time_axis: list[float] = []
    accelerations: list[float] = []
    commanded: list[float] = []
    policy_means: list[float] = []
    policy_log_stds: list[float] = []
    policy_plans: list[list[float]] = []
    wheel_speeds: list[float] = []
    throttle_angles: list[float] = []
    brake_torques: list[float] = []
    motor_voltages: list[float] = []
    motor_currents: list[float] = []
    slip_ratios: list[float] = []
    jerks: list[float] = []
    step_idx = 0

    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        plan, stats = select_action(policy, obs_tensor, deterministic=True)
        action_value = float(plan[0])
        obs, reward, terminated, truncated, info = env.step(action_value)
        time_axis.append(step_idx * env.config.dt)
        speeds.append(float(info.get("speed", env.speed)))
        references.append(float(info.get("reference_speed", reference[min(len(reference) - 1, len(references))])))
        actions.append(action_value)
        accelerations.append(float(info.get("acceleration", 0.0)))
        commanded.append(float(info.get("commanded_accel", 0.0)))
        wheel_speeds.append(float(info.get("wheel_speed", 0.0)))
        throttle_angles.append(float(info.get("throttle_angle", 0.0)))
        brake_torques.append(float(info.get("brake_torque", 0.0)))
        motor_voltages.append(float(info.get("motor_voltage", 0.0)))
        motor_currents.append(float(info.get("motor_current", 0.0)))
        slip_ratios.append(float(info.get("slip_ratio", 0.0)))
        jerks.append(float(info.get("jerk", 0.0)))
        policy_means.append(stats["pre_tanh_mean"])
        policy_log_stds.append(stats["log_std"])
        policy_plans.append(stats.get("plan", [action_value]))
        done = terminated or truncated
        step_idx += 1

    speeds_np = np.asarray(speeds)
    ref_np = np.asarray(references[: speeds_np.shape[0]])
    errors = speeds_np - ref_np
    rmse = float(np.sqrt(np.mean(errors**2)))
    smoothness = float(np.var(np.diff(actions)) if len(actions) > 1 else 0.0)
    threshold = 0.9 * max(abs(env.action_space.low[0]), abs(env.action_space.high[0]))
    extreme_fraction = float(np.mean(np.abs(actions) > threshold)) if actions else 0.0
    action_variance = float(np.var(actions)) if len(actions) > 1 else 0.0

    metrics = {
        "sequence_id": sequence_id,
        "rmse": rmse,
        "smoothness": smoothness,
        "extreme_fraction": extreme_fraction,
        "action_variance": action_variance,
    }

    trace = {
        "sequence_id": sequence_id,
        "time": time_axis,
        "speed": speeds,
        "reference": references,
        "acceleration": accelerations,
        "commanded_accel": commanded,
        "action": actions,
        "policy_pre_tanh_mean": policy_means,
        "policy_log_std": policy_log_stds,
        "policy_plan": policy_plans,
        "wheel_speed": wheel_speeds,
        "throttle_angle": throttle_angles,
        "brake_torque": brake_torques,
        "motor_voltage": motor_voltages,
        "motor_current": motor_currents,
        "slip_ratio": slip_ratios,
        "jerk": jerks,
    }
    return metrics, trace


def run_offline_evaluation(
    checkpoint: Path,
    dataset_path: Optional[str],
    output_path: Optional[Path],
    max_samples: Optional[int],
    plot_dir: Optional[Path] = None,
    device_str: Optional[str] = None,
) -> dict[str, float]:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy, env_cfg, _ = load_policy_from_checkpoint(checkpoint, device=device)
    dataset = _prepare_dataset(dataset_path, max_samples)
    env = LongitudinalEnv(env_cfg, randomization=RandomizationConfig(), seed=1234)

    per_sequence = []
    traces = []
    for idx, item in enumerate(dataset):
        reference = np.asarray(item["reference_speed"], dtype=np.float32)
        seq_id = f"offline_{idx:03d}"
        metrics, trace = _evaluate_sequence(env, policy, reference, device, seq_id)
        per_sequence.append(metrics)
        traces.append(trace)

    summary = {
        "count": len(per_sequence),
        "rmse_mean": float(np.mean([m["rmse"] for m in per_sequence])),
        "smoothness_mean": float(np.mean([m["smoothness"] for m in per_sequence])),
        "extreme_fraction_mean": float(np.mean([m["extreme_fraction"] for m in per_sequence])),
        "action_variance_mean": float(np.mean([m["action_variance"] for m in per_sequence])),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))

    if plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)
        for trace in traces:
            seq_path = plot_dir / f"{trace['sequence_id']}.png"
            plot_sequence_diagnostics(trace, seq_path)
        plot_summary(per_sequence, plot_dir / "offline_summary.png")

    return summary


def main() -> None:
    args = parse_args()
    summary = run_offline_evaluation(
        checkpoint=args.checkpoint,
        dataset_path=args.dataset,
        output_path=args.output,
        max_samples=args.max_samples,
        plot_dir=args.plot_dir,
        device_str=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


