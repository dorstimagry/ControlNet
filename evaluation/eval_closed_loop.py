#!/usr/bin/env python3
"""Closed-loop evaluation for SAC longitudinal policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from env.longitudinal_env import LongitudinalEnv
from evaluation.policy_loader import load_policy_from_checkpoint, select_action
from evaluation.plotting import plot_sequence_diagnostics, plot_summary
from utils.dynamics import RandomizationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop evaluation in the longitudinal environment.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/closed_loop_metrics.json"))
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional directory to save per-episode plots.",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def run_closed_loop_evaluation(
    checkpoint: Path,
    episodes: int,
    output_path: Optional[Path],
    plot_dir: Optional[Path] = None,
    device_str: Optional[str] = None,
) -> dict[str, float]:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy, env_cfg, _ = load_policy_from_checkpoint(checkpoint, device=device)
    env = LongitudinalEnv(env_cfg, randomization=RandomizationConfig(), seed=4321)

    episode_rewards = []
    avg_errors = []
    action_variances = []
    traces = []
    per_episode_metrics = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        reward_total = 0.0
        errors: list[float] = []
        actions: list[float] = []
        accelerations: list[float] = []
        jerks: list[float] = []
        wheel_speeds: list[float] = []
        V_cmds: list[float] = []
        V_maxes: list[float] = []
        brake_torques: list[float] = []
        brake_torque_maxes: list[float] = []
        motor_voltages: list[float] = []
        motor_currents: list[float] = []
        slip_ratios: list[float] = []
        policy_means: list[float] = []
        policy_log_stds: list[float] = []
        policy_plans: list[list[float]] = []
        speeds: list[float] = []
        references: list[float] = []
        time_axis: list[float] = []
        # Force data
        drive_torques: list[float] = []
        tire_forces: list[float] = []
        drag_forces: list[float] = []
        rolling_forces: list[float] = []
        grade_forces: list[float] = []
        net_forces: list[float] = []
        step_idx = 0

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            plan, stats = select_action(policy, obs_tensor, deterministic=True)
            action_value = float(plan[0])
            obs, reward, terminated, truncated, info = env.step(action_value)
            reward_total += reward
            errors.append(abs(info.get("speed_error", 0.0)))
            actions.append(action_value)
            accelerations.append(float(info.get("acceleration", 0.0)))
            jerks.append(float(info.get("jerk", 0.0)))
            wheel_speeds.append(float(info.get("wheel_speed", 0.0)))
            V_cmds.append(float(info.get("V_cmd", 0.0)))
            V_maxes.append(float(info.get("V_max", 800.0)))  # Default fallback
            brake_torques.append(float(info.get("brake_torque", 0.0)))
            brake_torque_maxes.append(float(info.get("brake_torque_max", 10000.0)))  # Default fallback
            motor_voltages.append(float(info.get("motor_voltage", 0.0)))
            motor_currents.append(float(info.get("motor_current", 0.0)))
            slip_ratios.append(float(info.get("slip_ratio", 0.0)))
            policy_means.append(stats["pre_tanh_mean"])
            policy_log_stds.append(stats["log_std"])
            policy_plans.append(stats.get("plan", [action_value]))
            speeds.append(float(info.get("speed", 0.0)))
            references.append(float(info.get("reference_speed", 0.0)))
            time_axis.append(step_idx * env.config.dt)
            # Collect force data
            drive_torques.append(float(info.get("drive_torque", 0.0)))
            tire_forces.append(float(info.get("tire_force", 0.0)))
            drag_forces.append(float(info.get("drag_force", 0.0)))
            rolling_forces.append(float(info.get("rolling_force", 0.0)))
            grade_forces.append(float(info.get("grade_force", 0.0)))
            net_forces.append(float(info.get("net_force", 0.0)))
            done = terminated or truncated
            step_idx += 1

        episode_rewards.append(reward_total)
        avg_errors.append(float(np.mean(errors)) if errors else 0.0)
        diffs = np.diff(actions)
        action_variances.append(float(np.var(diffs)) if len(diffs) else 0.0)

        episode_id = f"episode_{episode:03d}"
        per_episode_metrics.append(
            {
                "sequence_id": episode_id,
                "rmse": float(np.sqrt(np.mean((np.asarray(speeds) - np.asarray(references)) ** 2)))
                if speeds
                else 0.0,
                "smoothness": float(np.var(np.diff(actions)) if len(actions) > 1 else 0.0),
                "extreme_fraction": float(
                    np.mean(
                        np.abs(actions)
                        > 0.9 * max(abs(env.action_space.low[0]), abs(env.action_space.high[0]))
                    )
                )
                if actions
                else 0.0,
                "action_variance": float(np.var(actions)) if len(actions) > 1 else 0.0,
            }
        )
        traces.append(
            {
                "sequence_id": episode_id,
                "time": time_axis,
                "speed": speeds,
                "reference": references,
                "acceleration": accelerations,
                "action": actions,
                "policy_pre_tanh_mean": policy_means,
                "policy_log_std": policy_log_stds,
                "policy_plan": policy_plans,
                "jerk": jerks,
                "wheel_speed": wheel_speeds,
                "V_cmd": V_cmds,
                "V_max": V_maxes,
                "brake_torque": brake_torques,
                "brake_torque_max": brake_torque_maxes,
                "motor_voltage": motor_voltages,
                "motor_current": motor_currents,
                "slip_ratio": slip_ratios,
                "drive_torque": drive_torques,
                "tire_force": tire_forces,
                "drag_force": drag_forces,
                "rolling_force": rolling_forces,
                "grade_force": grade_forces,
                "net_force": net_forces,
            }
        )

    summary = {
        "episodes": episodes,
        "reward_mean": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "avg_abs_error": float(np.mean(avg_errors)),
        "action_delta_variance": float(np.mean(action_variances)),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))

    if plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)
        for trace in traces:
            plot_sequence_diagnostics(trace, plot_dir / f"{trace['sequence_id']}.png")
        plot_summary(per_episode_metrics, plot_dir / "closed_loop_summary.png")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_closed_loop_evaluation(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        output_path=args.output,
        plot_dir=args.plot_dir,
        device_str=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


