#!/usr/bin/env python3
"""Closed-loop evaluation for SAC longitudinal policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from env.longitudinal_env import LongitudinalEnv, LongitudinalEnvConfig
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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config file to override environment settings from checkpoint.",
    )
    return parser.parse_args()


def run_closed_loop_evaluation(
    checkpoint: Path,
    episodes: int,
    output_path: Optional[Path],
    plot_dir: Optional[Path] = None,
    device_str: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> dict[str, float]:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy, env_cfg, _ = load_policy_from_checkpoint(checkpoint, device=device)

    # Override environment config if specified
    if config_path is not None:
        with config_path.open() as fh:
            raw_config = yaml.safe_load(fh)
        if "env" in raw_config:
            env_cfg = LongitudinalEnvConfig(**raw_config["env"])
            print(f"Using config override from {config_path}")

    # Extract generator config if present
    generator_config = raw_config.get("generator", {})

    # Add vehicle randomization config to generator_config
    if "vehicle_randomization" in raw_config:
        generator_config["vehicle_randomization"] = raw_config["vehicle_randomization"]

    env = LongitudinalEnv(env_cfg, randomization=RandomizationConfig(), generator_config=generator_config, seed=4321)

    episode_rewards = []
    avg_errors = []
    action_variances = []
    traces = []
    per_episode_metrics = []
    vehicle_params_log = []

    for episode in tqdm(range(episodes), desc="Evaluating episodes"):
        obs, _ = env.reset()

        # Capture vehicle parameters for this episode
        vehicle_params = {
            "episode_id": episode,
            # Basic vehicle parameters
            "mass": float(env.params.mass),
            "rolling_coeff": float(env.params.rolling_coeff),
            "drag_area": float(env.params.drag_area),
            "actuator_tau": float(env.params.actuator_tau),
            "grade_rad": float(env.params.grade_rad),
            "air_density": float(env.params.air_density),
            # Extended motor parameters
            "motor_V_max": float(env.extended_params.motor.V_max),
            "motor_R": float(env.extended_params.motor.R),
            "motor_K_t": float(env.extended_params.motor.K_t),
            "motor_K_e": float(env.extended_params.motor.K_e),
            "motor_B_m": float(env.extended_params.motor.B_m),
            "motor_gear_ratio": float(env.extended_params.motor.gear_ratio),
            # Brake parameters
            "brake_tau": float(env.extended_params.brake.tau_br),
            "brake_T_max": float(env.extended_params.brake.T_br_max),
            "brake_p": float(env.extended_params.brake.p_br),
            "brake_kappa_c": float(env.extended_params.brake.kappa_c),
            "brake_mu": float(env.extended_params.brake.mu),
            # Body parameters (extended)
            "body_mass": float(env.extended_params.body.mass),
            "body_drag_area": float(env.extended_params.body.drag_area),
            "body_rolling_coeff": float(env.extended_params.body.rolling_coeff),
            "body_grade_rad": float(env.extended_params.body.grade_rad),
            "body_air_density": float(env.extended_params.body.air_density),
            # Wheel parameters
            "wheel_radius": float(env.extended_params.wheel.radius),
            "wheel_inertia": float(env.extended_params.wheel.inertia),
            "wheel_v_eps": float(env.extended_params.wheel.v_eps),
        }
        vehicle_params_log.append(vehicle_params)

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
            motor_voltages.append(float(info.get("back_emf_voltage", 0.0)))
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
                "vehicle_params": vehicle_params,
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
                "back_emf_voltage": motor_voltages,
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

        # Save vehicle parameters to a text file
        vehicle_params_file = plot_dir / "vehicle_parameters.txt"
        with vehicle_params_file.open("w") as f:
            f.write("Vehicle Parameters per Episode\n")
            f.write("=" * 50 + "\n\n")
            for params in vehicle_params_log:
                episode_id = params.pop("episode_id")
                f.write(f"Episode {episode_id}:\n")
                for key, value in sorted(params.items()):
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_closed_loop_evaluation(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        output_path=args.output,
        plot_dir=args.plot_dir,
        device_str=args.device,
        config_path=args.config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


