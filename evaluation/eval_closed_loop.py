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
from evaluation.pid_controller import PIDController
from evaluation.policy_loader import load_policy_from_checkpoint, select_action
from evaluation.plotting import plot_sequence_diagnostics, plot_summary
from utils.dynamics import RandomizationConfig

# Optional C++ ONNX inference
try:
    from evaluation.cpp_onnx_policy import CppOnnxPolicy
except ImportError:
    CppOnnxPolicy = None


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
    parser.add_argument(
        "--pid-kp",
        type=float,
        default=0.0,
        help="PID proportional gain (default: 0.0).",
    )
    parser.add_argument(
        "--pid-ki",
        type=float,
        default=0.0,
        help="PID integral gain (default: 0.0).",
    )
    parser.add_argument(
        "--pid-kd",
        type=float,
        default=0.0,
        help="PID derivative gain (default: 0.0).",
    )
    parser.add_argument(
        "--pid-integral-min",
        type=float,
        default=-0.1,
        help="PID integral saturation minimum (default: -0.1).",
    )
    parser.add_argument(
        "--pid-integral-max",
        type=float,
        default=0.1,
        help="PID integral saturation maximum (default: 0.1).",
    )
    parser.add_argument(
        "--pid-feedback-combined",
        action="store_true",
        help="Feed combined RL+PID action back to network observation (default: feed only RL action).",
    )
    parser.add_argument(
        "--use-cpp-inference",
        action="store_true",
        help="Enable C++ ONNX inference for comparison with Python inference.",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Path to ONNX model file (required if --use-cpp-inference is set).",
    )
    return parser.parse_args()


def run_closed_loop_evaluation(
    checkpoint: Path,
    episodes: int,
    output_path: Optional[Path],
    plot_dir: Optional[Path] = None,
    device_str: Optional[str] = None,
    config_path: Optional[Path] = None,
    pid_kp: float = 0.0,
    pid_ki: float = 0.0,
    pid_kd: float = 0.0,
    pid_integral_min: float = -0.1,
    pid_integral_max: float = 0.1,
    pid_feedback_rl_only: bool = True,
    use_cpp_inference: bool = False,
    onnx_model_path: Optional[Path] = None,
) -> dict[str, float]:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy, env_cfg, _ = load_policy_from_checkpoint(checkpoint, device=device)

    # Override environment config if specified
    raw_config = {}
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

    # Initialize PID controller
    pid_controller = PIDController(
        kp=pid_kp, 
        ki=pid_ki, 
        kd=pid_kd,
        integral_min=pid_integral_min,
        integral_max=pid_integral_max,
    )

    # Initialize C++ ONNX inference if requested
    cpp_policy = None
    if use_cpp_inference:
        if CppOnnxPolicy is None:
            raise ImportError(
                "C++ ONNX inference module not available. Please build it first:\n"
                "  cd evaluation/cpp_inference\n"
                "  mkdir -p build && cd build\n"
                "  cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..\n"
                "  cmake --build .\n"
            )
        if onnx_model_path is None:
            raise ValueError("--onnx-model is required when --use-cpp-inference is set")
        if not onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        try:
            cpp_policy = CppOnnxPolicy(onnx_model_path)
            print(f"[C++ Inference] Loaded ONNX model: {onnx_model_path}")
            print(f"[C++ Inference] obs_dim: {cpp_policy.obs_dim}, action_dim: {cpp_policy.action_dim}")
        except Exception as e:
            print(f"[Warning] Failed to load C++ ONNX inference: {e}")
            print("[Warning] Continuing with Python inference only")
            cpp_policy = None

    # Comparison tolerance
    COMPARISON_TOLERANCE = 1e-5

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
            "motor_b": float(env.extended_params.motor.b),
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

        # Reset PID controller at start of episode
        pid_controller.reset()

        done = False
        reward_total = 0.0
        errors: list[float] = []
        actions: list[float] = []  # Final combined actions
        rl_actions: list[float] = []  # RL policy actions
        pid_actions: list[float] = []  # PID controller actions
        # C++ inference tracking
        cpp_actions: list[float] = []  # C++ ONNX policy actions
        action_diffs: list[float] = []  # Absolute differences between Python and C++ actions
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
            # Get current speed and reference speed from observation for PID
            # Observation: [speed, prev_speed, prev_prev_speed, prev_action, refs...]
            current_speed = float(obs[0])
            reference_speed = float(obs[4])  # First reference speed (current)
            
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            plan, stats = select_action(policy, obs_tensor, deterministic=True)
            rl_action = float(plan[0])
            
            # Run C++ inference if enabled
            cpp_action = None
            action_diff = 0.0
            if cpp_policy is not None:
                try:
                    obs_array = np.array(obs, dtype=np.float32)
                    cpp_plan = cpp_policy.infer(obs_array)
                    cpp_action = float(cpp_plan[0])
                    action_diff = abs(rl_action - cpp_action)
                    cpp_actions.append(cpp_action)
                    action_diffs.append(action_diff)
                except Exception as e:
                    print(f"[Warning] C++ inference failed at step {step_idx}: {e}")
                    cpp_actions.append(float('nan'))
                    action_diffs.append(float('nan'))
            
            # Compute speed error and PID action
            speed_error = reference_speed - current_speed
            pid_action = pid_controller.compute(speed_error, env.config.dt)
            
            # Combine RL and PID actions (clipped to [-1, 1])
            # Use Python action for environment step (C++ is for verification)
            final_action = np.clip(rl_action + pid_action, -1.0, 1.0)
            
            # Track actions separately
            rl_actions.append(rl_action)
            pid_actions.append(pid_action)
            
            # Step environment with final action (for plant simulation)
            obs, reward, terminated, truncated, info = env.step(final_action)
            
            # Optionally override prev_action to feed only RL action back to network
            # This models PID as "part of the plant" - network only sees its own action
            if pid_feedback_rl_only:
                env._prev_action = rl_action
            reward_total += reward
            errors.append(abs(info.get("speed_error", 0.0)))
            actions.append(final_action)
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
            policy_plans.append(stats.get("plan", [rl_action]))
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

        # Compute C++ inference comparison metrics
        cpp_comparison = {}
        if cpp_policy is not None and len(cpp_actions) > 0:
            valid_diffs = [d for d in action_diffs if not np.isnan(d)]
            if valid_diffs:
                max_diff = float(np.max(valid_diffs))
                mean_diff = float(np.mean(valid_diffs))
                num_mismatches = sum(1 for d in valid_diffs if d > COMPARISON_TOLERANCE)
                mismatch_fraction = num_mismatches / len(valid_diffs) if valid_diffs else 0.0
                
                cpp_comparison = {
                    "max_action_diff": max_diff,
                    "mean_action_diff": mean_diff,
                    "num_mismatches": num_mismatches,
                    "mismatch_fraction": mismatch_fraction,
                    "total_steps": len(valid_diffs),
                }
                
                print(f"\n[C++ Comparison] Episode {episode}:")
                print(f"  Max difference: {max_diff:.2e}")
                print(f"  Mean difference: {mean_diff:.2e}")
                print(f"  Mismatches (> {COMPARISON_TOLERANCE:.1e}): {num_mismatches}/{len(valid_diffs)} ({mismatch_fraction*100:.2f}%)")
                if max_diff > COMPARISON_TOLERANCE:
                    print(f"  [Warning] Differences exceed tolerance!")
                else:
                    print(f"  [OK] All differences within tolerance")
            else:
                cpp_comparison = {"error": "No valid comparisons (all NaN)"}

        # Calculate maximum feasible speed from back-EMF constraint
        # v_max = V_max * r_w / (K_e * gear_ratio)
        # Apply configurable safety margin for conservative operation
        v_max_theoretical = (env.extended_params.motor.V_max * env.extended_params.wheel.radius / 
                            (env.extended_params.motor.K_e * env.extended_params.motor.gear_ratio))
        # Get safety factor from config (default to 0.75 if not specified)
        safety_factor = generator_config.get('speed_limit_safety_factor', 0.75)
        v_max_feasible = safety_factor * v_max_theoretical

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
        trace_data = {
            "sequence_id": episode_id,
            "vehicle_params": vehicle_params,
            "time": time_axis,
            "speed": speeds,
            "reference": references,
            "acceleration": accelerations,
            "action": actions,
            "rl_action": rl_actions,
            "pid_action": pid_actions,
            "policy_pre_tanh_mean": policy_means,
            "policy_log_std": policy_log_stds,
            "policy_plan": policy_plans,
            "jerk": jerks,
            "wheel_speed": wheel_speeds,
            "V_cmd": V_cmds,
            "V_max": V_maxes,
            "v_max_feasible": float(v_max_feasible),  # Generation max speed (with safety factor)
            "v_max_theoretical": float(v_max_theoretical),  # Theoretical max speed from back-EMF = V_max
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
        
        # Add C++ inference data if available
        if cpp_policy is not None:
            trace_data["cpp_action"] = cpp_actions
            trace_data["action_diff"] = action_diffs
            trace_data["cpp_comparison"] = cpp_comparison
        
        traces.append(trace_data)

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
    
    # Validate C++ inference arguments
    if args.use_cpp_inference and args.onnx_model is None:
        raise ValueError("--onnx-model is required when --use-cpp-inference is set")
    
    # Set default for pid_feedback_rl_only (default: True, unless --pid-feedback-combined is set)
    pid_feedback_rl_only = not getattr(args, 'pid_feedback_combined', False)
    
    summary = run_closed_loop_evaluation(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        output_path=args.output,
        plot_dir=args.plot_dir,
        device_str=args.device,
        config_path=args.config,
        pid_kp=args.pid_kp,
        pid_ki=args.pid_ki,
        pid_kd=args.pid_kd,
        pid_integral_min=args.pid_integral_min,
        pid_integral_max=args.pid_integral_max,
        pid_feedback_rl_only=pid_feedback_rl_only,
        use_cpp_inference=args.use_cpp_inference,
        onnx_model_path=args.onnx_model,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


