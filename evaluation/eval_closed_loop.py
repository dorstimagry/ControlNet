#!/usr/bin/env python3
"""Closed-loop evaluation for SAC longitudinal policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
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

# ClearML integration (optional)
try:
    import clearml
    from utils.clearml_logger import (
        init_clearml_task,
        log_metrics,
        log_config,
        upload_plot,
        log_matplotlib_figure,
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
    log_matplotlib_figure = None
    upload_artifact = None
    close_task = None


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
    parser.add_argument(
        "--force-export-onnx",
        action="store_true",
        help="Force re-export ONNX model from checkpoint before evaluation to ensure weight matching.",
    )
    parser.add_argument(
        "--step-functions",
        action="store_true",
        help="Evaluate over sharp step functions (no filtering) in addition to standard generator profiles.",
    )
    parser.add_argument(
        "--upload-plots-to-clearml",
        action="store_true",
        help="Upload plots to ClearML (default: False, plots are always saved offline).",
    )
    parser.add_argument(
        "--disable-clearml",
        action="store_true",
        help="Disable ClearML logging even if ClearML is installed",
    )
    return parser.parse_args()


def generate_step_function(start_speed: float, end_speed: float, length: int, dt: float) -> np.ndarray:
    """Generate a sharp step function profile with no filtering.
    
    Args:
        start_speed: Initial speed (m/s)
        end_speed: Target speed after step (m/s)
        length: Number of timesteps in the profile
        dt: Timestep duration (seconds)
    
    Returns:
        Array of speed values: [start_speed, end_speed, end_speed, ...]
    """
    profile = np.full(length, end_speed, dtype=np.float32)
    profile[0] = start_speed  # First timestep is start_speed, then jumps to end_speed
    return profile


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
    force_export_onnx: bool = False,
    step_functions: bool = False,
    upload_plots_to_clearml: bool = False,
    disable_clearml: bool = False,
) -> dict[str, float]:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy, env_cfg, _ = load_policy_from_checkpoint(checkpoint, device=device)
    
    # Initialize ClearML task for evaluation
    clearml_task = None
    if not disable_clearml and CLEARML_LOGGER_AVAILABLE and init_clearml_task is not None and clearml is not None:
        checkpoint_stem = checkpoint.stem
        clearml_task = init_clearml_task(
            task_name=f"Evaluation_{checkpoint_stem}",
            project="RL_Longitudinal",
            tags=["evaluation", "closed_loop"],
            task_type=clearml.Task.TaskTypes.testing,
        )
        if clearml_task:
            # Log evaluation configuration
            eval_config = {
                "checkpoint": str(checkpoint),
                "episodes": episodes,
                "device": str(device),
                "pid_kp": pid_kp,
                "pid_ki": pid_ki,
                "pid_kd": pid_kd,
                "pid_integral_min": pid_integral_min,
                "pid_integral_max": pid_integral_max,
                "pid_feedback_rl_only": pid_feedback_rl_only,
                "use_cpp_inference": use_cpp_inference,
                "step_functions": step_functions,
            }
            if config_path:
                eval_config["config_path"] = str(config_path)
            if onnx_model_path:
                eval_config["onnx_model_path"] = str(onnx_model_path)
    
    # Check if SysID is enabled
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    
    # Log checkpoint config to ClearML if available
    if clearml_task:
        checkpoint_config = checkpoint_data.get("config", {})
        if checkpoint_config:
            eval_config["checkpoint_config"] = checkpoint_config
        log_config(clearml_task, eval_config)
    
    # Log checkpoint config to ClearML if available
    if clearml_task:
        checkpoint_config = checkpoint_data.get("config", {})
        if checkpoint_config:
            eval_config["checkpoint_config"] = checkpoint_config
        log_config(clearml_task, eval_config)
    meta = checkpoint_data.get("meta", {})
    sysid_enabled_meta = meta.get("sysid_enabled", False)
    
    # Auto-detect SysID from model dimensions (more reliable than metadata)
    # Base observation: 6 (speed, prev_speed, prev_prev_speed, prev_action, speed_error, grade) + preview_steps
    # Note: Older checkpoints may have been trained with:
    #   - 4 base features (without speed_error and grade)
    #   - 5 base features (with speed_error but without grade)
    preview_steps = max(int(round(env_cfg.preview_horizon_s / env_cfg.dt)), 1)
    base_obs_dim_current = 6 + preview_steps  # Current environment (with speed_error and grade)
    base_obs_dim_without_grade = 5 + preview_steps  # Checkpoints with speed_error but without grade
    base_obs_dim_without_error = 4 + preview_steps  # Older checkpoints (without speed_error and grade)
    
    # Get obs_dim from policy weights (most reliable) or metadata
    policy_state = checkpoint_data.get("policy", {})
    if "net.0.weight" in policy_state:
        policy_obs_dim_from_weights = int(policy_state["net.0.weight"].shape[1])
    else:
        policy_obs_dim_from_weights = None
    
    policy_obs_dim = policy_obs_dim_from_weights if policy_obs_dim_from_weights else int(meta.get("obs_dim", base_obs_dim_current))
    z_dim_auto = None  # Initialize for scope
    
    # If policy expects more dimensions than base, SysID is enabled
    if policy_obs_dim > base_obs_dim_current:
        # Policy has more dims than current base -> SysID enabled (with speed_error)
        z_dim_auto = policy_obs_dim - base_obs_dim_current
        sysid_enabled = True
        print(f"[SysID] Auto-detected from model dimensions: obs_dim={policy_obs_dim}, base_obs_dim={base_obs_dim_current}, z_dim={z_dim_auto}")
        if not sysid_enabled_meta:
            print(f"[Warning] Checkpoint metadata says sysid_enabled=false, but model expects {policy_obs_dim}D obs (base={base_obs_dim_current}D). Using auto-detection.")
    elif policy_obs_dim == base_obs_dim_current:
        # Policy matches current base exactly -> No SysID (with speed_error and grade)
        z_dim_auto = 0
        sysid_enabled = False
        print(f"[SysID] Auto-detected: obs_dim={policy_obs_dim} matches base_obs_dim={base_obs_dim_current} -> No SysID")
        if sysid_enabled_meta:
            print(f"[Warning] Checkpoint metadata says sysid_enabled=true, but model expects {policy_obs_dim}D obs (matches base). Using auto-detection.")
    elif policy_obs_dim == base_obs_dim_without_grade:
        # Policy matches format without grade -> No SysID (with speed_error but without grade)
        z_dim_auto = 0
        sysid_enabled = False
        base_obs_dim_checkpoint = base_obs_dim_without_grade
        print(f"[SysID] Auto-detected: obs_dim={policy_obs_dim} matches base_obs_dim_without_grade={base_obs_dim_without_grade} -> No SysID (checkpoint trained without grade)")
    elif policy_obs_dim > base_obs_dim_without_error:
        # Policy has more dims than old base (without speed_error) -> Could be SysID or intermediate format
        # Check if it matches exactly the old base + speed_error (no SysID, no grade)
        if policy_obs_dim == base_obs_dim_without_error + 1:
            # This is just speed_error, not SysID
            z_dim_auto = 0
            sysid_enabled = False
            base_obs_dim_checkpoint = base_obs_dim_without_error
            print(f"[SysID] Auto-detected: obs_dim={policy_obs_dim} = base_obs_dim_without_error+1 -> No SysID (just speed_error added, no grade)")
        elif policy_obs_dim == base_obs_dim_without_error + 2:
            # This is speed_error + grade, no SysID
            z_dim_auto = 0
            sysid_enabled = False
            base_obs_dim_checkpoint = base_obs_dim_without_error
            print(f"[SysID] Auto-detected: obs_dim={policy_obs_dim} = base_obs_dim_without_error+2 -> No SysID (speed_error + grade added)")
        else:
            # This is SysID (with or without speed_error/grade)
            z_dim_auto = policy_obs_dim - base_obs_dim_without_error
            base_obs_dim_checkpoint = base_obs_dim_without_error
            sysid_enabled = True
            print(f"[SysID] Auto-detected from model dimensions: obs_dim={policy_obs_dim}, base_obs_dim={base_obs_dim_without_error} (without speed_error), z_dim={z_dim_auto}")
            if not sysid_enabled_meta:
                print(f"[Warning] Checkpoint metadata says sysid_enabled=false, but model expects {policy_obs_dim}D obs (base={base_obs_dim_without_error}D). Using auto-detection.")
    else:
        # Policy matches old base exactly -> No SysID (without speed_error and grade)
        z_dim_auto = 0
        sysid_enabled = sysid_enabled_meta
        print(f"[SysID] Auto-detected: obs_dim={policy_obs_dim} matches base_obs_dim_without_error={base_obs_dim_without_error} -> No SysID")
    
    # Initialize SysID components if enabled
    encoder = None
    encoder_norm = None
    feature_builder = None
    encoder_hidden = None
    prev_speed = 0.0
    prev_action = 0.0
    
    if sysid_enabled:
        from src.sysid import ContextEncoder, RunningNorm
        from src.sysid.encoder import FeatureBuilder
        from src.sysid.integration import compute_z_online
        
        config = checkpoint_data.get("config", {})
        sysid_config = config.get("sysid", {})
        
        # Determine z_dim: check encoder weights first (most reliable), then auto-detection, then config
        z_dim = None
        if "encoder" in checkpoint_data:
            # Extract z_dim from encoder weights (handles cases where base_obs_dim changed)
            encoder_state = checkpoint_data["encoder"]
            if "z_proj.weight" in encoder_state:
                z_dim_from_checkpoint = int(encoder_state["z_proj.weight"].shape[0])
                z_dim = z_dim_from_checkpoint
                print(f"[SysID] Using z_dim={z_dim} from checkpoint encoder weights")
                
                # Determine actual base_obs_dim from checkpoint: policy_obs_dim - z_dim
                # Always recalculate when we have encoder weights (most reliable source)
                base_obs_dim_checkpoint = policy_obs_dim - z_dim
                if base_obs_dim_checkpoint == base_obs_dim_without_error:
                    print(f"[SysID] Checkpoint was trained with base_obs_dim={base_obs_dim_checkpoint} (without speed_error and grade)")
                elif base_obs_dim_checkpoint == base_obs_dim_without_grade:
                    print(f"[SysID] Checkpoint was trained with base_obs_dim={base_obs_dim_checkpoint} (with speed_error but without grade)")
                elif base_obs_dim_checkpoint == base_obs_dim_current:
                    print(f"[SysID] Checkpoint was trained with base_obs_dim={base_obs_dim_checkpoint} (with speed_error and grade)")
                else:
                    print(f"[SysID] Checkpoint was trained with base_obs_dim={base_obs_dim_checkpoint} (unexpected format)")
        
        if z_dim is None:
            # Fall back to auto-detection or config
            z_dim = z_dim_auto if z_dim_auto is not None else int(sysid_config.get("dz", 12))
            if z_dim_auto is not None:
                print(f"[SysID] Using auto-detected z_dim={z_dim} (no encoder weights found)")
            else:
                print(f"[SysID] Using z_dim={z_dim} from config")
            # If we don't have base_obs_dim_checkpoint yet, infer it
            if base_obs_dim_checkpoint is None:
                base_obs_dim_checkpoint = policy_obs_dim - z_dim
        
        gru_hidden = int(sysid_config.get("gru_hidden", 64))
        
        encoder = ContextEncoder(input_dim=4, hidden_dim=gru_hidden, z_dim=z_dim)
        encoder_norm = RunningNorm(dim=4, eps=1e-6, clip=10.0)
        
        # Check if encoder weights exist in checkpoint
        if "encoder" not in checkpoint_data or "encoder_norm" not in checkpoint_data:
            raise ValueError(
                f"SysID is enabled (model expects {policy_obs_dim}D obs), but encoder weights "
                f"are missing from checkpoint. Checkpoint may have been saved incorrectly."
            )
        
        encoder.load_state_dict(checkpoint_data["encoder"])
        encoder_norm.load_state_dict(checkpoint_data["encoder_norm"])
        encoder.eval()
        encoder_norm.eval()
        encoder.to(device)
        encoder_norm.to(device)
        
        feature_builder = FeatureBuilder(dt=env_cfg.dt)
        
        print(f"[SysID] Enabled: z_dim={z_dim}, gru_hidden={gru_hidden}, policy_obs_dim={policy_obs_dim}, base_obs_dim_checkpoint={base_obs_dim_checkpoint}")

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
        
        # Force re-export ONNX model if requested to ensure weight matching
        if force_export_onnx:
            print(f"[Export] Re-exporting ONNX model from checkpoint to ensure weight matching...")
            import subprocess
            try:
                result = subprocess.run(
                    [
                        sys.executable,  # Use the same Python interpreter
                        "scripts/export_onnx.py",
                        "--checkpoint", str(checkpoint),
                        "--output", str(onnx_model_path),
                        "--no-validate",  # Skip validation for speed
                    ],
                    cwd=Path(__file__).parent.parent,  # Run from project root
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    print(f"[Export] Successfully re-exported ONNX model to: {onnx_model_path}")
                    if result.stdout:
                        print("[Export] Output:")
                        for line in result.stdout.strip().split('\n')[:10]:  # Show first 10 lines
                            print(f"  {line}")
                else:
                    print(f"[Export] Warning: Export failed with return code {result.returncode}")
                    print(f"[Export] Error: {result.stderr}")
                    print(f"[Export] Continuing with existing ONNX model")
            except subprocess.TimeoutExpired:
                print(f"[Export] Warning: Export timed out after 60 seconds")
                print(f"[Export] Continuing with existing ONNX model")
            except Exception as e:
                print(f"[Export] Warning: Failed to re-export ONNX model: {e}")
                print(f"[Export] Continuing with existing ONNX model")
        
        if not onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        try:
            cpp_policy = CppOnnxPolicy(onnx_model_path)
            print(f"[C++ Inference] Loaded ONNX model: {onnx_model_path}")
            print(f"[C++ Inference] obs_dim: {cpp_policy.obs_dim}, action_dim: {cpp_policy.action_dim}")
            if sysid_enabled:
                if cpp_policy.is_sysid_model:
                    print(f"[C++ Inference] SysID model detected: hidden_dim={cpp_policy.hidden_dim}")
                else:
                    print(f"[Warning] C++ module doesn't support SysID. Rebuild C++ module for SysID support.")
                    print(f"[Warning] C++ inference will be disabled for SysID models.")
                    cpp_policy = None  # Disable C++ inference if SysID not supported
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
        # Reset environment - for regular evaluation, start with no initial error and zero initial action
        # Temporarily disable randomization to ensure consistent initial conditions
        original_randomize_action = env.config.randomize_initial_action
        original_speed_error_range = env.config.initial_speed_error_range
        env.config.randomize_initial_action = False
        env.config.initial_speed_error_range = 0.0
        
        obs, _ = env.reset()
        
        # Get the initial reference speed (first reference in the preview horizon)
        initial_reference_speed = float(obs[4]) if len(obs) > 4 else float(env.reference[0]) if len(env.reference) > 0 else 0.0
        
        # Reset again with initial speed matching reference (no initial error)
        obs, _ = env.reset(options={"initial_speed": initial_reference_speed})
        
        # Ensure speed exactly matches reference (override any randomization)
        env.speed = initial_reference_speed
        
        # Ensure previous action is zero and previous speeds match initial speed
        # (Override any config-based randomization that might have been applied)
        env._prev_action = 0.0
        env.prev_speed = initial_reference_speed
        env._prev_prev_speed = initial_reference_speed
        
        # Restore original config values
        env.config.randomize_initial_action = original_randomize_action
        env.config.initial_speed_error_range = original_speed_error_range
        
        # Rebuild observation with correct values
        obs = env._build_observation()
        
        # Verify initial conditions are correct
        if abs(env.speed - initial_reference_speed) > 1e-6:
            print(f"[Warning] Speed mismatch after reset: speed={env.speed:.6f}, reference={initial_reference_speed:.6f}")
        if abs(env._prev_action) > 1e-6:
            print(f"[Warning] Non-zero initial action: {env._prev_action:.6f}")
        
        # Reset SysID encoder state for new episode
        if sysid_enabled:
            # Separate hidden states for Python and C++ inference
            py_encoder_hidden = encoder.reset(batch_size=1, device=device).squeeze(0)
            cpp_encoder_hidden = encoder.reset(batch_size=1, device=device).squeeze(0)
            feature_builder.reset()
            # Action history for SysID: track 3 previous actions to match FeatureBuilder bug
            # Due to FeatureBuilder's implementation, features at step t use action from step t-2
            # Initialize with initial speed (no error) and zero actions
            prev_speed = initial_reference_speed  # v_{t-1}, start with initial speed
            prev_action = 0.0  # u_{t-1}, most recent action (zero at start)
            prev_prev_action = 0.0  # u_{t-2}, used in features (due to FeatureBuilder bug)
            prev_prev_prev_action = 0.0  # u_{t-3}, used for du computation

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
        # SysID latent tracking
        z_latents: list[np.ndarray] = []  # Store z values if SysID enabled
        step_idx = 0

        while not done:
            # Get current speed and reference speed from observation for PID
            # Observation: [speed, prev_speed, prev_prev_speed, prev_action, refs...]
            current_speed = float(obs[0])
            reference_speed = float(obs[4])  # First reference speed (current)
            
            # Compute z_t if SysID enabled (for Python inference)
            if sysid_enabled:
                if encoder is None or encoder_norm is None or feature_builder is None:
                    raise RuntimeError(
                        f"SysID is enabled but encoder components are not initialized. "
                        f"obs_dim={len(obs)}, expected={policy_obs_dim}"
                    )
                py_encoder_hidden, z_t = compute_z_online(
                    encoder=encoder,
                    feature_builder=feature_builder,
                    encoder_norm=encoder_norm,
                    h_prev=py_encoder_hidden,
                    v_t=current_speed,
                    u_t=prev_action,
                    device=device
                )
                z_t_np = z_t.cpu().numpy()
                # Store z for analysis
                z_latents.append(z_t_np)
                
                # Adjust observation to match checkpoint's expected base_obs_dim
                # Current observation: [speed, prev_speed, prev_prev_speed, prev_action, speed_error, grade, ...preview]
                obs_adjusted = obs.copy()
                if base_obs_dim_checkpoint is not None and len(obs) == base_obs_dim_current:
                    if base_obs_dim_checkpoint == base_obs_dim_without_error:
                        # Remove speed_error (index 4) and grade (index 5)
                        # Keep: [speed, prev_speed, prev_prev_speed, prev_action, ...preview]
                        obs_adjusted = np.concatenate([obs[:4], obs[6:]])  # Skip speed_error and grade
                    elif base_obs_dim_checkpoint == base_obs_dim_without_grade:
                        # Remove grade (index 5) only
                        # Keep: [speed, prev_speed, prev_prev_speed, prev_action, speed_error, ...preview]
                        obs_adjusted = np.concatenate([obs[:5], obs[6:]])  # Skip grade at index 5
                
                # Augment observation with z_t
                obs_aug = np.concatenate([obs_adjusted, z_t_np])
                if len(obs_aug) != policy_obs_dim:
                    raise RuntimeError(
                        f"Observation dimension mismatch: obs={len(obs)}, obs_adjusted={len(obs_adjusted)}, z={len(z_t_np)}, "
                        f"augmented={len(obs_aug)}, expected={policy_obs_dim}, base_obs_dim_checkpoint={base_obs_dim_checkpoint}"
                    )
            else:
                obs_aug = obs
            
            obs_tensor = torch.as_tensor(obs_aug, dtype=torch.float32, device=device).unsqueeze(0)
            plan, stats = select_action(policy, obs_tensor, deterministic=True)
            rl_action = float(plan[0])
            
            # Run C++ inference if enabled (with independent state management)
            cpp_action = None
            action_diff = 0.0
            if cpp_policy is not None:
                try:
                    if sysid_enabled and hasattr(cpp_policy, 'is_sysid_model') and cpp_policy.is_sysid_model:
                        # For SysID, C++ needs all inputs
                        # Due to FeatureBuilder bug, features use action from t-2, not t-1
                        # So we pass prev_prev_action (u_{t-2}) where FeatureBuilder would use self.u_prev
                        cpp_plan, cpp_hidden_new = cpp_policy.infer_sysid(
                            base_obs=obs,
                            speed=current_speed,  # v_t
                            prev_action=prev_prev_action,  # u_{t-2} - matches FeatureBuilder's self.u_prev
                            prev_speed=prev_speed,  # v_{t-1}
                            prev_prev_action=prev_prev_prev_action,  # u_{t-3} - matches FeatureBuilder's self.u_prev_prev
                            hidden_state=cpp_encoder_hidden.cpu().numpy()
                        )
                        # Update C++ encoder hidden state
                        cpp_encoder_hidden = torch.from_numpy(cpp_hidden_new).to(device)
                    else:
                        obs_array = np.array(obs_aug, dtype=np.float32)
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
            
            # Update SysID history for next step
            if sysid_enabled:
                # Update action history: shift all back by one
                prev_prev_prev_action = prev_prev_action  # t-3 <- t-2
                prev_prev_action = prev_action  # t-2 <- t-1
                prev_action = rl_action  # t-1 <- t (current)
                prev_speed = current_speed
            
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
        
        # Add SysID latent data if available
        if sysid_enabled and len(z_latents) > 0:
            trace_data["z_latents"] = np.array(z_latents)
        
        # Add C++ inference data if available
        if cpp_policy is not None:
            trace_data["cpp_action"] = cpp_actions
            trace_data["action_diff"] = action_diffs
            trace_data["cpp_comparison"] = cpp_comparison
        
        traces.append(trace_data)

    # Step function evaluation (if enabled)
    step_traces = []
    if step_functions:
        # Use small speed values between 0 and 2 m/s with 10 evenly spaced values
        start_speeds = np.linspace(0.0, 2.0, 10).tolist()
        end_speeds = np.linspace(0.0, 2.0, 10).tolist()
        profile_length = env.config.max_episode_steps
        
        print(f"\n[Step Functions] Evaluating {len(start_speeds) * len(end_speeds)} step function combinations...")
        
        # Import required modules for validation
        from generator.adapter import extended_params_to_vehicle_capabilities
        from generator.feasibility import compute_max_feasible_speed
        
        invalid_combinations = []
        
        # Create progress bar for step function combinations
        total_combinations = len(start_speeds) * len(end_speeds)
        step_pbar = tqdm(total=total_combinations, desc="Step functions", unit="comb")
        
        for start_speed in start_speeds:
            for end_speed in end_speeds:
                # Generate step function profile
                step_profile = generate_step_function(start_speed, end_speed, profile_length, env.config.dt)
                
                # Reset environment with step function profile
                obs, _ = env.reset(options={"reference_profile": step_profile, "initial_speed": start_speed})
                
                # Ensure previous action is 0 and previous speeds are set to initial speed for step function evaluation
                # (Override any randomization that might have been applied)
                env._prev_action = 0.0
                env.prev_speed = start_speed
                env._prev_prev_speed = start_speed
                
                # Rebuild observation with correct values
                obs = env._build_observation()
                
                # Validate that vehicle can achieve the target speeds
                # Convert extended params to vehicle capabilities
                vehicle_caps = extended_params_to_vehicle_capabilities(env.extended_params, device=torch.device('cpu'))
                v_max_theoretical = compute_max_feasible_speed(vehicle_caps).item()
                
                # Check if speeds exceed maximum feasible speed
                max_speed_in_profile = max(start_speed, end_speed)
                if max_speed_in_profile > v_max_theoretical:
                    invalid_combinations.append({
                        "start": start_speed,
                        "end": end_speed,
                        "max_speed": max_speed_in_profile,
                        "v_max_theoretical": v_max_theoretical
                    })
                    print(f"[Warning] Step function {start_speed:.0f}→{end_speed:.0f} m/s exceeds vehicle max speed "
                          f"({v_max_theoretical:.2f} m/s). Vehicle cannot achieve this speed.")
                
                # Reset SysID encoder state for new episode
                if sysid_enabled:
                    py_encoder_hidden = encoder.reset(batch_size=1, device=device).squeeze(0)
                    cpp_encoder_hidden = encoder.reset(batch_size=1, device=device).squeeze(0)
                    feature_builder.reset()
                    prev_speed = start_speed
                    prev_action = 0.0
                    prev_prev_action = 0.0
                    prev_prev_prev_action = 0.0
                
                # Reset PID controller
                pid_controller.reset()
                
                # Track episode data
                done = False
                errors: list[float] = []
                actions: list[float] = []
                rl_actions: list[float] = []
                pid_actions: list[float] = []
                cpp_actions: list[float] = []
                action_diffs: list[float] = []
                accelerations: list[float] = []
                jerks: list[float] = []
                speeds: list[float] = []
                references: list[float] = []
                time_axis: list[float] = []
                z_latents: list[np.ndarray] = []
                step_idx = 0
                
                while not done:
                    current_speed = float(obs[0])
                    reference_speed = float(obs[4])
                    
                    # Compute z_t if SysID enabled
                    if sysid_enabled:
                        py_encoder_hidden, z_t = compute_z_online(
                            encoder=encoder,
                            feature_builder=feature_builder,
                            encoder_norm=encoder_norm,
                            h_prev=py_encoder_hidden,
                            v_t=current_speed,
                            u_t=prev_action,
                            device=device
                        )
                        z_t_np = z_t.cpu().numpy()
                        z_latents.append(z_t_np)
                        
                        # Adjust observation to match checkpoint's expected base_obs_dim
                        # Current observation: [speed, prev_speed, prev_prev_speed, prev_action, speed_error, grade, ...preview]
                        obs_adjusted = obs.copy()
                        if base_obs_dim_checkpoint is not None and len(obs) == base_obs_dim_current:
                            if base_obs_dim_checkpoint == base_obs_dim_without_error:
                                # Remove speed_error (index 4) and grade (index 5)
                                obs_adjusted = np.concatenate([obs[:4], obs[6:]])  # Skip speed_error and grade
                            elif base_obs_dim_checkpoint == base_obs_dim_without_grade:
                                # Remove grade (index 5) only
                                obs_adjusted = np.concatenate([obs[:5], obs[6:]])  # Skip grade
                        
                        obs_aug = np.concatenate([obs_adjusted, z_t_np])
                        if len(obs_aug) != policy_obs_dim:
                            raise RuntimeError(
                                f"Observation dimension mismatch: obs={len(obs)}, obs_adjusted={len(obs_adjusted)}, z={len(z_t_np)}, "
                                f"augmented={len(obs_aug)}, expected={policy_obs_dim}, base_obs_dim_checkpoint={base_obs_dim_checkpoint}"
                            )
                    else:
                        obs_aug = obs
                    
                    obs_tensor = torch.as_tensor(obs_aug, dtype=torch.float32, device=device).unsqueeze(0)
                    plan, stats = select_action(policy, obs_tensor, deterministic=True)
                    rl_action = float(plan[0])
                    
                    # C++ inference if enabled
                    if cpp_policy is not None:
                        try:
                            if sysid_enabled and hasattr(cpp_policy, 'is_sysid_model') and cpp_policy.is_sysid_model:
                                cpp_plan, cpp_hidden_new = cpp_policy.infer_sysid(
                                    base_obs=obs,
                                    speed=current_speed,
                                    prev_action=prev_prev_action,
                                    prev_speed=prev_speed,
                                    prev_prev_action=prev_prev_prev_action,
                                    hidden_state=cpp_encoder_hidden.cpu().numpy()
                                )
                                cpp_encoder_hidden = torch.from_numpy(cpp_hidden_new).to(device)
                            else:
                                obs_array = np.array(obs_aug, dtype=np.float32)
                                cpp_plan = cpp_policy.infer(obs_array)
                            cpp_action = float(cpp_plan[0])
                            action_diff = abs(rl_action - cpp_action)
                            cpp_actions.append(cpp_action)
                            action_diffs.append(action_diff)
                        except Exception as e:
                            cpp_actions.append(float('nan'))
                            action_diffs.append(float('nan'))
                    
                    # PID action
                    speed_error = reference_speed - current_speed
                    pid_action = pid_controller.compute(speed_error, env.config.dt)
                    
                    # Final action
                    final_action = np.clip(rl_action + pid_action, -1.0, 1.0)
                    
                    rl_actions.append(rl_action)
                    pid_actions.append(pid_action)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(final_action)
                    
                    # Update SysID history
                    if sysid_enabled:
                        prev_prev_prev_action = prev_prev_action
                        prev_prev_action = prev_action
                        prev_action = rl_action
                        prev_speed = current_speed
                    
                    if pid_feedback_rl_only:
                        env._prev_action = rl_action
                    
                    errors.append(abs(info.get("speed_error", 0.0)))
                    actions.append(final_action)
                    accelerations.append(float(info.get("acceleration", 0.0)))
                    jerks.append(float(info.get("jerk", 0.0)))
                    speeds.append(float(info.get("speed", 0.0)))
                    references.append(float(info.get("reference_speed", 0.0)))
                    time_axis.append(step_idx * env.config.dt)
                    done = terminated or truncated
                    step_idx += 1
                
                # Create trace data for this step function
                step_episode_id = f"step_{start_speed:.0f}_to_{end_speed:.0f}"
                step_trace = {
                    "sequence_id": step_episode_id,
                    "start_speed": start_speed,
                    "end_speed": end_speed,
                    "v_max_theoretical": v_max_theoretical,
                    "is_feasible": max_speed_in_profile <= v_max_theoretical,
                    "time": time_axis,
                    "speed": speeds,
                    "reference": references,
                    "acceleration": accelerations,
                    "action": actions,
                    "rl_action": rl_actions,
                    "pid_action": pid_actions,
                    "jerk": jerks,
                }
                
                if sysid_enabled and len(z_latents) > 0:
                    step_trace["z_latents"] = np.array(z_latents)
                
                if cpp_policy is not None:
                    step_trace["cpp_action"] = cpp_actions
                    step_trace["action_diff"] = action_diffs
                
                step_traces.append(step_trace)
                
                # Update progress bar
                step_pbar.update(1)
        
        # Close progress bar
        step_pbar.close()
        
        # Print summary of validation results
        if invalid_combinations:
            print(f"\n[Step Functions] Validation Summary:")
            print(f"  Total combinations evaluated: {len(start_speeds) * len(end_speeds)}")
            print(f"  Invalid combinations (exceed max speed): {len(invalid_combinations)}")
            print(f"  Valid combinations: {len(start_speeds) * len(end_speeds) - len(invalid_combinations)}")
            if len(invalid_combinations) > 0:
                print(f"\n  Invalid combinations:")
                for inv in invalid_combinations:
                    print(f"    {inv['start']:.0f}→{inv['end']:.0f} m/s (max feasible: {inv['v_max_theoretical']:.2f} m/s)")
        else:
            print(f"\n[Step Functions] All {len(start_speeds) * len(end_speeds)} combinations are feasible for the sampled vehicle.")

    summary = {
        "episodes": episodes,
        "reward_mean": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "avg_abs_error": float(np.mean(avg_errors)),
        "action_delta_variance": float(np.mean(action_variances)),
    }
    
    # Compute and print aggregated C++ comparison statistics
    if cpp_policy is not None:
        all_max_diffs = []
        all_mean_diffs = []
        all_mismatch_fractions = []
        total_steps_compared = 0
        total_mismatches = 0
        
        for trace in traces:
            cpp_comp = trace.get("cpp_comparison", {})
            if "max_action_diff" in cpp_comp:
                all_max_diffs.append(cpp_comp["max_action_diff"])
                all_mean_diffs.append(cpp_comp["mean_action_diff"])
                all_mismatch_fractions.append(cpp_comp["mismatch_fraction"])
                total_steps_compared += cpp_comp["total_steps"]
                total_mismatches += cpp_comp["num_mismatches"]
        
        if all_max_diffs:
            overall_max_diff = float(np.max(all_max_diffs))
            overall_mean_diff = float(np.mean(all_mean_diffs))
            overall_mismatch_fraction = total_mismatches / total_steps_compared if total_steps_compared > 0 else 0.0
            
            print("\n" + "=" * 70)
            print("[C++ vs Python Inference Comparison - Aggregated Statistics]")
            print("=" * 70)
            print(f"Total episodes compared: {len(all_max_diffs)}")
            print(f"Total steps compared: {total_steps_compared}")
            print(f"Overall max difference: {overall_max_diff:.4e}")
            print(f"Overall mean difference: {overall_mean_diff:.4e}")
            print(f"Per-episode mean diff - avg: {np.mean(all_mean_diffs):.4e}, std: {np.std(all_mean_diffs):.4e}")
            print(f"Per-episode max diff - avg: {np.mean(all_max_diffs):.4e}, std: {np.std(all_max_diffs):.4e}")
            print(f"Total mismatches (> {COMPARISON_TOLERANCE:.1e}): {total_mismatches}/{total_steps_compared} ({overall_mismatch_fraction*100:.2f}%)")
            
            if overall_max_diff > COMPARISON_TOLERANCE:
                print(f"[Warning] Differences exceed tolerance of {COMPARISON_TOLERANCE:.1e}")
            else:
                print(f"[OK] All differences within tolerance of {COMPARISON_TOLERANCE:.1e}")
            print("=" * 70 + "\n")
            
            # Add to summary
            summary["cpp_comparison"] = {
                "overall_max_diff": overall_max_diff,
                "overall_mean_diff": overall_mean_diff,
                "total_mismatches": total_mismatches,
                "total_steps": total_steps_compared,
                "mismatch_fraction": overall_mismatch_fraction,
                "per_episode_mean_diff_avg": float(np.mean(all_mean_diffs)),
                "per_episode_mean_diff_std": float(np.std(all_mean_diffs)),
                "per_episode_max_diff_avg": float(np.mean(all_max_diffs)),
                "per_episode_max_diff_std": float(np.std(all_max_diffs)),
            }


    # Log evaluation metrics to ClearML
    if clearml_task:
        # Log all summary metrics
        for metric_name, metric_value in summary.items():
            if isinstance(metric_value, (int, float)):
                log_metrics(clearml_task, {f"eval/{metric_name}": float(metric_value)}, step=0)
            elif isinstance(metric_value, dict):
                # Handle nested dictionaries (e.g., cpp_comparison)
                for nested_key, nested_value in metric_value.items():
                    if isinstance(nested_value, (int, float)):
                        log_metrics(clearml_task, {f"eval/{metric_name}/{nested_key}": float(nested_value)}, step=0)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        
        # Upload JSON summary to ClearML
        if clearml_task:
            upload_artifact(
                clearml_task,
                output_path,
                artifact_name="evaluation_summary",
                description="Evaluation metrics summary JSON",
            )

    if plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Fit PCA on all z latents if available (for per-episode plots)
        pca_model = None
        if sysid_enabled:
            # Check if any traces have z_latents
            z_available = any("z_latents" in t and t["z_latents"] is not None for t in traces)
            if z_available:
                from sklearn.decomposition import PCA
                # Collect all z latents
                all_z_for_pca = []
                for trace in traces:
                    if "z_latents" in trace and trace["z_latents"] is not None:
                        all_z_for_pca.append(trace["z_latents"])
                if len(all_z_for_pca) > 0:
                    z_concat = np.concatenate(all_z_for_pca, axis=0)
                    pca_model = PCA()
                    pca_model.fit(z_concat)
                    print(f"Fitted PCA on {z_concat.shape[0]} z samples for per-episode visualization")
        
        # Plot per-episode diagnostics (with PCA model if available)
        for trace in traces:
            plot_path = plot_dir / f"{trace['sequence_id']}.png"
            fig = plot_sequence_diagnostics(
                trace,
                plot_path,
                pca_model=pca_model
            )
            # Upload per-episode plot to ClearML as matplotlib figure (if enabled)
            if upload_plots_to_clearml and clearml_task and log_matplotlib_figure is not None:
                log_matplotlib_figure(
                    clearml_task,
                    fig,
                    title=f"Episode {trace['sequence_id']}",
                )
            plt.close(fig)  # Close after uploading
        
        summary_plot_path = plot_dir / "closed_loop_summary.png"
        fig = plot_summary(per_episode_metrics, summary_plot_path)
        if upload_plots_to_clearml and clearml_task and log_matplotlib_figure is not None:
            log_matplotlib_figure(
                clearml_task,
                fig,
                title="Closed Loop Summary",
            )
        plt.close(fig)  # Close after uploading
        
        # Plot comprehensive statistics across all episodes
        from evaluation.plotting import plot_profile_statistics, plot_z_latent_analysis
        profile_stats_path = plot_dir / "profile_statistics.png"
        fig = plot_profile_statistics(traces, profile_stats_path)
        if upload_plots_to_clearml and clearml_task and log_matplotlib_figure is not None:
            log_matplotlib_figure(
                clearml_task,
                fig,
                title="Profile Statistics",
            )
        plt.close(fig)  # Close after uploading
        
        # Plot z latent analysis if SysID is enabled
        if sysid_enabled:
            z_analysis_path = plot_dir / "z_latent_analysis.png"
            print("Creating z latent analysis...")
            fig = plot_z_latent_analysis(traces, vehicle_params_log, z_analysis_path)
            if upload_plots_to_clearml and clearml_task and log_matplotlib_figure is not None:
                log_matplotlib_figure(
                    clearml_task,
                    fig,
                    title="SysID Z Latent Analysis",
                )
            plt.close(fig)  # Close after uploading
        
        # Plot step function results if available
        if step_functions and len(step_traces) > 0:
            from evaluation.plotting import plot_step_function_results
            step_function_plot_path = plot_dir / "step_function_results.png"
            print("Creating step function results plot...")
            fig = plot_step_function_results(step_traces, step_function_plot_path)
            if upload_plots_to_clearml and clearml_task and log_matplotlib_figure is not None:
                log_matplotlib_figure(
                    clearml_task,
                    fig,
                    title="Step Function Results",
                )
            plt.close(fig)  # Close after uploading
        
        # Initial error step function analysis (if step_functions enabled)
        initial_error_traces = []
        if step_functions:
            print("\n[Initial Error Analysis] Evaluating throttle/brake response to different initial speed errors...")
            target_speed = 10.0  # Fixed target speed
            initial_errors = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]  # m/s error
            initial_speeds = [target_speed + err for err in initial_errors]
            profile_length = min(env.config.max_episode_steps, int(10.0 / env.config.dt))  # 10 seconds
            
            # Create progress bar for initial error analysis
            initial_error_pbar = tqdm(total=len(initial_errors), desc="Initial errors", unit="err")
            
            for initial_speed, initial_error in zip(initial_speeds, initial_errors):
                # Generate step function profile (constant target speed)
                step_profile = generate_step_function(initial_speed, target_speed, profile_length, env.config.dt)
                
                # Reset environment
                obs, _ = env.reset(options={"reference_profile": step_profile, "initial_speed": initial_speed})
                
                # Reset SysID encoder state
                if sysid_enabled:
                    py_encoder_hidden = encoder.reset(batch_size=1, device=device).squeeze(0)
                    feature_builder.reset()
                    prev_speed = initial_speed
                    prev_action = 0.0
                    prev_prev_action = 0.0
                    prev_prev_prev_action = 0.0
                
                # Reset PID controller
                pid_controller.reset()
                
                # Track episode data
                done = False
                actions: list[float] = []
                rl_actions: list[float] = []
                speeds: list[float] = []
                references: list[float] = []
                time_axis: list[float] = []
                step_idx = 0
                
                while not done and step_idx < profile_length:
                    current_speed = float(obs[0])
                    reference_speed = float(obs[4])
                    
                    # Compute z_t if SysID enabled
                    if sysid_enabled:
                        py_encoder_hidden, z_t = compute_z_online(
                            encoder=encoder,
                            feature_builder=feature_builder,
                            encoder_norm=encoder_norm,
                            h_prev=py_encoder_hidden,
                            v_t=current_speed,
                            u_t=prev_action,
                            device=device
                        )
                        z_t_np = z_t.cpu().numpy()
                        
                        # Adjust observation to match checkpoint's expected base_obs_dim
                        obs_adjusted = obs.copy()
                        if base_obs_dim_checkpoint is not None and len(obs) == base_obs_dim_current:
                            if base_obs_dim_checkpoint == base_obs_dim_without_error:
                                obs_adjusted = np.concatenate([obs[:4], obs[6:]])
                            elif base_obs_dim_checkpoint == base_obs_dim_without_grade:
                                obs_adjusted = np.concatenate([obs[:5], obs[6:]])
                        
                        obs_aug = np.concatenate([obs_adjusted, z_t_np])
                    else:
                        obs_aug = obs
                    
                    obs_tensor = torch.as_tensor(obs_aug, dtype=torch.float32, device=device).unsqueeze(0)
                    plan, stats = select_action(policy, obs_tensor, deterministic=True)
                    rl_action = float(plan[0])
                    
                    # PID action
                    speed_error = reference_speed - current_speed
                    pid_action = pid_controller.compute(speed_error, env.config.dt)
                    
                    # Final action
                    final_action = np.clip(rl_action + pid_action, -1.0, 1.0)
                    
                    rl_actions.append(rl_action)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(final_action)
                    
                    # Update SysID history
                    if sysid_enabled:
                        prev_prev_prev_action = prev_prev_action
                        prev_prev_action = prev_action
                        prev_action = rl_action
                        prev_speed = current_speed
                    
                    if pid_feedback_rl_only:
                        env._prev_action = rl_action
                    
                    actions.append(final_action)
                    speeds.append(float(info.get("speed", 0.0)))
                    references.append(float(info.get("reference_speed", 0.0)))
                    time_axis.append(step_idx * env.config.dt)
                    done = terminated or truncated
                    step_idx += 1
                
                # Create trace data
                initial_error_trace = {
                    "initial_error": initial_error,
                    "initial_speed": initial_speed,
                    "target_speed": target_speed,
                    "time": time_axis,
                    "speed": speeds,
                    "reference": references,
                    "action": actions,
                    "rl_action": rl_actions,
                }
                initial_error_traces.append(initial_error_trace)
                
                # Update progress bar
                initial_error_pbar.update(1)
            
            # Close progress bar
            initial_error_pbar.close()
            
            print(f"[Initial Error Analysis] Completed {len(initial_error_traces)} initial error evaluations.")
            
            # Plot initial error analysis
            if len(initial_error_traces) > 0:
                from evaluation.plotting import plot_initial_error_analysis
                initial_error_plot_path = plot_dir / "initial_error_analysis.png"
                print("Creating initial error analysis plot...")
                fig = plot_initial_error_analysis(initial_error_traces, initial_error_plot_path)
                if upload_plots_to_clearml and clearml_task and log_matplotlib_figure is not None:
                    log_matplotlib_figure(
                        clearml_task,
                        fig,
                        title="Initial Error Analysis",
                    )
                plt.close(fig)  # Close after uploading

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
    
    # Close ClearML task
    if clearml_task:
        close_task(clearml_task)
    
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
        force_export_onnx=args.force_export_onnx,
        step_functions=args.step_functions,
        upload_plots_to_clearml=args.upload_plots_to_clearml,
        disable_clearml=args.disable_clearml,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


