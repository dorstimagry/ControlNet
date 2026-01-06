"""Plotting utilities for SAC longitudinal evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_numpy(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float32)


def tanh_band(mu, sigma, p=0.68):
    """Compute the p-th percentile band for tanh-transformed Gaussian."""
    alpha_low = (1 - p) / 2
    alpha_high = 1 - alpha_low

    x_low = mu + sigma * norm.ppf(alpha_low)
    x_high = mu + sigma * norm.ppf(alpha_high)

    return np.tanh(x_low), np.tanh(x_high)


def plot_sequence_diagnostics(
    trace: Mapping[str, object],
    output_path: Path,
    pca_model: object | None = None
) -> None:
    """Create a multi-panel figure for a single trajectory.
    
    Args:
        trace: Episode trace data
        output_path: Path to save figure
        pca_model: Optional pre-fitted PCA model for z latent visualization
    """

    time = _to_numpy(trace["time"])
    speed = _to_numpy(trace["speed"])
    reference = _to_numpy(trace["reference"])
    acceleration = _to_numpy(trace["acceleration"])
    jerk = _to_numpy(trace.get("jerk", np.zeros_like(time)))
    V_cmd = _to_numpy(trace.get("V_cmd", np.zeros_like(time)))
    brake_torque = _to_numpy(trace.get("brake_torque", np.zeros_like(time)))
    brake_torque_max = float(trace.get("brake_torque_max", [10000.0])[0])  # Get max brake torque for this vehicle
    V_max = float(trace.get("V_max", [800.0])[0])  # Get max motor voltage for this vehicle

    # Get action arrays (final, RL, PID)
    action = _to_numpy(trace["action"])  # Final combined action
    rl_action = _to_numpy(trace.get("rl_action", action))  # RL action (fallback to final if not present)
    pid_action = _to_numpy(trace.get("pid_action", np.zeros_like(action)))  # PID action (default to zero if not present)
    
    # Convert actions to throttle/brake percentages
    # Throttle: max(0, action) * 100, Brake: max(0, -action) * 100
    throttle_pct_final = np.maximum(0, action) * 100.0
    brake_pct_final = np.maximum(0, -action) * 100.0
    throttle_pct_rl = np.maximum(0, rl_action) * 100.0
    brake_pct_rl = np.maximum(0, -rl_action) * 100.0
    throttle_pct_pid = np.maximum(0, pid_action) * 100.0
    brake_pct_pid = np.maximum(0, -pid_action) * 100.0
    policy_mean = _to_numpy(trace["policy_pre_tanh_mean"])
    policy_log_std = _to_numpy(trace["policy_log_std"])
    # Force data
    tire_force = _to_numpy(trace.get("tire_force", np.zeros_like(time)))
    drag_force = _to_numpy(trace.get("drag_force", np.zeros_like(time)))
    rolling_force = _to_numpy(trace.get("rolling_force", np.zeros_like(time)))
    grade_force = _to_numpy(trace.get("grade_force", np.zeros_like(time)))
    net_force = _to_numpy(trace.get("net_force", np.zeros_like(time)))

    # Motor data
    V_cmd = _to_numpy(trace.get("V_cmd", np.zeros_like(time)))  # input motor voltage
    back_emf_voltage = _to_numpy(trace.get("back_emf_voltage", np.zeros_like(time)))  # back-EMF voltage
    V_max = _to_numpy(trace.get("V_max", np.zeros_like(time)))
    effective_voltage = V_cmd - back_emf_voltage  # effective motor voltage

    # Check if we have z latents and PCA model
    has_z_latents = "z_latents" in trace and trace["z_latents"] is not None and pca_model is not None
    num_subplots = 10 if has_z_latents else 9
    figsize_height = 30 if has_z_latents else 27
    
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, figsize_height), sharex=True)

    # 1. Speed and ref speed
    v_max_feasible = trace.get("v_max_feasible", None)  # Generation max speed (with safety factor)
    v_max_theoretical = trace.get("v_max_theoretical", None)  # Theoretical max speed (back-EMF = V_max)
    axes[0].plot(time, speed, label="Speed")
    axes[0].plot(time, reference, label="Reference", linestyle="--")
    if v_max_theoretical is not None:
        axes[0].axhline(
            y=v_max_theoretical,
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.5,
            label=f"Theoretical max speed ({v_max_theoretical:.1f} m/s)",
        )
    if v_max_feasible is not None:
        axes[0].axhline(
            y=v_max_feasible,
            color="#d62728",
            linestyle=":",
            linewidth=2,
            label=f"Generation max speed ({v_max_feasible:.1f} m/s)",
        )

    # Set ylim based on speed and reference data only (not max feasible speed)
    if len(speed) > 0 and len(reference) > 0:
        y_min = min(np.min(speed), np.min(reference))
        y_max = max(np.max(speed), np.max(reference))
        y_range = y_max - y_min
        y_margin = 0.05 * y_range if y_range > 0 else 0.1
        axes[0].set_ylim(y_min - y_margin, y_max + y_margin)
    else:
        axes[0].set_ylim(0, 1.0)

    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Speed Tracking")

    # 2. Acceleration
    axes[1].plot(time, acceleration, label="Acceleration")
    axes[1].set_ylabel("Acceleration (m/s²)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)
    axes[1].set_title("Vehicle Acceleration")

    # 3. Jerk
    axes[2].plot(time, jerk, label="Jerk")
    axes[2].set_ylabel("Jerk (m/s³)")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)
    axes[2].set_title("Vehicle Jerk")

    # 4. Throttle %
    axes[3].plot(time, throttle_pct_final, label="Final throttle", color="#1b9e77", linewidth=2)
    axes[3].plot(time, throttle_pct_rl, label="RL throttle", color="#7570b3", linestyle="--", linewidth=1.5)
    axes[3].plot(time, throttle_pct_pid, label="PID throttle", color="#e7298a", linestyle="--", linewidth=1.5)
    axes[3].set_ylabel("Throttle %")
    axes[3].set_ylim(0, 100)
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.3)
    axes[3].set_title("Motor Throttle Command")

    # 5. Brake %
    axes[4].plot(time, brake_pct_final, label="Final brake", color="#d95f02", linewidth=2)
    axes[4].plot(time, brake_pct_rl, label="RL brake", color="#7570b3", linestyle="--", linewidth=1.5)
    axes[4].plot(time, brake_pct_pid, label="PID brake", color="#e7298a", linestyle="--", linewidth=1.5)
    axes[4].set_ylabel("Brake %")
    axes[4].set_ylim(0, 100)
    axes[4].legend(loc="upper right")
    axes[4].grid(alpha=0.3)
    axes[4].set_title("Brake Command")

    # 6. External forces (resistive forces acting on vehicle)
    axes[5].plot(time, drag_force, label="Drag force (N)", color="#ff7f0e")
    axes[5].plot(time, rolling_force, label="Rolling force (N)", color="#2ca02c")
    axes[5].plot(time, grade_force, label="Grade force (N)", color="#d62728")
    axes[5].set_ylabel("External Forces (N)")
    axes[5].legend(loc="upper right")
    axes[5].grid(alpha=0.3)
    axes[5].set_title("External Resistive Forces")

    # 7. Force balance (external + tire = net)
    total_external = -(drag_force + rolling_force + grade_force)  # Total external force on vehicle
    axes[6].plot(time, total_external, label="Total external (N)", color="#17becf", linewidth=2)
    axes[6].plot(time, tire_force, label="Tire force (N)", color="#1f77b4", linewidth=2)
    axes[6].plot(time, net_force, label="Net force (N)", color="#e377c2", linewidth=2)
    axes[6].set_ylabel("Total Forces (N)")
    axes[6].legend(loc="upper right")
    axes[6].grid(alpha=0.3)
    axes[6].set_title("Force Balance: External + Tire = Net")

    # 8. Motor state (voltages)
    axes[7].plot(time, V_cmd, label="Input voltage (V)", color="#2ca02c", linewidth=2)
    axes[7].plot(time, back_emf_voltage, label="Back-EMF (V)", color="#d62728", linewidth=2)
    axes[7].plot(time, effective_voltage, label="Effective voltage (V)", color="#1f77b4", linewidth=2)
    # Add horizontal line for max voltage (use first value since it's constant per episode)
    if len(V_max) > 0:
        axes[7].axhline(y=V_max[0], color="#ff7f0e", linestyle="--", linewidth=2, label=f"Max voltage ({V_max[0]:.1f}V)")
    axes[7].set_ylabel("Voltage (V)")
    axes[7].legend(loc="upper right")
    axes[7].grid(alpha=0.3)
    axes[7].set_title("Motor Electrical State")

    # 9. Policy mu post-tanh with 68th percentile band
    policy_std = np.exp(policy_log_std)  # Convert log std to std
    policy_mean_tanh = np.tanh(policy_mean)  # Post-tanh mean
    y_low, y_high = tanh_band(policy_mean, policy_std, p=0.68)  # 68th percentile band

    axes[8].plot(time, policy_mean_tanh, label="Policy μ (post-tanh)", color="#1f77b4", linewidth=2)
    axes[8].fill_between(time, y_low, y_high, alpha=0.3, color="#1f77b4", label="68th percentile")
    axes[8].set_ylabel("Policy μ (post-tanh)")
    axes[8].set_xlabel("Time (s)")
    axes[8].legend(loc="upper right")
    axes[8].grid(alpha=0.3)
    axes[8].set_title("Policy Output Distribution")

    # 10. Z Latent Principal Components (if available)
    if has_z_latents:
        z_latents = trace["z_latents"]  # Shape: (timesteps, z_dim)
        # Transform using PCA
        z_pca = pca_model.transform(z_latents)
        
        # Plot first 3 PCs
        n_pcs_to_plot = min(3, z_pca.shape[1])
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for i in range(n_pcs_to_plot):
            var_explained = pca_model.explained_variance_ratio_[i] * 100
            axes[9].plot(
                time[:len(z_pca)],  # Match z_pca length (in case of mismatch)
                z_pca[:, i],
                label=f"PC{i+1} ({var_explained:.1f}%)",
                color=colors[i],
                linewidth=2
            )
        axes[9].set_ylabel("PC Value")
        axes[9].set_xlabel("Time (s)")
        axes[9].legend(loc="upper right")
        axes[9].grid(alpha=0.3)
        axes[9].set_title("SysID Latent (z) Principal Components Over Time")

    fig.suptitle(f"Sequence diagnostics: {trace.get('sequence_id', 'unknown')}")
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_summary(metrics: list[Mapping[str, float]], output_path: Path) -> None:
    """Plot aggregate statistics across all sequences/episodes."""

    seq_ids = [str(m.get("sequence_id", idx)) for idx, m in enumerate(metrics)]
    rmse = np.asarray([m.get("rmse", np.nan) for m in metrics], dtype=np.float32)
    smooth = np.asarray([m.get("smoothness", np.nan) for m in metrics], dtype=np.float32)
    extreme = np.asarray([m.get("extreme_fraction", np.nan) for m in metrics], dtype=np.float32)
    action_var = np.asarray([m.get("action_variance", np.nan) for m in metrics], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(seq_ids, rmse)
    axes[0, 0].set_title("RMSE per sequence")
    axes[0, 0].set_ylabel("RMSE (m/s)")
    axes[0, 0].tick_params(axis="x", rotation=45, labelsize=8)

    axes[0, 1].bar(seq_ids, smooth, color="#ffb347")
    axes[0, 1].set_title("Smoothness penalty per sequence")
    axes[0, 1].set_ylabel("Variance of Δaction")
    axes[0, 1].tick_params(axis="x", rotation=45, labelsize=8)

    axes[1, 0].plot(seq_ids, extreme, marker="o", color="#8dd3c7")
    axes[1, 0].set_title("Extreme action fraction")
    axes[1, 0].set_ylabel("Fraction")
    axes[1, 0].tick_params(axis="x", rotation=45, labelsize=8)

    axes[1, 1].hist(action_var[~np.isnan(action_var)], bins=10, color="#80b1d3")
    axes[1, 1].set_title("Distribution of action variance")
    axes[1, 1].set_xlabel("Variance of Δaction")

    for ax in axes.flat:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feasibility_diagnostics(trace: dict, vehicle_caps, output_path: Path) -> None:
    """Plot feasibility diagnostics for a profile.

    Args:
        trace: Evaluation trace with original and feasible profiles
        vehicle_caps: VehicleCapabilities object
        output_path: Path to save the plot
    """
    from utils.data_utils import feasible_accel_bounds

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Extract data
    dt = 0.1  # Assume 10Hz
    original_v = _to_numpy(trace.get("original_reference", []))
    feasible_v = _to_numpy(trace.get("feasible_reference", []))
    time = np.arange(len(feasible_v)) * dt

    if len(original_v) == 0 or len(feasible_v) == 0:
        plt.close(fig)
        return

    # 1. Speed profiles comparison
    axes[0].plot(time, original_v, label="Original", color="#ff7f0e", linestyle="--", alpha=0.7)
    axes[0].plot(time, feasible_v, label="Feasible", color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Profile Feasibility: Original vs Feasible")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Acceleration bounds and requested acceleration
    grade_profile = np.zeros_like(feasible_v)  # Assume flat road for diagnostics

    a_req = np.zeros(len(feasible_v) - 1)
    a_min_bounds = np.zeros(len(feasible_v) - 1)
    a_max_bounds = np.zeros(len(feasible_v) - 1)

    for k in range(len(a_req)):
        a_req[k] = (feasible_v[k+1] - feasible_v[k]) / dt
        a_min_bounds[k], a_max_bounds[k] = feasible_accel_bounds(
            feasible_v[k], grade_profile[k], vehicle_caps, safety_margin=0.9
        )

    time_accel = time[:-1] + dt/2  # Center on acceleration intervals

    axes[1].fill_between(time_accel, a_min_bounds, a_max_bounds, alpha=0.3, color="#2ca02c",
                        label="Feasible range")
    axes[1].plot(time_accel, a_req, color="#d62728", linewidth=2, label="Requested")
    axes[1].set_ylabel("Acceleration (m/s²)")
    axes[1].set_title("Acceleration Feasibility Check")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Check for violations
    violations = 0
    for k in range(len(a_req)):
        if not (a_min_bounds[k] - 1e-3 <= a_req[k] <= a_max_bounds[k] + 1e-3):
            violations += 1
            axes[1].plot(time_accel[k], a_req[k], 'rx', markersize=8, markeredgewidth=2)

    # 3. Force analysis
    F_drive_max = np.zeros_like(feasible_v)
    F_brake_max = np.zeros_like(feasible_v)
    F_drag = np.zeros_like(feasible_v)
    F_roll = np.zeros_like(feasible_v)
    F_grade = np.zeros_like(feasible_v)

    for k in range(len(feasible_v)):
        F_drag[k] = 0.5 * vehicle_caps.rho * vehicle_caps.C_dA * feasible_v[k]**2
        F_roll[k] = vehicle_caps.C_r * vehicle_caps.m * 9.80665
        F_grade[k] = vehicle_caps.m * 9.80665 * np.sin(grade_profile[k])
        F_fric = vehicle_caps.mu * vehicle_caps.m * 9.80665

        F_drive_max[k] = 0.9 * min(vehicle_caps.T_drive_max / vehicle_caps.r_w, F_fric)
        F_brake_max[k] = 0.9 * min(vehicle_caps.T_brake_max / vehicle_caps.r_w, F_fric)

    axes[2].plot(time, F_drive_max, label="Max drive force", color="#1f77b4")
    axes[2].plot(time, F_brake_max, label="Max brake force", color="#d62728")
    axes[2].plot(time, F_drag + F_roll + F_grade, label="Resistive forces", color="#ff7f0e")
    axes[2].set_ylabel("Force (N)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Force Analysis")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    # Add text annotation with summary
    profile_feasible = trace.get("profile_feasible", True)
    max_adjustment = trace.get("max_profile_adjustment", 0.0)

    status_text = f"Feasible: {profile_feasible}\nMax adjustment: {max_adjustment:.3f} m/s\nAccel violations: {violations}"
    axes[0].text(0.02, 0.98, status_text, transform=axes[0].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_profile_statistics(traces: list[Mapping[str, object]], output_path: Path) -> None:
    """Plot comprehensive statistics across all episodes.
    
    Includes:
    - Speed distribution
    - Acceleration distribution  
    - Reference acceleration distribution
    - Policy uncertainty (std) vs speed
    - Policy uncertainty vs acceleration
    - Action distribution (RL actions)
    - Speed tracking error distribution
    
    Args:
        traces: List of trace dictionaries from all episodes
        output_path: Path to save the statistics plot
    """
    # Aggregate data from all traces
    all_speeds = []
    all_references = []
    all_accelerations = []
    all_ref_accelerations = []
    all_rl_actions = []
    all_policy_stds = []
    all_errors = []
    
    for trace in traces:
        speeds = _to_numpy(trace.get("speed", []))
        references = _to_numpy(trace.get("reference", []))
        accelerations = _to_numpy(trace.get("acceleration", []))
        rl_actions = _to_numpy(trace.get("rl_action", []))
        policy_log_std = _to_numpy(trace.get("policy_log_std", []))
        
        # Compute reference accelerations (derivative of reference speed)
        if len(references) > 1:
            time = _to_numpy(trace.get("time", []))
            dt = time[1] - time[0] if len(time) > 1 else 0.1
            ref_accel = np.diff(references) / dt
            all_ref_accelerations.extend(ref_accel)
        
        # Tracking errors
        if len(speeds) == len(references):
            errors = speeds - references
            all_errors.extend(errors)
        
        all_speeds.extend(speeds)
        all_references.extend(references)
        all_accelerations.extend(accelerations)
        all_rl_actions.extend(rl_actions)
        all_policy_stds.extend(np.exp(policy_log_std))  # Convert log std to std
    
    # Convert to numpy arrays
    all_speeds = np.array(all_speeds)
    all_references = np.array(all_references)
    all_accelerations = np.array(all_accelerations)
    all_ref_accelerations = np.array(all_ref_accelerations)
    all_rl_actions = np.array(all_rl_actions)
    all_policy_stds = np.array(all_policy_stds)
    all_errors = np.array(all_errors)
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. Speed distribution (actual)
    axes[0, 0].hist(all_speeds, bins=50, color="#1f77b4", alpha=0.7, edgecolor="black")
    axes[0, 0].set_xlabel("Speed (m/s)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title(f"Speed Distribution\n(μ={np.mean(all_speeds):.2f}, σ={np.std(all_speeds):.2f})")
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Reference speed distribution
    axes[0, 1].hist(all_references, bins=50, color="#ff7f0e", alpha=0.7, edgecolor="black")
    axes[0, 1].set_xlabel("Reference Speed (m/s)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title(f"Reference Speed Distribution\n(μ={np.mean(all_references):.2f}, σ={np.std(all_references):.2f})")
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Tracking error distribution
    axes[0, 2].hist(all_errors, bins=50, color="#d62728", alpha=0.7, edgecolor="black")
    axes[0, 2].set_xlabel("Error (m/s)")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title(f"Speed Tracking Error\n(μ={np.mean(all_errors):.3f}, σ={np.std(all_errors):.3f})")
    axes[0, 2].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[0, 2].grid(alpha=0.3)
    
    # 4. Acceleration distribution (actual)
    axes[1, 0].hist(all_accelerations, bins=50, color="#2ca02c", alpha=0.7, edgecolor="black")
    axes[1, 0].set_xlabel("Acceleration (m/s²)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title(f"Acceleration Distribution\n(μ={np.mean(all_accelerations):.3f}, σ={np.std(all_accelerations):.3f})")
    axes[1, 0].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Reference acceleration distribution
    axes[1, 1].hist(all_ref_accelerations, bins=50, color="#9467bd", alpha=0.7, edgecolor="black")
    axes[1, 1].set_xlabel("Reference Acceleration (m/s²)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title(f"Reference Acceleration Distribution\n(μ={np.mean(all_ref_accelerations):.3f}, σ={np.std(all_ref_accelerations):.3f})")
    axes[1, 1].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].grid(alpha=0.3)
    
    # 6. RL Action distribution
    axes[1, 2].hist(all_rl_actions, bins=50, color="#8c564b", alpha=0.7, edgecolor="black")
    axes[1, 2].set_xlabel("RL Action")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].set_title(f"RL Action Distribution\n(μ={np.mean(all_rl_actions):.3f}, σ={np.std(all_rl_actions):.3f})")
    axes[1, 2].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[1, 2].set_xlim(-1.05, 1.05)
    axes[1, 2].grid(alpha=0.3)
    
    # 7. Policy uncertainty vs (Speed, Acceleration) - 2D heatmap showing mean uncertainty
    # Match arrays to same length (acceleration is typically one element shorter due to diff)
    min_len = min(len(all_speeds), len(all_accelerations), len(all_policy_stds))
    speeds_matched = all_speeds[:min_len]
    accels_matched = all_accelerations[:min_len]
    stds_matched = all_policy_stds[:min_len]
    
    # Create 2D bins and compute mean uncertainty in each bin
    from scipy.stats import binned_statistic_2d
    
    # Define bin edges
    speed_bins = 40
    accel_bins = 40
    
    # Compute mean uncertainty in each (speed, acceleration) bin
    statistic, speed_edges, accel_edges, _ = binned_statistic_2d(
        speeds_matched, 
        accels_matched, 
        stds_matched,
        statistic='mean',
        bins=[speed_bins, accel_bins]
    )
    
    # Create meshgrid for plotting
    speed_centers = (speed_edges[:-1] + speed_edges[1:]) / 2
    accel_centers = (accel_edges[:-1] + accel_edges[1:]) / 2
    
    # Plot heatmap
    im = axes[2, 0].pcolormesh(speed_centers, accel_centers, statistic.T, 
                               cmap="viridis", shading='auto')
    axes[2, 0].set_xlabel("Speed (m/s)")
    axes[2, 0].set_ylabel("Acceleration (m/s²)")
    axes[2, 0].set_title("Mean Policy Uncertainty vs Speed & Acceleration")
    cbar = plt.colorbar(im, ax=axes[2, 0], label="Mean Policy Std Dev")
    axes[2, 0].grid(alpha=0.3, color='white', linewidth=0.5)
    
    # 8. Speed vs Acceleration distribution (2D histogram for context)
    h = axes[2, 1].hist2d(speeds_matched, accels_matched, bins=[40, 40], 
                          cmap="plasma", cmin=1)
    axes[2, 1].set_xlabel("Speed (m/s)")
    axes[2, 1].set_ylabel("Acceleration (m/s²)")
    axes[2, 1].set_title("Speed vs Acceleration Distribution")
    plt.colorbar(h[3], ax=axes[2, 1], label="Count")
    axes[2, 1].grid(alpha=0.3)
    
    # 9. Policy uncertainty distribution
    axes[2, 2].hist(all_policy_stds, bins=50, color="#e377c2", alpha=0.7, edgecolor="black")
    axes[2, 2].set_xlabel("Policy Std Dev (pre-tanh)")
    axes[2, 2].set_ylabel("Frequency")
    axes[2, 2].set_title(f"Policy Uncertainty Distribution\n(μ={np.mean(all_policy_stds):.3f}, σ={np.std(all_policy_stds):.3f})")
    axes[2, 2].grid(alpha=0.3)
    
    # Overall title
    fig.suptitle(f"Profile and Policy Statistics Across {len(traces)} Episodes", fontsize=16, y=0.995)
    
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_z_latent_analysis(
    traces: list[dict],
    vehicle_params_log: list[dict],
    output_path: Path,
) -> None:
    """
    Analyze SysID latent z to verify it represents vehicle dynamics.
    
    Creates comprehensive visualization showing:
    - PCA-reduced z colored by various vehicle parameters
    - Correlation between PCs and vehicle parameters
    - Z dimension statistics and distributions
    
    Args:
        traces: List of episode traces with z_latents field
        vehicle_params_log: List of vehicle parameters per episode
        output_path: Path to save the figure
    """
    from sklearn.decomposition import PCA
    from scipy.stats import pearsonr
    
    # Collect all z values and corresponding vehicle parameters
    all_z = []
    all_vehicle_params = []
    episode_labels = []
    
    for trace, veh_params in zip(traces, vehicle_params_log):
        if "z_latents" in trace and trace["z_latents"] is not None:
            z_array = trace["z_latents"]
            for z_step in z_array:
                all_z.append(z_step)
                all_vehicle_params.append(veh_params)
                episode_labels.append(veh_params["episode_id"])
    
    if len(all_z) == 0:
        print("Warning: No z latent data available for analysis")
        return
    
    # Convert to numpy arrays
    z_matrix = np.array(all_z)  # Shape: (N_steps, z_dim)
    episode_labels = np.array(episode_labels)
    
    print(f"Analyzing {z_matrix.shape[0]} z samples from {len(traces)} episodes")
    print(f"Z dimension: {z_matrix.shape[1]}")
    
    # Fit PCA
    pca = PCA()
    z_pca = pca.fit_transform(z_matrix)
    
    # Extract vehicle parameters into arrays
    param_names = [
        "mass", "drag_area", "rolling_coeff", "actuator_tau",
        "motor_gear_ratio", "motor_K_t", "motor_K_e", "motor_R", "motor_b",
        "brake_tau", "brake_T_max", "brake_mu",
        "wheel_radius", "wheel_inertia"
    ]
    
    param_arrays = {}
    for param_name in param_names:
        param_arrays[param_name] = np.array([vp[param_name] for vp in all_vehicle_params])
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    # ============== Row 1: PCA Scatter Plots ==============
    
    # 1. PCA colored by mass
    scatter1 = axes[0, 0].scatter(
        z_pca[:, 0], z_pca[:, 1],
        c=param_arrays["mass"],
        cmap="viridis",
        alpha=0.5,
        s=10
    )
    axes[0, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0, 0].set_title("Z Latent (PCA) colored by Mass")
    plt.colorbar(scatter1, ax=axes[0, 0], label="Mass (kg)")
    axes[0, 0].grid(alpha=0.3)
    
    # 2. PCA colored by drag_area
    scatter2 = axes[0, 1].scatter(
        z_pca[:, 0], z_pca[:, 1],
        c=param_arrays["drag_area"],
        cmap="plasma",
        alpha=0.5,
        s=10
    )
    axes[0, 1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0, 1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0, 1].set_title("Z Latent (PCA) colored by Drag Area")
    plt.colorbar(scatter2, ax=axes[0, 1], label="Drag Area (m²)")
    axes[0, 1].grid(alpha=0.3)
    
    # 3. PCA explained variance
    n_components = min(12, len(pca.explained_variance_ratio_))
    axes[0, 2].bar(
        range(1, n_components + 1),
        pca.explained_variance_ratio_[:n_components] * 100,
        color="steelblue",
        alpha=0.7,
        edgecolor="black"
    )
    axes[0, 2].plot(
        range(1, n_components + 1),
        np.cumsum(pca.explained_variance_ratio_[:n_components]) * 100,
        'ro-',
        linewidth=2,
        label="Cumulative"
    )
    axes[0, 2].set_xlabel("Principal Component")
    axes[0, 2].set_ylabel("Variance Explained (%)")
    axes[0, 2].set_title("PCA Explained Variance")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    axes[0, 2].set_xticks(range(1, n_components + 1))
    
    # ============== Row 2: More PCA Scatter + Correlations ==============
    
    # 4. PCA colored by motor gear ratio
    scatter4 = axes[1, 0].scatter(
        z_pca[:, 0], z_pca[:, 1],
        c=param_arrays["motor_gear_ratio"],
        cmap="coolwarm",
        alpha=0.5,
        s=10
    )
    axes[1, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1, 0].set_title("Z Latent (PCA) colored by Motor Gear Ratio")
    plt.colorbar(scatter4, ax=axes[1, 0], label="Gear Ratio")
    axes[1, 0].grid(alpha=0.3)
    
    # 5. PCA colored by episode (to show grouping)
    scatter5 = axes[1, 1].scatter(
        z_pca[:, 0], z_pca[:, 1],
        c=episode_labels,
        cmap="tab20",
        alpha=0.5,
        s=10
    )
    axes[1, 1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1, 1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1, 1].set_title("Z Latent (PCA) colored by Episode ID")
    plt.colorbar(scatter5, ax=axes[1, 1], label="Episode")
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Correlation heatmap: PCs vs vehicle parameters
    n_pcs_to_show = min(12, z_pca.shape[1])
    correlations = np.zeros((len(param_names), n_pcs_to_show))
    
    for i, param_name in enumerate(param_names):
        for j in range(n_pcs_to_show):
            # Compute correlation between PC and vehicle parameter
            corr, _ = pearsonr(z_pca[:, j], param_arrays[param_name])
            correlations[i, j] = corr
    
    im = axes[1, 2].imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1, 2].set_xlabel("Principal Component")
    axes[1, 2].set_ylabel("Vehicle Parameter")
    axes[1, 2].set_title("Correlation: PCs vs Vehicle Parameters")
    axes[1, 2].set_xticks(range(n_pcs_to_show))
    axes[1, 2].set_xticklabels([f"PC{i+1}" for i in range(n_pcs_to_show)], rotation=45)
    axes[1, 2].set_yticks(range(len(param_names)))
    axes[1, 2].set_yticklabels(param_names, fontsize=8)
    plt.colorbar(im, ax=axes[1, 2], label="Pearson r")
    
    # Add correlation values as text
    for i in range(len(param_names)):
        for j in range(n_pcs_to_show):
            text_color = "white" if abs(correlations[i, j]) > 0.5 else "black"
            axes[1, 2].text(j, i, f"{correlations[i, j]:.2f}",
                          ha="center", va="center", color=text_color, fontsize=6)
    
    # ============== Row 3: Distributions and Statistics ==============
    
    # 7. PC1 and PC2 distributions
    axes[2, 0].hist(z_pca[:, 0], bins=50, alpha=0.6, color="steelblue", label="PC1", edgecolor="black")
    axes[2, 0].hist(z_pca[:, 1], bins=50, alpha=0.6, color="coral", label="PC2", edgecolor="black")
    axes[2, 0].set_xlabel("Principal Component Value")
    axes[2, 0].set_ylabel("Frequency")
    axes[2, 0].set_title("PC1 and PC2 Distributions")
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)
    
    # 8. Z dimension statistics (box plot of raw z dimensions)
    z_dim = z_matrix.shape[1]
    axes[2, 1].boxplot(
        [z_matrix[:, i] for i in range(z_dim)],
        labels=[f"z{i+1}" for i in range(z_dim)],
        showfliers=False
    )
    axes[2, 1].set_xlabel("Z Dimension")
    axes[2, 1].set_ylabel("Value")
    axes[2, 1].set_title("Raw Z Dimension Distributions")
    axes[2, 1].grid(alpha=0.3, axis='y')
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    # 9. 2D density (PC1 vs PC2)
    h = axes[2, 2].hist2d(z_pca[:, 0], z_pca[:, 1], bins=50, cmap="Blues", cmin=1)
    axes[2, 2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[2, 2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[2, 2].set_title("Z Latent Density (PC1 vs PC2)")
    plt.colorbar(h[3], ax=axes[2, 2], label="Count")
    axes[2, 2].grid(alpha=0.3)
    
    # Overall title
    fig.suptitle(
        f"SysID Latent (z) Analysis: {len(traces)} Episodes, {z_matrix.shape[0]} Timesteps",
        fontsize=16,
        y=0.995
    )
    
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"Z latent analysis saved to {output_path}")


def plot_step_function_results(step_traces: list[dict], output_path: Path) -> None:
    """Plot all step function evaluation results in a single figure with multiple subplots.
    
    Creates a 10x5 grid of subplots with alternating rows:
    - Even rows (0, 2, 4, 6, 8): Speed tracking plots (reference step function and actual speed)
    - Odd rows (1, 3, 5, 7, 9): Actuation plots (action commands) directly below each speed plot
    
    Each column represents an end speed [0, 5, 10, 15, 20] m/s.
    Each row pair (speed + actuation) represents a start speed [0, 5, 10, 15, 20] m/s.
    
    Args:
        step_traces: List of trace dictionaries from step function evaluations.
                    Each trace should have 'start_speed', 'end_speed', 'time', 'speed', 'reference', 'action'
        output_path: Path to save the figure
    """
    start_speeds = [0.0, 5.0, 10.0, 15.0, 20.0]
    end_speeds = [0.0, 5.0, 10.0, 15.0, 20.0]
    
    # Create a dictionary for quick lookup: (start_speed, end_speed) -> trace
    trace_dict = {}
    for trace in step_traces:
        start = trace.get("start_speed")
        end = trace.get("end_speed")
        if start is not None and end is not None:
            trace_dict[(start, end)] = trace
    
    # Create figure with 10x5 subplots (alternating speed and actuation rows)
    fig, axes = plt.subplots(10, 5, figsize=(20, 40), sharex='col')
    
    # Iterate over all combinations
    for i, start_speed in enumerate(start_speeds):
        for j, end_speed in enumerate(end_speeds):
            key = (start_speed, end_speed)
            
            # Speed subplot (even rows: 0, 2, 4, 6, 8)
            ax_speed = axes[i * 2, j]
            
            if key in trace_dict:
                trace = trace_dict[key]
                time = _to_numpy(trace.get("time", []))
                speed = _to_numpy(trace.get("speed", []))
                reference = _to_numpy(trace.get("reference", []))
                action = _to_numpy(trace.get("action", []))
                v_max_theoretical = trace.get("v_max_theoretical")
                is_feasible = trace.get("is_feasible", True)
                
                # Plot reference (step function) as dashed line
                ax_speed.plot(time, reference, label="Reference", linestyle="--", color="#ff7f0e", linewidth=1.5)
                # Plot actual speed as solid line
                ax_speed.plot(time, speed, label="Speed", color="#1f77b4", linewidth=1.5)
                
                # Add max feasible speed line if available
                if v_max_theoretical is not None:
                    ax_speed.axhline(
                        y=v_max_theoretical,
                        color="#d62728",
                        linestyle=":",
                        linewidth=1,
                        alpha=0.7,
                        label=f"Max feasible ({v_max_theoretical:.1f} m/s)"
                    )
                
                # Set title with feasibility indicator
                title = f"Start: {start_speed:.0f} m/s → End: {end_speed:.0f} m/s"
                if not is_feasible:
                    title += " [INFEASIBLE]"
                ax_speed.set_title(title, fontsize=10, color="#d62728" if not is_feasible else "black")
                
                # Set labels only on edges
                if i == 4:  # Last speed row
                    # X-label will be on actuation plot below
                    pass
                if j == 0:  # Left column
                    ax_speed.set_ylabel("Speed (m/s)", fontsize=9)
                
                ax_speed.grid(alpha=0.3)
                ax_speed.legend(loc="upper right", fontsize=7)
                
                # Actuation subplot (odd rows: 1, 3, 5, 7, 9) - directly below speed plot
                ax_action = axes[i * 2 + 1, j]
                
                if len(action) > 0:
                    ax_action.plot(time, action, label="Action", color="#2ca02c", linewidth=1.5)
                    ax_action.axhline(y=0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
                    ax_action.set_ylim(-1.1, 1.1)
                    
                    # Set labels only on edges
                    if i == 4:  # Bottom row of actuation plots
                        ax_action.set_xlabel("Time (s)", fontsize=9)
                    if j == 0:  # Left column
                        ax_action.set_ylabel("Action", fontsize=9)
                    
                    ax_action.grid(alpha=0.3)
                    ax_action.legend(loc="upper right", fontsize=7)
                else:
                    ax_action.text(0.5, 0.5, "No action data", ha="center", va="center", transform=ax_action.transAxes)
                    ax_action.set_ylim(-1.1, 1.1)
                    if i == 4:
                        ax_action.set_xlabel("Time (s)", fontsize=9)
                    if j == 0:
                        ax_action.set_ylabel("Action", fontsize=9)
                    ax_action.grid(alpha=0.3)
            else:
                # No data for this combination
                ax_speed.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_speed.transAxes)
                ax_speed.set_title(f"Start: {start_speed:.0f} m/s → End: {end_speed:.0f} m/s", fontsize=10)
                
                # Empty actuation subplot
                ax_action = axes[i * 2 + 1, j]
                ax_action.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_action.transAxes)
                
                # Set labels only on edges
                if i == 4:
                    ax_action.set_xlabel("Time (s)", fontsize=9)
                if j == 0:
                    ax_speed.set_ylabel("Speed (m/s)", fontsize=9)
                    ax_action.set_ylabel("Action", fontsize=9)
                
                ax_speed.grid(alpha=0.3)
                ax_action.grid(alpha=0.3)
    
    # Overall title
    fig.suptitle("Step Function Evaluation Results", fontsize=16, y=0.995)
    
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"Step function results plot saved to {output_path}")


__all__ = ["plot_sequence_diagnostics", "plot_summary", "plot_feasibility_diagnostics", "plot_profile_statistics", "plot_z_latent_analysis", "plot_step_function_results"]


