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


def plot_sequence_diagnostics(trace: Mapping[str, object], output_path: Path) -> None:
    """Create a multi-panel figure for a single trajectory."""

    time = _to_numpy(trace["time"])
    speed = _to_numpy(trace["speed"])
    reference = _to_numpy(trace["reference"])
    acceleration = _to_numpy(trace["acceleration"])
    jerk = _to_numpy(trace.get("jerk", np.zeros_like(time)))
    V_cmd = _to_numpy(trace.get("V_cmd", np.zeros_like(time)))
    brake_torque = _to_numpy(trace.get("brake_torque", np.zeros_like(time)))
    brake_torque_max = float(trace.get("brake_torque_max", [10000.0])[0])  # Get max brake torque for this vehicle
    V_max = float(trace.get("V_max", [800.0])[0])  # Get max motor voltage for this vehicle

    # Convert to percentages (V_cmd is 0-V_max, brake torque uses actual vehicle max)
    throttle_pct = (V_cmd / V_max) * 100.0  # Convert to percentage using actual vehicle V_max
    brake_pct = (brake_torque / brake_torque_max) * 100.0  # Convert to percentage using actual vehicle max
    action = _to_numpy(trace["action"])
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

    fig, axes = plt.subplots(9, 1, figsize=(12, 27), sharex=True)

    # 1. Speed and ref speed
    axes[0].plot(time, speed, label="Speed")
    axes[0].plot(time, reference, label="Reference", linestyle="--")
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
    axes[3].plot(time, throttle_pct, label="Throttle %", color="#1b9e77")
    axes[3].set_ylabel("Throttle %")
    axes[3].set_ylim(0, 100)
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.3)
    axes[3].set_title("Motor Throttle Command")

    # 5. Brake %
    axes[4].plot(time, brake_pct, label="Brake %", color="#d95f02")
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


__all__ = ["plot_sequence_diagnostics", "plot_summary", "plot_feasibility_diagnostics"]


