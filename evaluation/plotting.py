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
    drive_torque = _to_numpy(trace.get("drive_torque", np.zeros_like(time)))
    tire_force = _to_numpy(trace.get("tire_force", np.zeros_like(time)))
    drag_force = _to_numpy(trace.get("drag_force", np.zeros_like(time)))
    rolling_force = _to_numpy(trace.get("rolling_force", np.zeros_like(time)))
    grade_force = _to_numpy(trace.get("grade_force", np.zeros_like(time)))
    net_force = _to_numpy(trace.get("net_force", np.zeros_like(time)))

    fig, axes = plt.subplots(7, 1, figsize=(12, 21), sharex=True)

    # 1. Speed and ref speed
    axes[0].plot(time, speed, label="Speed")
    axes[0].plot(time, reference, label="Reference", linestyle="--")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    # 2. Acceleration
    axes[1].plot(time, acceleration, label="Acceleration")
    axes[1].set_ylabel("Acceleration (m/s²)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    # 3. Jerk
    axes[2].plot(time, jerk, label="Jerk")
    axes[2].set_ylabel("Jerk (m/s³)")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    # 4. Throttle %
    axes[3].plot(time, throttle_pct, label="Throttle %", color="#1b9e77")
    axes[3].set_ylabel("Throttle %")
    axes[3].set_ylim(0, 100)
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.3)

    # 5. Brake %
    axes[4].plot(time, brake_pct, label="Brake %", color="#d95f02")
    axes[4].set_ylabel("Brake %")
    axes[4].set_ylim(0, 100)
    axes[4].legend(loc="upper right")
    axes[4].grid(alpha=0.3)

    # 6. Forces acting on the vehicle
    axes[5].plot(time, drive_torque, label="Drive torque (Nm)")
    axes[5].plot(time, tire_force, label="Tire force (N)")
    axes[5].plot(time, drag_force, label="Drag force (N)")
    axes[5].plot(time, rolling_force, label="Rolling force (N)")
    axes[5].plot(time, grade_force, label="Grade force (N)")
    axes[5].plot(time, net_force, label="Net force (N)", linewidth=2)
    axes[5].set_ylabel("Forces/Torques")
    axes[5].legend(loc="upper right", ncol=2)
    axes[5].grid(alpha=0.3)

    # 7. Policy mu post-tanh with 68th percentile band
    policy_std = np.exp(policy_log_std)  # Convert log std to std
    policy_mean_tanh = np.tanh(policy_mean)  # Post-tanh mean
    y_low, y_high = tanh_band(policy_mean, policy_std, p=0.68)  # 68th percentile band

    axes[6].plot(time, policy_mean_tanh, label="Policy μ (post-tanh)", color="#1f77b4", linewidth=2)
    axes[6].fill_between(time, y_low, y_high, alpha=0.3, color="#1f77b4", label="68th percentile")
    axes[6].set_ylabel("Policy μ (post-tanh)")
    axes[6].set_xlabel("Time (s)")
    axes[6].legend(loc="upper right")
    axes[6].grid(alpha=0.3)

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


__all__ = ["plot_sequence_diagnostics", "plot_summary"]


