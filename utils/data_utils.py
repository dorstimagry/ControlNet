"""Data-generation helpers that rely on Hugging Face datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import math
import numpy as np
import torch

try:  # pragma: no cover - import guard exercised indirectly
    from datasets import Dataset
except ImportError as exc:  # pragma: no cover - explicit guidance for users
    raise ImportError(
        "The 'datasets' package is required. Install via `pip install datasets`."
    ) from exc

try:
    from src.data.datasets import EVSequenceDataset, SequenceWindowConfig
except ImportError:
    # These classes will be imported when they're needed
    EVSequenceDataset = None
    SequenceWindowConfig = None


@dataclass(slots=True)
class SyntheticTrajectoryConfig:
    """Configuration for synthetic reference-speed generation."""

    num_sequences: int = 64
    sequence_length: int = 256
    dt: float = 0.1
    min_speed: float = 0.0
    max_speed: float = 30.0
    noise_std: float = 0.05
    max_accel: float = 2.5
    target_update_range: tuple[int, int] = (30, 120)
    smooth_noise_std: float = 0.05


class ReferenceTrajectoryGenerator:
    """Sample diverse reference-speed profiles for training/evaluation."""

    def __init__(
        self,
        dt: float = 0.1,
        min_speed: float = 0.0,
        max_speed: float = 30.0,
        max_accel: float = 2.5,
        target_update_range: tuple[int, int] = (30, 120),
        smooth_noise_std: float = 0.05,
    ) -> None:
        self.dt = dt
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.target_update_range = target_update_range
        self.smooth_noise_std = smooth_noise_std

    def sample(self, length: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return a single reference-speed profile of ``length`` samples."""

        rng = rng or np.random.default_rng()
        speeds = np.zeros(length, dtype=np.float32)
        current_speed = rng.uniform(self.min_speed, self.max_speed)
        target_speed = current_speed
        steps_remaining = 0
        accel_state = 0.0
        accel_gain = 0.5

        for idx in range(length):
            if steps_remaining <= 0:
                target_speed = rng.uniform(self.min_speed, self.max_speed)
                steps_remaining = int(
                    rng.integers(self.target_update_range[0], self.target_update_range[1])
                )
            steps_remaining -= 1

            desired_accel = np.clip(
                (target_speed - current_speed) * accel_gain,
                -self.max_accel,
                self.max_accel,
            )
            accel_state = 0.8 * accel_state + 0.2 * desired_accel
            accel_state = float(np.clip(accel_state, -self.max_accel, self.max_accel))
            current_speed = float(
                np.clip(current_speed + accel_state * self.dt, self.min_speed, self.max_speed)
            )
            speeds[idx] = current_speed

        if self.smooth_noise_std > 0:
            speeds += rng.normal(0.0, self.smooth_noise_std, size=length).astype(np.float32)
            kernel = max(3, int(round(0.5 / self.dt)))
            kernel += 1 - kernel % 2
            window = np.ones(kernel, dtype=np.float32) / kernel

            # Use edge padding instead of zero padding to avoid boundary artifacts
            pad = kernel // 2
            padded = np.pad(speeds, pad_width=pad, mode="edge")
            speeds = np.convolve(padded, window, mode="valid")

        return np.clip(speeds, self.min_speed, self.max_speed)


def build_synthetic_reference_dataset(
    config: SyntheticTrajectoryConfig,
    rng: np.random.Generator | None = None,
) -> Dataset:
    """Return a Hugging Face dataset containing synthetic reference trajectories."""

    rng = rng or np.random.default_rng()
    generator = ReferenceTrajectoryGenerator(
        dt=config.dt,
        min_speed=config.min_speed,
        max_speed=config.max_speed,
        max_accel=config.max_accel,
        target_update_range=config.target_update_range,
        smooth_noise_std=config.smooth_noise_std,
    )

    sequences: List[List[float]] = []
    metadata: List[dict[str, float]] = []
    for _ in range(config.num_sequences):
        profile = generator.sample(config.sequence_length, rng)
        if config.noise_std > 0:
            profile += rng.normal(0.0, config.noise_std, size=profile.shape).astype(np.float32)
        sequences.append(profile.tolist())
        metadata.append({"dt": config.dt, "length": config.sequence_length})

    return Dataset.from_dict(
        {
            "reference_speed": sequences,
            "metadata": metadata,
        }
    )


def ev_windows_to_dataset(
    data_path: Path,
    window: SequenceWindowConfig | None = None,
    state_features: Sequence[str] | None = None,
    max_samples: int | None = None,
) -> Dataset:
    """Convert :class:`EVSequenceDataset` windows into a Hugging Face dataset."""

    dataset = EVSequenceDataset(
        data_path=Path(data_path),
        window=window,
        state_features=state_features,
    )
    total = len(dataset)
    if total == 0:
        raise ValueError("EVSequenceDataset is empty; cannot create huggingface dataset")
    take = total if max_samples is None else min(total, max_samples)

    records = {
        "history_states": [],
        "history_actions": [],
        "future_states": [],
        "future_actions": [],
        "future_actuation": [],
        "loss_weight": [],
        "is_stationary": [],
    }

    for idx in range(take):
        sample = dataset[idx]
        records["history_states"].append(sample["history_states"].numpy().tolist())
        records["history_actions"].append(sample["history_actions"].numpy().tolist())
        records["future_states"].append(sample["future_states"].numpy().tolist())
        records["future_actions"].append(sample["future_actions"].numpy().tolist())
        records["future_actuation"].append(sample["future_actuation"].numpy().tolist())
        records["loss_weight"].append(float(sample["loss_weight"].item()))
        records["is_stationary"].append(int(sample["is_stationary"].item()))

    return Dataset.from_dict(records)


@dataclass(slots=True)
class VehicleCapabilities:
    """Vehicle parameters needed for feasibility calculations."""
    m: float  # mass (kg)
    r_w: float  # wheel radius (m)
    T_drive_max: float  # max wheel drive torque (Nm)
    T_brake_max: float  # max wheel brake torque (Nm)
    mu: float  # road friction coefficient
    C_dA: float  # drag area (m²·Cd)
    C_r: float  # rolling resistance coefficient
    rho: float = 1.225  # air density (kg/m³)


def feasible_accel_bounds(v: float, grade: float, vehicle: VehicleCapabilities,
                         safety_margin: float = 0.9) -> tuple[float, float]:
    """Compute feasible acceleration bounds for given speed and grade.

    Args:
        v: current speed (m/s)
        grade: road grade (radians)
        vehicle: vehicle capabilities object
        safety_margin: safety factor (0.9 = 90% of theoretical max)

    Returns:
        (a_min, a_max): minimum and maximum feasible accelerations (m/s²)
    """
    g = 9.80665  # gravity (m/s²)

    # Resistive forces
    F_drag = 0.5 * vehicle.rho * vehicle.C_dA * v * v
    F_roll = vehicle.C_r * vehicle.m * g
    F_grade = vehicle.m * g * math.sin(grade)
    F_fric = vehicle.mu * vehicle.m * g

    # Available longitudinal forces (with safety margin)
    F_drive_avail = safety_margin * min(vehicle.T_drive_max / vehicle.r_w, F_fric)
    F_brake_avail = safety_margin * min(vehicle.T_brake_max / vehicle.r_w, F_fric)

    # Feasible accelerations
    a_max = (F_drive_avail - F_drag - F_roll - F_grade) / vehicle.m
    a_min = - (F_brake_avail + F_drag + F_roll + F_grade) / vehicle.m

    return a_min, a_max


def project_profile_to_feasible(v0: np.ndarray, grade0: np.ndarray,
                               vehicle: VehicleCapabilities, dt: float,
                               safety_margin: float = 0.9, max_iters: int = 20,
                               tol: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Make a speed profile feasible by iterative acceleration clipping and integration.

    Args:
        v0: initial target speed array (length N+1)
        grade0: initial grade array (length N+1, radians)
        vehicle: vehicle capabilities object
        dt: timestep (s)
        safety_margin: safety factor for force calculations
        max_iters: maximum iterations
        tol: convergence tolerance (m/s)

    Returns:
        (v_feasible, grade_feasible): feasible speed and grade arrays
    """
    N = len(v0) - 1
    v = v0.copy()
    grade = grade0.copy()

    for iteration in range(max_iters):
        # Compute requested accelerations from current speed profile
        a_req = np.zeros(N)
        for k in range(N):
            a_req[k] = (v[k+1] - v[k]) / dt

        clipped = False

        # Compute feasible ranges and clip accelerations
        for k in range(N):
            vk = max(v[k], 0.0)  # ensure non-negative speed
            phi = grade[k]

            a_min, a_max = feasible_accel_bounds(vk, phi, vehicle, safety_margin)

            # If bounds are invalid, set to midpoint (shouldn't happen normally)
            if a_max < a_min:
                a_max = a_min = 0.5 * (a_max + a_min)

            # Clip requested acceleration
            new_a = np.clip(a_req[k], a_min, a_max)
            if abs(new_a - a_req[k]) > 1e-6:
                clipped = True
            a_req[k] = new_a

        # Integrate clipped accelerations to get new speed profile
        v_new = v.copy()
        for k in range(N):
            v_new[k+1] = max(0.0, v_new[k] + a_req[k] * dt)

        # Check convergence
        max_speed_change = np.max(np.abs(v_new - v))
        v = v_new

        # Convergence: no clipping needed and speed changes are small
        if max_speed_change < tol and not clipped:
            break

    return v, grade


@dataclass(slots=True)
class VehicleMotorCapabilities:
    """Vehicle motor parameters needed for initial speed feasibility."""
    # Motor parameters (required)
    r_w: float  # wheel radius (m)
    N_g: float  # gear ratio (motor speed / wheel speed)
    eta: float  # gearbox efficiency
    K_e: float  # back-EMF constant (V/(rad/s))
    K_t: float  # torque constant (Nm/A)
    R: float    # armature resistance (Ω)
    V_max: float  # max motor voltage (V)
    # Vehicle body parameters (required)
    mass: float
    C_dA: float  # drag area (m²·Cd)
    C_r: float   # rolling resistance coefficient
    # Optional parameters (with defaults)
    I_max: float | None = None  # max motor current (A), optional
    rho: float = 1.225  # air density (kg/m³)


def initial_target_feasible(v: float, grade: float, veh: VehicleMotorCapabilities,
                          safety_margin: float = 0.05) -> tuple[bool, float, float | None]:
    """Check if initial target speed and grade are motor-feasible for the vehicle.

    Args:
        v: target vehicle speed (m/s)
        grade: road grade (radians)
        veh: vehicle motor capabilities
        safety_margin: safety factor for voltage limits (0.05 = 5%)

    Returns:
        (feasible, V_needed, I_needed): feasibility flag, required voltage, required current
    """
    GRAVITY = 9.80665

    # Compute motor speed
    omega_w = v / max(veh.r_w, 1e-6)
    omega_m = veh.N_g * omega_w

    # Compute resistive forces
    F_drag = 0.5 * veh.rho * veh.C_dA * v * abs(v)
    F_roll = veh.C_r * veh.mass * GRAVITY
    F_grade = veh.mass * GRAVITY * math.sin(grade)

    F_resist = F_drag + F_roll + F_grade  # positive resists forward motion
    T_req_wheel = F_resist * veh.r_w      # wheel torque required
    T_req_motor = T_req_wheel / (veh.N_g * max(veh.eta, 1e-6))  # motor torque required

    # Compute required voltage
    T_req_motor_pos = max(0.0, T_req_motor)  # only positive torque requires voltage
    V_needed = veh.K_e * omega_m + (veh.R / max(veh.K_t, 1e-12)) * T_req_motor_pos

    # Check if motor can provide forward torque (not beyond no-load speed)
    can_drive = (veh.V_max - veh.K_e * omega_m) > 1e-6

    # Check current limit if available
    I_needed = None
    if veh.I_max is not None:
        I_needed = T_req_motor_pos / max(veh.K_t, 1e-12)
        if I_needed > veh.I_max:
            return False, V_needed, I_needed

    # Check voltage feasibility with safety margin
    feasible = (V_needed <= veh.V_max * (1.0 - safety_margin)) and can_drive

    return feasible, V_needed, I_needed


def adjust_initial_target(v0: float, grade0: float, veh: VehicleMotorCapabilities,
                         safety_margin: float = 0.05, v_min: float = 0.0, v_step: float = 0.5,
                         grade_step_deg: float = 0.5, max_iter_v: int = 50,
                         max_iter_grade: int = 20) -> tuple[float, float, float, float | None]:
    """Adjust initial target speed and grade until motor-feasible.

    Args:
        v0: initial target speed (m/s)
        grade0: initial grade (radians)
        veh: vehicle motor capabilities
        safety_margin: safety factor for voltage limits
        v_min: minimum allowed speed
        v_step: speed reduction step size
        grade_step_deg: grade adjustment step size (degrees)
        max_iter_v: max iterations for speed adjustment
        max_iter_grade: max iterations for grade adjustment

    Returns:
        (v_feasible, grade_feasible, V_needed, I_needed): adjusted values
    """
    v = v0
    grade = grade0

    # First try reducing speed
    for _ in range(max_iter_v):
        feasible, V_needed, I_needed = initial_target_feasible(v, grade, veh, safety_margin)
        if feasible:
            return v, grade, V_needed, I_needed
        v = max(v - v_step, v_min)

    # Then try adjusting grade (reduce uphill)
    grade_step = math.radians(abs(grade_step_deg))
    for _ in range(max_iter_grade):
        feasible, V_needed, I_needed = initial_target_feasible(v, grade, veh, safety_margin)
        if feasible:
            return v, grade, V_needed, I_needed

        # Reduce uphill grade: if grade > 0, reduce it; if grade < 0, make it less downhill
        if grade > 0:
            grade = max(grade - grade_step, -math.pi/6)  # don't exceed -30° downhill
        else:
            grade = grade + grade_step  # make less negative (less downhill)

    # Return best-effort values
    feasible, V_needed, I_needed = initial_target_feasible(v, grade, veh, safety_margin)
    return v, grade, V_needed, I_needed


__all__ = [
    "ReferenceTrajectoryGenerator",
    "SyntheticTrajectoryConfig",
    "VehicleCapabilities",
    "VehicleMotorCapabilities",
    "feasible_accel_bounds",
    "project_profile_to_feasible",
    "initial_target_feasible",
    "adjust_initial_target",
    "build_synthetic_reference_dataset",
    "ev_windows_to_dataset",
]


