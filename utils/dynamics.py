"""Longitudinal dynamics helpers and extended plant models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


GRAVITY: float = 9.80665  # m/s^2
DEFAULT_AIR_DENSITY: float = 1.225  # kg/m^3


# ---------------------------------------------------------------------------
# Baseline simple dynamics helpers (retained for backwards compatibility)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RandomizationConfig:
    """Ranges used to randomize vehicle parameters per episode."""

    mass_range: Tuple[float, float] = (800.0, 5500.0)  # kg
    rolling_coeff_range: Tuple[float, float] = (0.008, 0.02)
    drag_area_range: Tuple[float, float] = (0.6, 1.2)  # m^2 * C_d
    actuator_tau_range: Tuple[float, float] = (0.05, 0.30)  # seconds
    grade_range_deg: Tuple[float, float] = (-3.0, 3.0)
    speed_noise_range: Tuple[float, float] = (0.02, 0.2)  # m/s
    accel_noise_range: Tuple[float, float] = (0.05, 0.2)  # m/s^2
    air_density: float = DEFAULT_AIR_DENSITY


@dataclass(slots=True)
class VehicleParams:
    """Container for the physical parameters of the ego vehicle."""

    mass: float
    rolling_coeff: float
    drag_area: float
    actuator_tau: float
    grade_rad: float
    air_density: float = DEFAULT_AIR_DENSITY


def sample_vehicle_params(rng: np.random.Generator, config: RandomizationConfig) -> VehicleParams:
    """Sample a new :class:`VehicleParams` instance from the configured ranges."""

    def _sample(bounds: Tuple[float, float]) -> float:
        return float(rng.uniform(bounds[0], bounds[1]))

    grade_deg = _sample(config.grade_range_deg)
    return VehicleParams(
        mass=_sample(config.mass_range),
        rolling_coeff=_sample(config.rolling_coeff_range),
        drag_area=_sample(config.drag_area_range),
        actuator_tau=_sample(config.actuator_tau_range),
        grade_rad=np.deg2rad(grade_deg),
        air_density=config.air_density,
    )


def sample_sensor_noise(rng: np.random.Generator, config: RandomizationConfig) -> tuple[float, float]:
    """Return per-episode sensor noise scales for speed and acceleration."""

    speed_noise = float(rng.uniform(*config.speed_noise_range))
    accel_noise = float(rng.uniform(*config.accel_noise_range))
    return speed_noise, accel_noise


def aerodynamic_drag(speed: float, params: VehicleParams) -> float:
    """Aerodynamic drag force magnitude."""

    return 0.5 * params.air_density * params.drag_area * speed * speed


def rolling_resistance(params: VehicleParams) -> float:
    """Maximum rolling resistance force magnitude (at high speeds).

    Note: Actual rolling resistance in simulation varies with vehicle speed,
    going to zero as speed approaches zero.
    """

    return params.rolling_coeff * params.mass * GRAVITY


def grade_force(params: VehicleParams) -> float:
    """Force component due to road grade."""

    return params.mass * GRAVITY * np.sin(params.grade_rad)


def longitudinal_acceleration(speed: float, commanded_accel: float, params: VehicleParams) -> float:
    """Compute the actual longitudinal acceleration given the commanded input."""

    drive_force = params.mass * commanded_accel
    net_force = drive_force - aerodynamic_drag(speed, params) - rolling_resistance(params) - grade_force(params)
    return net_force / params.mass


# ---------------------------------------------------------------------------
# Extended plant configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MotorParams:
    R: float = 0.2  # motor resistance (Ω)
    K_e: float = 0.2  # back-EMF constant (V/(rad/s))
    K_t: float = 0.2  # torque constant (Nm/A)
    B_m: float = 1e-3  # motor damping (Nm/(rad/s))
    V_max: float = 400.0  # max motor voltage (V)
    gear_ratio: float = 10.0  # gear reduction ratio


@dataclass(slots=True)
class BrakeParams:
    T_br_max: float = 8000.0
    p_br: float = 1.2
    tau_br: float = 0.08
    kappa_c: float = 0.08
    mu: float = 0.9


@dataclass(slots=True)
class BodyParams:
    mass: float = 1400.0
    drag_area: float = 0.7
    rolling_coeff: float = 0.01
    grade_rad: float = 0.0
    air_density: float = DEFAULT_AIR_DENSITY


@dataclass(slots=True)
class WheelParams:
    radius: float = 0.30
    inertia: float = 1.2
    v_eps: float = 0.1


@dataclass(slots=True)
class ExtendedPlantParams:
    motor: MotorParams = MotorParams()
    brake: BrakeParams = BrakeParams()
    body: BodyParams = BodyParams()
    wheel: WheelParams = WheelParams()


@dataclass(slots=True)
class ExtendedPlantRandomization:
    mass_range: Tuple[float, float] = (1500.0, 3500.0)  # kg - updated range
    drag_area_range: Tuple[float, float] = (0.55, 0.85)
    rolling_coeff_range: Tuple[float, float] = (0.008, 0.02)
    mu_range: Tuple[float, float] = (0.5, 1.0)
    motor_Vmax_range: Tuple[float, float] = (200.0, 800.0)  # V - max motor voltage
    motor_R_range: Tuple[float, float] = (0.05, 0.5)  # Ω - motor resistance
    motor_K_range: Tuple[float, float] = (0.1, 0.4)  # Nm/A - motor constants
    gear_ratio_range: Tuple[float, float] = (4.0, 20.0)  # gear reduction ratio
    brake_tau_range: Tuple[float, float] = (0.04, 0.12)
    brake_Tmax_range: Tuple[float, float] = (5000.0, 10000.0)
    grade_deg_range: Tuple[float, float] = (-5.0, 5.0)


def sample_extended_params(rng: np.random.Generator, rand: ExtendedPlantRandomization) -> ExtendedPlantParams:
    """Sample plant parameters for the extended dynamics with rejection sampling for acceleration capability."""

    # Fixed parameters
    wheel_radius = 0.3  # m - fixed wheel radius
    eta_gb = 0.9  # gearbox efficiency

    # Rejection sampling loop to ensure 2.5-4.0 m/s² acceleration capability
    max_attempts = 100
    for attempt in range(max_attempts):
        # Sample mass and target acceleration
        mass = float(rng.uniform(*rand.mass_range))
        a_target = float(rng.uniform(2.5, 4.0))  # m/s²

        # Compute required wheel torque for target acceleration
        F_required = a_target * mass
        T_req = F_required * wheel_radius

        # Sample motor parameters
        V_max = float(rng.uniform(*rand.motor_Vmax_range))
        R = float(rng.uniform(*rand.motor_R_range))
        K_t = float(rng.uniform(*rand.motor_K_range))
        K_e = K_t  # SI units: K_e = K_t
        gear_ratio = float(rng.uniform(*rand.gear_ratio_range))

        # Compute theoretical stall torque
        i_stall = V_max / R
        tau_m_stall = K_t * i_stall
        T_wheel_stall = tau_m_stall * gear_ratio * eta_gb

        # Check if stall torque meets requirements (90%-130% of required)
        if 0.9 * T_req <= T_wheel_stall <= 1.3 * T_req:
            # Parameters accepted - create the parameter objects
            body = BodyParams(
                mass=mass,
                drag_area=float(rng.uniform(*rand.drag_area_range)),
                rolling_coeff=float(rng.uniform(*rand.rolling_coeff_range)),
                grade_rad=np.deg2rad(float(rng.uniform(*rand.grade_deg_range))),
            )
            motor = MotorParams(
                R=R,
                K_e=K_e,
                K_t=K_t,
                B_m=1e-3,  # fixed damping
                V_max=V_max,
                gear_ratio=gear_ratio,
            )
            brake = BrakeParams(
                T_br_max=float(rng.uniform(*rand.brake_Tmax_range)),
                p_br=1.2,
                tau_br=float(rng.uniform(*rand.brake_tau_range)),
                kappa_c=0.08,
                mu=float(rng.uniform(*rand.mu_range)),
            )
            return ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=WheelParams())

        # If parameters don't meet requirements, continue to next attempt

    # Fallback if rejection sampling fails (shouldn't happen with reasonable ranges)
    raise RuntimeError(f"Could not find suitable parameters after {max_attempts} attempts")


# ---------------------------------------------------------------------------
# Extended plant state and simulation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExtendedPlantState:
    speed: float
    position: float
    acceleration: float
    wheel_speed: float
    brake_torque: float
    slip_ratio: float
    action: float
    motor_current: float
    motor_voltage: float
    V_cmd: float  # commanded voltage
    # Forces and torques
    drive_torque: float
    tire_force: float
    drag_force: float
    rolling_force: float
    grade_force: float
    net_force: float
    held_by_brakes: bool  # True when vehicle is held at rest by brakes/static friction


class ExtendedPlant:
    """DC-motor throttle + nonlinear brake + wheel/vehicle longitudinal plant."""

    def __init__(self, params: ExtendedPlantParams):
        self.params = params
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, speed: float = 0.0, position: float = 0.0) -> ExtendedPlantState:
        self.speed = speed  # Allow negative speeds for testing reverse motion
        self.position = position
        self.acceleration = 0.0
        self.wheel_speed = self.speed / max(self.params.wheel.radius, 1e-3)
        self.brake_torque = 0.0
        self.motor_current = 0.0
        self.motor_voltage = 0.0
        self.V_cmd = 0.0  # commanded voltage
        self.last_action = 0.0
        self.slip_ratio = 0.0
        # Initialize forces
        self.drive_torque = 0.0
        self.tire_force = 0.0
        self.drag_force = 0.0
        self.rolling_force = 0.0
        self.grade_force = 0.0
        self.net_force = 0.0
        self.held_by_brakes = False
        self._current_grade_rad = None  # Current grade override (None = use body.grade_rad)
        return self.state

    # ------------------------------------------------------------------
    @property
    def state(self) -> ExtendedPlantState:
        return ExtendedPlantState(
            speed=self.speed,
            position=self.position,
            acceleration=self.acceleration,
            wheel_speed=self.wheel_speed,
            brake_torque=self.brake_torque,
            slip_ratio=self.slip_ratio,
            action=self.last_action,
            motor_current=self.motor_current,
            motor_voltage=self.motor_voltage,
            V_cmd=self.V_cmd,
            drive_torque=self.drive_torque,
            tire_force=self.tire_force,
            drag_force=self.drag_force,
            rolling_force=self.rolling_force,
            grade_force=self.grade_force,
            net_force=self.net_force,
            held_by_brakes=self.held_by_brakes,
        )

    # ------------------------------------------------------------------
    def step(self, action: float, dt: float, substeps: int = 1, grade_rad: float | None = None) -> ExtendedPlantState:
        """Advance the plant by ``dt`` seconds, optionally using sub-steps."""

        # Store the grade for this step (None means use default body.grade_rad)
        self._current_grade_rad = grade_rad

        dt = max(dt, 1e-6)
        sub_dt = dt / max(substeps, 1)
        clipped_action = float(np.clip(action, -1.0, 1.0))
        for _ in range(max(substeps, 1)):
            self._substep(clipped_action, sub_dt)
        self.last_action = clipped_action
        return self.state

    # ------------------------------------------------------------------
    def _substep(self, action: float, dt: float) -> None:
        motor = self.params.motor
        brake_params = self.params.brake
        wheel = self.params.wheel
        body = self.params.body

        # Action → motor voltage mapping (no regen)
        u = action
        u_th = max(0, u)  # throttle: u > 0
        self.V_cmd = u_th * motor.V_max  # commanded voltage

        # For braking (u < 0) or coasting (u = 0), motor voltage = 0 (no regen)
        if u <= 0:
            self.V_cmd = 0.0

        # Split action for braking (only when u < 0)
        brake_cmd = max(-u, 0.0)  # brake: u < 0

        # ===== DC MOTOR MODEL =====
        # motor electrical model (L=0 approximation)
        omega_m = motor.gear_ratio * self.wheel_speed  # motor speed (rad/s)
        self.motor_current = (self.V_cmd - motor.K_e * omega_m) / max(motor.R, 1e-6)
        tau_m = motor.K_t * self.motor_current - motor.B_m * omega_m  # motor torque

        # mechanical coupling with gear reduction
        eta_gb = 0.9  # gearbox efficiency (fixed)
        self.drive_torque = motor.gear_ratio * tau_m * eta_gb

        # Disable regeneration: no drive torque when not actively throttling
        if u <= 0:
            self.drive_torque = 0.0
            self.motor_current = 0.0  # Also zero out current for consistency

        # ===== BRAKE DYNAMICS =====
        T_br_cmd = brake_params.T_br_max * (brake_cmd ** brake_params.p_br)
        self.brake_torque += dt / max(brake_params.tau_br, 1e-4) * (T_br_cmd - self.brake_torque)

        # ===== PHYSICS-ROOTED HOLD & SLIP BRAKING MODEL =====

        # Constants for hold/slip logic
        v_lock_threshold = 0.05  # m/s - speed below which we consider "stopped"
        mu_s = brake_params.mu * 1.05  # static friction (slightly higher than kinetic)
        mu_k = brake_params.mu          # kinetic friction

        # 1. Compute drive torque and brake command
        tau_drive = self.drive_torque  # already signed (positive = forward)
        tau_brake_cmd = max(self.brake_torque, 0.0)  # magnitude (ensure non-negative)

        # 2. Compute brake torque (opposes current motion direction)
        # For stable behavior, brake torque should always oppose the vehicle's current motion
        # This prevents oscillations and ensures braking always slows the vehicle
        if abs(self.speed) > 0.01:  # Moving vehicle
            tau_brake = -np.sign(self.speed) * tau_brake_cmd  # Oppose motion direction
        else:  # Nearly stopped
            tau_brake = -tau_brake_cmd  # Default to opposing forward motion

        # 4. Net torque and attempted tyre force
        tau_net = tau_drive + tau_brake
        F_raw = tau_net / max(wheel.radius, 1e-6)

        # 5. Friction limits
        N_normal = body.mass * GRAVITY
        F_fric_static = mu_s * N_normal
        F_fric_kinetic = mu_k * N_normal

        # 6. External forces for holding logic
        F_drag = 0.5 * body.air_density * body.drag_area * self.speed * abs(self.speed)
        # Speed-dependent rolling resistance (goes to zero at zero speed)
        v_threshold = 0.1  # m/s - speed below which rolling resistance goes to zero
        roll_factor = min(1.0, abs(self.speed) / v_threshold)
        F_roll = body.rolling_coeff * body.mass * GRAVITY * roll_factor
        F_grade = body.mass * GRAVITY * np.sin(self._current_grade_rad if self._current_grade_rad is not None else body.grade_rad)
        F_ext = - (F_drag + F_roll + F_grade)
        F_req_to_hold = -F_ext  # tyre force needed to hold

        # 7. Decide hold vs slip
        brake_applied = tau_brake_cmd > 1e-3  # Check if brakes are actually applied

        if abs(self.speed) <= v_lock_threshold and brake_applied:
            # Nearly stopped with brakes applied: check if we can hold
            if (abs(F_req_to_hold) <= F_fric_static and
                abs(F_raw) >= abs(F_req_to_hold) * 0.8):  # Brake must be strong enough
                # Can hold - set tyre force to exactly balance externals
                self.tire_force = float(F_req_to_hold)
                self.wheel_speed = 0.0
                self.held_by_brakes = True
            else:
                # Cannot hold - kinetic friction limit from applied torque
                self.tire_force = float(np.clip(F_raw, -F_fric_kinetic, F_fric_kinetic))
                self.held_by_brakes = False
        else:
            # Moving or no brakes applied: kinetic friction limit
            self.tire_force = float(np.clip(F_raw, -F_fric_kinetic, F_fric_kinetic))
            self.held_by_brakes = False

        # 8. Wheel rotational dynamics (if not held)
        if not self.held_by_brakes:
            T_tire = self.tire_force * wheel.radius
            omega_dot = (tau_net - T_tire) / max(wheel.inertia, 1e-6)
            self.wheel_speed = self.wheel_speed + dt * omega_dot
            # Allow wheel to go to zero or negative (for reverse)
            self.wheel_speed = max(self.wheel_speed, -1e-6)  # small negative allowed

        # 9. Compute slip ratio (for diagnostics)
        wheel_linear_speed = self.wheel_speed * wheel.radius
        v_ref = max(abs(self.speed), wheel.v_eps)
        self.slip_ratio = (wheel_linear_speed - self.speed) / v_ref

        # vehicle longitudinal dynamics
        self.drag_force = 0.5 * body.air_density * body.drag_area * self.speed * abs(self.speed)
        # Speed-dependent rolling resistance (goes to zero at zero speed)
        v_threshold = 0.1  # m/s - speed below which rolling resistance goes to zero
        roll_factor = min(1.0, abs(self.speed) / v_threshold)
        self.rolling_force = body.rolling_coeff * body.mass * GRAVITY * roll_factor
        self.grade_force = body.mass * GRAVITY * np.sin(self._current_grade_rad if self._current_grade_rad is not None else body.grade_rad)

        # net force on vehicle
        self.net_force = self.tire_force - self.drag_force - self.rolling_force - self.grade_force

        if self.held_by_brakes:
            # Held by brakes - no acceleration, speed stays at zero
            self.acceleration = 0.0
            self.speed = 0.0
        else:
            # Normal dynamics
            self.acceleration = self.net_force / max(body.mass, 1e-6)
            self.speed = self.speed + dt * self.acceleration  # Allow negative speeds (reverse)

        self.position += dt * self.speed


__all__ = [
    "GRAVITY",
    "RandomizationConfig",
    "VehicleParams",
    "sample_vehicle_params",
    "sample_sensor_noise",
    "aerodynamic_drag",
    "rolling_resistance",
    "grade_force",
    "longitudinal_acceleration",
    "ExtendedPlantParams",
    "ExtendedPlantRandomization",
    "ExtendedPlant",
    "ExtendedPlantState",
    "sample_extended_params",
]


