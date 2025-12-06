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
    """Configuration for randomizing extended plant parameters."""

    # Basic vehicle parameters
    mass_range: Tuple[float, float] = (1500.0, 3500.0)  # kg
    drag_area_range: Tuple[float, float] = (0.55, 0.85)  # m²
    rolling_coeff_range: Tuple[float, float] = (0.008, 0.02)
    actuator_tau_range: Tuple[float, float] = (0.05, 0.30)  # seconds
    grade_deg_range: Tuple[float, float] = (-5.0, 5.0)  # degrees

    # Motor parameters
    motor_Vmax_range: Tuple[float, float] = (200.0, 800.0)  # V
    motor_R_range: Tuple[float, float] = (0.05, 0.5)  # Ω
    motor_K_range: Tuple[float, float] = (0.1, 0.4)  # Nm/A
    motor_Bm_range: Tuple[float, float] = (1e-5, 5e-3)  # Nm·s/rad

    # Gearbox
    gear_ratio_range: Tuple[float, float] = (4.0, 20.0)

    # Brake parameters
    brake_tau_range: Tuple[float, float] = (0.04, 0.12)  # seconds
    brake_Tmax_range: Tuple[float, float] = (5000.0, 10000.0)  # Nm
    brake_p_range: Tuple[float, float] = (0.6, 2.5)
    brake_kappa_range: Tuple[float, float] = (0.02, 0.25)
    mu_range: Tuple[float, float] = (0.5, 1.0)

    # Wheel parameters
    wheel_radius_range: Tuple[float, float] = (0.25, 0.45)  # m
    wheel_inertia_range: Tuple[float, float] = (0.5, 4.0)  # kg·m²

    # Efficiency
    eta_gb_range: Tuple[float, float] = (0.80, 0.98)

    @classmethod
    def from_config(cls, config: dict) -> 'ExtendedPlantRandomization':
        """Create ExtendedPlantRandomization from config dictionary."""
        if 'vehicle_randomization' not in config:
            return cls()  # Use defaults

        vr_config = config['vehicle_randomization']
        return cls(
            mass_range=tuple(vr_config.get('mass_range', (1500.0, 3500.0))),
            drag_area_range=tuple(vr_config.get('drag_area_range', (0.55, 0.85))),
            rolling_coeff_range=tuple(vr_config.get('rolling_coeff_range', (0.008, 0.02))),
            actuator_tau_range=tuple(vr_config.get('actuator_tau_range', (0.05, 0.30))),
            grade_deg_range=tuple(vr_config.get('grade_range_deg', (-5.0, 5.0))),
            motor_Vmax_range=tuple(vr_config.get('motor_Vmax_range', (200.0, 800.0))),
            motor_R_range=tuple(vr_config.get('motor_R_range', (0.05, 0.5))),
            motor_K_range=tuple(vr_config.get('motor_K_range', (0.1, 0.4))),
            motor_Bm_range=tuple(vr_config.get('motor_Bm_range', (1e-4, 5e-2))),
            gear_ratio_range=tuple(vr_config.get('gear_ratio_range', (4.0, 20.0))),
            brake_tau_range=tuple(vr_config.get('brake_tau_range', (0.04, 0.12))),
            brake_Tmax_range=tuple(vr_config.get('brake_Tmax_range', (5000.0, 10000.0))),
            brake_p_range=tuple(vr_config.get('brake_p_range', (0.6, 2.5))),
            brake_kappa_range=tuple(vr_config.get('brake_kappa_range', (0.02, 0.25))),
            mu_range=tuple(vr_config.get('mu_range', (0.5, 1.0))),
            wheel_radius_range=tuple(vr_config.get('wheel_radius_range', (0.25, 0.45))),
            wheel_inertia_range=tuple(vr_config.get('wheel_inertia_range', (0.5, 4.0))),
            eta_gb_range=tuple(vr_config.get('eta_gb_range', (0.80, 0.98))),
        )


def sample_extended_params(rng: np.random.Generator, rand: ExtendedPlantRandomization) -> ExtendedPlantParams:
    """Sample plant parameters for the extended dynamics with rejection sampling for acceleration capability."""

    # Sample additional parameters
    wheel_radius = float(rng.uniform(*rand.wheel_radius_range))  # m - wheel radius
    eta_gb = float(rng.uniform(*rand.eta_gb_range))  # gearbox efficiency

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
        # Sample motor viscous damping (log-uniform as recommended)
        B_m = 10 ** rng.uniform(np.log10(rand.motor_Bm_range[0]), np.log10(rand.motor_Bm_range[1]))
        gear_ratio = float(rng.uniform(*rand.gear_ratio_range))

        # Validate B_m sampling
        assert rand.motor_Bm_range[0] <= B_m <= rand.motor_Bm_range[1], f"B_m {B_m} out of range {rand.motor_Bm_range}"
        assert B_m >= 0, f"B_m {B_m} must be non-negative"

        # Relative magnitude check: viscous torque should be much smaller than electromagnetic torque
        omega_ref = 400.0  # rad/s (≈ 3800 rpm)
        I_ref = 100.0       # A (reasonable reference current)
        tau_visc = B_m * omega_ref
        tau_em = K_t * I_ref
        assert tau_visc < 0.2 * tau_em, f"Viscous torque {tau_visc:.4f} too large compared to EM torque {tau_em:.4f}"

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
                B_m=float(B_m),  # sampled damping
                V_max=V_max,
                gear_ratio=gear_ratio,
            )
            brake = BrakeParams(
                T_br_max=float(rng.uniform(*rand.brake_Tmax_range)),
                p_br=float(rng.uniform(*rand.brake_p_range)),
                tau_br=float(rng.uniform(*rand.brake_tau_range)),
                kappa_c=float(10 ** rng.uniform(np.log10(rand.brake_kappa_range[0]), np.log10(rand.brake_kappa_range[1]))),  # log-uniform
                mu=float(rng.uniform(*rand.mu_range)),
            )
            # Sample wheel parameters
            wheel_inertia = float(10 ** rng.uniform(np.log10(rand.wheel_inertia_range[0]), np.log10(rand.wheel_inertia_range[1])))  # log-uniform
            wheel = WheelParams(
                radius=wheel_radius,
                inertia=wheel_inertia,
                v_eps=0.1,  # keep fixed
            )
            return ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel)

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
    back_emf_voltage: float  # back-EMF voltage (K_e * omega_m)
    V_cmd: float  # commanded voltage (input motor voltage)
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
        # Initialize wheel angular speed (rad/s) and previous state
        self.wheel_omega = self.speed / max(self.params.wheel.radius, 1e-3)  # rad/s
        self.wheel_omega_prev = self.wheel_omega  # saved from previous substep
        self.brake_torque = 0.0
        self.motor_current = 0.0
        self.back_emf_voltage = 0.0  # back-EMF voltage
        self.V_cmd = 0.0  # commanded voltage (input motor voltage)
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
        # Initialize previous motor states for coupling
        self.motor_current_prev = 0.0
        self.tau_m_prev = 0.0
        self.drive_torque_prev = 0.0
        return self.state

    # ------------------------------------------------------------------
    @property
    def state(self) -> ExtendedPlantState:
        return ExtendedPlantState(
            speed=self.speed,
            position=self.position,
            acceleration=self.acceleration,
            wheel_speed=self.wheel_omega * self.params.wheel.radius,  # expose linear speed for API compatibility
            brake_torque=self.brake_torque,
            slip_ratio=self.slip_ratio,
            action=self.last_action,
            motor_current=self.motor_current,
            back_emf_voltage=self.back_emf_voltage,
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

        # ===== BRAKE TORQUE COMPUTATION (needed for motor coupling) =====
        # Compute brake torque (opposes current motion direction)
        # For stable behavior, brake torque should always oppose the vehicle's current motion
        tau_brake_cmd = max(self.brake_torque, 0.0)  # magnitude (ensure non-negative)
        if abs(self.speed) > 0.01:  # Moving vehicle
            tau_brake = -np.sign(self.speed) * tau_brake_cmd  # Oppose motion direction
        else:  # Nearly stopped
            tau_brake = -tau_brake_cmd  # Default to opposing forward motion

        # ===== FRICTION LIMITS (needed for motor coupling) =====
        # Constants for hold/slip logic
        mu_s = brake_params.mu * 1.05  # static friction (slightly higher than kinetic)
        mu_k = brake_params.mu          # kinetic friction
        N_normal = body.mass * GRAVITY
        F_fric_static = mu_s * N_normal
        F_fric_kinetic = mu_k * N_normal

        # Parameters for wheel dynamics (needed for motor coupling)
        B_w = 1e-3  # small viscous damping on wheel (Nm s / rad) - helps numeric stability

        # ===== DC MOTOR MODEL WITH INNER COUPLING =====
        # Inner motor↔mechanics coupling iteration (cheap fixed-point)

        # Parameters
        allow_regen = False  # set to True if regeneration is allowed
        eta_gb = 0.9  # gearbox efficiency (fixed)
        r_w = wheel.radius
        I_w = wheel.inertia
        gear = motor.gear_ratio
        K_t = motor.K_t
        K_e = motor.K_e
        R = motor.R
        B_m = motor.B_m

        # 1) compute motor torque from current wheel state
        omega_old = self.wheel_omega
        omega_m_old = gear * omega_old
        I_old = (self.V_cmd - K_e * omega_m_old) / max(R, 1e-9)
        if not allow_regen:
            I_old = max(I_old, 0.0)
        tau_m_old = K_t * I_old - B_m * omega_m_old
        drive_torque_old = gear * tau_m_old * eta_gb

        # Debug logging for passive torque (optional)
        debug_motor = False  # set to True for debugging
        if debug_motor:
            tau_viscous = -B_m * omega_m_old
            tau_em = K_t * I_old
            print(f"[B_m] B_m={B_m:.3e}, omega_m={omega_m_old:.2f}, "
                  f"tau_viscous={tau_viscous:.4f}, tau_em={tau_em:.4f}")

        # 2) tentative mechanical response using drive_torque_old
        tau_net_tent = drive_torque_old + tau_brake  # tau_brake computed earlier
        F_attempt_tent = tau_net_tent / max(r_w, 1e-9)
        F_tire_tent = float(np.clip(F_attempt_tent, -F_fric_kinetic, F_fric_kinetic))
        T_contact_tent = F_tire_tent * r_w
        omega_dot_tent = (tau_net_tent - T_contact_tent - B_w * omega_old) / max(I_w, 1e-9)
        omega_tent = omega_old + dt * omega_dot_tent

        # 3) recompute motor torque at tentative omega
        omega_m_tent = gear * omega_tent
        I_tent = (self.V_cmd - K_e * omega_m_tent) / max(R, 1e-9)
        if not allow_regen:
            I_tent = max(I_tent, 0.0)
        tau_m_tent = K_t * I_tent - B_m * omega_m_tent

        # 4) average torques (fixed-point update)
        tau_m = 0.5 * (tau_m_old + tau_m_tent)
        self.drive_torque = gear * tau_m * eta_gb

        # Store motor electrical quantities for diagnostics
        self.back_emf_voltage = K_e * omega_m_old  # use old for consistency with current approach
        self.motor_current = (self.V_cmd - self.back_emf_voltage) / max(R, 1e-9)
        if not allow_regen:
            self.motor_current = max(self.motor_current, 0.0)

        # Save previous states for next substep coupling
        self.tau_m_prev = tau_m
        self.motor_current_prev = self.motor_current
        self.drive_torque_prev = self.drive_torque

        # ===== BRAKE DYNAMICS =====
        T_br_cmd = brake_params.T_br_max * (brake_cmd ** brake_params.p_br)
        self.brake_torque += dt / max(brake_params.tau_br, 1e-4) * (T_br_cmd - self.brake_torque)

        # ===== PHYSICS-ROOTED HOLD & SLIP BRAKING MODEL =====

        # Constants for hold/slip logic
        v_lock_threshold = 0.05  # m/s - speed below which we consider "stopped"

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

        # 5. Friction limits (already computed earlier)

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

        # 8. Wheel rotational dynamics with stick/slip behavior

        # Parameters (tunable)
        B_w = 1e-3  # small viscous damping on wheel (Nm s / rad) - helps numeric stability
        slip_eps = 1e-3  # m/s threshold to decide slip vs stick (small)

        # Compute wheel linear speed and slip candidate
        wheel_v = self.wheel_omega * wheel.radius  # m/s linear rim speed
        v_rel = wheel_v - self.speed               # positive if wheel trying to drive faster than vehicle

        # Determine contact torque and whether we're slipping
        # (F_attempt is the tyre force that would be produced if full shaft torque transmitted)
        F_attempt = tau_net / max(wheel.radius, 1e-6)  # N, attempted tyre force from shaft torque

        # Friction limits already computed: F_fric_static, F_fric_kinetic

        # Decide if wheel is sliding: if wheel linear speed differs from vehicle speed beyond slip_eps OR applied torque exceeds static friction
        is_sliding = False
        # If wheel is nearly at same linear speed as vehicle, and required tyre force to hold <= static friction, then it's sticking.
        if abs(v_rel) > slip_eps:
            # clear sliding: wheel and vehicle speeds diverge
            is_sliding = True
        else:
            # wheel and vehicle locked kinematically -> check if required tyre force to satisfy net translational dynamics <= static friction
            # External forces computed above: F_ext = -(F_drag + F_roll + F_grade)
            # F_req_to_hold computed above: tyre force needed to hold vehicle (signed)
            # If the current attempted force would exceed static friction, slip will occur
            if abs(F_attempt) > F_fric_static + 1e-9:
                is_sliding = True
            else:
                is_sliding = False

        # Now branch on sliding vs sticking
        if is_sliding:
            # SLIDE: tyre force saturates at kinetic friction magnitude (direction opposes slip)
            # Determine transmitted tyre force from torque attempt but saturated by kinetic friction
            F_tire = float(np.clip(F_attempt, -F_fric_kinetic, F_fric_kinetic))
            self.tire_force = F_tire
            # Contact torque at wheel
            T_contact = self.tire_force * wheel.radius

            # Integrate wheel rotational dynamics with small viscous damping
            omega_dot = (tau_net - T_contact - B_w * self.wheel_omega) / max(wheel.inertia, 1e-6)
            self.wheel_omega = self.wheel_omega + dt * omega_dot
            # Update vehicle longitudinal dynamics as usual using self.tire_force (already set)
            self.held_by_brakes = False

        else:
            # STICK (no slip) — enforce kinematic constraint: wheel linear speed matches vehicle speed
            # The tyre force is determined by the shaft torque (F_attempt), but cannot exceed static friction
            # If F_attempt exceeds static friction, we should have detected slip above, so this shouldn't happen
            self.tire_force = float(np.clip(F_attempt, -F_fric_static, F_fric_static))
            # No rotational acceleration at the wheel (it rotates with the vehicle)
            # held_by_brakes remains as set earlier by brake logic above (if holding, True; else False)

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
            # In stick mode when held, wheel speed should also be zero
            self.wheel_omega = 0.0
        else:
            # Normal dynamics
            self.acceleration = self.net_force / max(body.mass, 1e-6)
            self.speed = self.speed + dt * self.acceleration  # Allow negative speeds (reverse)
            # In stick mode, wheel angular speed should match vehicle linear speed
            if not is_sliding:  # Only update if we're still in stick mode after speed update
                self.wheel_omega = self.speed / max(wheel.radius, 1e-6)

        self.position += dt * self.speed

        # 9. Compute slip ratio (for diagnostics) - now that speed is updated
        wheel_linear_speed = self.wheel_omega * wheel.radius
        v_ref = max(abs(self.speed), wheel.v_eps)
        self.slip_ratio = (wheel_linear_speed - self.speed) / v_ref

        # Save previous wheel state for next substep
        self.wheel_omega_prev = self.wheel_omega


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


