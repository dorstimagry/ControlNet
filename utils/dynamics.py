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
    R: float = 0.2  # armature resistance (Ω)
    L: float = 1e-3  # armature inductance (H) - CTMS model
    K_e: float = 0.2  # back-EMF constant (V/(rad/s))
    K_t: float = 0.2  # torque constant (Nm/A)
    b: float = 1e-3  # viscous friction (Nm·s/rad) - was B_m
    J: float = 1e-3  # rotor inertia (kg·m²) - CTMS model
    V_max: float = 400.0  # max motor voltage (V)
    gear_ratio: float = 10.0  # gear reduction ratio
    eta_gb: float = 0.9  # gearbox efficiency


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
class CreepParams:
    """EV-style creep torque parameters.
    
    Creep provides low-speed forward motion at zero throttle, mimicking
    ICE idle behavior without introducing idle RPMs or discontinuities.
    """
    a_max: float = 0.5      # [m/s²] maximum creep acceleration
    v_cutoff: float = 1.5   # [m/s] speed where creep fully fades out
    v_hold: float = 0.08    # [m/s] standstill region threshold


@dataclass(slots=True)
class ExtendedPlantParams:
    motor: MotorParams = MotorParams()
    brake: BrakeParams = BrakeParams()
    body: BodyParams = BodyParams()
    wheel: WheelParams = WheelParams()
    creep: CreepParams = CreepParams()


@dataclass(slots=True)
class ExtendedPlantRandomization:
    """Configuration for randomizing extended plant parameters.
    
    Ranges are based on realistic vehicle/motor specifications.
    Use log-uniform sampling for params spanning orders of magnitude (R, L, K_t, b, J_m).
    Use uniform sampling for other parameters.
    """

    # Basic vehicle parameters
    mass_range: Tuple[float, float] = (1500.0, 6000.0)  # kg - vehicle mass (per doc)
    drag_area_range: Tuple[float, float] = (0.2, 0.8)   # CdA (m²) - drag coefficient * area
    rolling_coeff_range: Tuple[float, float] = (0.007, 0.015)  # C_rr - rolling resistance
    actuator_tau_range: Tuple[float, float] = (0.05, 0.30)  # seconds - actuator time constant
    grade_deg_range: Tuple[float, float] = (-5.7, 5.7)  # degrees - ±10% grade

    # Motor electrical parameters (log-uniform recommended for R, L, K_t)
    motor_Vmax_range: Tuple[float, float] = (200.0, 800.0)  # V - motor max voltage
    motor_R_range: Tuple[float, float] = (0.02, 0.6)  # Ω - armature resistance (log-uniform)
    motor_L_range: Tuple[float, float] = (1e-4, 1e-2)  # H - armature inductance (log-uniform)
    motor_K_range: Tuple[float, float] = (0.05, 0.4)  # Nm/A and V·s/rad - K_t = K_e (log-uniform)
    motor_b_range: Tuple[float, float] = (1e-6, 5e-3)  # Nm·s/rad - viscous friction (log-uniform)
    motor_J_range: Tuple[float, float] = (1e-4, 1e-2)  # kg·m² - rotor inertia (log-uniform)

    # Gearbox
    gear_ratio_range: Tuple[float, float] = (4.0, 20.0)  # N - gear reduction ratio

    # Brake parameters
    brake_tau_range: Tuple[float, float] = (0.04, 0.12)  # seconds - brake time constant
    brake_accel_range: Tuple[float, float] = (8.0, 11.0)  # m/s² - max braking deceleration magnitude
    brake_p_range: Tuple[float, float] = (1.0, 1.8)  # brake exponent (per doc)
    brake_kappa_range: Tuple[float, float] = (0.02, 0.25)  # brake slip constant
    mu_range: Tuple[float, float] = (0.7, 1.0)  # tire friction coefficient

    # Wheel parameters
    wheel_radius_range: Tuple[float, float] = (0.26, 0.38)  # m - typical passenger wheel radii
    wheel_inertia_range: Tuple[float, float] = (0.5, 5.0)  # kg·m² - wheel + rotating assembly

    # Efficiency
    eta_gb_range: Tuple[float, float] = (0.85, 0.98)  # gearbox efficiency
    
    # Creep parameters (optional, None = use fixed default values)
    creep_a_max: float | None = None  # [m/s²] max creep acceleration (None = use default 0.5)
    creep_v_cutoff: float | None = None  # [m/s] creep fade speed (None = use default 1.5)
    creep_v_hold: float | None = None  # [m/s] standstill threshold (None = use default 0.08)
    
    # Feasibility thresholds (for rejection sampling)
    min_accel_from_rest: float = 2.5  # m/s² - minimum required acceleration at standstill
    min_brake_decel: float = 4.0  # m/s² - minimum braking deceleration (legacy, brake_accel_range replaces this)
    min_top_speed: float = 20.0  # m/s - minimum achievable top speed
    skip_feasibility_checks: bool = False  # If True, skip feasibility checks (fitted params mode)
    skip_sanity_checks: bool = False  # If True, skip sanity checks (viscous torque, stall current)

    @classmethod
    def from_config(cls, config: dict) -> 'ExtendedPlantRandomization':
        """Create ExtendedPlantRandomization from config dictionary."""
        if 'vehicle_randomization' not in config:
            return cls()  # Use defaults

        vr_config = config['vehicle_randomization']
        return cls(
            mass_range=tuple(vr_config.get('mass_range', (1500.0, 6000.0))),
            drag_area_range=tuple(vr_config.get('drag_area_range', (0.2, 0.8))),
            rolling_coeff_range=tuple(vr_config.get('rolling_coeff_range', (0.007, 0.015))),
            actuator_tau_range=tuple(vr_config.get('actuator_tau_range', (0.05, 0.30))),
            grade_deg_range=tuple(vr_config.get('grade_range_deg', (-5.7, 5.7))),
            motor_Vmax_range=tuple(vr_config.get('motor_Vmax_range', (200.0, 800.0))),
            motor_R_range=tuple(vr_config.get('motor_R_range', (0.02, 0.6))),
            motor_L_range=tuple(vr_config.get('motor_L_range', (1e-4, 1e-2))),
            motor_K_range=tuple(vr_config.get('motor_K_range', (0.05, 0.4))),
            # Support both old 'motor_Bm_range' and new 'motor_b_range' keys
            motor_b_range=tuple(vr_config.get('motor_b_range', vr_config.get('motor_Bm_range', (1e-6, 5e-3)))),
            motor_J_range=tuple(vr_config.get('motor_J_range', (1e-4, 1e-2))),
            gear_ratio_range=tuple(vr_config.get('gear_ratio_range', (4.0, 20.0))),
            brake_tau_range=tuple(vr_config.get('brake_tau_range', (0.04, 0.12))),
            brake_accel_range=tuple(vr_config.get('brake_accel_range', (8.0, 11.0))),
            brake_p_range=tuple(vr_config.get('brake_p_range', (1.0, 1.8))),
            brake_kappa_range=tuple(vr_config.get('brake_kappa_range', (0.02, 0.25))),
            mu_range=tuple(vr_config.get('mu_range', (0.7, 1.0))),
            wheel_radius_range=tuple(vr_config.get('wheel_radius_range', (0.26, 0.38))),
            wheel_inertia_range=tuple(vr_config.get('wheel_inertia_range', (0.5, 5.0))),
            eta_gb_range=tuple(vr_config.get('eta_gb_range', (0.85, 0.98))),
            # Creep parameters (optional, from top-level 'creep' key if present)
            creep_a_max=config.get('creep', {}).get('a_max'),
            creep_v_cutoff=config.get('creep', {}).get('v_cutoff'),
            creep_v_hold=config.get('creep', {}).get('v_hold'),
            # Feasibility thresholds
            min_accel_from_rest=vr_config.get('min_accel_from_rest', 2.5),
            min_brake_decel=vr_config.get('min_brake_decel', 4.0),
            min_top_speed=vr_config.get('min_top_speed', 20.0),
            skip_feasibility_checks=vr_config.get('skip_feasibility_checks', False),
            skip_sanity_checks=vr_config.get('skip_sanity_checks', False),
        )

    @classmethod
    def from_fitted_params(
        cls,
        fitted_params_path: str,
        spread_pct: float = 0.1,
    ) -> 'ExtendedPlantRandomization':
        """Create ExtendedPlantRandomization centered on fitted vehicle parameters.
        
        This factory method loads fitted parameters from a JSON file and creates
        a randomization config with ranges centered around the fitted values.
        
        Args:
            fitted_params_path: Path to fitted_params.json file
            spread_pct: Spread percentage around fitted means (default: 0.1 = ±10%)
            
        Returns:
            ExtendedPlantRandomization with ranges centered on fitted params
        """
        from pathlib import Path
        from fitting.randomization_config import (
            CenteredRandomizationConfig,
            create_extended_randomization_from_fitted,
        )
        
        return create_extended_randomization_from_fitted(
            Path(fitted_params_path),
            spread_pct=spread_pct,
        )


def compute_vehicle_capabilities(
    V_max: float, R: float, K_e: float, K_t: float, b: float, 
    gear_ratio: float, eta_gb: float, r_w: float, mass: float,
    CdA: float, C_rr: float, T_brake_max: float
) -> dict:
    """Compute derived vehicle capabilities for feasibility checking.
    
    Based on new_params_randomization.md specifications.
    
    Returns:
        Dictionary with:
        - i_stall: Stall current (A)
        - tau_wheel_drive_max: Max drive torque at wheel (Nm)
        - F_drive_max0: Max drive force at standstill (N)
        - v_no_load_max: Max no-load speed (m/s) - back-EMF limit
        - v_ss_level: Steady-state max speed on level ground (m/s)
        - a_max_from_rest: Max acceleration from standstill (m/s²)
        - a_brake_max: Max braking deceleration magnitude (m/s²)
    """
    g = GRAVITY
    
    # Stall current (at zero speed, V_cmd = V_max)
    i_stall = V_max / max(R, 1e-9)
    
    # Stall motor shaft torque
    tau_m_stall = K_t * i_stall
    
    # Stall wheel torque (accounting for gearbox)
    tau_wheel_drive_max = eta_gb * gear_ratio * tau_m_stall
    
    # Max drive force at standstill
    F_drive_max0 = tau_wheel_drive_max / max(r_w, 1e-6)
    
    # No-load max speed (back-EMF limit): V_max = K_e * omega_m
    omega_m_no_load = V_max / max(K_e, 1e-9)
    v_no_load_max = (omega_m_no_load / gear_ratio) * r_w
    
    # Resistive forces at standstill
    F_roll_0 = C_rr * mass * g
    
    # Max acceleration from rest (on level ground, no drag yet)
    a_max_from_rest = (F_drive_max0 - F_roll_0) / mass
    
    # Steady-state max speed under full throttle (on level ground)
    # Solve: F_drive_avail(v) = F_resist(v) numerically
    v_ss_level = _compute_steady_state_speed(
        V_max, R, K_e, K_t, b, gear_ratio, eta_gb, r_w, mass, CdA, C_rr, grade=0.0
    )
    
    # Max braking deceleration (at low speed on level ground)
    # Brake force + resistive forces
    F_brake_max = T_brake_max / max(r_w, 1e-6)
    # At low speed, resistive forces are minimal, so approx:
    a_brake_max = F_brake_max / mass
    
    return {
        'i_stall': i_stall,
        'tau_wheel_drive_max': tau_wheel_drive_max,
        'F_drive_max0': F_drive_max0,
        'v_no_load_max': v_no_load_max,
        'v_ss_level': v_ss_level,
        'a_max_from_rest': a_max_from_rest,
        'a_brake_max': a_brake_max,
    }


def _compute_steady_state_speed(
    V_max: float, R: float, K_e: float, K_t: float, b: float,
    gear_ratio: float, eta_gb: float, r_w: float, mass: float,
    CdA: float, C_rr: float, grade: float = 0.0, tol: float = 0.01
) -> float:
    """Compute steady-state max speed under full throttle using bisection.
    
    Finds v where F_drive_avail(v) = F_resist(v, grade).
    """
    g = GRAVITY
    
    def F_drive_avail(v: float) -> float:
        """Drive force available at speed v."""
        omega_m = gear_ratio * v / max(r_w, 1e-6)
        i_max_at_v = max(0.0, (V_max - K_e * omega_m) / max(R, 1e-9))
        tau_m_net = max(0.0, K_t * i_max_at_v - b * omega_m)
        tau_wheel = eta_gb * gear_ratio * tau_m_net
        return tau_wheel / max(r_w, 1e-6)
    
    def F_resist(v: float) -> float:
        """Resistive forces at speed v and grade."""
        F_aero = CdA * v * v
        F_roll = C_rr * mass * g * np.cos(grade)
        F_grade = mass * g * np.sin(grade)
        return F_aero + F_roll + F_grade
    
    # No-load max speed upper bound
    omega_m_no_load = V_max / max(K_e, 1e-9)
    v_hi = (omega_m_no_load / gear_ratio) * r_w
    v_lo = 0.0
    
    # Check if starting condition is feasible
    if F_drive_avail(0.0) < F_resist(0.0):
        return 0.0  # Can't even start
    
    # Bisection to find equilibrium
    for _ in range(60):
        v_mid = 0.5 * (v_lo + v_hi)
        if F_drive_avail(v_mid) >= F_resist(v_mid):
            v_lo = v_mid
        else:
            v_hi = v_mid
        if v_hi - v_lo < tol:
            break
    
    return v_lo


def compute_max_accel_at_speed(
    v: float, grade: float,
    V_max: float, R: float, K_e: float, K_t: float, b: float,
    gear_ratio: float, eta_gb: float, r_w: float, mass: float,
    CdA: float, C_rr: float
) -> float:
    """Compute maximum feasible acceleration at given speed and grade.
    
    Uses the no-regen model: i >= 0 always.
    """
    g = GRAVITY
    
    # Motor angular speed
    omega_m = gear_ratio * v / max(r_w, 1e-6)
    
    # Max current at this speed (no-regen: clamp to 0)
    i_max_at_v = max(0.0, (V_max - K_e * omega_m) / max(R, 1e-9))
    
    # Motor shaft torque (accounting for viscous)
    tau_m_net = max(0.0, K_t * i_max_at_v - b * omega_m)
    
    # Wheel drive torque and force
    tau_wheel = eta_gb * gear_ratio * tau_m_net
    F_drive = tau_wheel / max(r_w, 1e-6)
    
    # Resistive forces
    F_aero = CdA * v * abs(v)
    F_roll = C_rr * mass * g * np.cos(grade)
    F_grade = mass * g * np.sin(grade)
    F_resist = F_aero + F_roll + F_grade
    
    # Net force and acceleration
    F_net = F_drive - F_resist
    a_max = F_net / mass
    
    return a_max


def sample_extended_params(rng: np.random.Generator, rand: ExtendedPlantRandomization) -> ExtendedPlantParams:
    """Sample plant parameters for the extended dynamics with rejection sampling.
    
    Uses log-uniform sampling for electrical/mechanical params that span orders of magnitude.
    Rejection sampling ensures:
    - Sufficient acceleration capability from standstill
    - Sufficient braking capability
    - Reasonable top speed
    
    Based on new_params_randomization.md specifications.
    """

    def _log_uniform(lo: float, hi: float) -> float:
        """Sample from log-uniform distribution."""
        return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))

    # Rejection sampling loop
    max_attempts = 200
    for attempt in range(max_attempts):
        # Sample vehicle parameters (uniform)
        mass = float(rng.uniform(*rand.mass_range))
        CdA = float(rng.uniform(*rand.drag_area_range))
        C_rr = float(rng.uniform(*rand.rolling_coeff_range))
        
        # Sample wheel parameters
        wheel_radius = float(rng.uniform(*rand.wheel_radius_range))
        wheel_inertia = _log_uniform(*rand.wheel_inertia_range)
        
        # Sample motor electrical parameters
        # V_max - uniform (user requirement)
        V_max = float(rng.uniform(*rand.motor_Vmax_range))
        # R - log-uniform (spans orders of magnitude)
        R = _log_uniform(*rand.motor_R_range)
        # L - log-uniform
        L = _log_uniform(*rand.motor_L_range)
        # K_t, K_e - log-uniform (SI: K_t = K_e)
        K_t = _log_uniform(*rand.motor_K_range)
        K_e = K_t  # SI units: K_e = K_t for DC motor
        # b - log-uniform (viscous friction)
        b = _log_uniform(*rand.motor_b_range)
        # J - log-uniform (rotor inertia)
        J = _log_uniform(*rand.motor_J_range)
        
        # Gearbox parameters (uniform)
        gear_ratio = float(rng.uniform(*rand.gear_ratio_range))
        eta_gb = float(rng.uniform(*rand.eta_gb_range))
        
        # Brake parameters - compute T_brake_max from desired braking acceleration
        # a_brake_max = T_brake_max / (r_w * mass)
        # => T_brake_max = a_brake_max * r_w * mass
        desired_brake_accel = float(rng.uniform(*rand.brake_accel_range))
        T_brake_max = desired_brake_accel * wheel_radius * mass
        
        # Compute vehicle capabilities
        caps = compute_vehicle_capabilities(
            V_max=V_max, R=R, K_e=K_e, K_t=K_t, b=b,
            gear_ratio=gear_ratio, eta_gb=eta_gb, r_w=wheel_radius, mass=mass,
            CdA=CdA, C_rr=C_rr, T_brake_max=T_brake_max
        )
        
        # Feasibility checks per new_params_randomization.md section 9
        # Skip if explicitly disabled (fitted params mode)
        if not rand.skip_feasibility_checks:
            # Check 1: Minimum acceleration from rest
            if caps['a_max_from_rest'] < rand.min_accel_from_rest:
                continue
            
            # Check 2: Minimum top speed (no-load or steady-state)
            if caps['v_no_load_max'] < rand.min_top_speed:
                continue
            if caps['v_ss_level'] < rand.min_top_speed * 0.8:  # Allow some margin
                continue
            
            # Check 3: Braking deceleration - verify it matches our constraint
            # Since we set T_brake_max = desired_brake_accel * r_w * mass,
            # caps['a_brake_max'] should equal desired_brake_accel
            # Add small tolerance check for numerical precision
            if abs(caps['a_brake_max'] - desired_brake_accel) > 0.1:
                continue
        
        # Sanity checks - skip if flag is set (fitted params mode)
        if not rand.skip_sanity_checks:
            # Sanity check: viscous torque should be small compared to EM torque
            omega_ref = 300.0  # rad/s reference
            I_ref = min(V_max / R, 500.0)  # capped reference current
            tau_visc = b * omega_ref
            tau_em = K_t * I_ref
            if tau_visc > 0.2 * tau_em:
                continue  # Viscous damping too high
            
            # Sanity check: reasonable stall current (cap at 2000A for realism)
            i_stall = V_max / R
            if i_stall > 2000.0:
                continue
        
        # All checks passed - create parameter objects
        body = BodyParams(
            mass=mass,
            drag_area=CdA,
            rolling_coeff=C_rr,
            grade_rad=np.deg2rad(float(rng.uniform(*rand.grade_deg_range))),
        )
        motor = MotorParams(
            R=R,
            L=L,
            K_e=K_e,
            K_t=K_t,
            b=float(b),
            J=J,
            V_max=V_max,
            gear_ratio=gear_ratio,
            eta_gb=eta_gb,
        )
        brake = BrakeParams(
            T_br_max=T_brake_max,
            p_br=float(rng.uniform(*rand.brake_p_range)),
            tau_br=float(rng.uniform(*rand.brake_tau_range)),
            kappa_c=_log_uniform(*rand.brake_kappa_range),
            mu=float(rng.uniform(*rand.mu_range)),
        )
        wheel = WheelParams(
            radius=wheel_radius,
            inertia=wheel_inertia,
            v_eps=0.1,  # keep fixed
        )
        # Creep parameters: use from config if specified, otherwise use defaults
        creep = CreepParams(
            a_max=rand.creep_a_max if rand.creep_a_max is not None else 0.5,
            v_cutoff=rand.creep_v_cutoff if rand.creep_v_cutoff is not None else 1.5,
            v_hold=rand.creep_v_hold if rand.creep_v_hold is not None else 0.08,
        )
        return ExtendedPlantParams(motor=motor, brake=brake, body=body, wheel=wheel, creep=creep)

    # Fallback if rejection sampling fails
    raise RuntimeError(f"Could not find suitable parameters after {max_attempts} attempts. "
                       f"Consider adjusting parameter ranges in ExtendedPlantRandomization.")


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
    motor_omega: float  # motor shaft angular speed (rad/s) - CTMS model
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
    creep_torque: float  # Creep torque at motor shaft (Nm) - for diagnostics
    coupling_enabled: bool  # True when motor is coupled to wheel (False during braking)


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
        # Initialize motor angular speed (CTMS model) - synced to wheel initially
        self.motor_omega = self.wheel_omega * self.params.motor.gear_ratio  # motor shaft rad/s
        self.back_emf_voltage = self.params.motor.K_e * self.motor_omega  # back-EMF voltage
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
        self.creep_torque = 0.0  # Initialize creep torque
        self._current_grade_rad = None  # Current grade override (None = use body.grade_rad)
        # Initialize previous motor states for coupling
        self.motor_current_prev = 0.0
        self.tau_m_prev = 0.0
        self.drive_torque_prev = 0.0
        # Initialize coupling state (motor coupled to wheel)
        self._coupling_enabled = True
        self._coupling_enabled_prev = True
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
            motor_omega=self.motor_omega,
            back_emf_voltage=self.back_emf_voltage,
            V_cmd=self.V_cmd,
            drive_torque=self.drive_torque,
            tire_force=self.tire_force,
            drag_force=self.drag_force,
            rolling_force=self.rolling_force,
            grade_force=self.grade_force,
            net_force=self.net_force,
            held_by_brakes=self.held_by_brakes,
            creep_torque=self.creep_torque,
            coupling_enabled=self._coupling_enabled,
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
        """Single-DOF rigid coupling model: motor always coupled to wheel via gearbox.
        
        Based on coupling.md specification:
        - Motor always coupled to wheels (single rotational DOF)
        - Brake acts as torque at wheel, reflected to motor shaft
        - No regeneration: negative current clamped to zero
        - Uses combined inertia J_eq = J_m + (J_w + m*r_w²) / N²
        - omega_m is the single source of truth; speed derived from it
        """
        motor = self.params.motor
        brake_params = self.params.brake
        wheel = self.params.wheel
        body = self.params.body

        # ===== ACTION MAPPING =====
        u = action
        u_th = max(0, u)  # throttle: u > 0
        self.V_cmd = u_th * motor.V_max  # commanded voltage (positive forward)

        # For braking (u < 0) or coasting (u = 0), motor voltage = 0 (no regen)
        if u <= 0:
            self.V_cmd = 0.0

        # Brake command from negative action
        brake_cmd = max(-u, 0.0)

        # ===== MOTOR PARAMETERS =====
        L = motor.L
        R = motor.R
        K_e = motor.K_e
        K_t = motor.K_t
        b = motor.b  # viscous friction
        J_m = motor.J  # motor rotor inertia
        N = motor.gear_ratio
        eta = motor.eta_gb
        r_w = wheel.radius
        J_w = wheel.inertia
        
        # ===== CREEP TORQUE COMPUTATION =====
        # Compute EV-style creep behavior: low-speed forward motion at zero throttle
        # Creep is parameterized by max acceleration and fades with speed
        creep = self.params.creep
        
        # Step 1: Convert creep acceleration to motor torque (dynamic computation)
        # This ensures creep adapts to vehicle mass, gear ratio, etc.
        F_creep_max = body.mass * creep.a_max  # [N] max creep force
        T_wheel_creep_max = F_creep_max * r_w  # [Nm] max creep torque at wheel
        T_motor_creep_max = T_wheel_creep_max / (N * eta)  # [Nm] max creep torque at motor shaft
        
        # Step 2: Speed-dependent fade using gentler power function
        # Creep fades smoothly from full at v=0 to zero at v=v_cutoff
        # Use a gentler fade curve that maintains more torque at higher speeds
        # Use current motor omega to compute vehicle speed
        omega_m_current = self.motor_omega
        v_current = (omega_m_current / N) * r_w  # current vehicle speed from motor
        v_abs = abs(v_current)
        x = v_abs / max(creep.v_cutoff, 1e-6)  # normalized speed
        if x < 1.0:
            # Use a very gentle power fade: w = 1 - x^5
            # Original cubic smoothstep: w = 1 - 3x^2 + 2x^3 (maintains ~50% torque at x=0.5)
            # Power fade: w = 1 - x^5 (maintains ~97% torque at x=0.5, ~33% at x=0.925)
            # This allows vehicle to reach very close to v_cutoff before equilibrium
            # The fade starts immediately but decays very slowly, avoiding steep curves
            w_fade = 1.0 - x**5  # very gentle power fade
        else:
            w_fade = 0.0
        
        # Step 3: Brake dominance - creep must never fight the brake
        # Smoothly suppress creep as brake is applied
        w_brake_suppression = 1.0 - brake_cmd
        
        # Final creep torque at motor shaft
        T_creep_motor = T_motor_creep_max * w_fade * w_brake_suppression
        self.creep_torque = float(T_creep_motor)  # Store for diagnostics
        
        # Step 4: Convert creep torque to equivalent voltage addition
        # When coasting (u <= 0), add creep voltage to V_cmd
        # Creep provides a "virtual voltage" that produces the desired torque
        # From motor equation: tau_m = K_t * i, and i = (V - K_e*omega) / R
        # So: tau_m = K_t * (V - K_e*omega) / R
        # Solving for V given desired tau_m: V = (tau_m * R / K_t) + K_e*omega
        V_creep = (T_creep_motor * R / max(K_t, 1e-9)) + K_e * omega_m_current
        
        # Apply creep voltage only when not actively throttling (u <= 0)
        # This ensures smooth blending: throttle dominates when u > 0, creep fills in when u ≈ 0
        if u <= 0:
            self.V_cmd = max(V_creep, 0.0)  # Creep voltage (non-negative)

        # Combined inertia at motor shaft for single-DOF rigid coupling:
        # J_eff = J_m + (J_w + m * r_w^2) / N^2
        # 
        # The wheel and vehicle mass inertias are DIVIDED by N² (not multiplied)
        # because when the motor spins N times faster than the wheel, the 
        # reflected inertia is reduced by N² (from energy conservation).
        # 
        # Energy: 0.5 * J_w * ω_w² = 0.5 * J_w * (ω_m/N)² = 0.5 * (J_w/N²) * ω_m²
        # Similarly for vehicle mass: 0.5 * m * v² = 0.5 * (m*r_w²/N²) * ω_m²
        J_eq = J_m + (J_w + body.mass * r_w ** 2) / (N ** 2)

        # Current limits
        I_max = motor.V_max / max(R, 1e-9)  # Stall current
        allow_regen = False

        # ===== KINEMATICS FROM MOTOR STATE =====
        # omega_m is the single source of truth
        omega_m = self.motor_omega
        omega_w = omega_m / N  # wheel angular speed
        v_from_rot = omega_w * r_w  # vehicle speed derived from rotation

        # ===== COMPUTE TIRE FORCE =====
        # Tire force opposes motion (positive when opposing forward)
        # Use friction model based on current state
        mu_k = brake_params.mu
        N_normal = body.mass * GRAVITY
        F_fric_max = mu_k * N_normal

        # External resistive forces (for tire force calculation)
        F_drag = 0.5 * body.air_density * body.drag_area * v_from_rot * abs(v_from_rot)
        v_threshold = 0.1
        roll_factor = min(1.0, abs(v_from_rot) / v_threshold)
        F_roll = body.rolling_coeff * body.mass * GRAVITY * roll_factor
        grade_rad = self._current_grade_rad if self._current_grade_rad is not None else body.grade_rad
        F_grade = body.mass * GRAVITY * np.sin(grade_rad)

        # Tire force is what the ground applies to accelerate the vehicle
        # F_tire = m * a + F_resist (to achieve acceleration a)
        # In steady state or quasi-static, F_tire balances resistive forces
        # For now, compute based on what the motor can provide minus resistances
        # This will be updated iteratively

        # ===== COMPUTE BRAKE TORQUE AT WHEEL =====
        # Brake dynamics (first-order lag)
        T_br_cmd = brake_params.T_br_max * (brake_cmd ** brake_params.p_br)
        self.brake_torque += dt / max(brake_params.tau_br, 1e-4) * (T_br_cmd - self.brake_torque)

        # Brake torque magnitude
        tau_brake_mag = max(self.brake_torque, 0.0)

        # Brake torque opposes current motion direction
        # Sign convention: positive torque in motor equation accelerates motor forward
        # So brake should apply negative torque if moving forward, positive if backward
        # At rest, brake should hold the vehicle
        v_hold = 0.5  # velocity threshold for "nearly stopped" - wider for stability
        v_eps = 0.05  # smooth transition zone
        
        if abs(v_from_rot) < v_hold and tau_brake_mag > 100.0:
            # Nearly stopped with brakes applied: clamp motor omega to zero
            # This prevents oscillation at zero crossing
            self.held_by_brakes = True
            # Use a smooth factor that ramps brake effect to zero near zero speed
            # to prevent discontinuous torque jumps
            speed_factor = min(1.0, abs(v_from_rot) / max(v_eps, 1e-6))
            if v_from_rot > 0:
                tau_brake_wheel = tau_brake_mag * speed_factor
            elif v_from_rot < 0:
                tau_brake_wheel = -tau_brake_mag * speed_factor
            else:
                tau_brake_wheel = 0.0
        elif v_from_rot > v_eps:
            # Moving forward: brake opposes forward motion
            tau_brake_wheel = tau_brake_mag
            self.held_by_brakes = False
        elif v_from_rot < -v_eps:
            # Moving backward: brake opposes backward motion
            tau_brake_wheel = -tau_brake_mag
            self.held_by_brakes = False
        else:
            # In transition zone: smooth interpolation
            speed_factor = v_from_rot / v_eps  # -1 to +1
            tau_brake_wheel = tau_brake_mag * speed_factor
            self.held_by_brakes = tau_brake_mag > 100.0

        # ===== COMPUTE TIRE CONTACT TORQUE =====
        # Tire contact torque at wheel (positive opposes forward motion)
        # T_tire = F_resist * r_w where F_resist opposes motion
        T_tire = (F_drag + F_roll + F_grade) * r_w  # Resistive torque at wheel

        # ===== REFLECT WHEEL TORQUES TO MOTOR SHAFT =====
        # Total wheel opposing torque (at rest, only resistive loads)
        tau_wheel_opp = tau_brake_wheel + T_tire
        # Reflected to motor shaft
        tau_reflected = tau_wheel_opp / max(eta * N, 1e-12)

        # ===== ELECTRICAL INTEGRATION =====
        # Use internal substeps for motor integration
        motor_substeps = max(1, int(dt / (L / R / 10) + 1))
        motor_substeps = min(motor_substeps, 20)
        dt_motor = dt / motor_substeps

        i_curr = self.motor_current
        omega_m_curr = self.motor_omega

        # Track initial omega sign for zero-crossing detection
        omega_initial_sign = np.sign(omega_m_curr) if abs(omega_m_curr) > 1e-6 else 0
        
        for _ in range(motor_substeps):
            # Electrical: L di/dt = V - R*i - K_e*ω_m
            di_dt = (self.V_cmd - R * i_curr - K_e * omega_m_curr) / max(L, 1e-12)
            i_new = i_curr + dt_motor * di_dt

            # NO REGEN: clamp negative current to zero
            if not allow_regen:
                i_new = max(i_new, 0.0)
            i_new = min(i_new, I_max)

            # Mechanical: J_eq dω_m/dt = K_t*i - b*ω_m - τ_reflected
            tau_m_shaft = K_t * i_new
            domega_dt = (tau_m_shaft - b * omega_m_curr - tau_reflected) / max(J_eq, 1e-12)
            omega_m_new = omega_m_curr + dt_motor * domega_dt

            # Prevent sign change when brakes are applied (would cause oscillation)
            # If braking and omega is about to cross zero, clamp to zero
            if brake_cmd > 0.1:
                if omega_initial_sign > 0 and omega_m_new < 0:
                    omega_m_new = 0.0
                elif omega_initial_sign < 0 and omega_m_new > 0:
                    omega_m_new = 0.0
            
            # Clamp motor omega to reasonable range
            omega_m_max = motor.V_max / max(K_e, 1e-9) * 1.5
            omega_m_new = float(np.clip(omega_m_new, -omega_m_max, omega_m_max))

            i_curr = i_new
            omega_m_curr = omega_m_new

        # ===== UPDATE MOTOR STATE =====
        self.motor_current = float(i_curr)
        self.motor_omega = float(omega_m_curr)
        self.back_emf_voltage = K_e * self.motor_omega

        # ===== COMPUTE DRIVE TORQUE AT WHEEL =====
        tau_m_shaft = K_t * self.motor_current
        tau_drive_wheel = eta * N * tau_m_shaft  # positive forward

        # Save for diagnostics
        self.drive_torque = float(tau_drive_wheel)
        self.tau_m_prev = tau_m_shaft
        self.motor_current_prev = self.motor_current
        self.drive_torque_prev = self.drive_torque

        # ===== COMPUTE TIRE FORCE AND NET FORCE =====
        F_drive = tau_drive_wheel / r_w  # positive forward

        # Tire force is net of drive and brake at wheel
        # tau_brake_wheel is signed (positive opposes forward, negative opposes backward)
        # F_brake = tau_brake_wheel / r_w (signed brake force at tire contact patch)
        F_brake = tau_brake_wheel / r_w
        
        # Tire force = Drive - Brake (both signed)
        F_tire_raw = F_drive - F_brake
        
        # Apply friction limit
        self.tire_force = float(np.clip(F_tire_raw, -F_fric_max, F_fric_max))

        # Net longitudinal force
        self.drag_force = F_drag
        self.rolling_force = F_roll
        self.grade_force = F_grade
        self.net_force = self.tire_force - self.drag_force - self.rolling_force - self.grade_force

        # ===== DERIVE VEHICLE STATE FROM MOTOR (SINGLE SOURCE OF TRUTH) =====
        # Speed comes directly from omega_m
        v_old = self.speed
        v_new = (self.motor_omega / N) * r_w
        
        # ===== HELD BY BRAKES / ZERO SPEED CLAMPING =====
        # If brakes are applied and speed is very low, clamp to zero to prevent oscillation
        v_hold_clamp = 0.3  # threshold for clamping to zero when brakes applied
        if brake_cmd > 0.1 and abs(v_new) < v_hold_clamp:
            # Clamp motor omega to zero (vehicle held)
            self.motor_omega = 0.0
            self.motor_current = 0.0
            v_new = 0.0
            self.held_by_brakes = True
            self.acceleration = 0.0
        else:
            self.held_by_brakes = False
            # Compute acceleration from actual speed change (single-DOF consistent)
            self.acceleration = (v_new - v_old) / max(dt, 1e-6)
        
        self.speed = float(v_new)

        # Wheel omega follows motor
        self.wheel_omega = self.motor_omega / N

        # Update position
        self.position += self.speed * dt

        # Coupling is always enabled in single-DOF model
        self._coupling_enabled = True

        # Compute slip ratio (for diagnostics)
        # In single-DOF model, wheel and vehicle are rigidly coupled, so slip is minimal
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


