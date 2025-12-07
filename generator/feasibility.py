"""Feasibility projector for vehicle constraints: torque, current, voltage/back-EMF, braking, friction."""

from dataclasses import dataclass
import torch


@dataclass
class VehicleCapabilities:
    """Vehicle parameters for feasibility checking.
    
    Based on new_params_randomization.md specifications.
    All tensors are shape [B] for batch dimension.
    """
    m: torch.Tensor          # [B] mass (kg)
    r_w: torch.Tensor        # [B] wheel radius (m)
    gear_ratio: torch.Tensor # [B] gear ratio
    eta_gb: torch.Tensor     # [B] gearbox efficiency
    K_t: torch.Tensor        # [B] motor torque constant (Nm/A)
    K_e: torch.Tensor        # [B] motor back-EMF constant (V/(rad/s)) - SI: K_e = K_t
    R: torch.Tensor          # [B] motor resistance (ohm)
    i_max: torch.Tensor      # [B] max current (A) - typically V_max/R
    V_max: torch.Tensor      # [B] max voltage (V)
    b: torch.Tensor          # [B] motor viscous damping (Nm·s/rad)
    T_brake_max: torch.Tensor # [B] max brake torque at wheel (Nm)
    mu: torch.Tensor         # [B] tire friction coefficient
    C_dA: torch.Tensor       # [B] drag coefficient * area (CdA in m²)
    C_r: torch.Tensor        # [B] rolling resistance coefficient


@dataclass
class FeasibilityParams:
    """Parameters for feasibility projection."""
    safety_margin: float = 0.88  # Safety margin for voltage/current (0.85-0.95)
    max_projection_iters: int = 5  # Maximum iterations for projection
    convergence_tol: float = 1e-3  # Convergence tolerance for projection


def compute_max_feasible_speed(vehicle: VehicleCapabilities) -> torch.Tensor:
    """Compute maximum feasible speed where back-EMF = V_max.
    
    When back_EMF >= V_max, the motor cannot produce any drive torque.
    This gives a hard speed limit: v_max = V_max * r_w / (K_e * gear_ratio)
    
    Args:
        vehicle: Vehicle parameters
        
    Returns:
        Maximum feasible speeds [B] (m/s)
    """
    # back_EMF = K_e * gear_ratio * v / r_w
    # Set back_EMF = V_max and solve for v:
    # v_max = V_max * r_w / (K_e * gear_ratio)
    v_max = (vehicle.V_max * vehicle.r_w) / (vehicle.K_e * vehicle.gear_ratio)
    return v_max


def compute_resistive_forces(
    v: torch.Tensor,          # [B,H] speeds (m/s)
    grade: torch.Tensor,      # [B,H] grades (radians)
    vehicle: VehicleCapabilities
) -> torch.Tensor:
    """Compute total resistive forces at given speeds and grades.

    Args:
        v: Speed profile [B,H]
        grade: Grade profile [B,H]
        vehicle: Vehicle parameters

    Returns:
        Resistive forces [B,H] (N)
    """
    # Aerodynamic drag: 0.5 * rho * C_dA * v^2
    # Note: C_dA already includes rho * C_dA factor
    F_drag = 0.5 * vehicle.C_dA.unsqueeze(-1) * v**2

    # Rolling resistance: C_r * m * g * cos(grade)
    # Approximate g * cos(grade) ≈ 9.81 for small grades
    F_roll = vehicle.C_r.unsqueeze(-1) * vehicle.m.unsqueeze(-1) * 9.81

    # Grade force: m * g * sin(grade)
    F_grade = vehicle.m.unsqueeze(-1) * 9.81 * torch.sin(grade)

    return F_drag + F_roll + F_grade


def compute_feasible_accelerations(
    v: torch.Tensor,          # [B,H] speeds (m/s)
    grade: torch.Tensor,      # [B,H] grades (radians)
    vehicle: VehicleCapabilities,
    params: FeasibilityParams
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute feasible acceleration bounds for given speeds and grades.

    Args:
        v: Speed profile [B,H]
        grade: Grade profile [B,H]
        vehicle: Vehicle parameters
        params: Feasibility parameters

    Returns:
        Tuple of (a_min, a_max) tensors [B,H] for braking and drive limits
    """
    F_resist = compute_resistive_forces(v, grade, vehicle)

    # Maximum drive acceleration (torque/current/voltage limited)
    # T_wheel_max = i_max * K_t * eta_gb * gear_ratio
    T_wheel_max = (vehicle.i_max * vehicle.K_t * vehicle.eta_gb * vehicle.gear_ratio).unsqueeze(-1)
    a_drive_max_torque = (T_wheel_max / vehicle.r_w.unsqueeze(-1) - F_resist) / vehicle.m.unsqueeze(-1)

    # Voltage/back-EMF constraint for drive (accounting for b - viscous damping)
    # Motor model: tau_m = K_t * I - b * omega_m
    # where I = (V_cmd - back_EMF) / R, back_EMF = K_e * omega_m
    # omega_m = gear_ratio * v / r_w
    # T_wheel = tau_m * gear_ratio * eta_gb
    # F_drive = T_wheel / r_w
    # a_max = (F_drive - F_resist) / m
    
    # Compute motor angular speed and back-EMF
    omega_m = (vehicle.gear_ratio.unsqueeze(-1) * v) / vehicle.r_w.unsqueeze(-1)  # [B,H]
    back_EMF = vehicle.K_e.unsqueeze(-1) * omega_m  # [B,H]
    V_max_eff = params.safety_margin * vehicle.V_max.unsqueeze(-1)  # [B,1]
    
    # Current limited by voltage: I = (V_max - back_EMF) / R, clamped to >= 0 (no regen)
    I_available = torch.clamp((V_max_eff - back_EMF) / vehicle.R.unsqueeze(-1), min=0.0)  # [B,H]
    
    # Motor torque: tau_m = K_t * I - b * omega_m
    tau_m = (vehicle.K_t.unsqueeze(-1) * I_available - 
             vehicle.b.unsqueeze(-1) * omega_m)  # [B,H]
    
    # Wheel torque and drive force
    T_wheel = tau_m * vehicle.gear_ratio.unsqueeze(-1) * vehicle.eta_gb.unsqueeze(-1)  # [B,H]
    F_drive = T_wheel / vehicle.r_w.unsqueeze(-1)  # [B,H]
    
    # Maximum acceleration from voltage constraint
    a_drive_max_voltage = (F_drive - F_resist) / vehicle.m.unsqueeze(-1)  # [B,H]
    
    # Hard speed limit: when back_EMF >= V_max, motor cannot produce drive torque
    # Set a_drive_max to 0 (or negative if resistive forces dominate)
    back_EMF_limit = back_EMF >= V_max_eff
    a_drive_max_voltage = torch.where(
        back_EMF_limit,
        torch.clamp(-F_resist / vehicle.m.unsqueeze(-1), max=0.0),  # Can only decelerate
        a_drive_max_voltage
    )

    # Drive acceleration is limited by both torque and voltage
    a_drive_max = torch.min(a_drive_max_torque, a_drive_max_voltage)

    # Braking acceleration (deceleration) limits
    # Maximum braking torque provides minimum acceleration (most negative)
    a_brake_min_torque = (-T_wheel_max / vehicle.r_w.unsqueeze(-1) - F_resist) / vehicle.m.unsqueeze(-1)

    # Tire friction limit: a >= -mu * g
    a_brake_min_friction = -vehicle.mu.unsqueeze(-1) * 9.81

    # Braking limit is the more restrictive (less negative) constraint
    a_brake_min = torch.max(a_brake_min_torque, a_brake_min_friction)

    return a_brake_min, a_drive_max


def project_window_to_feasible(
    raw_window: torch.Tensor,      # [B,H] raw target speeds
    grade_window: torch.Tensor,    # [B,H] grades for the window
    vehicle: VehicleCapabilities,
    dt: float,
    params: FeasibilityParams
) -> torch.Tensor:
    """Project a window of target speeds to vehicle-feasible speeds.

    Args:
        raw_window: Raw target speeds [B,H]
        grade_window: Grade profile [B,H]
        vehicle: Vehicle parameters
        dt: Time step (s)
        params: Feasibility parameters

    Returns:
        Feasible target speeds [B,H]
    """
    # Initialize with raw window
    v_projected = raw_window.clone()
    v_current = raw_window[:, 0].clone()  # Current speed at start of window

    # Apply hard speed limit from back-EMF constraint
    v_max = compute_max_feasible_speed(vehicle)  # [B]
    v_max_expanded = v_max.unsqueeze(-1)  # [B, 1] to broadcast with [B, H]
    v_projected = torch.clamp(v_projected, min=torch.tensor(0.0, device=v_projected.device), max=v_max_expanded)  # Clamp to [0, v_max]
    v_current = torch.clamp(v_current, min=torch.tensor(0.0, device=v_current.device), max=v_max)  # Clamp current speed too

    # Iterative projection
    for iter_idx in range(params.max_projection_iters):
        # Compute raw accelerations from current projected speeds
        v_with_current = torch.cat([v_current.unsqueeze(-1), v_projected], dim=-1)
        a_raw = (v_with_current[:, 1:] - v_with_current[:, :-1]) / dt

        # Get feasible acceleration bounds
        a_brake_min, a_drive_max = compute_feasible_accelerations(
            v_projected, grade_window, vehicle, params
        )

        # Clip accelerations to feasible bounds
        a_clipped = torch.clamp(a_raw, a_brake_min, a_drive_max)

        # Reintegrate to get corrected speeds
        v_corrected = torch.zeros_like(v_projected)
        v_corrected[:, 0] = v_current
        for h in range(v_projected.shape[-1]):
            if h > 0:
                v_corrected[:, h] = v_corrected[:, h-1] + a_clipped[:, h-1] * dt

        # Check convergence
        speed_change = torch.abs(v_corrected - v_projected).max()
        v_projected = v_corrected

        if speed_change < params.convergence_tol:
            break

    # Final check: ensure zero-speed feasibility (voltage at zero accel)
    v_projected = enforce_voltage_at_zero_accel(
        v_projected, grade_window, vehicle, params
    )

    return v_projected


def enforce_voltage_at_zero_accel(
    v_profile: torch.Tensor,       # [B,H] speed profile
    grade_profile: torch.Tensor,   # [B,H] grade profile
    vehicle: VehicleCapabilities,
    params: FeasibilityParams
) -> torch.Tensor:
    """Ensure speeds are feasible even at zero acceleration (back-EMF check).

    Args:
        v_profile: Speed profile [B,H]
        grade_profile: Grade profile [B,H]
        vehicle: Vehicle parameters
        params: Feasibility parameters

    Returns:
        Corrected speed profile [B,H]
    """
    # Compute required voltage at zero acceleration using corrected motor model
    F_resist = compute_resistive_forces(v_profile, grade_profile, vehicle)

    # Motor model: tau_m = K_t * I - b * omega_m
    # where I = (V_cmd - back_EMF) / R, back_EMF = K_e * omega_m
    # omega_m = gear_ratio * v / r_w
    # At zero accel: T_wheel_req = r_w * F_resist (to overcome resistance)
    # T_wheel = tau_m * gear_ratio * eta_gb
    # So: tau_m_req = T_wheel_req / (gear_ratio * eta_gb) = r_w * F_resist / (gear_ratio * eta_gb)
    
    omega_m = (vehicle.gear_ratio.unsqueeze(-1) * v_profile) / vehicle.r_w.unsqueeze(-1)  # [B,H]
    back_EMF = vehicle.K_e.unsqueeze(-1) * omega_m  # [B,H]
    
    T_wheel_req = vehicle.r_w.unsqueeze(-1) * F_resist  # [B,H]
    tau_m_req = T_wheel_req / (vehicle.gear_ratio.unsqueeze(-1) * vehicle.eta_gb.unsqueeze(-1))  # [B,H]
    
    # Solve for required voltage: tau_m = K_t * (V_req - back_EMF) / R - b * omega_m
    # Rearranging: V_req = (tau_m + b * omega_m) * R / K_t + back_EMF
    V_req_zero = ((tau_m_req + vehicle.b.unsqueeze(-1) * omega_m) * vehicle.R.unsqueeze(-1) / 
                  vehicle.K_t.unsqueeze(-1) + back_EMF)  # [B,H]

    V_max_eff = params.safety_margin * vehicle.V_max.unsqueeze(-1)

    # Find samples where voltage exceeds limit
    voltage_violation = V_req_zero > V_max_eff

    if voltage_violation.any():
        # For violating samples, reduce speed via bisection
        v_corrected = v_profile.clone()

        # Process each batch element separately
        for b in range(v_profile.shape[0]):
            for h in range(v_profile.shape[1]):
                if voltage_violation[b, h]:
                    v_corrected[b, h] = reduce_speed_until_voltage_ok(
                        v_profile[b, h],
                        grade_profile[b, h],
                        vehicle.m[b].item(),
                        vehicle.r_w[b].item(),
                        vehicle.gear_ratio[b].item(),
                        vehicle.eta_gb[b].item(),
                        vehicle.K_t[b].item(),
                        vehicle.K_e[b].item(),
                        vehicle.R[b].item(),
                        vehicle.b[b].item(),
                        F_resist[b, h].item(),
                        V_max_eff[b, 0].item(),  # Broadcast to scalar
                        v_min=0.0,
                        max_iters=20,
                        tol=1e-3
                    )

        return v_corrected

    return v_profile


def reduce_speed_until_voltage_ok(
    v_guess: float,
    grade: float,
    m: float,
    r_w: float,
    gear_ratio: float,
    eta_gb: float,
    K_t: float,
    K_e: float,
    R: float,
    b: float,
    F_resist: float,
    V_max_eff: float,
    v_min: float = 0.0,
    max_iters: int = 20,
    tol: float = 1e-3
) -> float:
    """Use bisection to find maximum feasible speed given voltage constraint.

    Args:
        v_guess: Initial speed guess
        grade: Road grade
        m: Vehicle mass
        r_w: Wheel radius
        gear_ratio: Gear ratio
        eta_gb: Gearbox efficiency
        K_t: Motor torque constant
        K_e: Motor back-EMF constant
        R: Motor resistance
        b: Motor viscous damping (Nm·s/rad)
        F_resist: Resistive force
        V_max_eff: Maximum effective voltage
        v_min: Minimum speed to consider
        max_iters: Maximum bisection iterations
        tol: Convergence tolerance

    Returns:
        Feasible speed
    """
    # Bisection between v_min and v_guess
    v_low = v_min
    v_high = max(v_guess, v_min + 1e-3)  # Ensure v_high > v_low

    for _ in range(max_iters):
        v_mid = 0.5 * (v_low + v_high)

        # Compute required voltage at this speed using corrected motor model
        omega_m = gear_ratio * v_mid / r_w
        back_EMF = K_e * omega_m
        
        # At zero accel: T_wheel_req = r_w * F_resist
        T_wheel_req = r_w * F_resist
        tau_m_req = T_wheel_req / (gear_ratio * eta_gb)
        
        # Solve for required voltage: tau_m = K_t * (V_req - back_EMF) / R - b * omega_m
        # Rearranging: V_req = (tau_m + b * omega_m) * R / K_t + back_EMF
        V_req = (tau_m_req + b * omega_m) * R / K_t + back_EMF

        if V_req > V_max_eff:
            # Voltage too high, reduce speed
            v_high = v_mid
        else:
            # Voltage OK, try higher speed
            v_low = v_mid

        if v_high - v_low < tol:
            break

    return v_low
