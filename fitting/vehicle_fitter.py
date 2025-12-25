"""Vehicle parameter fitting from trip data using trajectory simulation.

Fits vehicle dynamics parameters by simulating full trip trajectories and
minimizing velocity error (not instantaneous acceleration error).

This is more robust because:
1. Errors compound over time, forcing physically consistent parameters
2. Velocity integration smooths measurement noise
3. Tests the model's ability to predict real vehicle behavior
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

LOGGER = logging.getLogger(__name__)

# Physical constants (matching utils/dynamics.py)
GRAVITY: float = 9.80665  # m/s²
AIR_DENSITY: float = 1.225  # kg/m³


@dataclass(slots=True)
class FittedVehicleParams:
    """Fitted vehicle parameters matching ExtendedPlant model.
    
    ALL 18 parameters are fitted by default.
    """
    
    # === BODY PARAMETERS ===
    mass: float  # kg - vehicle mass
    drag_area: float  # m² - CdA (drag coefficient * frontal area)
    rolling_coeff: float  # dimensionless - rolling resistance coefficient
    
    # === MOTOR PARAMETERS ===
    motor_V_max: float  # V - maximum motor voltage
    motor_R: float  # Ω - armature resistance
    motor_L: float  # H - armature inductance
    motor_K: float  # Nm/A = V·s/rad - K_t = K_e (torque/back-EMF constant)
    motor_b: float  # Nm·s/rad - viscous friction
    motor_J: float  # kg·m² - rotor inertia
    
    # === DRIVETRAIN PARAMETERS ===
    gear_ratio: float  # N - gear reduction ratio
    eta_gb: float  # gearbox efficiency
    
    # === BRAKE PARAMETERS ===
    brake_T_max: float  # Nm - maximum brake torque at wheel
    brake_tau: float  # s - brake time constant
    brake_p: float  # brake exponent
    brake_kappa: float  # brake slip constant
    mu: float  # tire friction coefficient
    
    # === WHEEL PARAMETERS ===
    wheel_radius: float  # m - wheel radius
    wheel_inertia: float  # kg·m² - wheel + rotating assembly
    
    # === FITTING METADATA ===
    fit_loss: float = 0.0  # Final loss value (velocity MSE)
    num_samples: int = 0  # Number of data points used
    r_squared: float = 0.0  # Coefficient of determination for velocity
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "FittedVehicleParams":
        """Create from dictionary with backward compatibility."""
        # Handle very old files with motor_force_coeff
        if "motor_force_coeff" in d:
            d.pop("motor_force_coeff")
        
        # Default values for backward compatibility
        defaults = {
            "motor_V_max": 400.0,
            "motor_R": 0.2,
            "motor_L": 1e-3,
            "motor_K": 0.2,
            "motor_b": 1e-3,
            "motor_J": 1e-3,
            "gear_ratio": 10.0,
            "eta_gb": 0.92,
            "brake_T_max": 15000.0,
            "brake_tau": 0.08,
            "brake_p": 1.2,
            "brake_kappa": 0.08,
            "mu": 0.9,
            "wheel_radius": 0.346,
            "wheel_inertia": 1.5,
        }
        for key, val in defaults.items():
            if key not in d:
                d[key] = val
        return cls(**d)
    
    def save(self, path: Path) -> None:
        """Save fitted parameters to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved fitted params to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "FittedVehicleParams":
        """Load fitted parameters from JSON file."""
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)
    
    def to_extended_plant_params(self) -> Dict:
        """Convert to ExtendedPlantParams-compatible dictionary."""
        return {
            "mass": self.mass,
            "drag_area": self.drag_area,
            "rolling_coeff": self.rolling_coeff,
            "motor": {
                "V_max": self.motor_V_max,
                "R": self.motor_R,
                "L": self.motor_L,
                "K_t": self.motor_K,
                "K_e": self.motor_K,  # K_t = K_e for DC motors
                "b": self.motor_b,
                "J": self.motor_J,
            },
            "gear_ratio": self.gear_ratio,
            "eta_gb": self.eta_gb,
            "brake": {
                "T_max": self.brake_T_max,
                "tau": self.brake_tau,
                "p": self.brake_p,
                "kappa_c": self.brake_kappa,
                "mu": self.mu,
            },
            "wheel": {
                "radius": self.wheel_radius,
                "inertia": self.wheel_inertia,
            },
        }


@dataclass(slots=True)
class FitterConfig:
    """Configuration for vehicle parameter fitting.
    
    ALL parameters are fitted by default. Use same min/max bounds to fix a parameter.
    """
    
    # === SIMULATION SETTINGS ===
    dt: float = 0.1  # s - simulation timestep (will be estimated from data)
    
    # === INITIAL GUESSES (all parameters) ===
    mass_init: float = 1900.0  # kg
    drag_area_init: float = 0.65  # m² (Cd~0.29 * A~2.2m²)
    rolling_coeff_init: float = 0.011  # typical EV
    motor_V_max_init: float = 360.0  # V
    motor_R_init: float = 0.1  # Ω
    motor_L_init: float = 1e-3  # H - inductance
    motor_K_init: float = 0.15  # Nm/A
    motor_b_init: float = 1e-3  # Nm·s/rad - viscous friction
    motor_J_init: float = 1e-3  # kg·m² - rotor inertia
    gear_ratio_init: float = 7.98  # gear ratio
    eta_gb_init: float = 0.92  # gearbox efficiency
    brake_T_max_init: float = 15000.0  # Nm
    brake_tau_init: float = 0.08  # s - brake time constant
    brake_p_init: float = 1.2  # brake exponent
    brake_kappa_init: float = 0.08  # brake slip constant
    mu_init: float = 0.9  # tire friction coefficient
    wheel_radius_init: float = 0.346  # m
    wheel_inertia_init: float = 1.5  # kg·m²
    
    # === PARAMETER BOUNDS (all fitted by default) ===
    mass_bounds: Tuple[float, float] = (1800.0, 2300.0)  # kg
    drag_area_bounds: Tuple[float, float] = (0.4, 1.2)   # CdA (m²)
    rolling_coeff_bounds: Tuple[float, float] = (0.006, 0.020)  # C_rr
    motor_V_max_bounds: Tuple[float, float] = (320.0, 400.0)  # V
    motor_R_bounds: Tuple[float, float] = (0.02, 0.5)  # Ω
    motor_L_bounds: Tuple[float, float] = (1e-5, 1e-1)  # H
    motor_K_bounds: Tuple[float, float] = (0.05, 0.5)  # Nm/A
    motor_b_bounds: Tuple[float, float] = (1e-6, 1e-1)  # Nm·s/rad
    motor_J_bounds: Tuple[float, float] = (1e-2, 1e-1)  # kg·m²
    gear_ratio_bounds: Tuple[float, float] = (4.3, 11.0)  # N
    eta_gb_bounds: Tuple[float, float] = (0.85, 0.99)  # efficiency
    brake_T_max_bounds: Tuple[float, float] = (10000.0, 20000.0)  # Nm
    brake_tau_bounds: Tuple[float, float] = (0.01, 0.5)  # s
    brake_p_bounds: Tuple[float, float] = (0.5, 3.0)  # exponent
    brake_kappa_bounds: Tuple[float, float] = (0.01, 0.3)  # slip constant
    mu_bounds: Tuple[float, float] = (0.5, 1.2)  # friction coefficient
    wheel_radius_bounds: Tuple[float, float] = (0.315, 0.34)  # m
    wheel_inertia_bounds: Tuple[float, float] = (1.0, 2.0)  # kg·m²
    
    # === SEGMENT FILTERING ===
    min_speed: float = 0.0  # m/s - minimum speed to include
    max_speed: float = 20.0  # m/s - maximum speed to include (filter unrealistic values)
    max_accel: float = 3.0  # m/s² - filter extreme accelerations
    min_segment_length: int = 150  # minimum timesteps per segment
    max_segment_length: int = 150  # maximum timesteps per segment (memory)
    downsampling_factor: int = 1  # downsample data by taking every Nth sample (1 = no downsampling)
    max_zero_speed_fraction: float = 0.05  # maximum fraction of segments with zero/near-zero speed (0.05 = 5%)
    zero_speed_eps: float = 0.1  # epsilon for zero speed threshold (m/s) - segments with mean speed < eps are considered zero-speed
    
    # === OPTIMIZATION SETTINGS ===
    max_iter: int = 200  # iterations per optimization call
    tolerance: float = 1e-9
    
    # === TRAJECTORY BATCHING ===
    segments_per_batch: int = 10  # number of trip segments per batch (smaller = faster)
    num_epochs: int = 1  # number of passes over all segments
    shuffle_segments: bool = True
    validation_fraction: float = 0.1  # fraction of segments for validation
    
    # === WARMUP ===
    use_warmup: bool = False  # enable warmup to find better initial guess
    warmup_samples: int = 10  # number of random parameter sets to try
    warmup_seed: int = 42  # random seed for warmup sampling
    
    # === BARRIER FUNCTIONS ===
    use_barrier: bool = False  # enable interior-point barrier method to avoid active constraints
    barrier_mu: float = 0.01  # barrier parameter μ (smaller = stronger barrier, keeps params away from boundaries)


@dataclass
class TripSegment:
    """A single trip segment for trajectory simulation."""
    trip_id: str
    speed: np.ndarray      # measured speeds (m/s) - target
    throttle: np.ndarray   # throttle input (0-100)
    brake: np.ndarray      # brake input (0-100)
    grade: np.ndarray      # road grade (rad)
    dt: float              # timestep (s)
    
    @property
    def length(self) -> int:
        return len(self.speed)
    
    @property
    def initial_speed(self) -> float:
        return float(self.speed[0])


class VehicleParamFitter:
    """Fits vehicle dynamics parameters by simulating full trajectories.
    
    Uses quasi-steady-state DC motor model:
    
        Motor current (quasi-steady): i = (V_cmd - K_e * ω_m) / R
        Motor torque: τ_m = K_t * i
        Wheel torque: τ_w = η_gb * N * τ_m
        Drive force: F_drive = τ_w / r
        
    The key difference from single-step fitting is that we:
    1. Start with the measured initial speed v[0]
    2. Simulate forward: v[t+1] = v[t] + a(params, v[t], inputs[t]) * dt
    3. Minimize MSE between simulated v and measured v over full trajectory
    
    This forces parameters to be physically consistent over time.
    """
    
    PARAM_NAMES = [
        "mass", "drag_area", "rolling_coeff",
        "motor_V_max", "motor_R", "motor_L", "motor_K", "motor_b", "motor_J",
        "gear_ratio", "eta_gb",
        "brake_T_max", "brake_tau", "brake_p", "brake_kappa", "mu",
        "wheel_radius", "wheel_inertia",
    ]
    
    def __init__(self, config: Optional[FitterConfig] = None):
        self.config = config or FitterConfig()
        self._segments: List[TripSegment] = []
        self._current_loss: float = 0.0
        self._bounds: Optional[List[Tuple[float, float]]] = None  # Store bounds for barrier computation
    
    def _barrier_penalty(self, params: np.ndarray, bounds: List[Tuple[float, float]]) -> float:
        """Compute logarithmic barrier penalty to keep parameters away from boundaries.
        
        Barrier function: -μ * Σ (log(x_i - l_i) + log(u_i - x_i))
        
        Args:
            params: Parameter vector
            bounds: List of (min, max) tuples for each parameter
            
        Returns:
            Barrier penalty value (positive, added to loss)
        """
        if not self.config.use_barrier:
            return 0.0
        
        mu = self.config.barrier_mu
        eps = 1e-10  # Small epsilon to prevent log(0) and handle numerical issues
        
        penalty_sum = 0.0
        for i, (x, (l, u)) in enumerate(zip(params, bounds)):
            # Skip barrier for fixed parameters (min == max)
            if l == u:
                continue
            
            # Ensure we're within bounds (with small margin for numerical stability)
            if x <= l + eps or x >= u - eps:
                # Return large penalty if too close to boundary
                return 1e10
            
            # Compute barrier terms: log(x - l) + log(u - x)
            # Both terms are negative (since x is between l and u)
            # We negate the sum and multiply by mu to get positive penalty
            penalty_sum += np.log(x - l) + np.log(u - x)
        
        # Return -mu * sum (negative because we want to add this to loss)
        # Since penalty_sum is negative, -mu * penalty_sum is positive
        return -mu * penalty_sum
    
    def load_trip_data(self, data_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
        """Load trip data from .pt file.
        
        Args:
            data_path: Path to all_trips_data.pt file
            
        Returns:
            Dictionary with trip_id -> {speed, acceleration, throttle, brake, angle, time}
        """
        import torch
        
        raw = torch.load(data_path, weights_only=False)
        trips = {}
        
        for key, value in raw.items():
            if key == "metadata":
                continue
            if not isinstance(value, dict):
                continue
            
            # Extract required fields
            try:
                trip_data = {
                    "speed": np.asarray(value["speed"], dtype=np.float64),
                    "acceleration": np.asarray(value["acceleration"], dtype=np.float64),
                    "throttle": np.asarray(value["throttle"], dtype=np.float64),
                    "brake": np.asarray(value["brake"], dtype=np.float64),
                    "angle": np.asarray(value.get("angle", np.zeros_like(value["speed"])), dtype=np.float64),
                }
                # Try to get time or estimate dt
                if "time" in value:
                    trip_data["time"] = np.asarray(value["time"], dtype=np.float64)
                trips[key] = trip_data
            except KeyError as e:
                LOGGER.warning(f"Trip {key} missing field {e}, skipping")
                continue
        
        LOGGER.info(f"Loaded {len(trips)} trips from {data_path}")
        return trips
    
    def _estimate_dt(self, trips: Dict[str, Dict[str, np.ndarray]]) -> float:
        """Estimate timestep from trip data."""
        dts = []
        for trip_id, data in trips.items():
            if "time" in data:
                t = data["time"]
                if len(t) > 1:
                    dt_trip = np.median(np.diff(t))
                    if 0.01 < dt_trip < 1.0:  # reasonable range
                        dts.append(dt_trip)
        
        if dts:
            return float(np.median(dts))
        return self.config.dt  # fallback to config default
    
    def _create_segments(
        self, trips: Dict[str, Dict[str, np.ndarray]], dt: float
    ) -> List[TripSegment]:
        """Create trip segments for trajectory simulation.
        
        Splits trips into segments, filtering for quality data.
        """
        cfg = self.config
        segments = []
        
        for trip_id, data in trips.items():
            v = data["speed"]
            a = data["acceleration"]
            th = data["throttle"]
            br = data["brake"]
            grade = data["angle"]
            
            n = len(v)
            if n < cfg.min_segment_length:
                continue
            
            # Find contiguous valid regions
            valid = (
                (v >= cfg.min_speed) &
                (v <= cfg.max_speed) &
                (np.abs(a) <= cfg.max_accel) &
                np.isfinite(v) &
                np.isfinite(a) &
                np.isfinite(th) &
                np.isfinite(br) &
                np.isfinite(grade)
            )
            
            # Split into contiguous segments
            segment_start = None
            for i in range(n):
                if valid[i] and segment_start is None:
                    segment_start = i
                elif (not valid[i] or i == n - 1) and segment_start is not None:
                    end = i if not valid[i] else i + 1
                    length = end - segment_start
                    
                    if length >= cfg.min_segment_length:
                        # Possibly split long segments
                        for seg_start in range(segment_start, end, cfg.max_segment_length):
                            seg_end = min(seg_start + cfg.max_segment_length, end)
                            seg_len = seg_end - seg_start
                            
                            if seg_len >= cfg.min_segment_length:
                                segments.append(TripSegment(
                                    trip_id=f"{trip_id}_{seg_start}",
                                    speed=v[seg_start:seg_end].copy(),
                                    throttle=th[seg_start:seg_end].copy(),
                                    brake=br[seg_start:seg_end].copy(),
                                    grade=grade[seg_start:seg_end].copy(),
                                    dt=dt,
                                ))
                    
                    segment_start = None
        
        LOGGER.info(f"Created {len(segments)} segments from {len(trips)} trips")
        return segments
    
    def _downsample_segments(
        self,
        segments: List[TripSegment],
        factor: int,
    ) -> List[TripSegment]:
        """Downsample segments by taking every Nth sample.
        
        Args:
            segments: List of trip segments
            factor: Downsampling factor (1 = no downsampling, 2 = every 2nd sample, etc.)
            
        Returns:
            List of downsampled segments (dt is updated accordingly)
        """
        if factor <= 1:
            return segments
        
        downsampled = []
        for segment in segments:
            # Take every Nth sample
            indices = np.arange(0, segment.length, factor)
            
            if len(indices) < self.config.min_segment_length:
                # Skip segments that become too short after downsampling
                continue
            
            downsampled.append(TripSegment(
                trip_id=segment.trip_id,
                speed=segment.speed[indices].copy(),
                throttle=segment.throttle[indices].copy(),
                brake=segment.brake[indices].copy(),
                grade=segment.grade[indices].copy(),
                dt=segment.dt * factor,  # Update dt to reflect new sampling rate
            ))
        
        LOGGER.info(f"Downsampled {len(segments)} segments to {len(downsampled)} segments (factor={factor})")
        return downsampled
    
    def _filter_zero_speed_segments(
        self,
        segments: List[TripSegment],
        max_fraction: float,
        eps: float,
    ) -> List[TripSegment]:
        """Filter segments to limit the fraction of zero-speed segments.
        
        A segment is considered "zero-speed" if its median speed is below eps.
        This prevents the dataset from being dominated by stationary/idle segments.
        Using median instead of mean makes the classification more robust to outliers.
        
        Args:
            segments: List of trip segments
            max_fraction: Maximum fraction of segments that can be zero-speed (0.05 = 5%)
            eps: Epsilon threshold for zero speed (m/s)
            
        Returns:
            Filtered list of segments
        """
        if max_fraction >= 1.0:
            return segments  # No filtering needed
        
        # Classify segments as zero-speed or not
        zero_speed_segments = []
        non_zero_segments = []
        
        for segment in segments:
            median_speed = np.median(segment.speed)
            if median_speed < eps:
                zero_speed_segments.append(segment)
            else:
                non_zero_segments.append(segment)
        
        n_total = len(segments)
        n_zero = len(zero_speed_segments)
        current_fraction = n_zero / n_total if n_total > 0 else 0.0
        
        if current_fraction <= max_fraction:
            # Already within limit, no filtering needed
            return segments
        
        # Need to filter: keep all non-zero segments, limit zero-speed segments
        max_zero_allowed = int(n_total * max_fraction)
        
        # If we have too many zero-speed segments, randomly sample to keep only max_zero_allowed
        if n_zero > max_zero_allowed:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_zero, size=max_zero_allowed, replace=False)
            zero_speed_segments = [zero_speed_segments[i] for i in indices]
        
        filtered = non_zero_segments + zero_speed_segments
        LOGGER.info(
            f"Filtered zero-speed segments: {n_zero}/{n_total} ({current_fraction:.1%}) -> "
            f"{len(zero_speed_segments)}/{len(filtered)} ({len(zero_speed_segments)/len(filtered):.1%})"
        )
        return filtered
    
    def _compute_acceleration(
        self,
        params: np.ndarray,
        speed: float,
        throttle: float,
        brake: float,
        grade: float,
    ) -> float:
        """Compute acceleration for a single timestep.
        
        Uses DC motor model with quasi-steady-state assumption.
        All 18 parameters are extracted even if some have minimal effect.
        """
        (mass, drag_area, rolling_coeff,
         V_max, R, L, K, b, J,
         gear_ratio, eta,
         brake_T_max, brake_tau, brake_p, brake_kappa, mu,
         r_w, wheel_inertia) = params
        
        # Motor speed from wheel speed
        omega_m = gear_ratio * speed / r_w
        
        # Commanded voltage
        V_cmd = (max(throttle, 0.0) / 100.0) * V_max
        
        # Motor current (quasi-steady, no regen)
        back_emf = K * omega_m
        motor_current = max((V_cmd - back_emf) / R, 0.0)
        
        # Motor torque with viscous friction loss
        motor_torque = K * motor_current - b * omega_m
        motor_torque = max(motor_torque, 0.0)
        
        # Wheel torque through gearbox
        wheel_torque = eta * gear_ratio * motor_torque
        
        # Drive force
        F_drive = wheel_torque / r_w
        
        # Brake force with nonlinear characteristic
        brake_cmd = max(brake, 0.0) / 100.0
        F_brake = brake_T_max * (brake_cmd ** brake_p) / r_w
        
        # Aerodynamic drag
        F_drag = 0.5 * AIR_DENSITY * drag_area * speed * abs(speed)
        
        # Rolling resistance
        cos_grade = np.cos(grade)
        F_roll = rolling_coeff * mass * GRAVITY * cos_grade
        
        # Grade resistance
        sin_grade = np.sin(grade)
        F_grade = mass * GRAVITY * sin_grade
        
        # Net force and acceleration
        # Effective mass includes rotational inertia of wheels and motor
        effective_mass = mass + (4 * wheel_inertia + J * gear_ratio**2) / (r_w**2)
        F_net = F_drive - F_brake - F_drag - F_roll - F_grade
        a = F_net / effective_mass
        
        return a
    
    def _simulate_segment(
        self,
        params: np.ndarray,
        segment: TripSegment,
    ) -> np.ndarray:
        """Simulate a full trip segment and return predicted velocities.
        
        Args:
            params: Parameter vector
            segment: Trip segment with inputs and initial conditions
            
        Returns:
            Simulated velocity trajectory
        """
        n = segment.length
        v_sim = np.zeros(n)
        v_sim[0] = segment.initial_speed
        
        for t in range(n - 1):
            a = self._compute_acceleration(
                params,
                v_sim[t],
                segment.throttle[t],
                segment.brake[t],
                segment.grade[t],
            )
            # Euler integration with speed clamp (no negative speeds)
            v_sim[t + 1] = max(v_sim[t] + a * segment.dt, 0.0)
        
        return v_sim
    
    def _trajectory_loss(
        self,
        params: np.ndarray,
        segments: List[TripSegment],
    ) -> float:
        """Compute velocity MSE over all trajectory segments.
        
        Args:
            params: Parameter vector
            segments: List of trip segments to simulate
            
        Returns:
            Mean squared velocity error (with barrier penalty if enabled)
        """
        total_se = 0.0
        total_samples = 0
        
        for segment in segments:
            v_sim = self._simulate_segment(params, segment)
            se = np.sum((v_sim - segment.speed) ** 2)
            total_se += se
            total_samples += segment.length
        
        mse = total_se / total_samples if total_samples > 0 else 0.0
        
        # Add barrier penalty if enabled
        if self.config.use_barrier and self._bounds is not None:
            barrier_penalty = self._barrier_penalty(params, self._bounds)
            mse += barrier_penalty
        
        self._current_loss = mse
        return mse
    
    def _trajectory_loss_with_numerical_gradient(
        self,
        params: np.ndarray,
        segments: List[TripSegment],
        eps: float = 1e-6,
    ) -> Tuple[float, np.ndarray]:
        """Compute loss and numerical gradient.
        
        Uses central finite differences for gradient approximation.
        """
        loss = self._trajectory_loss(params, segments)
        
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            loss_plus = self._trajectory_loss(params_plus, segments)
            loss_minus = self._trajectory_loss(params_minus, segments)
            
            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        return loss, grad
    
    def _save_checkpoint(
        self,
        log_path: Path,
        params: np.ndarray,
        loss: float,
        epoch: int,
        batch: int,
    ) -> None:
        """Save current best parameters to log file."""
        checkpoint = {
            "epoch": epoch,
            "batch": batch,
            "loss": float(loss),
            "rmse": float(np.sqrt(loss)),
            "params": {name: float(val) for name, val in zip(self.PARAM_NAMES, params)},
        }
        with open(log_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
    
    def _sample_random_params(
        self,
        bounds: List[Tuple[float, float]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample random parameters uniformly within bounds.
        
        Args:
            bounds: List of (min, max) tuples for each parameter
            rng: Random number generator
            
        Returns:
            Random parameter vector
        """
        params = np.array([
            rng.uniform(low=bound[0], high=bound[1]) for bound in bounds
        ])
        return params
    
    def _run_warmup(
        self,
        bounds: List[Tuple[float, float]],
        val_segments: List[TripSegment],
        num_samples: int,
        seed: int,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Run warmup to find best initial parameters.
        
        Randomly samples parameter sets and evaluates on validation set.
        
        Args:
            bounds: Parameter bounds
            val_segments: Validation segments for evaluation
            num_samples: Number of random samples to try
            seed: Random seed
            verbose: Print progress
            
        Returns:
            Tuple of (best_params, best_loss)
        """
        if verbose:
            print(f"\nWarmup: Evaluating {num_samples} random parameter sets on validation set...")
        
        rng = np.random.default_rng(seed)
        best_loss = float('inf')
        best_params = None
        
        warmup_iter = range(num_samples)
        if verbose and tqdm is not None:
            warmup_iter = tqdm(warmup_iter, desc="  Warmup", position=0)
        
        for i in warmup_iter:
            # Sample random parameters
            params = self._sample_random_params(bounds, rng)
            
            # Evaluate on validation set
            val_loss = self._trajectory_loss(params, val_segments)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params.copy()
            
            if verbose and tqdm is not None:
                warmup_iter.set_postfix({
                    "best_RMSE": f"{np.sqrt(best_loss):.3f}",
                    "current_RMSE": f"{np.sqrt(val_loss):.3f}"
                })
        
        if verbose:
            print(f"Warmup complete: best validation RMSE = {np.sqrt(best_loss):.4f} m/s")
            print("Best warmup parameters:")
            for name, val in zip(self.PARAM_NAMES, best_params):
                print(f"  {name}: {val:.4f}")
        
        return best_params, best_loss
    
    def _plot_speed_histograms(
        self,
        train_segments: List[TripSegment],
        val_segments: List[TripSegment],
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot histograms of speed values for train and validation sets."""
        if not MATPLOTLIB_AVAILABLE:
            LOGGER.warning("matplotlib not available, skipping histogram plot")
            return
        
        train_speeds = np.concatenate([s.speed for s in train_segments])
        val_speeds = np.concatenate([s.speed for s in val_segments])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training set histogram
        ax1.hist(train_speeds, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Speed (m/s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Training Set Speed Distribution\n({len(train_segments)} segments, {len(train_speeds):,} samples)')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(train_speeds.mean(), color='red', linestyle='--', label=f'Mean: {train_speeds.mean():.2f} m/s')
        ax1.legend()
        
        # Validation set histogram
        ax2.hist(val_speeds, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Speed (m/s)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Validation Set Speed Distribution\n({len(val_segments)} segments, {len(val_speeds):,} samples)')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(val_speeds.mean(), color='red', linestyle='--', label=f'Mean: {val_speeds.mean():.2f} m/s')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            LOGGER.info(f"Saved speed histograms to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    def _plot_validation_trips(
        self,
        params: np.ndarray,
        val_segments: List[TripSegment],
        save_path: Path,
        max_trips: int = 5,
    ) -> None:
        """Plot GT vs simulated speed for validation trips."""
        if not MATPLOTLIB_AVAILABLE:
            LOGGER.warning("matplotlib not available, skipping validation plot")
            return
        
        n_trips = min(max_trips, len(val_segments))
        selected_segments = val_segments[:n_trips]
        
        fig, axes = plt.subplots(n_trips, 1, figsize=(12, 3 * n_trips), sharex=True)
        if n_trips == 1:
            axes = [axes]
        
        for idx, segment in enumerate(selected_segments):
            ax = axes[idx]
            v_sim = self._simulate_segment(params, segment)
            time = np.arange(segment.length) * segment.dt
            
            ax.plot(time, segment.speed, 'b-', label='GT Speed', alpha=0.8, linewidth=2)
            ax.plot(time, v_sim, 'r--', label='Simulated', alpha=0.8, linewidth=2)
            ax.set_ylabel('Velocity (m/s)')
            ax.set_title(f'Validation Trip: {segment.trip_id}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add RMSE text
            rmse = np.sqrt(np.mean((v_sim - segment.speed) ** 2))
            ax.text(0.02, 0.98, f'RMSE: {rmse:.3f} m/s', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        LOGGER.info(f"Saved validation trip comparison to {save_path}")
        plt.close(fig)
    
    def fit(
        self,
        data_path: Path,
        verbose: bool = True,
        log_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[np.ndarray, float], None]] = None,
    ) -> FittedVehicleParams:
        """Fit vehicle parameters by minimizing trajectory velocity error.

        Args:
            data_path: Path to trip data (.pt file)
            verbose: Print progress
            log_path: Path to save best-so-far parameters (updated on each improvement)
            progress_callback: Optional callback called with (best_params, best_loss) when new best is found

        Returns:
            Fitted vehicle parameters
        """
        cfg = self.config
        
        # Default log path next to data
        if log_path is None:
            log_path = data_path.parent / "fitting_checkpoint.json"
        
        # Load and process data
        if verbose:
            print(f"Loading data from {data_path}...")
        
        trips = self.load_trip_data(data_path)
        
        # Estimate dt from data
        dt = self._estimate_dt(trips)
        if verbose:
            print(f"Estimated dt: {dt:.4f} s")
        
        # Create segments
        all_segments = self._create_segments(trips, dt)
        
        if len(all_segments) == 0:
            raise ValueError("No valid segments found in data")
        
        # Downsample if requested
        if cfg.downsampling_factor > 1:
            if verbose:
                print(f"Downsampling data by factor {cfg.downsampling_factor}...")
            all_segments = self._downsample_segments(all_segments, cfg.downsampling_factor)
            # Update dt to reflect downsampling
            dt = dt * cfg.downsampling_factor
            if verbose:
                print(f"Updated dt after downsampling: {dt:.4f} s")
        
        # Filter zero-speed segments to limit their fraction
        if cfg.max_zero_speed_fraction < 1.0:
            if verbose:
                print(f"Filtering zero-speed segments (max fraction: {cfg.max_zero_speed_fraction:.1%}, eps: {cfg.zero_speed_eps:.3f} m/s)...")
            all_segments = self._filter_zero_speed_segments(
                all_segments,
                max_fraction=cfg.max_zero_speed_fraction,
                eps=cfg.zero_speed_eps,
            )
        
        # Split into train/validation
        rng = np.random.default_rng(42)
        n_total = len(all_segments)
        n_val = max(1, int(n_total * cfg.validation_fraction))
        n_train = n_total - n_val
        
        perm = rng.permutation(n_total)
        train_segments = [all_segments[i] for i in perm[:n_train]]
        val_segments = [all_segments[i] for i in perm[n_train:]]
        self._segments = all_segments
        
        train_samples = sum(s.length for s in train_segments)
        val_samples = sum(s.length for s in val_segments)
        
        if verbose:
            print(f"Created {n_total} segments: {n_train} train, {n_val} validation")
            print(f"Train samples: {train_samples:,}, Validation samples: {val_samples:,}")
            speeds = np.concatenate([s.speed for s in all_segments])
            print(f"Speed range: [{speeds.min():.2f}, {speeds.max():.2f}] m/s")
        
        # Plot speed histograms for train/val split
        if verbose:
            hist_path = data_path.parent / "speed_histograms.png"
            self._plot_speed_histograms(train_segments, val_segments, save_path=hist_path)
        
        # Initial parameters (all 18)
        x0 = np.array([
            cfg.mass_init,
            cfg.drag_area_init,
            cfg.rolling_coeff_init,
            cfg.motor_V_max_init,
            cfg.motor_R_init,
            cfg.motor_L_init,
            cfg.motor_K_init,
            cfg.motor_b_init,
            cfg.motor_J_init,
            cfg.gear_ratio_init,
            cfg.eta_gb_init,
            cfg.brake_T_max_init,
            cfg.brake_tau_init,
            cfg.brake_p_init,
            cfg.brake_kappa_init,
            cfg.mu_init,
            cfg.wheel_radius_init,
            cfg.wheel_inertia_init,
        ])
        
        # Bounds (all 18)
        bounds = [
            cfg.mass_bounds,
            cfg.drag_area_bounds,
            cfg.rolling_coeff_bounds,
            cfg.motor_V_max_bounds,
            cfg.motor_R_bounds,
            cfg.motor_L_bounds,
            cfg.motor_K_bounds,
            cfg.motor_b_bounds,
            cfg.motor_J_bounds,
            cfg.gear_ratio_bounds,
            cfg.eta_gb_bounds,
            cfg.brake_T_max_bounds,
            cfg.brake_tau_bounds,
            cfg.brake_p_bounds,
            cfg.brake_kappa_bounds,
            cfg.mu_bounds,
            cfg.wheel_radius_bounds,
            cfg.wheel_inertia_bounds,
        ]
        
        # Store bounds for barrier function computation
        self._bounds = bounds
        
        # Warmup: find better initial guess by random sampling
        if cfg.use_warmup:
            warmup_params, warmup_loss = self._run_warmup(
                bounds=bounds,
                val_segments=val_segments,
                num_samples=cfg.warmup_samples,
                seed=cfg.warmup_seed,
                verbose=verbose,
            )
            # Use warmup result as initial guess
            x0 = warmup_params
            if verbose:
                print(f"Using warmup parameters as initial guess (val_RMSE={np.sqrt(warmup_loss):.4f} m/s)")
        
        # Batch setup (training segments only)
        batch_size = min(cfg.segments_per_batch, n_train)
        num_batches = (n_train + batch_size - 1) // batch_size
        
        if verbose:
            print(f"\nOptimization: {cfg.num_epochs} epochs, {num_batches} batches/epoch")
            print(f"Parameters: {self.PARAM_NAMES}")
            print(f"Checkpoint log: {log_path}")
            print("-" * 70)
        
        best_val_loss = float('inf')
        best_params = x0.copy()
        
        # Progress tracking
        epoch_iter = range(1, cfg.num_epochs + 1)
        if verbose and tqdm is not None:
            epoch_iter = tqdm(epoch_iter, desc="Epochs", position=0)
        
        for epoch in epoch_iter:
            # Shuffle training segments for this epoch
            if cfg.shuffle_segments:
                train_indices = rng.permutation(n_train)
            else:
                train_indices = np.arange(n_train)
            
            epoch_train_losses = []
            
            # Batch iteration
            batch_iter = range(num_batches)
            if verbose and tqdm is not None:
                batch_iter = tqdm(batch_iter, desc=f"  Epoch {epoch}", leave=False, position=1)
            
            for b in batch_iter:
                i_start = b * batch_size
                i_end = min((b + 1) * batch_size, n_train)
                batch_indices = train_indices[i_start:i_end]
                batch_segments = [train_segments[i] for i in batch_indices]
                
                # Optimize on this batch
                result = minimize(
                    self._trajectory_loss_with_numerical_gradient,
                    x0,
                    args=(batch_segments,),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    options={"maxiter": cfg.max_iter, "ftol": cfg.tolerance, "disp": False},
                )
                
                train_loss = result.fun
                epoch_train_losses.append(train_loss)
                
                # Evaluate on validation set
                val_loss = self._trajectory_loss(result.x, val_segments)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = result.x.copy()
                    # Save checkpoint
                    self._save_checkpoint(log_path, best_params, best_val_loss, epoch, b)
                    if verbose:
                        print(f"\n  New best: val_RMSE={np.sqrt(best_val_loss):.4f} m/s (saved)")
                        # Plot validation trips comparison
                        val_plot_path = data_path.parent / f"validation_trips_epoch{epoch}_batch{b}.png"
                        self._plot_validation_trips(best_params, val_segments, val_plot_path, max_trips=5)

                    # Call progress callback if provided
                    if progress_callback is not None:
                        progress_callback(best_params, best_val_loss)
                
                # Warm start next batch
                x0 = result.x
                
                if verbose and tqdm is not None:
                    batch_iter.set_postfix({
                        "train": f"{np.sqrt(train_loss):.3f}",
                        "val": f"{np.sqrt(val_loss):.3f}",
                        "best_val": f"{np.sqrt(best_val_loss):.3f}"
                    })
            
            # Epoch summary
            epoch_mean_train = np.mean(epoch_train_losses)
            epoch_val_loss = self._trajectory_loss(x0, val_segments)
            if verbose:
                if tqdm is not None:
                    epoch_iter.set_postfix({
                        "train": f"{np.sqrt(epoch_mean_train):.3f}",
                        "val": f"{np.sqrt(epoch_val_loss):.3f}",
                        "best_val": f"{np.sqrt(best_val_loss):.3f}"
                    })
                else:
                    print(f"Epoch {epoch}: train={np.sqrt(epoch_mean_train):.4f}, val={np.sqrt(epoch_val_loss):.4f}, best_val={np.sqrt(best_val_loss):.4f}")
        
        # Final evaluation on ALL segments
        if verbose:
            print("\nFinal evaluation...")
        
        train_loss = self._trajectory_loss(best_params, train_segments)
        val_loss = self._trajectory_loss(best_params, val_segments)
        all_loss = self._trajectory_loss(best_params, all_segments)
        
        # Compute R² for velocity on all segments
        all_v_measured = np.concatenate([s.speed for s in all_segments])
        all_v_simulated = np.concatenate([
            self._simulate_segment(best_params, s) for s in all_segments
        ])
        
        ss_res = np.sum((all_v_measured - all_v_simulated) ** 2)
        ss_tot = np.sum((all_v_measured - np.mean(all_v_measured)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        if verbose:
            print("-" * 70)
            print(f"Train RMSE: {np.sqrt(train_loss):.4f} m/s")
            print(f"Val RMSE:   {np.sqrt(val_loss):.4f} m/s")
            print(f"All RMSE:   {np.sqrt(all_loss):.4f} m/s")
            print(f"Velocity R²: {r_squared:.4f}")
            print("\nFitted parameters:")
            for name, val in zip(self.PARAM_NAMES, best_params):
                print(f"  {name}: {val:.4f}")
        
        total_samples = train_samples + val_samples
        
        # Create result (all 18 params from optimization)
        fitted = FittedVehicleParams(
            mass=best_params[0],
            drag_area=best_params[1],
            rolling_coeff=best_params[2],
            motor_V_max=best_params[3],
            motor_R=best_params[4],
            motor_L=best_params[5],
            motor_K=best_params[6],
            motor_b=best_params[7],
            motor_J=best_params[8],
            gear_ratio=best_params[9],
            eta_gb=best_params[10],
            brake_T_max=best_params[11],
            brake_tau=best_params[12],
            brake_p=best_params[13],
            brake_kappa=best_params[14],
            mu=best_params[15],
            wheel_radius=best_params[16],
            wheel_inertia=best_params[17],
            fit_loss=val_loss,
            num_samples=total_samples,
            r_squared=r_squared,
        )
        
        return fitted
    
    def evaluate_fit(
        self,
        params: FittedVehicleParams,
        data_path: Optional[Path] = None,
        segments: Optional[List[TripSegment]] = None,
    ) -> Dict:
        """Evaluate fit quality on data.
        
        Args:
            params: Fitted parameters to evaluate
            data_path: Path to data (if segments not provided)
            segments: Pre-loaded segments (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if segments is None:
            if data_path is None:
                if self._segments:
                    segments = self._segments
                else:
                    raise ValueError("Must provide data_path or segments")
            else:
                trips = self.load_trip_data(data_path)
                dt = self._estimate_dt(trips)
                segments = self._create_segments(trips, dt)
        
        # Convert params to array (all 18)
        param_array = np.array([
            params.mass,
            params.drag_area,
            params.rolling_coeff,
            params.motor_V_max,
            params.motor_R,
            params.motor_L,
            params.motor_K,
            params.motor_b,
            params.motor_J,
            params.gear_ratio,
            params.eta_gb,
            params.brake_T_max,
            params.brake_tau,
            params.brake_p,
            params.brake_kappa,
            params.mu,
            params.wheel_radius,
            params.wheel_inertia,
        ])
        
        # Simulate all segments
        all_v_measured = []
        all_v_simulated = []
        segment_errors = []
        
        for segment in segments:
            v_sim = self._simulate_segment(param_array, segment)
            all_v_measured.append(segment.speed)
            all_v_simulated.append(v_sim)
            
            # Per-segment metrics
            seg_mse = np.mean((v_sim - segment.speed) ** 2)
            seg_rmse = np.sqrt(seg_mse)
            segment_errors.append({
                "trip_id": segment.trip_id,
                "length": segment.length,
                "mse": seg_mse,
                "rmse": seg_rmse,
                "max_error": np.max(np.abs(v_sim - segment.speed)),
            })
        
        all_v_measured = np.concatenate(all_v_measured)
        all_v_simulated = np.concatenate(all_v_simulated)
        
        # Global metrics
        mse = np.mean((all_v_measured - all_v_simulated) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_v_measured - all_v_simulated))
        
        ss_res = np.sum((all_v_measured - all_v_simulated) ** 2)
        ss_tot = np.sum((all_v_measured - np.mean(all_v_measured)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r_squared": r_squared,
            "num_segments": len(segments),
            "total_samples": len(all_v_measured),
            "segment_errors": segment_errors,
        }
    
    def plot_segment_comparison(
        self,
        params: FittedVehicleParams,
        segment_idx: int = 0,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot measured vs simulated velocity for a segment.
        
        Args:
            params: Fitted parameters
            segment_idx: Index of segment to plot
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            LOGGER.warning("matplotlib not available for plotting")
            return
        
        if not self._segments:
            raise ValueError("No segments loaded. Call fit() first.")
        
        segment = self._segments[segment_idx]
        
        param_array = np.array([
            params.mass, params.drag_area, params.rolling_coeff,
            params.motor_V_max, params.motor_R, params.motor_L,
            params.motor_K, params.motor_b, params.motor_J,
            params.gear_ratio, params.eta_gb,
            params.brake_T_max, params.brake_tau, params.brake_p,
            params.brake_kappa, params.mu,
            params.wheel_radius, params.wheel_inertia,
        ])
        
        v_sim = self._simulate_segment(param_array, segment)
        time = np.arange(segment.length) * segment.dt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # Velocity comparison
        ax = axes[0]
        ax.plot(time, segment.speed, 'b-', label='Measured', alpha=0.8)
        ax.plot(time, v_sim, 'r--', label='Simulated', alpha=0.8)
        ax.set_ylabel('Velocity (m/s)')
        ax.legend()
        ax.set_title(f'Segment: {segment.trip_id}')
        ax.grid(True, alpha=0.3)
        
        # Velocity error
        ax = axes[1]
        error = v_sim - segment.speed
        ax.plot(time, error, 'g-', alpha=0.8)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel('Velocity Error (m/s)')
        ax.grid(True, alpha=0.3)
        
        # Inputs
        ax = axes[2]
        ax.plot(time, segment.throttle, 'b-', label='Throttle', alpha=0.7)
        ax.plot(time, segment.brake, 'r-', label='Brake', alpha=0.7)
        ax.plot(time, np.degrees(segment.grade), 'g-', label='Grade (°)', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Input')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            LOGGER.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
