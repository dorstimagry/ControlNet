"""Centered randomization configuration from fitted vehicle parameters.

This module provides utilities to create parameter randomization ranges
centered around fitted vehicle parameters, enabling RL training on
distributions close to a target vehicle.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from fitting.vehicle_fitter import FittedVehicleParams

LOGGER = logging.getLogger(__name__)


def _make_range(
    mean: float,
    spread_pct: float,
    bounds: Optional[Tuple[float, float]] = None,
    enforce_positivity: bool = False,
) -> Tuple[float, float]:
    """Create a range centered on mean with given spread percentage.
    
    Args:
        mean: Center value
        spread_pct: Spread as fraction (0.1 = ±10%)
        bounds: Optional (min, max) to clamp the range
        enforce_positivity: If True, only enforce positivity (low >= 0), ignore other bounds
        
    Returns:
        Tuple of (low, high)
    """
    low = mean * (1.0 - spread_pct)
    high = mean * (1.0 + spread_pct)
    
    if enforce_positivity:
        # Only enforce positivity constraint
        low = max(low, 0.0)
        # high can be anything positive
    elif bounds is not None:
        low = max(low, bounds[0])
        high = min(high, bounds[1])
    
    # Ensure low < high
    if low >= high:
        if enforce_positivity:
            high = max(low + 1e-6, mean * 1.01)  # Ensure positive and reasonable
        elif bounds is not None:
            low, high = bounds
        else:
            high = low + 1e-6
    
    return (low, high)


def _make_log_range(
    mean: float,
    spread_pct: float,
    bounds: Optional[Tuple[float, float]] = None,
    enforce_positivity: bool = False,
) -> Tuple[float, float]:
    """Create a range for log-uniform sampling centered on mean.
    
    For parameters sampled with log-uniform distribution, we want the
    geometric mean to be the fitted value.
    
    Args:
        mean: Center value (geometric mean)
        spread_pct: Spread as fraction in log space
        bounds: Optional (min, max) to clamp the range
        enforce_positivity: If True, only enforce positivity (low > 0), ignore other bounds
        
    Returns:
        Tuple of (low, high)
    """
    # Multiplicative factor: e.g., 10% spread -> 1.2x factor
    factor = 1.0 + spread_pct * 2
    
    low = mean / factor
    high = mean * factor
    
    if enforce_positivity:
        # Only enforce positivity constraint (must be > 0 for log-uniform)
        low = max(low, 1e-10)  # Small positive value
        # high can be anything positive
    elif bounds is not None:
        low = max(low, bounds[0])
        high = min(high, bounds[1])
    
    if low >= high:
        if enforce_positivity:
            high = max(low * 1.01, mean * 1.01)  # Ensure positive and reasonable
        elif bounds is not None:
            low, high = bounds
        else:
            high = low * 1.01
    
    return (low, high)


@dataclass
class CenteredRandomizationConfig:
    """Configuration for creating centered randomization from fitted params.
    
    This stores the parameters needed to build an ExtendedPlantRandomization
    instance centered on fitted vehicle parameters.
    
    All parameters match the actual ExtendedPlant model used in RL training.
    """
    
    # === BODY PARAMETERS (fitted) ===
    mass: float
    drag_area: float
    rolling_coeff: float
    
    # === MOTOR PARAMETERS (fitted) ===
    motor_V_max: float  # V - maximum motor voltage
    motor_R: float  # Ω - armature resistance
    motor_K: float  # Nm/A = V·s/rad - K_t = K_e
    gear_ratio: float  # N - gear reduction ratio
    
    # === BRAKE PARAMETERS (fitted) ===
    brake_T_max: float  # Nm - maximum brake torque at wheel
    
    # === WHEEL PARAMETERS ===
    wheel_radius: float  # m - wheel radius
    
    # === FIXED PARAMETERS (from fitted or defaults) ===
    motor_L: float = 1e-3  # H - armature inductance
    motor_b: float = 1e-3  # Nm·s/rad - viscous friction
    motor_J: float = 1e-3  # kg·m² - rotor inertia
    eta_gb: float = 0.9  # gearbox efficiency
    brake_tau: float = 0.08  # s - brake time constant
    brake_p: float = 1.2  # brake exponent
    brake_kappa: float = 0.08  # brake slip constant
    mu: float = 0.9  # tire friction coefficient
    wheel_inertia: float = 1.5  # kg·m² - wheel + rotating assembly
    
    # === SPREAD CONFIGURATION ===
    spread_pct: float = 0.1  # ±10% default
    
    # Use different spreads for different parameter types (optional)
    mass_spread_pct: Optional[float] = None
    drag_spread_pct: Optional[float] = None
    rolling_spread_pct: Optional[float] = None
    motor_spread_pct: Optional[float] = None
    brake_spread_pct: Optional[float] = None
    wheel_spread_pct: Optional[float] = None
    
    @classmethod
    def from_fitted_params(
        cls,
        fitted: FittedVehicleParams,
        spread_pct: float = 0.1,
        **overrides,
    ) -> "CenteredRandomizationConfig":
        """Create config from fitted vehicle parameters.
        
        Args:
            fitted: Fitted vehicle parameters
            spread_pct: Default spread percentage for all parameters
            **overrides: Override any field (e.g., mass_spread_pct=0.05)
            
        Returns:
            CenteredRandomizationConfig instance
        """
        return cls(
            # Body params
            mass=fitted.mass,
            drag_area=fitted.drag_area,
            rolling_coeff=fitted.rolling_coeff,
            # Motor params
            motor_V_max=fitted.motor_V_max,
            motor_R=fitted.motor_R,
            motor_K=fitted.motor_K,
            gear_ratio=fitted.gear_ratio,
            # Brake params
            brake_T_max=fitted.brake_T_max,
            # Wheel params
            wheel_radius=fitted.wheel_radius,
            # Fixed params from fitted
            motor_L=fitted.motor_L,
            motor_b=fitted.motor_b,
            motor_J=fitted.motor_J,
            eta_gb=fitted.eta_gb,
            brake_tau=fitted.brake_tau,
            brake_p=fitted.brake_p,
            brake_kappa=fitted.brake_kappa,
            mu=fitted.mu,
            wheel_inertia=fitted.wheel_inertia,
            # Spread
            spread_pct=spread_pct,
            **overrides,
        )
    
    def _get_spread(self, param_name: str) -> float:
        """Get spread for a specific parameter."""
        specific = getattr(self, f"{param_name}_spread_pct", None)
        return specific if specific is not None else self.spread_pct
    
    def to_extended_randomization_dict(self) -> Dict:
        """Convert to a dictionary suitable for ExtendedPlantRandomization.from_config().
        
        When used with fitted parameters, this creates ranges centered on fitted values
        with only positivity constraints (no hard bounds) and permissive feasibility
        thresholds, since the target speed profile generator handles feasibility.
        
        Returns:
            Dictionary with vehicle_randomization key for config loading
        """
        # Build ranges for each parameter, centered on fitted values
        # Use enforce_positivity=True to only enforce positivity, not hard bounds
        
        # === BODY PARAMETERS ===
        mass_range = _make_range(
            self.mass, 
            self._get_spread("mass"),
            enforce_positivity=True,  # Only enforce mass > 0
        )
        
        drag_area_range = _make_range(
            self.drag_area,
            self._get_spread("drag"),
            enforce_positivity=True,  # Only enforce CdA > 0
        )
        
        rolling_coeff_range = _make_range(
            self.rolling_coeff,
            self._get_spread("rolling"),
            enforce_positivity=True,  # Only enforce C_rr > 0
        )
        
        # === MOTOR PARAMETERS ===
        motor_spread = self._get_spread("motor")
        
        motor_Vmax_range = _make_range(
            self.motor_V_max,
            motor_spread,
            enforce_positivity=True,  # Only enforce V_max > 0
        )
        
        motor_R_range = _make_log_range(
            self.motor_R,
            motor_spread,
            enforce_positivity=True,  # Only enforce R > 0
        )
        
        motor_L_range = _make_log_range(
            self.motor_L,
            motor_spread,
            enforce_positivity=True,  # Only enforce L > 0
        )
        
        motor_K_range = _make_log_range(
            self.motor_K,
            motor_spread,
            enforce_positivity=True,  # Only enforce K > 0
        )
        
        motor_b_range = _make_log_range(
            self.motor_b,
            motor_spread,
            enforce_positivity=True,  # Only enforce b > 0
        )
        
        motor_J_range = _make_log_range(
            self.motor_J,
            motor_spread,
            enforce_positivity=True,  # Only enforce J > 0
        )
        
        gear_ratio_range = _make_range(
            self.gear_ratio,
            motor_spread,
            enforce_positivity=True,  # Only enforce gear_ratio > 0
        )
        
        eta_gb_range = _make_range(
            self.eta_gb,
            motor_spread * 0.3,  # Efficiency shouldn't vary much
            enforce_positivity=True,  # Only enforce eta_gb > 0
        )
        
        # === BRAKE PARAMETERS ===
        brake_spread = self._get_spread("brake")
        
        brake_tau_range = _make_range(
            self.brake_tau,
            brake_spread,
            enforce_positivity=True,  # Only enforce tau > 0
        )
        
        brake_Tmax_range = _make_range(
            self.brake_T_max,
            brake_spread,
            enforce_positivity=True,  # Only enforce T_max > 0
        )
        
        # Compute brake acceleration range from torque range
        # a_brake = T_brake / (r_w * mass)
        brake_accel_low = brake_Tmax_range[0] / (self.wheel_radius * self.mass)
        brake_accel_high = brake_Tmax_range[1] / (self.wheel_radius * self.mass)
        brake_accel_range = (brake_accel_low, brake_accel_high)
        
        brake_p_range = _make_range(
            self.brake_p,
            brake_spread * 0.5,
            enforce_positivity=True,  # Only enforce p > 0
        )
        
        brake_kappa_range = _make_log_range(
            self.brake_kappa,
            brake_spread,
            enforce_positivity=True,  # Only enforce kappa > 0
        )
        
        mu_range = _make_range(
            self.mu,
            brake_spread * 0.5,
            enforce_positivity=True,  # Only enforce mu > 0
        )
        
        # === WHEEL PARAMETERS ===
        wheel_spread = self._get_spread("wheel") * 0.5  # Tighter for wheel
        
        wheel_radius_range = _make_range(
            self.wheel_radius,
            wheel_spread,
            enforce_positivity=True,  # Only enforce radius > 0
        )
        
        wheel_inertia_range = _make_log_range(
            self.wheel_inertia,
            self._get_spread("wheel"),
            enforce_positivity=True,  # Only enforce inertia > 0
        )
        
        return {
            "vehicle_randomization": {
                # Body
                "mass_range": list(mass_range),
                "drag_area_range": list(drag_area_range),
                "rolling_coeff_range": list(rolling_coeff_range),
                # Motor
                "motor_Vmax_range": list(motor_Vmax_range),
                "motor_R_range": list(motor_R_range),
                "motor_L_range": list(motor_L_range),
                "motor_K_range": list(motor_K_range),
                "motor_b_range": list(motor_b_range),
                "motor_J_range": list(motor_J_range),
                "gear_ratio_range": list(gear_ratio_range),
                "eta_gb_range": list(eta_gb_range),
                # Brake
                "brake_tau_range": list(brake_tau_range),
                "brake_accel_range": list(brake_accel_range),
                "brake_p_range": list(brake_p_range),
                "brake_kappa_range": list(brake_kappa_range),
                "mu_range": list(mu_range),
                # Wheel
                "wheel_radius_range": list(wheel_radius_range),
                "wheel_inertia_range": list(wheel_inertia_range),
                # Grade is environmental, not vehicle-specific
                "grade_range_deg": [-5.7, 5.7],
                # Actuator tau (use default spread)
                "actuator_tau_range": [0.05, 0.30],
                # Feasibility thresholds - set very permissive since profile generator handles feasibility
                "min_accel_from_rest": 0.1,  # Very permissive (was 2.5)
                "min_brake_decel": 0.1,  # Very permissive (was 4.0)
                "min_top_speed": 0.1,  # Very permissive (was 20.0)
                # Skip feasibility and sanity checks when using fitted params (profile generator handles feasibility)
                "skip_feasibility_checks": True,
                "skip_sanity_checks": True,
            }
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            # Body params
            "mass": self.mass,
            "drag_area": self.drag_area,
            "rolling_coeff": self.rolling_coeff,
            # Motor params
            "motor_V_max": self.motor_V_max,
            "motor_R": self.motor_R,
            "motor_K": self.motor_K,
            "gear_ratio": self.gear_ratio,
            # Brake params
            "brake_T_max": self.brake_T_max,
            # Wheel params
            "wheel_radius": self.wheel_radius,
            # Fixed params
            "motor_L": self.motor_L,
            "motor_b": self.motor_b,
            "motor_J": self.motor_J,
            "eta_gb": self.eta_gb,
            "brake_tau": self.brake_tau,
            "brake_p": self.brake_p,
            "brake_kappa": self.brake_kappa,
            "mu": self.mu,
            "wheel_inertia": self.wheel_inertia,
            # Spread
            "spread_pct": self.spread_pct,
            "mass_spread_pct": self.mass_spread_pct,
            "drag_spread_pct": self.drag_spread_pct,
            "rolling_spread_pct": self.rolling_spread_pct,
            "motor_spread_pct": self.motor_spread_pct,
            "brake_spread_pct": self.brake_spread_pct,
            "wheel_spread_pct": self.wheel_spread_pct,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "CenteredRandomizationConfig":
        """Create from dictionary."""
        return cls(**d)
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved centered randomization config to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "CenteredRandomizationConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_extended_randomization_from_fitted(
    fitted_params_path: Path,
    spread_pct: float = 0.1,
):
    """Create ExtendedPlantRandomization from fitted parameters file.
    
    This is a convenience function for use in training scripts.
    
    Args:
        fitted_params_path: Path to fitted_params.json
        spread_pct: Spread percentage around fitted means
        
    Returns:
        ExtendedPlantRandomization instance
    """
    from utils.dynamics import ExtendedPlantRandomization
    
    fitted = FittedVehicleParams.load(fitted_params_path)
    config = CenteredRandomizationConfig.from_fitted_params(fitted, spread_pct)
    rand_dict = config.to_extended_randomization_dict()
    
    return ExtendedPlantRandomization.from_config(rand_dict)


__all__ = [
    "CenteredRandomizationConfig",
    "create_extended_randomization_from_fitted",
]
