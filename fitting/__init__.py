"""Vehicle parameter fitting module.

This module provides tools for fitting vehicle dynamics parameters from
real trip data, enabling targeted RL training on distributions close to
specific vehicles.
"""

from fitting.vehicle_fitter import (
    FittedVehicleParams,
    VehicleParamFitter,
    FitterConfig,
)
from fitting.randomization_config import (
    CenteredRandomizationConfig,
    create_extended_randomization_from_fitted,
)

__all__ = [
    "FittedVehicleParams",
    "VehicleParamFitter",
    "FitterConfig",
    "CenteredRandomizationConfig",
    "create_extended_randomization_from_fitted",
]

