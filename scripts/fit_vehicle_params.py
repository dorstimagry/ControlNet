#!/usr/bin/env python3
"""Fit vehicle dynamics parameters from trip data using trajectory simulation.

ALL 18 vehicle parameters are fitted by default. Use same min/max bounds to fix
a parameter to a specific value.

Usage:
    python scripts/fit_vehicle_params.py \
        --data data/processed/NiroEV/NIROEV_HA_02/all_trips_data.pt

    # Fix a parameter (use same value for min and max):
    python scripts/fit_vehicle_params.py \
        --data data/processed/NiroEV/NIROEV_HA_02/all_trips_data.pt \
        --mass-bounds 1885 1885  # Fix mass to 1885 kg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fitting import VehicleParamFitter, FitterConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit ALL vehicle dynamics parameters from trip data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data", type=Path, required=True,
        help="Path to all_trips_data.pt file",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for fitted_params.json (default: same dir as data)",
    )
    
    # Optimization settings
    opt = parser.add_argument_group("Optimization")
    opt.add_argument("--epochs", type=int, default=1, help="Training epochs (default: 5)")
    opt.add_argument("--batch-size", type=int, default=10, help="Segments per batch (default: 10)")
    opt.add_argument("--max-iter", type=int, default=30, help="Optimizer iterations per batch (default: 30)")
    opt.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction (default: 0.1)")
    opt.add_argument("--downsample", type=int, default=1, 
                     help="Downsample data by taking every Nth sample (default: 1 = no downsampling)")
    opt.add_argument("--warmup", action="store_true",
                     help="Enable warmup: randomly sample parameters and use best as initial guess")
    opt.add_argument("--warmup-samples", type=int, default=10,
                     help="Number of random parameter sets to try in warmup (default: 10)")
    opt.add_argument("--warmup-seed", type=int, default=42,
                     help="Random seed for warmup sampling (default: 42)")
    
    # Data filtering
    filter_group = parser.add_argument_group("Data Filtering")
    filter_group.add_argument("--min-speed", type=float, default=0.0,
                             help="Minimum speed to include in m/s (default: 0.5)")
    filter_group.add_argument("--max-speed", type=float, default=20.0,
                             help="Maximum speed to include in m/s (default: 200.0)")
    filter_group.add_argument("--max-accel", type=float, default=4.0,
                             help="Maximum acceleration magnitude in m/s² (default: 6.0)")
    filter_group.add_argument("--max-zero-speed-fraction", type=float, default=0.05,
                             help="Maximum fraction of segments with zero/near-zero speed (default: 0.05 = 5%%)")
    filter_group.add_argument("--zero-speed-eps", type=float, default=0.1,
                             help="Epsilon threshold for zero speed in m/s (default: 0.1)")
    
    # ALL parameter bounds - use same min/max to fix a parameter
    bounds = parser.add_argument_group("Parameter Bounds (use same min/max to fix)")
    
    # Body parameters
    bounds.add_argument("--mass-bounds", type=float, nargs=2, default=[1800, 2300],
                        metavar=("MIN", "MAX"), help="Vehicle mass in kg")
    bounds.add_argument("--drag-area-bounds", type=float, nargs=2, default=[0.4, 1.2],
                        metavar=("MIN", "MAX"), help="Drag area CdA in m²")
    bounds.add_argument("--rolling-coeff-bounds", type=float, nargs=2, default=[0.006, 0.020],
                        metavar=("MIN", "MAX"), help="Rolling resistance coefficient")
    
    # Motor parameters
    bounds.add_argument("--vmax-bounds", type=float, nargs=2, default=[320, 400],
                        metavar=("MIN", "MAX"), help="Motor max voltage in V")
    bounds.add_argument("--motor-r-bounds", type=float, nargs=2, default=[0.02, 0.5],
                        metavar=("MIN", "MAX"), help="Motor resistance in Ω")
    bounds.add_argument("--motor-l-bounds", type=float, nargs=2, default=[1e-5, 0.1],
                        metavar=("MIN", "MAX"), help="Motor inductance in H")
    bounds.add_argument("--motor-k-bounds", type=float, nargs=2, default=[0.05, 0.5],
                        metavar=("MIN", "MAX"), help="Motor torque constant in Nm/A")
    bounds.add_argument("--motor-b-bounds", type=float, nargs=2, default=[1e-6, 0.1],
                        metavar=("MIN", "MAX"), help="Motor viscous friction in Nm·s/rad")
    bounds.add_argument("--motor-j-bounds", type=float, nargs=2, default=[1e-2, 0.1],
                        metavar=("MIN", "MAX"), help="Motor rotor inertia in kg·m²")
    
    # Drivetrain parameters
    bounds.add_argument("--gear-bounds", type=float, nargs=2, default=[4.3, 11.0],
                        metavar=("MIN", "MAX"), help="Gear ratio")
    bounds.add_argument("--eta-gb-bounds", type=float, nargs=2, default=[0.85, 0.99],
                        metavar=("MIN", "MAX"), help="Gearbox efficiency")
    
    # Brake parameters
    bounds.add_argument("--brake-tmax-bounds", type=float, nargs=2, default=[10000, 20000],
                        metavar=("MIN", "MAX"), help="Max brake torque in Nm")
    bounds.add_argument("--brake-tau-bounds", type=float, nargs=2, default=[0.01, 0.5],
                        metavar=("MIN", "MAX"), help="Brake time constant in s")
    bounds.add_argument("--brake-p-bounds", type=float, nargs=2, default=[0.5, 3.0],
                        metavar=("MIN", "MAX"), help="Brake exponent")
    bounds.add_argument("--brake-kappa-bounds", type=float, nargs=2, default=[0.01, 0.3],
                        metavar=("MIN", "MAX"), help="Brake slip constant")
    bounds.add_argument("--mu-bounds", type=float, nargs=2, default=[0.5, 1.2],
                        metavar=("MIN", "MAX"), help="Tire friction coefficient")
    
    # Wheel parameters
    bounds.add_argument("--wheel-radius-bounds", type=float, nargs=2, default=[0.315, 0.34],
                        metavar=("MIN", "MAX"), help="Wheel radius in m")
    bounds.add_argument("--wheel-inertia-bounds", type=float, nargs=2, default=[1.0, 2.0],
                        metavar=("MIN", "MAX"), help="Wheel inertia in kg·m²")
    
    # Initial guesses (optional)
    init = parser.add_argument_group("Initial Guesses (optional)")
    init.add_argument("--mass-init", type=float, default=None)
    init.add_argument("--drag-area-init", type=float, default=None)
    init.add_argument("--rolling-coeff-init", type=float, default=None)
    init.add_argument("--vmax-init", type=float, default=None)
    init.add_argument("--motor-r-init", type=float, default=None)
    init.add_argument("--motor-l-init", type=float, default=None)
    init.add_argument("--motor-k-init", type=float, default=None)
    init.add_argument("--motor-b-init", type=float, default=None)
    init.add_argument("--motor-j-init", type=float, default=None)
    init.add_argument("--gear-init", type=float, default=None)
    init.add_argument("--eta-gb-init", type=float, default=None)
    init.add_argument("--brake-tmax-init", type=float, default=None)
    init.add_argument("--brake-tau-init", type=float, default=None)
    init.add_argument("--brake-p-init", type=float, default=None)
    init.add_argument("--brake-kappa-init", type=float, default=None)
    init.add_argument("--mu-init", type=float, default=None)
    init.add_argument("--wheel-radius-init", type=float, default=None)
    init.add_argument("--wheel-inertia-init", type=float, default=None)
    
    parser.add_argument("-v", "--verbose", action="store_true")
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        return 1
    
    if args.output is None:
        args.output = args.data.parent / "fitted_params.json"
    
    # Build config with all 18 parameter bounds
    config_kwargs = dict(
        segments_per_batch=args.batch_size,
        num_epochs=args.epochs,
        max_iter=args.max_iter,
        validation_fraction=args.val_fraction,
        downsampling_factor=args.downsample,
        use_warmup=args.warmup,
        warmup_samples=args.warmup_samples,
        warmup_seed=args.warmup_seed,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        max_accel=args.max_accel,
        max_zero_speed_fraction=args.max_zero_speed_fraction,
        zero_speed_eps=args.zero_speed_eps,
        # All bounds
        mass_bounds=tuple(args.mass_bounds),
        drag_area_bounds=tuple(args.drag_area_bounds),
        rolling_coeff_bounds=tuple(args.rolling_coeff_bounds),
        motor_V_max_bounds=tuple(args.vmax_bounds),
        motor_R_bounds=tuple(args.motor_r_bounds),
        motor_L_bounds=tuple(args.motor_l_bounds),
        motor_K_bounds=tuple(args.motor_k_bounds),
        motor_b_bounds=tuple(args.motor_b_bounds),
        motor_J_bounds=tuple(args.motor_j_bounds),
        gear_ratio_bounds=tuple(args.gear_bounds),
        eta_gb_bounds=tuple(args.eta_gb_bounds),
        brake_T_max_bounds=tuple(args.brake_tmax_bounds),
        brake_tau_bounds=tuple(args.brake_tau_bounds),
        brake_p_bounds=tuple(args.brake_p_bounds),
        brake_kappa_bounds=tuple(args.brake_kappa_bounds),
        mu_bounds=tuple(args.mu_bounds),
        wheel_radius_bounds=tuple(args.wheel_radius_bounds),
        wheel_inertia_bounds=tuple(args.wheel_inertia_bounds),
    )
    
    # Add initial guesses if provided
    if args.mass_init is not None:
        config_kwargs["mass_init"] = args.mass_init
    if args.drag_area_init is not None:
        config_kwargs["drag_area_init"] = args.drag_area_init
    if args.rolling_coeff_init is not None:
        config_kwargs["rolling_coeff_init"] = args.rolling_coeff_init
    if args.vmax_init is not None:
        config_kwargs["motor_V_max_init"] = args.vmax_init
    if args.motor_r_init is not None:
        config_kwargs["motor_R_init"] = args.motor_r_init
    if args.motor_l_init is not None:
        config_kwargs["motor_L_init"] = args.motor_l_init
    if args.motor_k_init is not None:
        config_kwargs["motor_K_init"] = args.motor_k_init
    if args.motor_b_init is not None:
        config_kwargs["motor_b_init"] = args.motor_b_init
    if args.motor_j_init is not None:
        config_kwargs["motor_J_init"] = args.motor_j_init
    if args.gear_init is not None:
        config_kwargs["gear_ratio_init"] = args.gear_init
    if args.eta_gb_init is not None:
        config_kwargs["eta_gb_init"] = args.eta_gb_init
    if args.brake_tmax_init is not None:
        config_kwargs["brake_T_max_init"] = args.brake_tmax_init
    if args.brake_tau_init is not None:
        config_kwargs["brake_tau_init"] = args.brake_tau_init
    if args.brake_p_init is not None:
        config_kwargs["brake_p_init"] = args.brake_p_init
    if args.brake_kappa_init is not None:
        config_kwargs["brake_kappa_init"] = args.brake_kappa_init
    if args.mu_init is not None:
        config_kwargs["mu_init"] = args.mu_init
    if args.wheel_radius_init is not None:
        config_kwargs["wheel_radius_init"] = args.wheel_radius_init
    if args.wheel_inertia_init is not None:
        config_kwargs["wheel_inertia_init"] = args.wheel_inertia_init
    
    config = FitterConfig(**config_kwargs)
    
    fitter = VehicleParamFitter(config)
    fitted = fitter.fit(args.data, verbose=True)
    fitted.save(args.output)
    
    print("\n" + "=" * 60)
    print("FITTED VEHICLE PARAMETERS (all 18)")
    print("=" * 60)
    print("Body:")
    print(f"  mass:          {fitted.mass:.1f} kg")
    print(f"  drag_area:     {fitted.drag_area:.4f} m²")
    print(f"  rolling_coeff: {fitted.rolling_coeff:.5f}")
    print("Motor:")
    print(f"  V_max:         {fitted.motor_V_max:.1f} V")
    print(f"  R:             {fitted.motor_R:.4f} Ω")
    print(f"  L:             {fitted.motor_L:.2e} H")
    print(f"  K:             {fitted.motor_K:.4f} Nm/A")
    print(f"  b:             {fitted.motor_b:.2e} Nm·s/rad")
    print(f"  J:             {fitted.motor_J:.2e} kg·m²")
    print("Drivetrain:")
    print(f"  gear_ratio:    {fitted.gear_ratio:.2f}")
    print(f"  eta_gb:        {fitted.eta_gb:.4f}")
    print("Brake:")
    print(f"  T_max:         {fitted.brake_T_max:.1f} Nm")
    print(f"  tau:           {fitted.brake_tau:.4f} s")
    print(f"  p:             {fitted.brake_p:.4f}")
    print(f"  kappa:         {fitted.brake_kappa:.4f}")
    print(f"  mu:            {fitted.mu:.4f}")
    print("Wheel:")
    print(f"  radius:        {fitted.wheel_radius:.4f} m")
    print(f"  inertia:       {fitted.wheel_inertia:.4f} kg·m²")
    print("-" * 60)
    print(f"Val RMSE: {fitted.fit_loss**0.5:.4f} m/s")
    print(f"R² score: {fitted.r_squared:.4f}")
    print("=" * 60)
    print(f"\nSaved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
