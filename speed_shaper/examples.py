"""Example usage of the speed profile shaper."""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from src.shaper_math import shape_speed_profile


def example_basic_usage():
    """Basic usage example with constant weights."""
    
    print("Example 1: Basic usage with constant weights")
    print("=" * 60)
    
    # Create a simple profile with a step change
    dt = 0.1  # 100 ms time step
    r = np.concatenate([
        np.ones(20) * 10.0,  # 10 m/s for 2 seconds
        np.ones(20) * 5.0,   # Step to 5 m/s
        np.ones(21) * 15.0   # Step to 15 m/s
    ])
    
    # Initial conditions (starting from rest with smooth start)
    meas = {
        'v_meas': r[0],
        'a_meas': 0.0,
        'j_meas': 0.0
    }
    
    # Weight parameters (balanced smoothing)
    weight_params = {
        'wE_start': 10.0, 'wE_end': 10.0, 'lamE': 0.0,  # constant error weight
        'wA_start': 5.0,  'wA_end': 5.0,  'lamA': 0.0,  # constant accel weight
        'wJ_start': 5.0,  'wJ_end': 5.0,  'lamJ': 0.0,  # constant jerk weight
    }
    
    # Solve
    v = shape_speed_profile(r, dt, meas, weight_params, enforce_terminal=False)
    
    print(f"Profile length: {len(v)} samples ({len(v)*dt:.1f} seconds)")
    print(f"Initial speed: {v[0]:.2f} m/s (constraint: {meas['v_meas']:.2f})")
    print(f"Initial accel: {(v[1]-v[0])/dt:.2f} m/s² (constraint: {meas['a_meas']:.2f})")
    print(f"Max speed: {v.max():.2f} m/s")
    print(f"Min speed: {v.min():.2f} m/s")
    print()


def example_time_varying_weights():
    """Example with time-varying weights (high smoothing at start, tight tracking at end)."""
    
    print("Example 2: Time-varying weights")
    print("=" * 60)
    
    # Create a profile
    dt = 0.1
    t = np.arange(100) * dt
    r = 10.0 + 5.0 * np.sin(2 * np.pi * t / 5.0)
    
    meas = {
        'v_meas': r[0],
        'a_meas': 0.0,
        'j_meas': 0.0
    }
    
    # Time-varying weights: start with high smoothing, end with tight tracking
    weight_params = {
        'wE_start': 5.0,   'wE_end': 50.0,  'lamE': 2.0,   # increase error weight
        'wA_start': 20.0,  'wA_end': 2.0,   'lamA': 2.0,   # decrease accel weight
        'wJ_start': 20.0,  'wJ_end': 2.0,   'lamJ': 2.0,   # decrease jerk weight
    }
    
    v = shape_speed_profile(r, dt, meas, weight_params, enforce_terminal=False)
    
    # Compute tracking error over time
    error = np.abs(v - r)
    early_error = error[:20].mean()
    late_error = error[-20:].mean()
    
    print(f"Early tracking error (0-2s): {early_error:.3f} m/s (high smoothing)")
    print(f"Late tracking error (8-10s): {late_error:.3f} m/s (tight tracking)")
    print(f"Ratio: {early_error/late_error:.1f}x looser at start")
    print()


def example_terminal_constraint():
    """Example showing effect of terminal constraint."""
    
    print("Example 3: Terminal constraint effect")
    print("=" * 60)
    
    # Create profile with significant endpoint deviation
    dt = 0.1
    r = np.concatenate([
        np.linspace(0, 15, 30),
        np.linspace(15, 8, 30),
        np.ones(11) * 8.0
    ])
    
    meas = {
        'v_meas': r[0],
        'a_meas': 0.0,
        'j_meas': 0.0
    }
    
    weight_params = {
        'wE_start': 10.0, 'wE_end': 10.0, 'lamE': 0.0,
        'wA_start': 10.0, 'wA_end': 10.0, 'lamA': 0.0,
        'wJ_start': 10.0, 'wJ_end': 10.0, 'lamJ': 0.0,
    }
    
    # Solve without and with terminal constraint
    v_no_term = shape_speed_profile(r, dt, meas, weight_params, enforce_terminal=False)
    v_with_term = shape_speed_profile(r, dt, meas, weight_params, enforce_terminal=True)
    
    print(f"Target endpoint: {r[-1]:.2f} m/s")
    print(f"Without terminal constraint: {v_no_term[-1]:.2f} m/s (error: {abs(v_no_term[-1]-r[-1]):.3f})")
    print(f"With terminal constraint:    {v_with_term[-1]:.2f} m/s (error: {abs(v_with_term[-1]-r[-1]):.2e})")
    print()


def example_performance():
    """Performance demonstration on large problem."""
    
    print("Example 4: Performance on large problem")
    print("=" * 60)
    
    import time
    
    # Large problem
    N = 1000
    dt = 0.02  # 50 Hz
    t = np.arange(N + 1) * dt
    r = 15.0 + 8.0 * np.sin(2 * np.pi * t / 10.0) + 3.0 * np.cos(2 * np.pi * t / 3.0)
    r = np.clip(r, 0, 30)
    
    meas = {
        'v_meas': r[0],
        'a_meas': 0.0,
        'j_meas': 0.0
    }
    
    weight_params = {
        'wE_start': 20.0, 'wE_end': 10.0, 'lamE': 1.0,
        'wA_start': 5.0,  'wA_end': 15.0, 'lamA': 0.5,
        'wJ_start': 5.0,  'wJ_end': 10.0, 'lamJ': 0.3,
    }
    
    start = time.time()
    v = shape_speed_profile(r, dt, meas, weight_params, enforce_terminal=False)
    elapsed = time.time() - start
    
    print(f"Problem size: N={N} ({N+1} variables, {(N+1)*dt:.1f} seconds)")
    print(f"Solve time: {elapsed*1000:.2f} ms")
    print(f"Real-time factor: {((N+1)*dt / elapsed):.0f}x faster than real-time")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("SPEED PROFILE SHAPER - USAGE EXAMPLES")
    print("=" * 60)
    print()
    
    example_basic_usage()
    example_time_varying_weights()
    example_terminal_constraint()
    example_performance()
    
    print("=" * 60)
    print("For interactive GUI, run:")
    print("  python -m src.gui_matplotlib coarse   # Discontinuous steps")
    print("  python -m src.gui_matplotlib smooth   # High acceleration/jerk")
    print()
    print("Sliders update plots automatically (no button needed)")
    print("=" * 60)
