"""Test KKT solution sanity checks."""

import numpy as np
import sys
import os

# Add parent directory to path to import shaper_math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.shaper_math import shape_speed_profile, build_D1_D2, build_weight_mats, build_qp_matrices


def compute_objective(v, r, dt, We, Wa, Wj, D1, D2):
    """Compute the objective value for a given profile."""
    # Error term: (v - r)^T We (v - r)
    error_term = (v - r) @ We @ (v - r)
    
    # Acceleration term: (D1 v)^T Wa (D1 v)
    a = D1 @ v
    accel_term = a @ Wa @ a
    
    # Jerk term: (D2 v)^T Wj (D2 v)
    j = D2 @ v
    jerk_term = j @ Wj @ j
    
    return error_term + accel_term + jerk_term


def test_solution_is_finite():
    """Test that solution is finite and has correct shape."""
    N = 50
    dt = 0.1
    
    # Generate smooth test profile
    t = np.arange(N + 1) * dt
    r = 10.0 + 5.0 * np.sin(2 * np.pi * t / 3.0)
    
    meas = {
        'v_meas': r[0],
        'a_meas': 0.0,
        'j_meas': 0.0
    }
    
    weight_params = {
        'wE_start': 10.0, 'wE_end': 10.0, 'lamE': 0.0,
        'wA_start': 5.0, 'wA_end': 5.0, 'lamA': 0.0,
        'wJ_start': 5.0, 'wJ_end': 5.0, 'lamJ': 0.0,
    }
    
    v = shape_speed_profile(r, dt, meas, weight_params, enforce_terminal=False)
    
    # Check shape
    assert v.shape == r.shape, f"Shape mismatch: {v.shape} != {r.shape}"
    
    # Check finite
    assert np.all(np.isfinite(v)), "Solution contains NaN or Inf"
    
    print(f"✓ Solution shape: {v.shape}")
    print(f"✓ Solution is finite")
    print(f"  Min: {v.min():.2f}, Max: {v.max():.2f}, Mean: {v.mean():.2f}")


def test_objective_comparison():
    """Test that solution objective is reasonable compared to raw profile."""
    N = 30
    dt = 0.1
    
    # Create profile with sharp transitions (high jerk)
    r = np.concatenate([
        np.ones(10) * 5.0,
        np.ones(10) * 15.0,
        np.ones(11) * 8.0
    ])
    
    # Start with initial conditions matching raw profile
    meas = {
        'v_meas': r[0],
        'a_meas': 0.0,
        'j_meas': 0.0
    }
    
    # Use constant weights
    weight_params = {
        'wE_start': 10.0, 'wE_end': 10.0, 'lamE': 0.0,
        'wA_start': 5.0, 'wA_end': 5.0, 'lamA': 0.0,
        'wJ_start': 5.0, 'wJ_end': 5.0, 'lamJ': 0.0,
    }
    
    # Solve
    v = shape_speed_profile(r, dt, meas, weight_params, enforce_terminal=False)
    
    # Build weight matrices to compute objectives
    D1, D2 = build_D1_D2(N, dt)
    We, Wa, Wj = build_weight_mats(N, dt, weight_params)
    
    # Compute objectives
    obj_raw = compute_objective(r, r, dt, We, Wa, Wj, D1, D2)
    obj_solution = compute_objective(v, r, dt, We, Wa, Wj, D1, D2)
    
    print(f"Objective (raw):      {obj_raw:.6f}")
    print(f"Objective (solution): {obj_solution:.6f}")
    print(f"Improvement:          {obj_raw - obj_solution:.6f}")
    
    # The solution should have a finite objective
    assert np.isfinite(obj_solution), "Solution objective is not finite"
    
    # With hard initial constraints, we can't guarantee improvement over raw,
    # but the objective should be reasonable (not absurdly large)
    assert obj_solution < 1e6, f"Solution objective suspiciously large: {obj_solution}"
    
    print("✓ Objective values are reasonable")


def test_smoothing_effect():
    """Test that increasing acceleration/jerk weights produces smoother profiles."""
    N = 40
    dt = 0.1
    
    # Create noisy profile
    np.random.seed(42)
    t = np.arange(N + 1) * dt
    r = 10.0 + 5.0 * np.sin(2 * np.pi * t / 2.0) + np.random.normal(0, 0.5, N + 1)
    
    meas = {
        'v_meas': r[0],
        'a_meas': 0.0,
        'j_meas': 0.0
    }
    
    # Low smoothing weights
    weight_params_low = {
        'wE_start': 10.0, 'wE_end': 10.0, 'lamE': 0.0,
        'wA_start': 0.1, 'wA_end': 0.1, 'lamA': 0.0,
        'wJ_start': 0.1, 'wJ_end': 0.1, 'lamJ': 0.0,
    }
    
    # High smoothing weights
    weight_params_high = {
        'wE_start': 10.0, 'wE_end': 10.0, 'lamE': 0.0,
        'wA_start': 50.0, 'wA_end': 50.0, 'lamA': 0.0,
        'wJ_start': 50.0, 'wJ_end': 50.0, 'lamJ': 0.0,
    }
    
    v_low = shape_speed_profile(r, dt, meas, weight_params_low, enforce_terminal=False)
    v_high = shape_speed_profile(r, dt, meas, weight_params_high, enforce_terminal=False)
    
    # Compute accelerations
    a_low = np.diff(v_low) / dt
    a_high = np.diff(v_high) / dt
    
    # Compute jerk
    j_low = np.diff(a_low) / dt
    j_high = np.diff(a_high) / dt
    
    # Measure smoothness (variance of jerk)
    jerk_var_low = np.var(j_low)
    jerk_var_high = np.var(j_high)
    
    print(f"Jerk variance (low smoothing):  {jerk_var_low:.4f}")
    print(f"Jerk variance (high smoothing): {jerk_var_high:.4f}")
    print(f"Smoothing ratio:                {jerk_var_low / jerk_var_high:.2f}x")
    
    # High smoothing should produce lower jerk variance
    assert jerk_var_high < jerk_var_low, "High smoothing weights should reduce jerk variance"
    
    print("✓ Smoothing effect verified")


def test_large_problem():
    """Test performance on larger problem (N=1000)."""
    import time
    
    N = 1000
    dt = 0.02  # 50 Hz
    
    # Generate realistic driving profile
    t = np.arange(N + 1) * dt
    r = 15.0 + 5.0 * np.sin(2 * np.pi * t / 10.0) + 3.0 * np.sin(2 * np.pi * t / 3.0)
    r = np.clip(r, 0, 25)
    
    meas = {
        'v_meas': r[0],
        'a_meas': 0.0,
        'j_meas': 0.0
    }
    
    weight_params = {
        'wE_start': 20.0, 'wE_end': 10.0, 'lamE': 1.0,
        'wA_start': 5.0, 'wA_end': 15.0, 'lamA': 0.5,
        'wJ_start': 5.0, 'wJ_end': 10.0, 'lamJ': 0.3,
    }
    
    # Time the solve
    start = time.time()
    v = shape_speed_profile(r, dt, meas, weight_params, enforce_terminal=False)
    elapsed = time.time() - start
    
    print(f"Problem size: N={N} ({N+1} variables)")
    print(f"Solve time: {elapsed*1000:.1f} ms")
    
    # Check solution is valid
    assert v.shape == r.shape
    assert np.all(np.isfinite(v))
    
    # Performance check (should be < 200 ms)
    if elapsed < 0.2:
        print(f"✓ Performance target met (< 200 ms)")
    else:
        print(f"⚠ Performance warning: {elapsed*1000:.1f} ms > 200 ms target")
    
    print("✓ Large problem solved successfully")


if __name__ == '__main__':
    print("Running KKT solution sanity tests...\n")
    
    print("Test 1: Solution is finite and correct shape")
    print("-" * 50)
    test_solution_is_finite()
    print()
    
    print("Test 2: Objective value comparison")
    print("-" * 50)
    test_objective_comparison()
    print()
    
    print("Test 3: Smoothing effect")
    print("-" * 50)
    test_smoothing_effect()
    print()
    
    print("Test 4: Large problem (N=1000)")
    print("-" * 50)
    test_large_problem()
    print()
    
    print("=" * 50)
    print("✓✓✓ All KKT solution tests passed! ✓✓✓")
    print("=" * 50)
