"""Test box constraints on acceleration and jerk."""

import numpy as np
import sys
import os

# Add parent directory to path to import shaper_math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.shaper_math import shape_speed_profile, build_D1_D2


def test_acceleration_bounds_satisfied():
    """Test that acceleration bounds are enforced."""
    print("Test 1: Acceleration bounds satisfied")
    print("-" * 70)
    
    # Setup
    N = 30
    dt = 0.1
    np.random.seed(42)
    r = np.random.uniform(5.0, 15.0, N + 1)
    
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
    
    # Set moderate acceleration bounds
    a_min, a_max = -2.0, 2.0
    
    # Solve with acceleration bounds
    v = shape_speed_profile(
        r, dt, meas, weight_params,
        enable_accel_bounds=True,
        a_min=a_min, a_max=a_max
    )
    
    # Compute acceleration
    D1, _ = build_D1_D2(N, dt)
    a = D1 @ v
    
    # Check bounds with small tolerance
    tol = 1e-6
    a_min_viol = np.min(a - a_min)
    a_max_viol = np.max(a - a_max)
    
    print(f"  Acceleration bounds: [{a_min:.2f}, {a_max:.2f}]")
    print(f"  Actual range: [{a.min():.4f}, {a.max():.4f}]")
    print(f"  Min violation: {a_min_viol:.2e} (should be >= {-tol:.0e})")
    print(f"  Max violation: {a_max_viol:.2e} (should be <= {tol:.0e})")
    
    assert a_min_viol >= -tol, f"Acceleration minimum violated: {a_min_viol}"
    assert a_max_viol <= tol, f"Acceleration maximum violated: {a_max_viol}"
    
    # Check initial conditions still hold
    assert abs(v[0] - meas['v_meas']) < 1e-8
    assert abs((v[1] - v[0])/dt - meas['a_meas']) < 1e-8
    
    print("  ✓ PASS: All acceleration constraints satisfied")
    print()


def test_jerk_bounds_satisfied():
    """Test that jerk bounds are enforced."""
    print("Test 2: Jerk bounds satisfied")
    print("-" * 70)
    
    # Setup
    N = 30
    dt = 0.1
    np.random.seed(123)
    r = np.random.uniform(5.0, 15.0, N + 1)
    
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
    
    # Set moderate jerk bounds
    j_min, j_max = -5.0, 5.0
    
    # Solve with jerk bounds
    v = shape_speed_profile(
        r, dt, meas, weight_params,
        enable_jerk_bounds=True,
        j_min=j_min, j_max=j_max
    )
    
    # Compute jerk
    D1, D2 = build_D1_D2(N, dt)
    j = D2 @ v
    
    # Check bounds with small tolerance
    tol = 1e-6
    j_min_viol = np.min(j - j_min)
    j_max_viol = np.max(j - j_max)
    
    print(f"  Jerk bounds: [{j_min:.2f}, {j_max:.2f}]")
    print(f"  Actual range: [{j.min():.4f}, {j.max():.4f}]")
    print(f"  Min violation: {j_min_viol:.2e} (should be >= {-tol:.0e})")
    print(f"  Max violation: {j_max_viol:.2e} (should be <= {tol:.0e})")
    
    assert j_min_viol >= -tol, f"Jerk minimum violated: {j_min_viol}"
    assert j_max_viol <= tol, f"Jerk maximum violated: {j_max_viol}"
    
    print("  ✓ PASS: All jerk constraints satisfied")
    print()


def test_both_constraints_together():
    """Test that both acceleration and jerk bounds work simultaneously."""
    print("Test 3: Both acceleration and jerk bounds together")
    print("-" * 70)
    
    # Setup
    N = 40
    dt = 0.1
    # Create profile with sharp transitions
    r = np.concatenate([
        np.ones(10) * 5.0,
        np.ones(15) * 15.0,
        np.ones(16) * 8.0
    ])
    
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
    
    # Set both bounds
    a_min, a_max = -3.0, 3.0
    j_min, j_max = -8.0, 8.0
    
    # Solve with both constraints
    v = shape_speed_profile(
        r, dt, meas, weight_params,
        enable_accel_bounds=True, a_min=a_min, a_max=a_max,
        enable_jerk_bounds=True, j_min=j_min, j_max=j_max
    )
    
    # Compute acceleration and jerk
    D1, D2 = build_D1_D2(N, dt)
    a = D1 @ v
    j = D2 @ v
    
    # Check both bounds
    tol = 1e-6
    
    a_ok = (a.min() >= a_min - tol) and (a.max() <= a_max + tol)
    j_ok = (j.min() >= j_min - tol) and (j.max() <= j_max + tol)
    
    print(f"  Acceleration: [{a.min():.4f}, {a.max():.4f}] ∈ [{a_min}, {a_max}]")
    print(f"  Jerk:         [{j.min():.4f}, {j.max():.4f}] ∈ [{j_min}, {j_max}]")
    
    assert a_ok, f"Acceleration bounds violated"
    assert j_ok, f"Jerk bounds violated"
    
    print("  ✓ PASS: Both constraints satisfied simultaneously")
    print()


def test_infeasibility_detection():
    """Test that infeasible bounds are detected."""
    print("Test 4: Infeasibility detection")
    print("-" * 70)
    
    # Setup
    N = 20
    dt = 0.1
    r = np.ones(N + 1) * 10.0
    
    # Set initial acceleration that violates bounds
    meas = {
        'v_meas': 10.0,
        'a_meas': 5.0,  # Outside bounds!
        'j_meas': 0.0
    }
    
    weight_params = {
        'wE_start': 10.0, 'wE_end': 10.0, 'lamE': 0.0,
        'wA_start': 5.0, 'wA_end': 5.0, 'lamA': 0.0,
        'wJ_start': 5.0, 'wJ_end': 5.0, 'lamJ': 0.0,
    }
    
    # Try to solve with incompatible bounds
    a_min, a_max = -2.0, 2.0  # a_meas = 5.0 is outside!
    
    try:
        v = shape_speed_profile(
            r, dt, meas, weight_params,
            enable_accel_bounds=True,
            a_min=a_min, a_max=a_max
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Caught expected error:")
        print(f"    {str(e)}")
        assert "a_meas" in str(e).lower() or "acceleration" in str(e).lower()
        print("  ✓ PASS: Infeasibility detected with clear error message")
    
    print()


def test_backward_compatibility():
    """Test that old code without new parameters still works."""
    print("Test 5: Backward compatibility (no bounds)")
    print("-" * 70)
    
    # Setup
    N = 25
    dt = 0.1
    np.random.seed(456)
    r = np.random.uniform(5.0, 15.0, N + 1)
    
    meas = {
        'v_meas': r[0],
        'a_meas': 0.5,
        'j_meas': -0.2
    }
    
    weight_params = {
        'wE_start': 10.0, 'wE_end': 10.0, 'lamE': 0.0,
        'wA_start': 5.0, 'wA_end': 5.0, 'lamA': 0.0,
        'wJ_start': 5.0, 'wJ_end': 5.0, 'lamJ': 0.0,
    }
    
    # Call without new parameters (uses KKT solver)
    v = shape_speed_profile(r, dt, meas, weight_params)
    
    # Check constraints
    tol = 1e-8
    assert abs(v[0] - meas['v_meas']) < tol
    assert abs((v[1] - v[0])/dt - meas['a_meas']) < tol
    assert abs((v[2] - 2*v[1] + v[0])/(dt*dt) - meas['j_meas']) < tol
    
    print("  ✓ PASS: Old API works (backward compatible)")
    print()


def test_performance_with_osqp():
    """Test performance with OSQP on larger problem."""
    print("Test 6: Performance with OSQP (N=1000)")
    print("-" * 70)
    
    import time
    
    N = 1000
    dt = 0.02
    t = np.arange(N + 1) * dt
    r = 15.0 + 5.0 * np.sin(2 * np.pi * t / 10.0)
    
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
    
    # Time the solve with both constraints
    start = time.time()
    v = shape_speed_profile(
        r, dt, meas, weight_params,
        enable_accel_bounds=True, a_min=-4.0, a_max=4.0,
        enable_jerk_bounds=True, j_min=-10.0, j_max=10.0
    )
    elapsed = time.time() - start
    
    print(f"  Problem size: N={N} ({N+1} variables)")
    print(f"  Solve time: {elapsed*1000:.1f} ms")
    
    # Verify solution is valid
    assert v.shape == r.shape
    assert np.all(np.isfinite(v))
    
    # Verify constraints
    D1, D2 = build_D1_D2(N, dt)
    a = D1 @ v
    j = D2 @ v
    
    a_ok = (a.min() >= -4.0 - 1e-6) and (a.max() <= 4.0 + 1e-6)
    j_ok = (j.min() >= -10.0 - 1e-6) and (j.max() <= 10.0 + 1e-6)
    
    assert a_ok and j_ok, "Constraints violated in large problem"
    
    if elapsed < 0.1:
        print(f"  ✓ PASS: Excellent performance ({elapsed*1000:.1f} ms < 100 ms target)")
    else:
        print(f"  ✓ PASS: Acceptable performance ({elapsed*1000:.1f} ms)")
    
    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("TESTING BOX CONSTRAINTS ON ACCELERATION AND JERK")
    print("=" * 70)
    print()
    
    try:
        test_acceleration_bounds_satisfied()
        test_jerk_bounds_satisfied()
        test_both_constraints_together()
        test_infeasibility_detection()
        test_backward_compatibility()
        test_performance_with_osqp()
        
        print("=" * 70)
        print("✓✓✓ ALL BOX CONSTRAINT TESTS PASSED! ✓✓✓")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise
