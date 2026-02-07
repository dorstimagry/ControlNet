"""Test hard equality constraints are satisfied after QP solve."""

import numpy as np
import sys
import os

# Add parent directory to path to import shaper_math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.shaper_math import shape_speed_profile


def test_initial_constraints_no_terminal():
    """Test that initial v, a, j constraints are satisfied without terminal constraint."""
    # Setup small test case
    N = 10
    dt = 0.1
    
    # Random raw profile
    np.random.seed(42)
    r = np.random.uniform(5.0, 15.0, N + 1)
    
    # Initial conditions
    v_meas = 1.5
    a_meas = 0.5
    j_meas = -0.2
    
    meas = {
        'v_meas': v_meas,
        'a_meas': a_meas,
        'j_meas': j_meas
    }
    
    # Weight parameters (moderate values)
    weight_params = {
        'wE_start': 10.0, 'wE_end': 10.0, 'lamE': 0.0,
        'wA_start': 5.0, 'wA_end': 5.0, 'lamA': 0.0,
        'wJ_start': 5.0, 'wJ_end': 5.0, 'lamJ': 0.0,
    }
    
    # Solve
    v = shape_speed_profile(r, dt, meas, weight_params, enforce_terminal=False)
    
    # Verify constraints
    tol = 1e-8
    
    # Constraint 1: v[0] = v_meas
    err_v0 = abs(v[0] - v_meas)
    print(f"v[0] constraint error: {err_v0:.2e}")
    assert err_v0 < tol, f"v[0] constraint violated: {err_v0} >= {tol}"
    
    # Constraint 2: (v[1] - v[0])/dt = a_meas
    a_computed = (v[1] - v[0]) / dt
    err_a0 = abs(a_computed - a_meas)
    print(f"a[0] constraint error: {err_a0:.2e}")
    assert err_a0 < tol, f"a[0] constraint violated: {err_a0} >= {tol}"
    
    # Constraint 3: (v[2] - 2*v[1] + v[0])/dt^2 = j_meas
    j_computed = (v[2] - 2*v[1] + v[0]) / (dt * dt)
    err_j0 = abs(j_computed - j_meas)
    print(f"j[0] constraint error: {err_j0:.2e}")
    assert err_j0 < tol, f"j[0] constraint violated: {err_j0} >= {tol}"
    
    print("✓ All initial constraints satisfied (no terminal)")


def test_initial_and_terminal_constraints():
    """Test that initial v, a, j and terminal v constraints are satisfied."""
    # Setup small test case
    N = 10
    dt = 0.1
    
    # Random raw profile
    np.random.seed(123)
    r = np.random.uniform(5.0, 15.0, N + 1)
    
    # Initial conditions
    v_meas = 2.0
    a_meas = -0.3
    j_meas = 0.1
    
    meas = {
        'v_meas': v_meas,
        'a_meas': a_meas,
        'j_meas': j_meas
    }
    
    # Weight parameters
    weight_params = {
        'wE_start': 10.0, 'wE_end': 10.0, 'lamE': 0.0,
        'wA_start': 5.0, 'wA_end': 5.0, 'lamA': 0.0,
        'wJ_start': 5.0, 'wJ_end': 5.0, 'lamJ': 0.0,
    }
    
    # Solve with terminal constraint
    v = shape_speed_profile(r, dt, meas, weight_params, enforce_terminal=True)
    
    # Verify constraints
    tol = 1e-8
    
    # Constraint 1: v[0] = v_meas
    err_v0 = abs(v[0] - v_meas)
    print(f"v[0] constraint error: {err_v0:.2e}")
    assert err_v0 < tol, f"v[0] constraint violated: {err_v0} >= {tol}"
    
    # Constraint 2: (v[1] - v[0])/dt = a_meas
    a_computed = (v[1] - v[0]) / dt
    err_a0 = abs(a_computed - a_meas)
    print(f"a[0] constraint error: {err_a0:.2e}")
    assert err_a0 < tol, f"a[0] constraint violated: {err_a0} >= {tol}"
    
    # Constraint 3: (v[2] - 2*v[1] + v[0])/dt^2 = j_meas
    j_computed = (v[2] - 2*v[1] + v[0]) / (dt * dt)
    err_j0 = abs(j_computed - j_meas)
    print(f"j[0] constraint error: {err_j0:.2e}")
    assert err_j0 < tol, f"j[0] constraint violated: {err_j0} >= {tol}"
    
    # Constraint 4: v[N] = r[N]
    err_vN = abs(v[N] - r[N])
    print(f"v[N] constraint error: {err_vN:.2e}")
    assert err_vN < tol, f"v[N] constraint violated: {err_vN} >= {tol}"
    
    print("✓ All initial and terminal constraints satisfied")


def test_time_varying_weights():
    """Test with time-varying weights (exponential decay)."""
    N = 20
    dt = 0.1
    
    # Piecewise profile
    r = np.concatenate([
        np.linspace(0, 10, 5),
        np.ones(6) * 10,
        np.linspace(10, 5, 5),
        np.linspace(5, 15, 6)
    ])
    
    # Initial conditions
    meas = {
        'v_meas': r[0],
        'a_meas': 0.0,
        'j_meas': 0.0
    }
    
    # Time-varying weights
    weight_params = {
        'wE_start': 50.0, 'wE_end': 10.0, 'lamE': 2.0,
        'wA_start': 1.0, 'wA_end': 20.0, 'lamA': 1.5,
        'wJ_start': 1.0, 'wJ_end': 10.0, 'lamJ': 1.0,
    }
    
    # Solve
    v = shape_speed_profile(r, dt, meas, weight_params, enforce_terminal=False)
    
    # Verify constraints
    tol = 1e-8
    
    err_v0 = abs(v[0] - meas['v_meas'])
    err_a0 = abs((v[1] - v[0]) / dt - meas['a_meas'])
    err_j0 = abs((v[2] - 2*v[1] + v[0]) / (dt*dt) - meas['j_meas'])
    
    print(f"Time-varying weights test:")
    print(f"  v[0] error: {err_v0:.2e}")
    print(f"  a[0] error: {err_a0:.2e}")
    print(f"  j[0] error: {err_j0:.2e}")
    
    assert err_v0 < tol, f"v[0] constraint violated"
    assert err_a0 < tol, f"a[0] constraint violated"
    assert err_j0 < tol, f"j[0] constraint violated"
    
    print("✓ Time-varying weights test passed")


if __name__ == '__main__':
    print("Running constraint tests...\n")
    
    print("Test 1: Initial constraints (no terminal)")
    print("-" * 50)
    test_initial_constraints_no_terminal()
    print()
    
    print("Test 2: Initial and terminal constraints")
    print("-" * 50)
    test_initial_and_terminal_constraints()
    print()
    
    print("Test 3: Time-varying weights")
    print("-" * 50)
    test_time_varying_weights()
    print()
    
    print("=" * 50)
    print("✓✓✓ All constraint tests passed! ✓✓✓")
    print("=" * 50)
