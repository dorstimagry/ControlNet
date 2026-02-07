# Speed Profile Shaper

A quadratic programming (QP) based speed profile shaper that smooths raw target speed profiles while enforcing **hard equality constraints** on initial velocity, acceleration, and jerk.

## Features

- **Hard initial derivative constraints**: Exactly satisfies initial conditions v(0), a(0), j(0)
- **Box constraints**: Optional hard bounds on acceleration and jerk (via OSQP)
- **Time-varying weights**: Exponential interpolation schedules for each penalty term
- **Dual solver support**: KKT (fast, equality-only) or OSQP (general QP with inequalities)
- **Interactive GUI**: Real-time tuning with matplotlib widgets
- **Fast**: Solves N=1000 in ~6 ms (KKT) or ~50-100 ms (OSQP)

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

Requirements:
- numpy
- scipy  
- matplotlib
- osqp

## Usage

### Interactive GUI

Launch the interactive GUI to visualize and tune the shaper:

```bash
# Coarse example (discontinuous steps with large jumps) - default
python -m src.gui_matplotlib

# Or explicitly:
python -m src.gui_matplotlib coarse

# Smooth example (smooth profile with high acceleration/jerk)
python -m src.gui_matplotlib smooth
```

The GUI provides:
- **2 example profiles**:
  - `coarse`: Discontinuous steps with large instantaneous jumps (19+ m/s jumps)
  - `smooth`: Continuous profile with aggressive acceleration (58+ m/s²) and jerk (1000+ m/s³)
- **9 sliders** to control time-varying weights for error, acceleration, and jerk terms
- **Auto-update**: Plot refreshes automatically when sliders change (no button needed)
- **Terminal constraint toggle** to optionally enforce v[N] = r[N]
- **Box constraints** on acceleration and jerk (checkboxes + bound sliders)
- **Real-time plotting** of shaped profile, acceleration, jerk, and constraint bounds

#### Weight Parameters

Each term (error, acceleration, jerk) has 3 parameters:

- **w_start**: Weight at t=0
- **w_end**: Target weight as t→∞  
- **λ (lambda)**: Transition rate (range: -5 to +5)

Weight schedule: `w(t) = w_end + (w_start - w_end) * exp(-λ * t)`

**Important:** Lambda only has an effect when `w_start ≠ w_end`.

**How it works:**
- **Positive λ (recommended)**: Weights transition FROM `w_start` TOWARD `w_end`
  - Example: `w_start=30, w_end=10, λ=+2` → weights decrease 30→10
  - Example: `w_start=5, w_end=20, λ=+2` → weights increase 5→20
  - Higher |λ| = faster transition
  
- **λ = 0**: Constant weight stays at `w_start`
  
- **Negative λ (advanced)**: Weights move AWAY from `w_end`
  - Can cause exponential growth/decay away from target
  - May clip to zero if weights go negative
  - Generally not recommended; use positive λ with swapped start/end instead

**Transition speed:**
- `λ = 0.5`: Gentle (63% complete in 2 seconds)
- `λ = 1.0`: Moderate (63% complete in 1 second)
- `λ = 3.0`: Fast (95% complete in 1 second)

**Default values** demonstrate typical patterns:
- **Error**: start=30, end=10, λ=+1.0 (tight early → loose later)
- **Accel**: start=2, end=10, λ=-0.5 (loose early → tight later, using negative λ for growth)
- **Jerk**: start=2, end=10, λ=-0.5 (loose early → tight later, using negative λ for growth)

**Note:** For growth patterns (increasing weights), you can either:
1. Use negative λ (as in defaults), but watch for instability
2. Set w_start low, w_end high, with positive λ (more stable)

#### Tuning Tips

- **High error weight** → Tight tracking of raw profile
- **High acceleration weight** → Smooth velocity slopes
- **High jerk weight** → No sharp corners or discontinuities

### Programmatic API

```python
from src.shaper_math import shape_speed_profile
import numpy as np

# Raw target speed profile
dt = 0.1  # time step (seconds)
r = np.array([0, 5, 10, 15, 15, 15, 10, 5])  # m/s

# Initial conditions
meas = {
    'v_meas': 0.0,   # initial velocity (m/s)
    'a_meas': 0.0,   # initial acceleration (m/s²)
    'j_meas': 0.0    # initial jerk (m/s³)
}

# Weight parameters
weight_params = {
    'wE_start': 20.0, 'wE_end': 10.0, 'lamE': 1.0,  # error
    'wA_start': 5.0,  'wA_end': 15.0, 'lamA': 0.5,  # acceleration
    'wJ_start': 5.0,  'wJ_end': 10.0, 'lamJ': 0.3,  # jerk
}

# Solve
v_shaped = shape_speed_profile(r, dt, meas, weight_params, enforce_terminal=False)

print(f"Shaped profile: {v_shaped}")
```

### Box Constraints on Acceleration and Jerk

Enable hard bounds on acceleration and/or jerk to enforce physical or comfort limits:

```python
from src.shaper_math import shape_speed_profile
import numpy as np

# Setup (same as above)
dt = 0.1
r = np.array([0, 5, 10, 15, 15, 15, 10, 5])
meas = {'v_meas': 0.0, 'a_meas': 0.0, 'j_meas': 0.0}
weight_params = {
    'wE_start': 20.0, 'wE_end': 10.0, 'lamE': 1.0,
    'wA_start': 5.0,  'wA_end': 15.0, 'lamA': 0.5,
    'wJ_start': 5.0,  'wJ_end': 10.0, 'lamJ': 0.3,
}

# Solve with box constraints
v_shaped = shape_speed_profile(
    r, dt, meas, weight_params,
    enforce_terminal=False,
    # Acceleration constraints
    enable_accel_bounds=True,
    a_min=-3.0,  # m/s² (minimum acceleration, typically negative)
    a_max=3.0,   # m/s² (maximum acceleration, typically positive)
    # Jerk constraints  
    enable_jerk_bounds=True,
    j_min=-8.0,  # m/s³ (minimum jerk)
    j_max=8.0    # m/s³ (maximum jerk)
)
```

**Key Points:**
- When enabled, constraints are **hard bounds** strictly enforced in the QP solution
- Uses OSQP solver (slightly slower than KKT but still fast: ~10-50ms for N=1000)
- Bounds must be **compatible with initial conditions**:
  - If `enable_accel_bounds=True`, must have `a_min ≤ a_meas ≤ a_max`
  - If `enable_jerk_bounds=True`, must have `j_min ≤ j_meas ≤ j_max`
  - Incompatible bounds raise `ValueError` with clear message
- Without any bounds enabled, uses original KKT solver (backward compatible)
- Typical values:
  - Comfortable car: `a ∈ [-3, 3] m/s²`, `j ∈ [-5, 5] m/s³`
  - Aggressive driving: `a ∈ [-6, 6] m/s²`, `j ∈ [-10, 10] m/s³`

**GUI Usage:**
- Check "Enable accel bounds" or "Enable jerk bounds" 
- Adjust `a_min`, `a_max`, `j_min`, `j_max` sliders
- Constraint bounds shown as dashed lines on acceleration/jerk plot
- Plot updates automatically

## Mathematical Formulation

### Objective

Minimize weighted sum of:
1. **Speed tracking error**: `(v - r)ᵀ Wₑ (v - r)`
2. **Acceleration penalty**: `(D₁v)ᵀ Wₐ (D₁v)` where `D₁` is forward difference operator
3. **Jerk penalty**: `(D₂v)ᵀ Wⱼ (D₂v)` where `D₂` is second difference operator

### Constraints

Hard equality constraints (always enforced):
1. `v[0] = v_meas` (initial velocity)
2. `(v[1] - v[0])/dt = a_meas` (initial acceleration)  
3. `(v[2] - 2v[1] + v[0])/dt² = j_meas` (initial jerk)
4. `v[N] = r[N]` (optional terminal constraint)

Hard inequality constraints (optional box constraints):
- Acceleration: `a_min ≤ D₁v ≤ a_max` (element-wise)
- Jerk: `j_min ≤ D₂v ≤ j_max` (element-wise)

### Solution Method

**Without box constraints** (default):  
Solved via KKT system for equality-constrained QP:

```
[H   Aᵀ] [v ]   [f]
[A   0 ] [ν ] = [b]
```

where:
- `H = 2(Wₑ + D₁ᵀWₐD₁ + D₂ᵀWⱼD₂)` (Hessian)
- `f = 2Wₑr` (gradient)
- `A`, `b` encode linear equality constraints

Solved using `scipy.sparse.linalg.spsolve` for efficiency.

**With box constraints**:  
Solved via OSQP (convex QP solver) with both equality and inequality constraints:

```
min   ½vᵀHv - fᵀv
s.t.  Av = b           (equality constraints)
      Gv ≤ h           (inequality constraints)
```

where `G` and `h` encode the box constraints in standard form.

## Testing

Run unit tests:

```bash
# Test constraint satisfaction
python tests/test_constraints.py

# Test solution quality and performance
python tests/test_kkt_solution.py

# Test box constraints (acceleration/jerk bounds)
python tests/test_box_constraints.py
```

All tests verify:
- Equality constraints satisfied to 1e-8 tolerance (1e-7 with OSQP)
- Box constraints satisfied when enabled
- Infeasibility detection works correctly
- Solution is finite and has correct dimensions
- Smoothing effect of weights
- Backward compatibility (old API still works)
- Performance on large problems (N=1000)

## Project Structure

```
speed_shaper/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── shaper_math.py      # Core QP solver (KKT + OSQP)
│   └── gui_matplotlib.py   # Interactive GUI
└── tests/
    ├── __init__.py
    ├── test_constraints.py     # Constraint verification
    ├── test_kkt_solution.py    # Solution quality tests
    └── test_box_constraints.py # Box constraint tests
```

## Performance

Typical solve times on standard laptop:

**Without box constraints (KKT solver):**
- N=50: < 1 ms
- N=100: ~2 ms
- N=1000: ~6 ms

**With box constraints (OSQP solver):**
- N=50: ~5 ms
- N=100: ~10 ms
- N=1000: ~50-100 ms

Both well below the 200 ms target for N=1000. Use KKT when constraints aren't needed for best performance.

## License

This module is part of the DiffDynamics project.
