"""Core mathematical functions for speed profile shaping via QP with hard constraints."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Tuple, Dict, Any


def build_D1_D2(N: int, dt: float) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build sparse difference operators for acceleration and jerk.
    
    Args:
        N: Number of time steps (creates N+1 decision variables v[0]..v[N])
        dt: Time step size (seconds)
    
    Returns:
        D1: Forward difference operator (N x N+1) for acceleration
            (D1 @ v)_k = (v[k+1] - v[k]) / dt
        D2: Second difference operator (N-1 x N+1) for jerk
            (D2 @ v)_k = (v[k+2] - 2*v[k+1] + v[k]) / dt^2
    """
    # D1: forward difference for acceleration
    # Row k has: -1/dt at column k, +1/dt at column k+1
    # Shape: (N, N+1)
    diag_main = -np.ones(N) / dt
    diag_upper = np.ones(N) / dt
    D1 = sp.diags([diag_main, diag_upper], offsets=[0, 1], shape=(N, N+1), format='csr')
    
    # D2: second difference for jerk
    # Row k has: +1/dt^2 at column k, -2/dt^2 at column k+1, +1/dt^2 at column k+2
    # Shape: (N-1, N+1)
    dt_sq = dt * dt
    diag_0 = np.ones(N-1) / dt_sq
    diag_1 = -2.0 * np.ones(N-1) / dt_sq
    diag_2 = np.ones(N-1) / dt_sq
    D2 = sp.diags([diag_0, diag_1, diag_2], offsets=[0, 1, 2], shape=(N-1, N+1), format='csr')
    
    return D1, D2


def weight_schedule(t: np.ndarray, w_start: float, w_end: float, lam: float) -> np.ndarray:
    """Compute time-varying weight schedule using exponential interpolation.
    
    Formula: w(t) = w_end + (w_start - w_end) * exp(-lam * t)
    
    Args:
        t: Time array (seconds)
        w_start: Initial weight at t=0
        w_end: Final weight at t=infinity (for lam>0) or t=-infinity (for lam<0)
        lam: Decay/growth rate (1/seconds). 
             - If lam ≈ 0: returns constant w_start
             - If lam > 0: exponential decay from w_start toward w_end
             - If lam < 0: exponential growth from w_start toward w_end
    
    Returns:
        Array of weights, clipped to non-negative
    """
    if abs(lam) <= 1e-12:
        # Constant weight case (lambda near zero)
        return w_start * np.ones_like(t)
    else:
        w = w_end + (w_start - w_end) * np.exp(-lam * t)
        return np.clip(w, 0.0, None)


def build_weight_mats(N: int, dt: float, params: Dict[str, Any]) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Build diagonal weight matrices for error, acceleration, and jerk terms.
    
    Args:
        N: Number of time steps
        dt: Time step size
        params: Dictionary with keys:
            - 'wE_start', 'wE_end', 'lamE': error weights
            - 'wA_start', 'wA_end', 'lamA': acceleration weights
            - 'wJ_start', 'wJ_end', 'lamJ': jerk weights
    
    Returns:
        We: Diagonal matrix (N+1 x N+1) for speed error
        Wa: Diagonal matrix (N x N) for acceleration
        Wj: Diagonal matrix (N-1 x N-1) for jerk
    """
    # Time arrays for each term
    t_e = np.arange(N + 1) * dt  # Speed error at all N+1 points
    t_a = np.arange(N) * dt      # Acceleration at N intervals
    t_j = np.arange(N - 1) * dt  # Jerk at N-1 intervals
    
    # Compute weight schedules
    w_e = weight_schedule(t_e, params['wE_start'], params['wE_end'], params['lamE'])
    w_a = weight_schedule(t_a, params['wA_start'], params['wA_end'], params['lamA'])
    w_j = weight_schedule(t_j, params['wJ_start'], params['wJ_end'], params['lamJ'])
    
    # Create diagonal sparse matrices
    We = sp.diags(w_e, format='csr')
    Wa = sp.diags(w_a, format='csr')
    Wj = sp.diags(w_j, format='csr')
    
    return We, Wa, Wj


def build_qp_matrices(r: np.ndarray, dt: float, We: sp.csr_matrix, Wa: sp.csr_matrix, 
                      Wj: sp.csr_matrix, D1: sp.csr_matrix, D2: sp.csr_matrix) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Build QP Hessian and gradient vector.
    
    Objective: min (v-r)^T We (v-r) + (D1 v)^T Wa (D1 v) + (D2 v)^T Wj (D2 v)
    
    Standard form: min (1/2) v^T H v - f^T v
    
    Args:
        r: Raw target speed profile (N+1,)
        dt: Time step size
        We: Error weight matrix (N+1 x N+1)
        Wa: Acceleration weight matrix (N x N)
        Wj: Jerk weight matrix (N-1 x N-1)
        D1: Acceleration operator (N x N+1)
        D2: Jerk operator (N-1 x N+1)
    
    Returns:
        H: Hessian matrix for QP (N+1 x N+1)
        f: Gradient vector (N+1,)
    """
    # H = 2 * (We + D1^T Wa D1 + D2^T Wj D2)
    H = 2.0 * (We + D1.T @ Wa @ D1 + D2.T @ Wj @ D2)
    
    # f = 2 * We @ r
    f = 2.0 * (We @ r)
    
    return H.tocsr(), f


def build_constraints(N: int, dt: float, v_meas: float, a_meas: float, j_meas: float,
                      enforce_terminal: bool, rN: float) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Build linear equality constraint matrix and vector: A v = b.
    
    Constraints:
        1) v[0] = v_meas
        2) (v[1] - v[0])/dt = a_meas
        3) (v[2] - 2*v[1] + v[0])/dt^2 = j_meas
        4) v[N] = rN (optional)
    
    Args:
        N: Number of time steps
        dt: Time step size
        v_meas: Measured initial velocity
        a_meas: Measured initial acceleration
        j_meas: Measured initial jerk
        enforce_terminal: If True, add terminal constraint v[N] = rN
        rN: Terminal target speed (used if enforce_terminal=True)
    
    Returns:
        A: Constraint matrix (3 or 4 rows, N+1 columns)
        b: Constraint vector (3 or 4 elements)
    """
    if N < 2:
        raise ValueError(f"N must be >= 2 to enforce jerk constraint, got N={N}")
    
    num_constraints = 4 if enforce_terminal else 3
    
    # Build constraint matrix using COO format for easy construction
    rows = []
    cols = []
    data = []
    
    # Constraint 0: v[0] = v_meas
    rows.append(0)
    cols.append(0)
    data.append(1.0)
    
    # Constraint 1: (v[1] - v[0])/dt = a_meas
    # => -v[0]/dt + v[1]/dt = a_meas
    rows.extend([1, 1])
    cols.extend([0, 1])
    data.extend([-1.0/dt, 1.0/dt])
    
    # Constraint 2: (v[2] - 2*v[1] + v[0])/dt^2 = j_meas
    dt_sq = dt * dt
    rows.extend([2, 2, 2])
    cols.extend([0, 1, 2])
    data.extend([1.0/dt_sq, -2.0/dt_sq, 1.0/dt_sq])
    
    # Constraint 3 (optional): v[N] = rN
    if enforce_terminal:
        rows.append(3)
        cols.append(N)
        data.append(1.0)
    
    A = sp.coo_matrix((data, (rows, cols)), shape=(num_constraints, N+1)).tocsr()
    
    # Build constraint vector
    b = np.array([v_meas, a_meas, j_meas])
    if enforce_terminal:
        b = np.append(b, rN)
    
    return A, b


def build_ineq_constraints(D1: sp.csr_matrix, D2: sp.csr_matrix,
                           a_min: float, a_max: float, j_min: float, j_max: float,
                           enable_accel: bool = True, enable_jerk: bool = True) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Build inequality constraint matrix for acceleration and jerk box constraints.
    
    Converts box constraints to standard form G v <= h:
    - Acceleration: a_min <= D1 v <= a_max
    - Jerk: j_min <= D2 v <= j_max
    
    Standard form:
        D1 v <= a_max * ones
       -D1 v <= -a_min * ones  (equivalent to D1 v >= a_min)
        D2 v <= j_max * ones
       -D2 v <= -j_min * ones
    
    Args:
        D1: Acceleration operator (N x N+1)
        D2: Jerk operator (N-1 x N+1)
        a_min: Minimum acceleration (m/s²)
        a_max: Maximum acceleration (m/s²)
        j_min: Minimum jerk (m/s³)
        j_max: Maximum jerk (m/s³)
        enable_accel: If True, include acceleration constraints
        enable_jerk: If True, include jerk constraints
    
    Returns:
        G: Inequality constraint matrix (sparse CSR)
        h: Upper bound vector
    
    Raises:
        ValueError: If bounds are invalid (min >= max or non-finite)
    """
    # Validate bounds
    if enable_accel:
        if not (np.isfinite(a_min) and np.isfinite(a_max)):
            raise ValueError(f"Acceleration bounds must be finite: a_min={a_min}, a_max={a_max}")
        if a_min >= a_max:
            raise ValueError(f"Invalid acceleration bounds: a_min={a_min} >= a_max={a_max}")
    
    if enable_jerk:
        if not (np.isfinite(j_min) and np.isfinite(j_max)):
            raise ValueError(f"Jerk bounds must be finite: j_min={j_min}, j_max={j_max}")
        if j_min >= j_max:
            raise ValueError(f"Invalid jerk bounds: j_min={j_min} >= j_max={j_max}")
    
    # Build constraint blocks
    G_blocks = []
    h_blocks = []
    
    if enable_accel:
        N = D1.shape[0]
        # D1 v <= a_max
        G_blocks.append(D1)
        h_blocks.append(a_max * np.ones(N))
        # -D1 v <= -a_min
        G_blocks.append(-D1)
        h_blocks.append(-a_min * np.ones(N))
    
    if enable_jerk:
        N_jerk = D2.shape[0]
        # D2 v <= j_max
        G_blocks.append(D2)
        h_blocks.append(j_max * np.ones(N_jerk))
        # -D2 v <= -j_min
        G_blocks.append(-D2)
        h_blocks.append(-j_min * np.ones(N_jerk))
    
    if not G_blocks:
        # No constraints enabled, return empty
        n = D1.shape[1]
        return sp.csr_matrix((0, n)), np.array([])
    
    # Stack all blocks
    G = sp.vstack(G_blocks, format='csr')
    h = np.concatenate(h_blocks)
    
    return G, h


def solve_qp_osqp(H: sp.csr_matrix, f: np.ndarray, 
                  Aeq: sp.csr_matrix, beq: np.ndarray,
                  G: sp.csr_matrix, h: np.ndarray) -> np.ndarray:
    """Solve QP with equality and inequality constraints using OSQP.
    
    Solves:
        min (1/2) v^T H v - f^T v
        s.t. Aeq v = beq  (equality constraints)
             G v <= h     (inequality constraints)
    
    Maps to OSQP form: min 0.5 x^T P x + q^T x  s.t.  l <= A x <= u
    
    Args:
        H: Hessian matrix (n x n)
        f: Gradient vector (n,)
        Aeq: Equality constraint matrix (m_eq x n)
        beq: Equality constraint vector (m_eq,)
        G: Inequality constraint matrix (m_ineq x n)
        h: Inequality upper bounds (m_ineq,)
    
    Returns:
        v: Optimal solution (n,)
    
    Raises:
        RuntimeError: If OSQP fails to solve
    """
    import osqp
    
    n = H.shape[0]
    
    # Convert to OSQP format
    # P = H (must be CSC for OSQP)
    P = H.tocsc()
    
    # q = -f (OSQP uses +q^T x, we use -f^T v)
    q = -f
    
    # Stack equality and inequality constraints
    # Equalities: l = u = beq
    # Inequalities: l = -inf, u = h
    if G.shape[0] > 0:
        A_osqp = sp.vstack([Aeq, G], format='csc')
        l = np.concatenate([beq, -np.inf * np.ones(len(h))])
        u = np.concatenate([beq, h])
    else:
        # Only equality constraints
        A_osqp = Aeq.tocsc()
        l = beq
        u = beq
    
    # Create OSQP problem
    prob = osqp.OSQP()
    
    # Setup problem
    prob.setup(
        P=P, q=q, A=A_osqp, l=l, u=u,
        eps_abs=1e-9,
        eps_rel=1e-9,
        max_iter=20000,
        polish=True,
        verbose=False
    )
    
    # Solve
    result = prob.solve()
    
    # Check status - accept both 'solved' and 'solved inaccurate'
    # ('solved inaccurate' can happen for large problems but is often acceptable)
    if result.info.status not in ['solved', 'solved inaccurate']:
        raise RuntimeError(f"OSQP failed to solve: status = {result.info.status}")
    
    return result.x


def validate_bounds_feasibility(a_meas: float, j_meas: float,
                                a_min: float, a_max: float,
                                j_min: float, j_max: float,
                                enable_accel: bool, enable_jerk: bool) -> None:
    """Validate that initial conditions are compatible with box constraints.
    
    Args:
        a_meas: Measured initial acceleration
        j_meas: Measured initial jerk
        a_min, a_max: Acceleration bounds
        j_min, j_max: Jerk bounds
        enable_accel: If True, check acceleration bounds
        enable_jerk: If True, check jerk bounds
    
    Raises:
        ValueError: If initial conditions violate enabled bounds
    """
    if enable_accel:
        if not (a_min <= a_meas <= a_max):
            raise ValueError(
                f"Initial acceleration a_meas={a_meas:.3f} violates bounds "
                f"[{a_min:.3f}, {a_max:.3f}]. "
                f"Cannot satisfy hard initial condition v'(0)=a_meas with enabled acceleration bounds."
            )
    
    if enable_jerk:
        if not (j_min <= j_meas <= j_max):
            raise ValueError(
                f"Initial jerk j_meas={j_meas:.3f} violates bounds "
                f"[{j_min:.3f}, {j_max:.3f}]. "
                f"Cannot satisfy hard initial condition v''(0)=j_meas with enabled jerk bounds."
            )


def solve_kkt(H: sp.csr_matrix, f: np.ndarray, A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
    """Solve equality-constrained QP via KKT system.
    
    Solves:
        min (1/2) v^T H v - f^T v
        s.t. A v = b
    
    KKT system:
        [H   A^T] [v  ]   [f]
        [A    0 ] [nu ] = [b]
    
    Args:
        H: Hessian matrix (n x n)
        f: Gradient vector (n,)
        A: Constraint matrix (m x n)
        b: Constraint vector (m,)
    
    Returns:
        v: Primal solution (n,)
    """
    n = H.shape[0]
    m = A.shape[0]
    
    # Add small regularization for numerical stability
    H_reg = H + 1e-9 * sp.eye(n, format='csr')
    
    # Build KKT matrix
    # Top row: [H_reg, A^T]
    # Bottom row: [A, 0]
    zero_block = sp.csr_matrix((m, m))
    KKT = sp.bmat([[H_reg, A.T], [A, zero_block]], format='csr')
    
    # Build RHS
    rhs = np.concatenate([f, b])
    
    # Solve sparse linear system
    try:
        solution = spsolve(KKT, rhs)
    except Exception as e:
        raise RuntimeError(f"KKT solve failed: {e}")
    
    # Extract primal solution (first n components)
    v = solution[:n]
    
    return v


def shape_speed_profile(r: np.ndarray, dt: float, meas: Dict[str, float], 
                        weight_params: Dict[str, float], enforce_terminal: bool = False,
                        enable_accel_bounds: bool = False, a_min: float = -np.inf, a_max: float = np.inf,
                        enable_jerk_bounds: bool = False, j_min: float = -np.inf, j_max: float = np.inf) -> np.ndarray:
    """Orchestrate QP-based speed profile shaping with hard initial derivative constraints.
    
    Args:
        r: Raw target speed profile (N+1,)
        dt: Time step size (seconds)
        meas: Dictionary with 'v_meas', 'a_meas', 'j_meas' initial conditions
        weight_params: Dictionary with weight schedule parameters:
            'wE_start', 'wE_end', 'lamE',
            'wA_start', 'wA_end', 'lamA',
            'wJ_start', 'wJ_end', 'lamJ'
        enforce_terminal: If True, enforce v[N] = r[N]
        enable_accel_bounds: If True, enforce acceleration box constraints
        a_min: Minimum acceleration (m/s²), only used if enable_accel_bounds=True
        a_max: Maximum acceleration (m/s²), only used if enable_accel_bounds=True
        enable_jerk_bounds: If True, enforce jerk box constraints
        j_min: Minimum jerk (m/s³), only used if enable_jerk_bounds=True
        j_max: Maximum jerk (m/s³), only used if enable_jerk_bounds=True
    
    Returns:
        v: Shaped speed profile (N+1,)
    
    Raises:
        ValueError: If bounds are incompatible with initial conditions
    """
    N = len(r) - 1
    
    if N < 2:
        raise ValueError(f"Profile too short: need at least 3 points (N >= 2), got {N+1} points")
    
    # Build differential operators
    D1, D2 = build_D1_D2(N, dt)
    
    # Build weight matrices
    We, Wa, Wj = build_weight_mats(N, dt, weight_params)
    
    # Build QP matrices
    H, f = build_qp_matrices(r, dt, We, Wa, Wj, D1, D2)
    
    # Build equality constraints
    Aeq, beq = build_constraints(
        N, dt, 
        meas['v_meas'], meas['a_meas'], meas['j_meas'],
        enforce_terminal, r[N]
    )
    
    # Check if inequality constraints are enabled
    use_osqp = enable_accel_bounds or enable_jerk_bounds
    
    if use_osqp:
        # Validate feasibility of bounds with initial conditions
        validate_bounds_feasibility(
            meas['a_meas'], meas['j_meas'],
            a_min, a_max, j_min, j_max,
            enable_accel_bounds, enable_jerk_bounds
        )
        
        # Build inequality constraints
        G, h = build_ineq_constraints(
            D1, D2, a_min, a_max, j_min, j_max,
            enable_accel_bounds, enable_jerk_bounds
        )
        
        # Solve with OSQP
        v = solve_qp_osqp(H, f, Aeq, beq, G, h)
    else:
        # Use existing KKT solver (backward compatible)
        v = solve_kkt(H, f, Aeq, beq)
    
    return v
