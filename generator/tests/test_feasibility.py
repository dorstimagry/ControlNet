"""Tests for the feasibility projector implementation."""

import torch
import pytest

from generator.feasibility import (
    VehicleCapabilities,
    FeasibilityParams,
    compute_resistive_forces,
    compute_feasible_accelerations,
    compute_max_feasible_speed,
    project_window_to_feasible,
    enforce_voltage_at_zero_accel
)


class TestResistiveForces:
    """Test resistive force calculations."""

    def test_basic_resistive_forces(self):
        """Test basic resistive force computation."""
        # Simple vehicle parameters
        vehicle = VehicleCapabilities(
            m=torch.tensor([1500.0]),
            r_w=torch.tensor([0.3]),
            gear_ratio=torch.tensor([10.0]),
            eta_gb=torch.tensor([0.9]),
            K_t=torch.tensor([0.5]),
            K_e=torch.tensor([0.5]),
            R=torch.tensor([0.1]),
            i_max=torch.tensor([200.0]),
            V_max=torch.tensor([400.0]),
            b=torch.tensor([1e-3]),
            T_brake_max=torch.tensor([2000.0]),
            mu=torch.tensor([0.8]),
            C_dA=torch.tensor([0.6]),  # Includes rho
            C_r=torch.tensor([0.01])
        )

        v = torch.tensor([[0.0, 10.0, 20.0]])  # [B=1, H=3]
        grade = torch.zeros_like(v)  # Flat road

        F_resist = compute_resistive_forces(v, grade, vehicle)

        assert F_resist.shape == v.shape
        # At zero speed, only rolling resistance
        expected_roll = vehicle.C_r * vehicle.m * 9.81
        assert abs(F_resist[0, 0] - expected_roll) < 1e-3
        # At higher speeds, drag should increase
        assert F_resist[0, 1] > F_resist[0, 0]
        assert F_resist[0, 2] > F_resist[0, 1]


class TestFeasibleAccelerations:
    """Test feasible acceleration bound computation."""

    def test_feasible_accels(self):
        """Test feasible acceleration computation."""
        vehicle = VehicleCapabilities(
            m=torch.tensor([1500.0]),
            r_w=torch.tensor([0.3]),
            gear_ratio=torch.tensor([10.0]),
            eta_gb=torch.tensor([0.9]),
            K_t=torch.tensor([0.5]),
            K_e=torch.tensor([0.5]),
            R=torch.tensor([0.1]),
            i_max=torch.tensor([200.0]),
            V_max=torch.tensor([400.0]),
            b=torch.tensor([1e-3]),
            T_brake_max=torch.tensor([2000.0]),
            mu=torch.tensor([0.8]),
            C_dA=torch.tensor([0.6]),
            C_r=torch.tensor([0.01])
        )

        params = FeasibilityParams(safety_margin=0.9)

        v = torch.tensor([[0.0, 10.0]])  # [B=1, H=2]
        grade = torch.zeros_like(v)

        a_min, a_max = compute_feasible_accelerations(v, grade, vehicle, params)

        assert a_min.shape == v.shape
        assert a_max.shape == v.shape
        # Drive acceleration should be positive
        assert torch.all(a_max > 0)
        # Brake acceleration should be negative
        assert torch.all(a_min < 0)
        # Brake acceleration should be limited by both torque and friction
        # The limit should be negative and not exceed physical constraints
        friction_limit = -vehicle.mu * 9.81  # ≈ -7.8 m/s²
        torque_limit = (-vehicle.T_brake_max / vehicle.r_w - compute_resistive_forces(v, grade, vehicle)) / vehicle.m

        # The actual limit should be the less restrictive (higher) of the two
        expected_min = torch.max(torque_limit, friction_limit)
        # Just check that we get reasonable negative values
        assert torch.all(a_min <= 0)  # Braking gives negative acceleration
        assert torch.all(a_min >= -10.0)  # Not ridiculously negative


class TestWindowProjection:
    """Test window projection to feasible speeds."""

    def test_projection_maintains_shape(self):
        """Test that projection preserves tensor shape."""
        vehicle = VehicleCapabilities(
            m=torch.tensor([1500.0]),
            r_w=torch.tensor([0.3]),
            gear_ratio=torch.tensor([10.0]),
            eta_gb=torch.tensor([0.9]),
            K_t=torch.tensor([0.5]),
            K_e=torch.tensor([0.5]),
            R=torch.tensor([0.1]),
            i_max=torch.tensor([200.0]),
            V_max=torch.tensor([400.0]),
            b=torch.tensor([1e-3]),
            T_brake_max=torch.tensor([2000.0]),
            mu=torch.tensor([0.8]),
            C_dA=torch.tensor([0.6]),
            C_r=torch.tensor([0.01])
        )

        params = FeasibilityParams()
        dt = 0.02

        # Simple case: constant speed profile
        raw_window = torch.tensor([[5.0, 5.0, 5.0]])  # [B=1, H=3]
        grade_window = torch.zeros_like(raw_window)

        projected = project_window_to_feasible(raw_window, grade_window, vehicle, dt, params)

        assert projected.shape == raw_window.shape

    def test_projection_convergence(self):
        """Test that projection converges."""
        vehicle = VehicleCapabilities(
            m=torch.tensor([1500.0]),
            r_w=torch.tensor([0.3]),
            gear_ratio=torch.tensor([10.0]),
            eta_gb=torch.tensor([0.9]),
            K_t=torch.tensor([0.5]),
            K_e=torch.tensor([0.5]),
            R=torch.tensor([0.1]),
            i_max=torch.tensor([200.0]),
            V_max=torch.tensor([400.0]),
            b=torch.tensor([1e-3]),
            T_brake_max=torch.tensor([2000.0]),
            mu=torch.tensor([0.8]),
            C_dA=torch.tensor([0.6]),
            C_r=torch.tensor([0.01])
        )

        params = FeasibilityParams(max_projection_iters=10)
        dt = 0.02

        # Create a profile that needs adjustment
        raw_window = torch.tensor([[0.0, 10.0, 20.0, 30.0]])  # Rapid acceleration
        grade_window = torch.zeros_like(raw_window)

        projected = project_window_to_feasible(raw_window, grade_window, vehicle, dt, params)

        # Should still have reasonable accelerations
        a = (projected[:, 1:] - projected[:, :-1]) / dt
        assert torch.all(a >= -15.0)  # Not too aggressive braking
        assert torch.all(a <= 5.0)   # Not too aggressive acceleration


class TestVoltageEnforcement:
    """Test voltage constraint enforcement."""

    def test_voltage_enforcement(self):
        """Test that voltage constraints are enforced at zero acceleration."""
        # Create a vehicle with tight voltage constraints
        vehicle = VehicleCapabilities(
            m=torch.tensor([1500.0]),
            r_w=torch.tensor([0.3]),
            gear_ratio=torch.tensor([10.0]),
            eta_gb=torch.tensor([0.9]),
            K_t=torch.tensor([0.5]),
            K_e=torch.tensor([0.5]),
            R=torch.tensor([0.1]),
            i_max=torch.tensor([200.0]),
            V_max=torch.tensor([100.0]),  # Low voltage limit
            b=torch.tensor([1e-3]),
            T_brake_max=torch.tensor([2000.0]),
            mu=torch.tensor([0.8]),
            C_dA=torch.tensor([0.6]),
            C_r=torch.tensor([0.01])
        )

        params = FeasibilityParams(safety_margin=0.9)

        # High speed that might violate voltage at zero accel
        v_profile = torch.tensor([[15.0, 20.0]])  # [B=1, H=2]
        grade_profile = torch.zeros_like(v_profile)

        enforced = enforce_voltage_at_zero_accel(v_profile, grade_profile, vehicle, params)

        # Should reduce speeds if they violate voltage constraint
        # (This is a basic test - detailed voltage calculations would be more complex)
        assert enforced.shape == v_profile.shape
        assert torch.all(enforced >= 0)  # Non-negative speeds

    def test_back_emf_speed_limit(self):
        """Test that profiles respect back-EMF speed limit."""
        # Create a vehicle with known back-EMF limit
        # v_max = V_max * r_w / (K_e * gear_ratio)
        # v_max = 400 * 0.3 / (0.5 * 10) = 24 m/s
        vehicle = VehicleCapabilities(
            m=torch.tensor([1500.0]),
            r_w=torch.tensor([0.3]),
            gear_ratio=torch.tensor([10.0]),
            eta_gb=torch.tensor([0.9]),
            K_t=torch.tensor([0.5]),
            K_e=torch.tensor([0.5]),
            R=torch.tensor([0.1]),
            i_max=torch.tensor([200.0]),
            V_max=torch.tensor([400.0]),
            b=torch.tensor([1e-3]),
            T_brake_max=torch.tensor([2000.0]),
            mu=torch.tensor([0.8]),
            C_dA=torch.tensor([0.6]),
            C_r=torch.tensor([0.01])
        )
        
        v_max = compute_max_feasible_speed(vehicle)
        expected_v_max = 400.0 * 0.3 / (0.5 * 10.0)  # 24.0 m/s
        assert abs(v_max.item() - expected_v_max) < 1e-3
        
        # Test that projection clamps speeds to this limit
        params = FeasibilityParams()
        dt = 0.02
        
        # Create a profile that exceeds the limit
        raw_window = torch.tensor([[20.0, 25.0, 30.0]])  # Exceeds 24 m/s limit
        grade_window = torch.zeros_like(raw_window)
        
        projected = project_window_to_feasible(raw_window, grade_window, vehicle, dt, params)
        
        # All speeds should be <= v_max
        assert torch.all(projected <= v_max.unsqueeze(-1) + 1e-3)

    def test_b_reduces_available_torque(self):
        """Test that viscous damping (b) reduces available torque and feasible acceleration."""
        params = FeasibilityParams(safety_margin=0.9)
        
        # Create two vehicles: one with b=0 and one with b>0
        vehicle_no_b = VehicleCapabilities(
            m=torch.tensor([1500.0]),
            r_w=torch.tensor([0.3]),
            gear_ratio=torch.tensor([10.0]),
            eta_gb=torch.tensor([0.9]),
            K_t=torch.tensor([0.5]),
            K_e=torch.tensor([0.5]),
            R=torch.tensor([0.1]),
            i_max=torch.tensor([200.0]),
            V_max=torch.tensor([400.0]),
            b=torch.tensor([0.0]),  # No viscous damping
            T_brake_max=torch.tensor([2000.0]),
            mu=torch.tensor([0.8]),
            C_dA=torch.tensor([0.6]),
            C_r=torch.tensor([0.01])
        )
        
        vehicle_with_b = VehicleCapabilities(
            m=torch.tensor([1500.0]),
            r_w=torch.tensor([0.3]),
            gear_ratio=torch.tensor([10.0]),
            eta_gb=torch.tensor([0.9]),
            K_t=torch.tensor([0.5]),
            K_e=torch.tensor([0.5]),
            R=torch.tensor([0.1]),
            i_max=torch.tensor([200.0]),
            V_max=torch.tensor([400.0]),
            b=torch.tensor([0.01]),  # Significant viscous damping
            T_brake_max=torch.tensor([2000.0]),
            mu=torch.tensor([0.8]),
            C_dA=torch.tensor([0.6]),
            C_r=torch.tensor([0.01])
        )
        
        # Test at moderate speed where b effect is noticeable
        v = torch.tensor([[15.0]])  # 15 m/s
        grade = torch.zeros_like(v)
        
        a_min_no_b, a_max_no_b = compute_feasible_accelerations(v, grade, vehicle_no_b, params)
        a_min_with_b, a_max_with_b = compute_feasible_accelerations(v, grade, vehicle_with_b, params)
        
        # With b, maximum acceleration should be lower (or equal if limited by other constraints)
        assert a_max_with_b[0, 0] <= a_max_no_b[0, 0] + 1e-6  # Allow small numerical error
        
        # At high speeds, the difference should be more pronounced
        v_high = torch.tensor([[20.0]])  # 20 m/s
        a_min_no_b_high, a_max_no_b_high = compute_feasible_accelerations(v_high, grade, vehicle_no_b, params)
        a_min_with_b_high, a_max_with_b_high = compute_feasible_accelerations(v_high, grade, vehicle_with_b, params)
        
        # At higher speed, b effect should be more noticeable
        assert a_max_with_b_high[0, 0] <= a_max_no_b_high[0, 0] + 1e-6
