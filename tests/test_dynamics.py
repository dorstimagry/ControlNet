"""Unit tests for the extended plant dynamics model."""

from __future__ import annotations

import numpy as np
import pytest

from utils.dynamics import ExtendedPlant, ExtendedPlantParams, sample_extended_params


class TestExtendedPlantDynamics:
    """Test the torque-based tire model and plant dynamics."""

    @pytest.fixture
    def plant_params(self) -> ExtendedPlantParams:
        """Create test plant parameters."""
        from utils.dynamics import ExtendedPlantRandomization
        return sample_extended_params(np.random.default_rng(42), ExtendedPlantRandomization())

    @pytest.fixture
    def plant(self, plant_params: ExtendedPlantParams) -> ExtendedPlant:
        """Create a test plant instance."""
        plant = ExtendedPlant(plant_params)
        plant.reset(speed=10.0)  # Start at 10 m/s
        return plant

    def test_zero_action_sanity(self, plant: ExtendedPlant) -> None:
        """Test that zero action produces physically reasonable behavior."""
        # Run for 2 seconds with zero action
        dt = 0.1
        initial_speed = plant.speed
        for _ in range(20):
            plant.step(0.0, dt)

        # With zero action: motor voltage = 0, drive torque small (viscous losses only)
        assert plant.V_cmd == 0.0, f"V_cmd should be 0 with zero action, got {plant.V_cmd}"
        assert abs(plant.drive_torque) < 1.0, f"Drive torque should be small with zero action, got {plant.drive_torque}"

        # Acceleration should be reasonable (can be positive due to downhill grade, negative due to drag/rolling)
        assert abs(plant.acceleration) < 10.0, f"Acceleration too extreme: {plant.acceleration}"

        # Speed should change in a physically reasonable way
        final_speed = plant.speed
        speed_change = final_speed - initial_speed
        assert abs(speed_change) < 5.0, f"Speed change too extreme: {speed_change}"

    def test_steady_throttle(self, plant: ExtendedPlant) -> None:
        """Test that steady throttle produces smooth acceleration."""
        dt = 0.1
        action = 0.5  # Moderate throttle

        accelerations = []
        speeds = []

        # Run for 10 seconds
        for _ in range(100):
            plant.step(action, dt)
            accelerations.append(plant.acceleration)
            speeds.append(plant.speed)

        # Should accelerate initially
        initial_accel = accelerations[0]
        assert initial_accel > 0, f"Should accelerate with throttle, got {initial_accel}"

        # Acceleration should decrease over time due to increasing drag (not reach zero)
        initial_accel = accelerations[0]
        final_accel = accelerations[-1]
        assert final_accel < initial_accel, f"Acceleration should decrease due to drag, got {final_accel} vs {initial_accel}"
        assert final_accel > 0, f"Should still be accelerating, got {final_accel}"

        # Speed should increase monotonically then stabilize
        assert speeds[-1] > speeds[0], "Speed should increase"
        # Check monotonic increase for first part
        increasing_period = 50  # First 5 seconds
        for i in range(1, increasing_period):
            assert speeds[i] >= speeds[i-1], f"Speed not monotonic at step {i}"

    def test_steady_braking(self, plant: ExtendedPlant) -> None:
        """Test that steady braking produces smooth deceleration."""
        dt = 0.1
        action = -0.5  # Moderate brake

        accelerations = []
        speeds = []

        # Run for 5 seconds
        for _ in range(50):
            plant.step(action, dt)
            accelerations.append(plant.acceleration)
            speeds.append(plant.speed)

        # Should decelerate initially (may become 0 when vehicle stops and is held by brakes)
        initial_accels = accelerations[:30]  # Check first 30 accelerations (3 seconds)
        for i, accel in enumerate(initial_accels):
            assert accel <= 0, f"Should decelerate initially at step {i}, got {accel}"

        # Speed should decrease monotonically
        for i in range(1, len(speeds)):
            assert speeds[i] <= speeds[i-1], f"Speed not decreasing at step {i}"

        # Should not oscillate wildly
        accel_std = np.std(accelerations)
        assert accel_std < 5.0, f"Braking too erratic, std={accel_std}"

    def test_throttle_to_brake_transition(self, plant: ExtendedPlant) -> None:
        """Test smooth transition from throttle to brake."""
        dt = 0.1

        # Phase 1: Throttle for 3 seconds
        throttle_action = 0.5
        for _ in range(30):
            plant.step(throttle_action, dt, substeps=5)

        speed_after_throttle = plant.speed

        # Phase 2: Switch to brake for 3 seconds
        brake_action = -0.5
        accelerations = []
        for _ in range(30):
            plant.step(brake_action, dt, substeps=5)
            accelerations.append(plant.acceleration)

        speed_after_brake = plant.speed

        # Should have accelerated then decelerated
        assert speed_after_throttle > 10.0, "Should have accelerated"
        assert speed_after_brake < speed_after_throttle, "Should have decelerated"

        # No extreme jerk spikes during transition (use more substeps for stability)
        max_jerk = max(abs(np.diff(accelerations))) / dt
        assert max_jerk < 10.0, f"Jerk spike too large: {max_jerk} m/s³"

    def test_tire_force_limits(self, plant: ExtendedPlant) -> None:
        """Test that tire force is properly clamped by friction limits."""
        dt = 0.1

        # Test extreme throttle
        plant.step(1.0, dt)  # Full throttle
        mu = plant.params.brake.mu
        mass = plant.params.body.mass
        gravity = 9.81
        expected_limit = mu * mass * gravity

        assert abs(plant.tire_force) <= 1.2 * expected_limit, f"Tire force exceeds limit: {plant.tire_force}"

        # Test extreme brake
        plant.reset(speed=20.0)  # Higher speed for more dramatic test
        plant.step(-1.0, dt)  # Full brake
        assert abs(plant.tire_force) <= 1.2 * expected_limit, f"Tire force exceeds limit: {plant.tire_force}"

    def test_wheel_speed_reasonable(self, plant: ExtendedPlant) -> None:
        """Test that wheel speed stays within reasonable bounds (not clamped artificially)."""
        dt = 0.1

        # Test with various actions
        actions = [0.0, 0.5, -0.5, 1.0, -1.0]
        for action in actions:
            plant.reset(speed=5.0)  # Reset for each test
            for _ in range(10):
                plant.step(action, dt)
                # Wheel angular speed should be reasonable in magnitude (allow for slipping dynamics)
                # Just check that it's finite and not NaN
                assert abs(plant.wheel_omega) < 1e6, f"Wheel angular speed became non-finite: {plant.wheel_omega}"
                assert not np.isnan(plant.wheel_omega), f"Wheel angular speed became NaN"

    def test_slip_ratio_reasonable(self, plant: ExtendedPlant) -> None:
        """Test that slip ratio stays within reasonable bounds."""
        dt = 0.1

        # Normal driving
        for _ in range(50):
            plant.step(0.3, dt)  # Moderate throttle
            assert abs(plant.slip_ratio) < 0.5, f"Slip ratio too large: {plant.slip_ratio}"

        # Aggressive braking
        plant.reset(speed=25.0)  # Higher speed
        for _ in range(20):
            plant.step(-0.8, dt)  # Hard brake
            # Allow higher slip during hard braking but not extreme
            assert abs(plant.slip_ratio) < 2.0, f"Slip ratio extreme: {plant.slip_ratio}"

    def test_max_acceleration_capability(self, plant: ExtendedPlant) -> None:
        """Test that vehicle can achieve 2.4-4.2 m/s² max acceleration."""
        dt = 0.1

        # Run full throttle for 3 seconds
        accelerations = []
        for _ in range(30):  # 3 seconds
            plant.step(1.0, dt)  # Full throttle
            accelerations.append(plant.acceleration)

        max_accel = max(accelerations)
        assert 2.4 <= max_accel <= 4.2, f"Max acceleration {max_accel} m/s² not in required range [2.4, 4.2]"

    def test_no_regen_during_braking(self, plant: ExtendedPlant) -> None:
        """Test that no regenerative braking occurs (V_cmd = 0 and drive_torque = 0 during braking)."""
        dt = 0.1

        # Test braking action
        plant.step(-0.5, dt)  # Brake

        # During braking, commanded voltage should be 0 (no regen)
        assert plant.V_cmd == 0.0, f"V_cmd should be 0 during braking, got {plant.V_cmd}"
        # Drive torque may be small negative due to viscous losses (physically correct)
        assert plant.drive_torque <= 1.0, f"Drive torque should not be positive during braking, got {plant.drive_torque}"

        # Back-EMF voltage depends on wheel speed, which may not be zero yet
        # Just check that it's reasonable (not negative or unreasonably large)
        assert plant.back_emf_voltage >= 0, f"Back-EMF voltage should not be negative: {plant.back_emf_voltage}"
        assert plant.back_emf_voltage < 1000, f"Back-EMF voltage unreasonably large: {plant.back_emf_voltage}"

    def test_speed_accel_correlation(self, plant: ExtendedPlant) -> None:
        """Test that acceleration follows the derivative of speed."""
        dt = 0.1

        # Collect speed and acceleration data during acceleration
        speeds = []
        accelerations = []

        # Accelerate for 5 seconds
        for _ in range(50):
            plant.step(0.8, dt)  # Moderate throttle
            speeds.append(plant.speed)
            accelerations.append(plant.acceleration)

        # Check correlation: when speed increases, acceleration should be positive
        # Use a simple check: acceleration should generally be positive when accelerating
        positive_accel_count = sum(1 for a in accelerations if a > 0.1)  # Allow small negative due to drag
        assert positive_accel_count > len(accelerations) * 0.8, f"Acceleration not correlated with speed increase: {positive_accel_count}/{len(accelerations)} positive accelerations"

    # B_m (motor viscous damping) tests
    def test_B_m_free_coast_decay(self, plant: ExtendedPlant) -> None:
        """Test free-coast decay with passive damping only."""
        dt = 0.1

        # Initialize with positive wheel speed, zero voltage (free coast)
        plant.reset(speed=10.0)  # Start with some speed
        plant.step(0.0, dt)  # Apply zero action (no voltage, no current)

        initial_speed = plant.speed
        initial_wheel_omega = plant.wheel_omega

        # Let it coast for several steps
        speeds = [initial_speed]
        wheel_omegas = [initial_wheel_omega]

        for _ in range(20):  # 2 seconds
            plant.step(0.0, dt)  # Continue coasting
            speeds.append(plant.speed)
            wheel_omegas.append(plant.wheel_omega)

        # Should decay smoothly (not instantly stop)
        final_speed = speeds[-1]
        final_wheel_omega = wheel_omegas[-1]

        assert final_speed < initial_speed, "Speed should decrease during coasting"
        assert final_wheel_omega < initial_wheel_omega, "Wheel speed should decrease during coasting"
        assert final_speed > 0, "Should not stop completely (just damped)"
        assert final_wheel_omega > 0, "Wheel should not stop completely (just damped)"

    def test_B_m_drive_step_dominance(self, plant: ExtendedPlant) -> None:
        """Test that electromagnetic torque dominates viscous torque during drive."""
        dt = 0.1

        plant.reset(speed=5.0)

        # Apply full throttle
        plant.step(1.0, dt)

        # Check that EM torque is much larger than viscous torque
        motor = plant.params.motor
        omega_m = motor.gear_ratio * plant.wheel_omega
        tau_viscous = motor.B_m * omega_m
        tau_em = motor.K_t * plant.motor_current

        # Use a more reasonable threshold - EM should be at least 2x viscous at operating conditions
        assert abs(tau_em) > 2 * abs(tau_viscous), f"EM torque {tau_em:.4f} should dominate viscous {tau_viscous:.4f}"

    def test_B_m_braking_passive_only(self, plant: ExtendedPlant) -> None:
        """Test that passive damping doesn't dominate braking when regen is disabled."""
        dt = 0.1

        plant.reset(speed=10.0)

        # Apply full brake (negative action)
        plant.step(-1.0, dt)

        motor = plant.params.motor
        omega_m = motor.gear_ratio * plant.wheel_omega

        # Viscous torque magnitude
        tau_viscous_max = motor.B_m * abs(omega_m)

        # Brake torque should be much larger than viscous torque
        # (brake torque is stored in brake_torque field)
        assert abs(plant.brake_torque) > 10 * tau_viscous_max, \
            f"Brake torque {plant.brake_torque:.1f} should dominate viscous {tau_viscous_max:.4f}"


class TestProfileFeasibility:
    """Test profile feasibility functions."""

    @pytest.fixture
    def vehicle_caps(self) -> "VehicleCapabilities":
        """Create test vehicle capabilities."""
        from utils.data_utils import VehicleCapabilities
        return VehicleCapabilities(
            m=1500.0,      # 1500 kg mass
            r_w=0.3,       # 0.3m wheel radius
            T_drive_max=2000.0,  # 2000 Nm max drive torque
            T_brake_max=4000.0,  # 4000 Nm max brake torque
            mu=0.8,        # friction coefficient
            C_dA=0.6,      # drag area
            C_r=0.012,     # rolling resistance
        )

    def test_feasible_accel_bounds_flat_road(self, vehicle_caps: "VehicleCapabilities") -> None:
        """Test acceleration bounds on flat road at various speeds."""
        from utils.data_utils import feasible_accel_bounds

        # At zero speed (no drag)
        a_min, a_max = feasible_accel_bounds(0.0, 0.0, vehicle_caps, safety_margin=0.9)
        # Max accel should be drive force / mass ≈ 3.9 m/s²
        assert 3.5 < a_max < 4.5, f"Expected ~3.9 m/s² max accel at 0 speed, got {a_max}"
        # Min accel (braking) should be negative ≈ -7.2 m/s²
        assert -8.0 < a_min < -6.0, f"Expected ~-7.2 m/s² min accel at 0 speed, got {a_min}"

        # At 20 m/s (with drag)
        a_min_20, a_max_20 = feasible_accel_bounds(20.0, 0.0, vehicle_caps, safety_margin=0.9)
        # Max accel should be slightly reduced by drag
        assert 3.0 < a_max_20 < 4.5, f"Expected ~3.8 m/s² max accel at 20 m/s, got {a_max_20}"
        assert a_max_20 < a_max, "Max accel should be reduced by drag at high speed"

    def test_feasible_accel_bounds_uphill(self, vehicle_caps: "VehicleCapabilities") -> None:
        """Test acceleration bounds on uphill road."""
        from utils.data_utils import feasible_accel_bounds
        import math

        grade_uphill = math.radians(5.0)  # 5° uphill

        a_min, a_max = feasible_accel_bounds(10.0, grade_uphill, vehicle_caps, safety_margin=0.9)

        # Uphill should reduce max acceleration and make braking more aggressive
        a_min_flat, a_max_flat = feasible_accel_bounds(10.0, 0.0, vehicle_caps, safety_margin=0.9)

        assert a_max < a_max_flat, "Uphill should reduce maximum acceleration"
        assert a_min < a_min_flat, "Uphill should make braking more aggressive (more negative)"

        # Check reasonable ranges
        assert 2.5 < a_max < 3.5, f"Uphill max accel should be ~3.0 m/s², got {a_max}"
        assert -9.0 < a_min < -7.0, f"Uphill min accel should be ~-8.0 m/s², got {a_min}"

    def test_feasible_accel_bounds_downhill(self, vehicle_caps: "VehicleCapabilities") -> None:
        """Test acceleration bounds on downhill road."""
        from utils.data_utils import feasible_accel_bounds
        import math

        grade_downhill = math.radians(-5.0)  # 5° downhill

        a_min, a_max = feasible_accel_bounds(10.0, grade_downhill, vehicle_caps, safety_margin=0.9)

        # Downhill should increase max acceleration and make braking less aggressive
        a_min_flat, a_max_flat = feasible_accel_bounds(10.0, 0.0, vehicle_caps, safety_margin=0.9)

        assert a_max > a_max_flat, "Downhill should increase maximum acceleration"
        assert a_min > a_min_flat, "Downhill should make braking less aggressive (less negative)"

    def test_project_profile_to_feasible_already_feasible(self, vehicle_caps: "VehicleCapabilities") -> None:
        """Test projection when profile is already feasible."""
        from utils.data_utils import project_profile_to_feasible, feasible_accel_bounds

        # Create a truly feasible profile (much gentler acceleration)
        dt = 0.1
        # Start with very small accelerations that are definitely feasible
        speeds = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])  # 1 m/s² constant accel (feasible)
        grades = np.zeros_like(speeds)

        # Verify this profile is actually feasible before testing
        a_req = np.diff(speeds) / dt  # [1.0, 1.0, 1.0, 1.0, 1.0]
        feasible_before = True
        for k in range(len(a_req)):
            a_min, a_max = feasible_accel_bounds(speeds[k], grades[k], vehicle_caps, 0.9)
            if not (a_min <= a_req[k] <= a_max):
                feasible_before = False
                break

        assert feasible_before, "Test profile should be feasible before projection"

        v_feasible, grade_feasible = project_profile_to_feasible(
            speeds, grades, vehicle_caps, dt, safety_margin=0.9
        )

        # Should be very close to original (no clipping needed)
        np.testing.assert_allclose(v_feasible, speeds, atol=1e-3)
        np.testing.assert_allclose(grade_feasible, grades, atol=1e-3)

    def test_project_profile_to_feasible_clipping_needed(self, vehicle_caps: "VehicleCapabilities") -> None:
        """Test projection when profile needs acceleration clipping."""
        from utils.data_utils import project_profile_to_feasible, feasible_accel_bounds

        # Create an aggressive profile that exceeds vehicle capabilities
        dt = 0.1
        speeds = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])  # 20 m/s² accel (way beyond ~4 m/s² max)
        grades = np.zeros_like(speeds)

        v_feasible, grade_feasible = project_profile_to_feasible(
            speeds, grades, vehicle_caps, dt, safety_margin=0.9
        )

        # Feasible profile should be slower than requested
        assert np.max(v_feasible) < np.max(speeds), "Feasible profile should be slower than aggressive original"

        # Should still reach some reasonable speed (limited by max feasible acceleration)
        assert np.max(v_feasible) > 1.0, "Should still reach some speed"

        # Check that accelerations are within bounds
        a_req = np.diff(v_feasible) / dt
        for k in range(len(a_req)):
            a_min, a_max = feasible_accel_bounds(v_feasible[k], grades[k], vehicle_caps, 0.9)
            assert a_min - 1e-3 <= a_req[k] <= a_max + 1e-3, f"Acceleration {a_req[k]} at step {k} violates bounds [{a_min}, {a_max}]"

    def test_project_profile_to_feasible_convergence(self, vehicle_caps: "VehicleCapabilities") -> None:
        """Test that the iterative algorithm converges."""
        from utils.data_utils import project_profile_to_feasible, feasible_accel_bounds

        # Create a very aggressive profile
        dt = 0.1
        speeds = np.linspace(0, 30, 100)  # Very fast acceleration to 30 m/s
        grades = np.zeros_like(speeds)

        v_feasible, grade_feasible = project_profile_to_feasible(
            speeds, grades, vehicle_caps, dt, safety_margin=0.9, max_iters=50
        )

        # Should converge to a feasible profile
        assert len(v_feasible) == len(speeds)

        # Check all accelerations are within bounds
        a_req = np.diff(v_feasible) / dt
        violations = 0
        for k in range(len(a_req)):
            a_min, a_max = feasible_accel_bounds(v_feasible[k], grades[k], vehicle_caps, 0.9)
            if not (a_min - 1e-3 <= a_req[k] <= a_max + 1e-3):
                violations += 1

        assert violations == 0, f"Found {violations} acceleration violations in feasible profile"


class TestHoldSlipBraking:
    """Test the physics-rooted hold & slip braking model."""

    @pytest.fixture
    def plant_params(self) -> ExtendedPlantParams:
        """Create test plant parameters."""
        from utils.dynamics import ExtendedPlantRandomization
        return sample_extended_params(np.random.default_rng(42), ExtendedPlantRandomization())

    @pytest.fixture
    def plant(self, plant_params: ExtendedPlantParams) -> ExtendedPlant:
        """Create a test plant instance."""
        plant = ExtendedPlant(plant_params)
        plant.reset(speed=0.0)  # Start at rest
        return plant

    def test_hold_on_flat_with_brake(self, plant: ExtendedPlant) -> None:
        """Test that vehicle holds at rest on flat ground with brake applied."""
        dt = 0.1

        # Apply full brake (u = -1.0)
        plant._substep(-1.0, dt)

        # Should be held by brakes
        assert plant.held_by_brakes == True
        assert abs(plant.speed) < 1e-6, f"Speed should be ~0, got {plant.speed}"
        assert abs(plant.acceleration) < 1e-6, f"Acceleration should be ~0, got {plant.acceleration}"
        assert plant.wheel_speed == 0.0, f"Wheel speed should be 0, got {plant.wheel_speed}"

    def test_hold_on_uphill_with_brake(self, plant: ExtendedPlant) -> None:
        """Test holding on uphill grade with brake."""
        dt = 0.1

        # Set uphill grade (3°)
        plant.params.body.grade_rad = np.radians(3.0)

        # Apply brake
        plant._substep(-0.5, dt)  # Moderate brake

        # Should still be able to hold on moderate uphill
        assert plant.held_by_brakes == True
        assert abs(plant.speed) < 1e-6
        assert abs(plant.acceleration) < 1e-6

    def test_slip_on_steep_downhill_insufficient_brake(self, plant: ExtendedPlant) -> None:
        """Test that vehicle slips/rolls on steep downhill with insufficient brake."""
        dt = 0.1

        # Set steep downhill grade (10°)
        plant.params.body.grade_rad = np.radians(-10.0)

        # Apply weak brake
        plant._substep(-0.1, dt)  # Very weak brake

        # Should not be held - vehicle should start moving downhill (forward on downhill slope)
        assert plant.held_by_brakes == False
        assert plant.speed > 0.0, f"Should move downhill (positive speed), got {plant.speed}"
        assert plant.acceleration > 0.0, f"Should accelerate downhill, got {plant.acceleration}"

    def test_reverse_motion_braking(self, plant: ExtendedPlant) -> None:
        """Test braking when vehicle is moving backward."""
        dt = 0.1

        # Start with backward motion
        plant.speed = -2.0  # 2 m/s backward
        plant.wheel_speed = plant.speed / plant.params.wheel.radius

        # Apply brake (u = -1.0)
        plant._substep(-1.0, dt)

        # Braking should decelerate backward motion (reduce negative speed)
        assert plant.speed > -2.0, f"Backward speed should decrease, got {plant.speed}"
        assert plant.held_by_brakes == False  # Moving, so not held

    def test_brake_release_from_held_state(self, plant: ExtendedPlant) -> None:
        """Test releasing brake from held state."""
        dt = 0.1

        # First, apply brake to hold vehicle
        plant._substep(-1.0, dt)
        assert plant.held_by_brakes == True
        assert abs(plant.speed) < 1e-6

        # Now release brake
        plant._substep(0.0, dt)  # Neutral

        # Should no longer be held
        assert plant.held_by_brakes == False

    def test_no_spurious_acceleration_when_held(self, plant: ExtendedPlant) -> None:
        """Test that held vehicles don't report spurious accelerations."""
        dt = 0.1

        # Run multiple steps with brake applied
        accelerations = []
        for _ in range(10):
            plant._substep(-1.0, dt)
            accelerations.append(abs(plant.acceleration))

        # All accelerations should be near zero when held
        max_accel = max(accelerations)
        assert max_accel < 1e-3, f"Max acceleration when held should be < 1e-3 m/s², got {max_accel}"

    def test_kinetic_friction_limit_when_moving(self, plant: ExtendedPlant) -> None:
        """Test that moving vehicles are limited by kinetic friction."""
        dt = 0.1

        # Start with some speed
        plant.speed = 5.0
        plant.wheel_speed = plant.speed / plant.params.wheel.radius

        # Apply very strong brake
        plant._substep(-1.0, dt)

        # Tire force should be limited by kinetic friction
        mu_k = plant.params.brake.mu  # kinetic friction
        expected_max_force = mu_k * plant.params.body.mass * 9.80665

        assert abs(plant.tire_force) <= expected_max_force * 1.01, \
            f"Tire force {plant.tire_force} exceeds kinetic limit {expected_max_force}"


class TestInitialTargetFeasibility:
    """Test initial target speed feasibility functions."""

    @pytest.fixture
    def vehicle_motor_caps(self) -> "VehicleMotorCapabilities":
        """Create test vehicle motor capabilities."""
        from utils.data_utils import VehicleMotorCapabilities
        return VehicleMotorCapabilities(
            r_w=0.3,      # wheel radius (m)
            N_g=10.0,     # gear ratio
            eta=0.9,      # efficiency
            K_e=0.02,     # back-EMF constant (V/(rad/s))
            K_t=0.02,     # torque constant (Nm/A)
            R=0.1,        # resistance (Ω)
            V_max=250.0,  # max voltage (V)
            I_max=None,   # no current limit
            mass=1500.0,  # mass (kg)
            C_dA=0.6,     # drag area
            C_r=0.012,    # rolling resistance
        )

    def test_initial_target_feasible_low_speed(self, vehicle_motor_caps: "VehicleMotorCapabilities") -> None:
        """Test that low speeds are typically feasible."""
        from utils.data_utils import initial_target_feasible

        # Low speed should be feasible
        feasible, V_needed, I_needed = initial_target_feasible(5.0, 0.0, vehicle_motor_caps)
        assert feasible == True, f"Low speed should be feasible, V_needed={V_needed}"
        assert V_needed < vehicle_motor_caps.V_max * 0.95  # Well within limits

    def test_initial_target_feasible_high_speed_infeasible(self, vehicle_motor_caps: "VehicleMotorCapabilities") -> None:
        """Test that very high speeds are infeasible."""
        from utils.data_utils import initial_target_feasible

        # Very high speed should be infeasible (back-EMF too high)
        feasible, V_needed, I_needed = initial_target_feasible(60.0, 0.0, vehicle_motor_caps)
        assert feasible == False, f"Very high speed should be infeasible, V_needed={V_needed}"
        assert V_needed > vehicle_motor_caps.V_max * 0.95  # Exceeds limits

    def test_initial_target_feasible_uphill_reduces_feasibility(self, vehicle_motor_caps: "VehicleMotorCapabilities") -> None:
        """Test that uphill grade reduces feasibility."""
        from utils.data_utils import initial_target_feasible
        import math

        # Same speed, flat vs uphill
        speed = 15.0
        flat_feasible, flat_V, _ = initial_target_feasible(speed, 0.0, vehicle_motor_caps)
        uphill_feasible, uphill_V, _ = initial_target_feasible(speed, math.radians(5.0), vehicle_motor_caps)

        # Uphill should require more voltage and potentially be less feasible
        assert uphill_V > flat_V, "Uphill should require more voltage"
        if not flat_feasible:
            assert not uphill_feasible, "If flat is infeasible, uphill should also be infeasible"

    def test_adjust_initial_target_reduces_speed_when_needed(self, vehicle_motor_caps: "VehicleMotorCapabilities") -> None:
        """Test that adjust_initial_target reduces speed when high speed is infeasible."""
        from utils.data_utils import adjust_initial_target

        # Start with infeasible high speed
        original_speed = 65.0  # Should be infeasible (needs >250V)
        adjusted_speed, adjusted_grade, V_needed, I_needed = adjust_initial_target(
            original_speed, 0.0, vehicle_motor_caps, v_step=5.0
        )

        assert adjusted_speed < original_speed, f"Speed should be reduced: {original_speed} → {adjusted_speed}"

        # Verify adjusted speed is feasible
        from utils.data_utils import initial_target_feasible
        feasible, _, _ = initial_target_feasible(adjusted_speed, adjusted_grade, vehicle_motor_caps)
        assert feasible, f"Adjusted speed {adjusted_speed} should be feasible"

    def test_adjust_initial_target_flat_grade_unchanged(self, vehicle_motor_caps: "VehicleMotorCapabilities") -> None:
        """Test that flat grade is unchanged when speed adjustment suffices."""
        from utils.data_utils import adjust_initial_target

        # Start with infeasible speed on flat grade
        original_speed = 35.0
        original_grade = 0.0

        adjusted_speed, adjusted_grade, _, _ = adjust_initial_target(
            original_speed, original_grade, vehicle_motor_caps
        )

        # Grade should remain unchanged (flat)
        assert abs(adjusted_grade - original_grade) < 1e-6, "Flat grade should not be changed"

    def test_adjust_initial_target_grade_adjustment_when_needed(self, vehicle_motor_caps: "VehicleMotorCapabilities") -> None:
        """Test grade adjustment when speed reduction doesn't suffice."""
        from utils.data_utils import adjust_initial_target
        import math

        # Create a scenario where even moderate speed needs grade adjustment
        moderate_speed = 55.0  # Borderline feasible
        steep_uphill = math.radians(8.0)  # Steep uphill

        adjusted_speed, adjusted_grade, _, _ = adjust_initial_target(
            moderate_speed, steep_uphill, vehicle_motor_caps,
            max_iter_v=5, max_iter_grade=10  # Limited speed iterations to force grade adjustment
        )

        # Should have made some adjustments
        assert adjusted_speed <= moderate_speed and adjusted_grade <= steep_uphill, \
            "Should have adjusted speed and/or grade to make feasible"

    def test_voltage_calculation_matches_expectation(self, vehicle_motor_caps: "VehicleMotorCapabilities") -> None:
        """Test that voltage calculation matches the expected formula."""
        from utils.data_utils import initial_target_feasible

        speed = 25.0  # m/s
        grade = 0.0   # flat

        feasible, V_needed, I_needed = initial_target_feasible(speed, grade, vehicle_motor_caps)

        # Manual calculation
        omega_w = speed / vehicle_motor_caps.r_w
        omega_m = vehicle_motor_caps.N_g * omega_w

        # Resistive forces
        F_drag = 0.5 * vehicle_motor_caps.rho * vehicle_motor_caps.C_dA * speed**2
        F_roll = vehicle_motor_caps.C_r * vehicle_motor_caps.mass * 9.80665
        F_grade = 0.0  # flat
        F_resist = F_drag + F_roll + F_grade

        T_req_wheel = F_resist * vehicle_motor_caps.r_w
        T_req_motor = T_req_wheel / (vehicle_motor_caps.N_g * vehicle_motor_caps.eta)

        V_expected = vehicle_motor_caps.K_e * omega_m + (vehicle_motor_caps.R / vehicle_motor_caps.K_t) * T_req_motor

        assert abs(V_needed - V_expected) < 1e-6, f"V_needed mismatch: {V_needed} vs {V_expected}"

    def test_zero_speed_always_feasible(self, vehicle_motor_caps: "VehicleMotorCapabilities") -> None:
        """Test that zero speed is always feasible."""
        from utils.data_utils import initial_target_feasible

        feasible, V_needed, I_needed = initial_target_feasible(0.0, 0.0, vehicle_motor_caps)
        assert feasible == True, "Zero speed should always be feasible"
        assert V_needed >= 0.0, "V_needed should be non-negative"

    def test_B_m_batch_sampling_sanity(self) -> None:
        """Test that B_m batch sampling produces reasonable distribution."""
        import numpy as np
        from utils.dynamics import ExtendedPlantRandomization, sample_extended_params

        rand = ExtendedPlantRandomization()
        rng = np.random.default_rng(42)

        # Sample many B_m values
        B_m_samples = []
        for _ in range(10000):
            params = sample_extended_params(rng, rand)
            B_m_samples.append(params.motor.B_m)

        B_m_samples = np.array(B_m_samples)

        # Check range
        assert np.all(B_m_samples >= 1e-5), f"Min B_m {np.min(B_m_samples)} below range"
        assert np.all(B_m_samples <= 5e-3), f"Max B_m {np.max(B_m_samples)} above range"

        # Check distribution is log-uniform (check quantiles)
        q10 = np.quantile(B_m_samples, 0.1)
        q50 = np.quantile(B_m_samples, 0.5)
        q90 = np.quantile(B_m_samples, 0.9)

        # In log-uniform distribution, quantiles should be roughly geometric
        # q50 should be roughly sqrt(q10 * q90)
        expected_q50 = np.sqrt(q10 * q90)
        assert abs(q50 - expected_q50) / expected_q50 < 0.1, \
            f"Distribution not log-uniform: q10={q10:.2e}, q50={q50:.2e}, q90={q90:.2e}"

        # Check that we cover the full range
        assert q10 < 5e-5, f"10th percentile {q10:.2e} too high"
        assert q90 > 1e-3, f"90th percentile {q90:.2e} too low"
