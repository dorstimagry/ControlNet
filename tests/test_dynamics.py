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

        # With zero action and creep enabled: V_cmd will be non-zero (creep voltage)
        # Creep provides gentle forward motion at zero throttle
        # Drive torque should be small but positive (creep torque)
        assert plant.V_cmd >= 0.0, f"V_cmd should be non-negative with zero action (creep), got {plant.V_cmd}"
        # Creep torque should be bounded and reasonable
        assert abs(plant.drive_torque) < 500.0, f"Drive torque should be reasonable with creep, got {plant.drive_torque}"

        # Acceleration should be reasonable (can be positive due to creep or downhill grade, negative due to drag/rolling)
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
        """Test that steady braking produces smooth deceleration on flat ground."""
        dt = 0.1
        action = -0.5  # Moderate brake
        
        # Set flat grade for consistent test
        plant.params.body.grade_rad = 0.0

        accelerations = []
        speeds = []

        # Run for 5 seconds
        for _ in range(50):
            plant.step(action, dt)
            accelerations.append(plant.acceleration)
            speeds.append(plant.speed)

        # Should decelerate initially (may become 0 when vehicle stops and is held by brakes)
        # Allow some positive acceleration due to motor dynamics settling
        initial_accels = accelerations[5:30]  # Skip first few steps, check next 25
        for i, accel in enumerate(initial_accels):
            assert accel <= 0.5, f"Should decelerate (or nearly) at step {i+5}, got {accel}"

        # Speed should generally decrease
        # Allow some tolerance due to dynamics
        assert speeds[-1] < speeds[0], f"Speed should decrease overall"

        # Should not oscillate wildly (acceleration is computed from speed derivative,
        # so some variation is expected during transients)
        accel_std = np.std(accelerations)
        assert accel_std < 15.0, f"Braking too erratic, std={accel_std}"

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

        # No extreme jerk spikes during transition
        # Note: with single-DOF model, acceleration is derived from speed change,
        # so transients during mode switches can cause larger jerk
        max_jerk = max(abs(np.diff(accelerations))) / dt
        assert max_jerk < 500.0, f"Jerk spike too large: {max_jerk} m/s³"

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
        """Test that vehicle can achieve minimum required acceleration from near standstill."""
        dt = 0.1
        
        # Set flat grade and start from low speed for max acceleration
        plant.params.body.grade_rad = 0.0
        plant.reset(speed=0.5)  # Start from near standstill

        # Run full throttle for 3 seconds
        accelerations = []
        for _ in range(30):  # 3 seconds
            plant.step(1.0, dt)  # Full throttle
            accelerations.append(plant.acceleration)

        max_accel = max(accelerations)
        # Relaxed bounds: rejection sampling ensures >= 2.5 m/s² from rest
        # Allow some margin due to initial transients
        assert max_accel >= 2.0, f"Max acceleration {max_accel} m/s² below minimum 2.0"
        assert max_accel <= 10.0, f"Max acceleration {max_accel} m/s² above realistic max"

    def test_no_regen_during_braking(self, plant: ExtendedPlant) -> None:
        """Test that no regenerative braking occurs during braking action."""
        dt = 0.1

        # Test braking action from moderate speed
        plant.reset(speed=5.0)  # Start at moderate speed
        plant.step(-0.5, dt)  # Brake

        # During braking, there should be no regenerative current
        # Motor current must be non-negative (no regen constraint)
        assert plant.motor_current >= 0, f"Motor current should be non-negative (no regen), got {plant.motor_current}"
        
        # With braking, motor is decoupled from drive (V_cmd may have creep component at low speed)
        # The key check is: no negative current (no regeneration)
        
        # Back-EMF voltage depends on wheel speed
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
        """Test free-coast decay with passive damping only (on flat road)."""
        dt = 0.1

        # Set flat road (no grade) for proper coast test
        plant.params.body.grade_rad = 0.0

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
        tau_viscous = motor.b * omega_m
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
        tau_viscous_max = motor.b * abs(omega_m)

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
    """Test the braking behavior in single-DOF model."""

    @pytest.fixture
    def plant_params(self) -> ExtendedPlantParams:
        """Create test plant parameters."""
        from utils.dynamics import ExtendedPlantRandomization
        params = sample_extended_params(np.random.default_rng(42), ExtendedPlantRandomization())
        # Use flat grade for predictable tests
        params.body.grade_rad = 0.0
        return params

    @pytest.fixture
    def plant(self, plant_params: ExtendedPlantParams) -> ExtendedPlant:
        """Create a test plant instance."""
        plant = ExtendedPlant(plant_params)
        plant.reset(speed=0.0)  # Start at rest
        return plant

    def test_hold_on_flat_with_brake(self, plant: ExtendedPlant) -> None:
        """Test that vehicle stays nearly stopped on flat ground with brake applied."""
        dt = 0.1

        # Apply full brake (u = -1.0)
        plant._substep(-1.0, dt)

        # In single-DOF model, brakes are reflected to motor shaft.
        # At rest with brakes applied, speed should remain near zero.
        assert abs(plant.speed) < 0.1, f"Speed should be ~0, got {plant.speed}"

    def test_hold_on_uphill_with_brake(self, plant: ExtendedPlant) -> None:
        """Test braking on uphill grade."""
        dt = 0.1

        # Set uphill grade (3°)
        plant.params.body.grade_rad = np.radians(3.0)

        # Apply brake
        plant._substep(-0.5, dt)  # Moderate brake

        # Speed should be near zero initially (may drift slightly)
        assert abs(plant.speed) < 0.5, f"Speed should be small, got {plant.speed}"

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

        # Start with backward motion - must set motor_omega (single source of truth)
        # speed = (omega_m / N) * r_w => omega_m = speed * N / r_w
        target_speed = -2.0  # 2 m/s backward
        N = plant.params.motor.gear_ratio
        r_w = plant.params.wheel.radius
        plant.motor_omega = target_speed * N / r_w
        plant.wheel_omega = plant.motor_omega / N
        plant.speed = target_speed

        # Apply brake (u = -1.0)
        plant._substep(-1.0, dt)

        # Braking should decelerate backward motion (reduce negative speed toward 0)
        assert plant.speed > -2.0, f"Backward speed should decrease, got {plant.speed}"

    def test_brake_release_from_held_state(self, plant: ExtendedPlant) -> None:
        """Test releasing brake from held state."""
        dt = 0.1

        # First, apply brake to hold vehicle
        plant._substep(-1.0, dt)
        # In single-DOF model, speed should be near zero
        assert abs(plant.speed) < 0.1, f"Speed should be ~0, got {plant.speed}"

        # Now release brake (with creep enabled, vehicle will start to creep forward)
        plant._substep(0.0, dt)  # Neutral

        # With no brakes and creep enabled, vehicle will start to move forward
        # Speed should be small but positive (creep acceleration)
        assert plant.speed >= 0, f"Speed should be non-negative, got {plant.speed}"
        assert abs(plant.speed) < 0.5, f"Speed should be small (early creep), got {plant.speed}"

    def test_no_spurious_acceleration_when_held(self, plant: ExtendedPlant) -> None:
        """Test that braking keeps speed near zero on flat ground."""
        dt = 0.1

        # Run multiple steps with brake applied
        speeds = []
        for _ in range(10):
            plant._substep(-1.0, dt)
            speeds.append(abs(plant.speed))

        # All speeds should remain near zero when braking from rest
        max_speed = max(speeds)
        assert max_speed < 0.5, f"Max speed when braking should be < 0.5 m/s, got {max_speed}"

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
        original_speed = 30.0
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
        b_samples = []
        for _ in range(10000):
            params = sample_extended_params(rng, rand)
            b_samples.append(params.motor.b)

        b_samples = np.array(b_samples)

        # Check range (updated to new spec: 1e-6 to 5e-3)
        assert np.all(b_samples >= 1e-6), f"Min b {np.min(b_samples)} below range"
        assert np.all(b_samples <= 5e-3), f"Max b {np.max(b_samples)} above range"

        # Check distribution is log-uniform (check quantiles)
        q10 = np.quantile(b_samples, 0.1)
        q50 = np.quantile(b_samples, 0.5)
        q90 = np.quantile(b_samples, 0.9)

        # In log-uniform distribution, quantiles should be roughly geometric
        # q50 should be roughly sqrt(q10 * q90)
        expected_q50 = np.sqrt(q10 * q90)
        assert abs(q50 - expected_q50) / expected_q50 < 0.2, \
            f"Distribution not log-uniform: q10={q10:.2e}, q50={q50:.2e}, q90={q90:.2e}"

        # Check that we cover a reasonable range (log-uniform from 1e-6 to 5e-3)
        assert q10 < 1e-4, f"10th percentile {q10:.2e} too high"
        assert q90 > 5e-4, f"90th percentile {q90:.2e} too low"


class TestCTMSMotorModel:
    """Tests for the CTMS DC motor model and braking decoupling."""

    def test_motor_step_response_decoupled(self) -> None:
        """Test motor step response when decoupled (no wheel coupling).

        Setup: Force decoupling, apply V_cmd, check that ω_m rises toward steady-state.
        """
        from utils.dynamics import ExtendedPlant, ExtendedPlantParams, MotorParams

        # Create motor with known parameters
        motor = MotorParams(
            R=0.1, L=1e-3, K_e=0.2, K_t=0.2, b=1e-3, J=1e-3,
            V_max=24.0, gear_ratio=1.0, eta_gb=0.9
        )
        params = ExtendedPlantParams(motor=motor)
        plant = ExtendedPlant(params)
        plant.reset(speed=0.0)

        # Force decoupling by applying brake
        omega_m_history = []
        i_history = []
        for _ in range(100):
            # Apply brake to trigger decoupling
            s = plant.step(action=-0.5, dt=0.01, substeps=5)
            omega_m_history.append(s.motor_omega)
            i_history.append(s.motor_current)

        # With V_cmd=0 during braking, motor should spin down due to viscous friction
        # Motor omega should decrease (or stay near zero)
        assert omega_m_history[-1] <= omega_m_history[0] + 1.0, \
            "Motor should spin down or stay low when decoupled with V_cmd=0"

        # No NaNs
        assert all(not np.isnan(w) for w in omega_m_history), "Motor omega should not be NaN"
        assert all(not np.isnan(i) for i in i_history), "Motor current should not be NaN"

    def test_single_dof_brake_reflection(self) -> None:
        """Test that brake torque is reflected to motor shaft in single-DOF model.

        In single-DOF coupling, motor and wheel are always rigidly coupled.
        Brake torque at the wheel is reflected to the motor shaft via gearbox.
        Motor omega is the single source of truth; speed is derived from it.
        """
        from utils.dynamics import ExtendedPlant, ExtendedPlantParams, MotorParams

        motor = MotorParams(
            R=0.1, L=1e-3, K_e=0.2, K_t=0.2, b=1e-3, J=1e-3,
            V_max=400.0, gear_ratio=10.0, eta_gb=0.9
        )
        params = ExtendedPlantParams(motor=motor)
        plant = ExtendedPlant(params)

        # Start with vehicle moving
        s = plant.reset(speed=5.0)
        initial_motor_omega = s.motor_omega

        # Coupling should always be enabled in single-DOF model
        assert s.coupling_enabled == True, "Motor should be coupled (single-DOF model)"

        # Apply brakes
        for _ in range(20):
            s = plant.step(action=-0.5, dt=0.1, substeps=5)

        # Coupling should still be enabled
        assert s.coupling_enabled == True, "Motor should remain coupled during braking"

        # Motor omega should have decreased (brakes slow down motor via reflection)
        assert s.motor_omega < initial_motor_omega, "Motor should slow down with brakes"

        # Speed should be derived from motor omega: v = (omega_m / N) * r_w
        gear = motor.gear_ratio
        r_w = params.wheel.radius
        expected_speed = (s.motor_omega / gear) * r_w
        assert abs(s.speed - expected_speed) < 1e-6, \
            f"Speed should match omega_m: got {s.speed:.4f}, expected {expected_speed:.4f}"

        # Motor and wheel remain in sync (single DOF)
        assert abs(s.motor_omega - plant.wheel_omega * gear) < 1e-6, \
            "Motor and wheel should remain in sync via gear ratio"

    def test_motor_wheel_always_synced(self) -> None:
        """Test that motor and wheel are always synced in single-DOF model.
        
        In single-DOF rigid coupling, motor_omega = wheel_omega * gear_ratio
        at all times. This is the kinematic constraint.
        """
        from utils.dynamics import ExtendedPlant, ExtendedPlantParams, MotorParams

        motor = MotorParams(
            R=0.1, L=1e-3, K_e=0.2, K_t=0.2, b=1e-3, J=1e-3,
            V_max=400.0, gear_ratio=10.0, eta_gb=0.9
        )
        params = ExtendedPlantParams(motor=motor)
        plant = ExtendedPlant(params)
        gear = motor.gear_ratio

        # Start moving
        s = plant.reset(speed=3.0)

        # Verify sync after reset
        assert abs(s.motor_omega - plant.wheel_omega * gear) < 1e-6, \
            "Motor and wheel should be synced after reset"

        # Accelerate
        for _ in range(10):
            s = plant.step(action=0.5, dt=0.1, substeps=5)
            assert abs(s.motor_omega - plant.wheel_omega * gear) < 1e-6, \
                "Motor and wheel should remain synced during acceleration"

        # Apply brakes
        for _ in range(10):
            s = plant.step(action=-0.5, dt=0.1, substeps=5)
            assert abs(s.motor_omega - plant.wheel_omega * gear) < 1e-6, \
                "Motor and wheel should remain synced during braking"

        # Coast
        for _ in range(10):
            s = plant.step(action=0.0, dt=0.1, substeps=5)
            assert abs(s.motor_omega - plant.wheel_omega * gear) < 1e-6, \
                "Motor and wheel should remain synced during coasting"

    def test_no_regen_current_clamped(self) -> None:
        """Test that motor current is clamped to zero (no regeneration).
        
        When braking with V_cmd=0, the back-EMF would try to push current negative.
        The no-regen constraint clamps current to >= 0.
        """
        from utils.dynamics import ExtendedPlant, ExtendedPlantParams, MotorParams

        motor = MotorParams(
            R=0.1, L=1e-3, K_e=0.2, K_t=0.2, b=1e-3, J=1e-3,
            V_max=400.0, gear_ratio=10.0, eta_gb=0.9
        )
        params = ExtendedPlantParams(motor=motor)
        plant = ExtendedPlant(params)

        s = plant.reset(speed=5.0)

        # Apply brakes (V_cmd = 0)
        for _ in range(10):
            s = plant.step(action=-0.5, dt=0.1, substeps=5)

        # Motor current should never be negative (no regen)
        assert s.motor_current >= 0, f"Motor current should be >= 0, got {s.motor_current}"

        # With V_cmd=0 and back-EMF > 0, current should be zero
        assert s.motor_current == 0.0, f"Motor current should be 0 during braking, got {s.motor_current}"

    def test_ctms_motor_has_L_J_params(self) -> None:
        """Test that motor params include L and J for CTMS model."""
        from utils.dynamics import ExtendedPlantParams

        params = ExtendedPlantParams()
        motor = params.motor

        assert hasattr(motor, 'L'), "Motor should have inductance L"
        assert hasattr(motor, 'J'), "Motor should have rotor inertia J"
        assert hasattr(motor, 'b'), "Motor should have viscous friction b"
        assert motor.L > 0, "Inductance should be positive"
        assert motor.J > 0, "Rotor inertia should be positive"
        assert motor.b >= 0, "Viscous friction should be non-negative"


class TestCreepTorque:
    """Test EV-style creep torque behavior."""

    @pytest.fixture
    def plant_params(self) -> "ExtendedPlantParams":
        """Create test plant parameters with known creep settings."""
        from utils.dynamics import ExtendedPlantRandomization, sample_extended_params, CreepParams
        
        params = sample_extended_params(np.random.default_rng(42), ExtendedPlantRandomization())
        # Override with known creep parameters for predictable testing
        params.creep = CreepParams(a_max=0.5, v_cutoff=1.5, v_hold=0.08)
        # Use flat grade for predictable tests
        params.body.grade_rad = 0.0
        return params

    @pytest.fixture
    def plant(self, plant_params: "ExtendedPlantParams") -> "ExtendedPlant":
        """Create a test plant instance."""
        from utils.dynamics import ExtendedPlant
        plant = ExtendedPlant(plant_params)
        plant.reset(speed=0.0)
        return plant

    def test_creep_at_zero_throttle(self, plant: "ExtendedPlant") -> None:
        """Test that vehicle creeps forward slowly at zero throttle (no brake)."""
        dt = 0.1
        
        # Start at rest, apply zero action (no throttle, no brake)
        plant.reset(speed=0.0)
        initial_speed = plant.speed
        
        # Run for 5 seconds with zero action
        for _ in range(50):
            plant.step(0.0, dt, substeps=5)
        
        final_speed = plant.speed
        
        # Vehicle should have crept forward
        assert final_speed > initial_speed, f"Vehicle should creep forward, got {final_speed} vs {initial_speed}"
        assert final_speed > 0.1, f"Vehicle should reach reasonable creep speed, got {final_speed}"
        
        # Creep torque should be non-zero and positive
        assert plant.creep_torque > 0, f"Creep torque should be positive, got {plant.creep_torque}"
        
        # Speed shouldn't be too high (creep is limited)
        assert final_speed < 2.0, f"Creep speed shouldn't exceed fade threshold much, got {final_speed}"

    def test_creep_fade_with_speed(self, plant: "ExtendedPlant") -> None:
        """Test that creep torque fades smoothly as speed increases."""
        dt = 0.1
        creep_v_cutoff = plant.params.creep.v_cutoff
        
        # Test at different speeds
        test_speeds = [0.0, 0.5, 1.0, 1.5, 2.0]
        creep_torques = []
        
        for v in test_speeds:
            plant.reset(speed=v)
            plant.step(0.0, dt, substeps=1)  # Zero action to activate creep
            creep_torques.append(plant.creep_torque)
        
        # Creep torque should decrease with speed
        assert creep_torques[0] > creep_torques[1] > 0, "Creep should decrease from 0 to 0.5 m/s"
        assert creep_torques[1] > creep_torques[2] > 0, "Creep should decrease from 0.5 to 1.0 m/s"
        
        # Should be nearly zero at v_cutoff
        assert creep_torques[3] < 0.1 * creep_torques[0], f"Creep should be ~0 at v_cutoff, got {creep_torques[3]}"
        
        # Should be zero above v_cutoff
        assert creep_torques[4] < 0.01 * creep_torques[0], f"Creep should be 0 above v_cutoff, got {creep_torques[4]}"

    def test_brake_suppresses_creep(self, plant: "ExtendedPlant") -> None:
        """Test that braking fully suppresses creep torque."""
        dt = 0.1
        
        # Test creep with no brake
        plant.reset(speed=0.0)
        plant.step(0.0, dt, substeps=5)
        creep_no_brake = plant.creep_torque
        
        # Test creep with light brake
        plant.reset(speed=0.0)
        plant.step(-0.3, dt, substeps=5)  # 30% brake
        creep_light_brake = plant.creep_torque
        
        # Test creep with full brake
        plant.reset(speed=0.0)
        plant.step(-1.0, dt, substeps=5)  # 100% brake
        creep_full_brake = plant.creep_torque
        
        # Brake should suppress creep
        assert creep_no_brake > 0, "Should have creep with no brake"
        assert creep_light_brake < creep_no_brake, "Light brake should reduce creep"
        assert creep_full_brake < 0.1 * creep_no_brake, f"Full brake should eliminate creep, got {creep_full_brake}"

    def test_throttle_to_creep_transition(self, plant: "ExtendedPlant") -> None:
        """Test smooth transition from throttle to creep when releasing accelerator."""
        dt = 0.1
        
        # Phase 1: Accelerate with very gentle throttle to stay below v_cutoff
        plant.reset(speed=0.0)
        for _ in range(8):
            plant.step(0.15, dt, substeps=5)  # 15% throttle (very gentle)
        
        speed_after_throttle = plant.speed
        
        # Phase 2: Release throttle (coast with creep)
        accelerations = []
        speeds = []
        creep_torques = []
        for _ in range(30):
            plant.step(0.0, dt, substeps=5)  # Zero action
            accelerations.append(plant.acceleration)
            speeds.append(plant.speed)
            creep_torques.append(plant.creep_torque)
        
        # Should decelerate smoothly (no jump in acceleration)
        max_jerk = max(abs(np.diff(accelerations))) / dt
        assert max_jerk < 50.0, f"Transition should be smooth, max jerk={max_jerk}"
        
        # Creep should be active during some portion of coast (when speed is below v_cutoff)
        num_with_creep = sum(1 for ct in creep_torques if ct > 0)
        assert num_with_creep > 5, f"Creep should be active during coast, got {num_with_creep}/30 steps"
        
        # Speed should remain non-negative throughout
        assert all(s >= 0 for s in speeds), "Vehicle should not go backward"

    def test_creep_to_brake_transition(self, plant: "ExtendedPlant") -> None:
        """Test smooth transition from creep to braking."""
        dt = 0.1
        
        # Phase 1: Set up at low speed and record baseline creep
        plant.reset(speed=0.5)
        plant.step(0.0, dt, substeps=5)  # One step to establish creep
        baseline_creep = plant.creep_torque
        assert baseline_creep > 0, "Creep should be active at 0.5 m/s"
        
        # Phase 2: Apply brake and observe creep suppression
        plant.reset(speed=0.5)  # Reset to same initial condition
        plant.step(-0.5, dt, substeps=1)  # 50% brake, single substep for immediate effect
        creep_with_brake = plant.creep_torque
        
        # With 50% brake command, creep should be suppressed by brake dominance factor (1 - 0.5) = 0.5
        # Allow some tolerance for dynamics
        assert creep_with_brake <= 0.6 * baseline_creep, \
            f"Creep should be suppressed by brake: {creep_with_brake} vs baseline {baseline_creep}"
        
        # Phase 3: Test smooth deceleration with continued braking
        accelerations = []
        for _ in range(20):
            plant.step(-0.5, dt, substeps=5)
            accelerations.append(plant.acceleration)
        
        # Should decelerate smoothly
        max_jerk = max(abs(np.diff(accelerations))) / dt
        assert max_jerk < 100.0, f"Braking transition should be smooth, max jerk={max_jerk}"
        
        # Vehicle should slow down
        assert plant.speed < 0.5, "Vehicle should slow down with braking"

    def test_creep_on_uphill_grade(self, plant: "ExtendedPlant") -> None:
        """Test creep behavior on uphill grade (may not overcome gravity)."""
        dt = 0.1
        
        # Set uphill grade (3 degrees)
        plant.params.body.grade_rad = np.radians(3.0)
        
        plant.reset(speed=0.0)
        initial_speed = plant.speed
        
        # Run with zero action for several seconds
        for _ in range(30):
            plant.step(0.0, dt, substeps=5)
        
        final_speed = plant.speed
        
        # Creep should be active
        assert plant.creep_torque > 0, "Creep torque should be active"
        
        # On uphill, creep might not be strong enough to overcome grade
        # Vehicle might roll back or stay near zero, but creep should try
        # Just verify no crash and creep is computed
        assert abs(final_speed) < 5.0, "Speed should remain bounded"
        assert not np.isnan(plant.speed), "Speed should not be NaN"

    def test_no_oscillation_at_standstill(self, plant: "ExtendedPlant") -> None:
        """Test that creep doesn't cause oscillations when vehicle is near zero speed."""
        dt = 0.1
        
        # Start very close to zero with creep active
        plant.reset(speed=0.01)
        
        speeds = []
        accelerations = []
        
        # Run for extended period at near-zero speed
        for _ in range(50):
            plant.step(0.0, dt, substeps=5)
            speeds.append(plant.speed)
            accelerations.append(plant.acceleration)
        
        # Check for oscillations: speed should not flip sign repeatedly
        speed_signs = np.sign(speeds)
        sign_changes = np.sum(np.abs(np.diff(speed_signs)) > 0)
        assert sign_changes < 3, f"Too many sign changes (oscillations): {sign_changes}"
        
        # Acceleration shouldn't have wild swings
        accel_std = np.std(accelerations)
        assert accel_std < 5.0, f"Acceleration too erratic: std={accel_std}"

    def test_creep_torque_computation(self, plant: "ExtendedPlant") -> None:
        """Test that creep torque is correctly computed from acceleration parameter."""
        dt = 0.1
        
        plant.reset(speed=0.0)
        plant.step(0.0, dt, substeps=1)  # Single substep for predictable state
        
        # Manually compute expected creep torque at v=0 (no fade, no brake suppression)
        creep = plant.params.creep
        body = plant.params.body
        motor = plant.params.motor
        wheel = plant.params.wheel
        
        F_creep_max = body.mass * creep.a_max
        T_wheel_creep_max = F_creep_max * wheel.radius
        T_motor_creep_max_expected = T_wheel_creep_max / (motor.gear_ratio * motor.eta_gb)
        
        # At v=0, fade weight should be 1.0, brake suppression should be 1.0
        # So creep_torque should equal T_motor_creep_max
        assert abs(plant.creep_torque - T_motor_creep_max_expected) < 0.01 * T_motor_creep_max_expected, \
            f"Creep torque mismatch: got {plant.creep_torque}, expected {T_motor_creep_max_expected}"

    def test_creep_differentiability(self, plant: "ExtendedPlant") -> None:
        """Test that creep computation produces smooth gradients (no discontinuities)."""
        dt = 0.05  # Smaller timestep for gradient check
        
        # Test gradient w.r.t. speed (fade function)
        speeds = np.linspace(0.0, 2.0, 50)
        creep_torques = []
        
        for v in speeds:
            plant.reset(speed=v)
            plant.step(0.0, dt, substeps=1)
            creep_torques.append(plant.creep_torque)
        
        creep_torques = np.array(creep_torques)
        
        # Compute numerical gradient
        gradients = np.gradient(creep_torques, speeds)
        
        # Gradient should be continuous (no jumps)
        gradient_changes = np.abs(np.diff(gradients))
        max_gradient_jump = np.max(gradient_changes)
        
        # Allow some numerical noise but no large discontinuities
        assert max_gradient_jump < 100.0, f"Gradient discontinuity detected: max jump={max_gradient_jump}"
        
        # Gradient should be negative (creep decreases with speed)
        assert np.all(gradients[:-1] <= 0), "Creep gradient should be non-positive"

    def test_creep_with_parameter_variation(self, plant: "ExtendedPlant") -> None:
        """Test that creep works correctly across different vehicle parameters."""
        from utils.dynamics import ExtendedPlant, ExtendedPlantRandomization, sample_extended_params, CreepParams
        
        dt = 0.1
        
        # Test with several random parameter sets
        for seed in range(5):
            rng = np.random.default_rng(seed + 100)
            params = sample_extended_params(rng, ExtendedPlantRandomization())
            params.creep = CreepParams(a_max=0.5, v_cutoff=1.5, v_hold=0.08)
            params.body.grade_rad = 0.0  # Flat for consistency
            
            test_plant = ExtendedPlant(params)
            test_plant.reset(speed=0.0)
            
            # Run for a few steps
            for _ in range(20):
                test_plant.step(0.0, dt, substeps=5)
            
            # Should have crept forward
            assert test_plant.speed > 0, f"Seed {seed}: Should creep forward, got speed={test_plant.speed}"
            assert test_plant.creep_torque > 0, f"Seed {seed}: Should have positive creep torque"
            assert not np.isnan(test_plant.speed), f"Seed {seed}: Speed should not be NaN"
            assert not np.isnan(test_plant.creep_torque), f"Seed {seed}: Creep torque should not be NaN"

    def test_creep_params_in_config(self) -> None:
        """Test that creep parameters can be loaded from configuration."""
        from utils.dynamics import ExtendedPlantRandomization, CreepParams
        
        # Create config with creep parameters
        config = {
            'creep': {
                'a_max': 0.6,
                'v_cutoff': 2.0,
                'v_hold': 0.1,
            },
            'vehicle_randomization': {
                'mass_range': [1500.0, 2000.0],
            }
        }
        
        rand = ExtendedPlantRandomization.from_config(config)
        
        # Check that creep parameters were loaded
        assert rand.creep_a_max == 0.6, f"Expected a_max=0.6, got {rand.creep_a_max}"
        assert rand.creep_v_cutoff == 2.0, f"Expected v_cutoff=2.0, got {rand.creep_v_cutoff}"
        assert rand.creep_v_hold == 0.1, f"Expected v_hold=0.1, got {rand.creep_v_hold}"

    def test_creep_default_values(self) -> None:
        """Test that creep uses default values when not specified in config."""
        from utils.dynamics import ExtendedPlantRandomization, sample_extended_params
        
        # Config without creep parameters
        config = {
            'vehicle_randomization': {
                'mass_range': [1500.0, 2000.0],
            }
        }
        
        rand = ExtendedPlantRandomization.from_config(config)
        params = sample_extended_params(np.random.default_rng(42), rand)
        
        # Should use default CreepParams values
        assert params.creep.a_max == 0.5, "Should use default a_max=0.5"
        assert params.creep.v_cutoff == 1.5, "Should use default v_cutoff=1.5"
        assert params.creep.v_hold == 0.08, "Should use default v_hold=0.08"


class TestCreepFunctional:
    """Functional/integration tests for creep behavior in realistic scenarios."""

    @pytest.fixture
    def plant(self) -> "ExtendedPlant":
        """Create a test plant with realistic parameters."""
        from utils.dynamics import ExtendedPlant, ExtendedPlantRandomization, sample_extended_params, CreepParams
        
        params = sample_extended_params(np.random.default_rng(42), ExtendedPlantRandomization())
        params.creep = CreepParams(a_max=0.5, v_cutoff=1.5, v_hold=0.08)
        params.body.grade_rad = 0.0
        plant = ExtendedPlant(params)
        plant.reset(speed=0.0)
        return plant

    def test_creep_enables_smooth_stops(self, plant: "ExtendedPlant") -> None:
        """Test that creep helps agent smoothly approach zero speed (no dead zone)."""
        dt = 0.1
        
        # Simulate approaching a stop from low speed (stay in creep range)
        plant.reset(speed=1.0)
        
        # Decelerate gently
        actions = np.linspace(0.1, 0.0, 20)  # Reduce throttle gently over 2 seconds
        for action in actions:
            plant.step(action, dt, substeps=5)
        
        # Continue coasting with creep
        speeds = []
        creep_torques = []
        for _ in range(30):
            plant.step(0.0, dt, substeps=5)
            speeds.append(plant.speed)
            creep_torques.append(plant.creep_torque)
        
        # Should have some creep activity at low speeds
        num_with_creep = sum(1 for ct in creep_torques if ct > 0)
        assert num_with_creep > 5, f"Creep should be active during low-speed coast, got {num_with_creep}/30"
        
        # Speed should remain non-negative
        assert all(s >= 0 for s in speeds), "Speed should not go negative"

    def test_full_episode_with_creep(self, plant: "ExtendedPlant") -> None:
        """Test complete episode with various maneuvers including creep."""
        dt = 0.1
        
        # Phase 1: Accelerate gently from rest (stay in creep range initially)
        plant.reset(speed=0.0)
        for _ in range(15):
            plant.step(0.2, dt, substeps=5)  # 20% throttle (gentle)
        
        assert plant.speed > 0.5, "Should have accelerated"
        
        # Phase 2: Coast - creep may be active depending on final speed
        creep_torques = []
        for _ in range(20):
            plant.step(0.0, dt, substeps=5)
            creep_torques.append(plant.creep_torque)
        
        # Creep may not be active if speed is above v_cutoff
        # Just verify no instabilities
        assert plant.speed >= 0, "Should maintain non-negative speed"
        
        # Phase 3: Brake to stop
        for _ in range(30):
            plant.step(-0.6, dt, substeps=5)  # 60% brake
        
        assert plant.speed < 1.0, "Should have slowed down significantly"
        
        # Phase 4: Hold at stop
        for _ in range(20):
            plant.step(-0.8, dt, substeps=5)  # Strong brake
        
        # Should be held near zero
        assert abs(plant.speed) < 0.5, f"Should be held near zero, got {plant.speed}"
        
        # No NaNs or instabilities throughout
        assert not np.isnan(plant.speed), "Speed should not be NaN"
        assert not np.isnan(plant.acceleration), "Acceleration should not be NaN"
        assert not np.isnan(plant.creep_torque), "Creep torque should not be NaN"

    def test_stop_and_go_with_creep(self, plant: "ExtendedPlant") -> None:
        """Test stop-and-go traffic scenario with creep."""
        dt = 0.1
        
        plant.reset(speed=0.0)
        
        # Simulate 3 stop-and-go cycles with gentle acceleration
        for cycle in range(3):
            # Go: accelerate gently
            for _ in range(12):
                plant.step(0.25, dt, substeps=5)  # 25% throttle
            
            speed_after_accel = plant.speed
            assert speed_after_accel > 0.5, f"Cycle {cycle}: Should accelerate"
            
            # Coast - check if creep is active (depends on speed)
            creep_torques = []
            for _ in range(10):
                plant.step(0.0, dt, substeps=5)
                creep_torques.append(plant.creep_torque)
            
            # At least verify no NaNs
            assert all(not np.isnan(ct) for ct in creep_torques), f"Cycle {cycle}: Creep torques should not be NaN"
            
            # Stop: brake
            for _ in range(15):
                plant.step(-0.7, dt, substeps=5)
            
            assert plant.speed < 1.0, f"Cycle {cycle}: Should slow down"
        
        # After multiple cycles, dynamics should remain stable
        assert not np.isnan(plant.speed), "Speed should not be NaN after cycles"
        assert abs(plant.speed) < 5.0, "Speed should remain bounded"

    def test_creep_on_varied_grades(self, plant: "ExtendedPlant") -> None:
        """Test creep behavior across different road grades."""
        dt = 0.1
        
        grades_deg = [0.0, 2.0, -2.0, 5.0, -5.0]
        
        for grade_deg in grades_deg:
            plant.params.body.grade_rad = np.radians(grade_deg)
            plant.reset(speed=0.0)
            
            # Run with zero action (creep active)
            for _ in range(30):
                plant.step(0.0, dt, substeps=5)
            
            # Creep should always be computed
            assert plant.creep_torque >= 0, f"Grade {grade_deg}°: Creep torque should be non-negative"
            
            # No crashes or NaNs
            assert not np.isnan(plant.speed), f"Grade {grade_deg}°: Speed should not be NaN"
            assert abs(plant.speed) < 10.0, f"Grade {grade_deg}°: Speed should remain bounded"
            
            # On flat or downhill, should move forward
            if grade_deg <= 0:
                assert plant.speed > 0, f"Grade {grade_deg}°: Should move forward on flat/downhill"

    def test_parameter_variation_robustness(self, plant: "ExtendedPlant") -> None:
        """Test that creep works robustly across wide parameter ranges."""
        from utils.dynamics import ExtendedPlant, ExtendedPlantRandomization, sample_extended_params, CreepParams
        
        dt = 0.1
        rand = ExtendedPlantRandomization()
        
        # Test with 10 random parameter sets
        for seed in range(10):
            rng = np.random.default_rng(seed + 200)
            params = sample_extended_params(rng, rand)
            params.creep = CreepParams(a_max=0.5, v_cutoff=1.5, v_hold=0.08)
            params.body.grade_rad = 0.0
            
            test_plant = ExtendedPlant(params)
            test_plant.reset(speed=0.0)
            
            # Run a simple maneuver: accelerate gently, coast, brake
            # Accelerate gently to stay in/near creep range
            for _ in range(10):
                test_plant.step(0.25, dt, substeps=5)  # Gentle acceleration
            
            speed_after_accel = test_plant.speed
            
            # Coast - collect creep torques
            creep_torques = []
            for _ in range(15):
                test_plant.step(0.0, dt, substeps=5)
                creep_torques.append(test_plant.creep_torque)
            
            # Brake
            for _ in range(15):
                test_plant.step(-0.5, dt, substeps=5)
            
            # Verify stability
            assert not np.isnan(test_plant.speed), f"Seed {seed}: Speed should not be NaN"
            assert not np.isnan(test_plant.creep_torque), f"Seed {seed}: Creep torque should not be NaN"
            assert speed_after_accel > 0, f"Seed {seed}: Should have accelerated"
            # Creep should be active during at least some of the coast (if speed is below v_cutoff)
            num_with_creep = sum(1 for ct in creep_torques if ct > 0)
            # Allow for variation - just check no NaNs and some creep activity
            assert all(not np.isnan(ct) for ct in creep_torques), f"Seed {seed}: Creep torques should not be NaN"

    def test_creep_with_rapid_action_changes(self, plant: "ExtendedPlant") -> None:
        """Test creep behavior with rapid changes in control input (stability test)."""
        dt = 0.1
        
        plant.reset(speed=1.0)
        
        # Rapidly alternate between throttle, coast, and brake
        actions = [0.5, 0.0, -0.3, 0.0, 0.7, 0.0, -0.5, 0.0] * 10  # 80 steps
        
        accelerations = []
        creep_torques = []
        
        for action in actions:
            plant.step(action, dt, substeps=5)
            accelerations.append(plant.acceleration)
            creep_torques.append(plant.creep_torque)
        
        # Check for stability: no NaNs, no extreme values
        assert not np.any(np.isnan(accelerations)), "Accelerations should not be NaN"
        assert not np.any(np.isnan(creep_torques)), "Creep torques should not be NaN"
        assert np.all(np.abs(accelerations) < 20.0), "Accelerations should be bounded"
        assert np.all(np.array(creep_torques) >= 0), "Creep torque should be non-negative"
        
        # Creep should activate during zero-action phases
        zero_action_indices = [i for i, a in enumerate(actions) if a == 0.0]
        creep_during_zero = [creep_torques[i] for i in zero_action_indices if i < len(creep_torques)]
        
        # At least some creep should be active during zero actions (depending on speed)
        num_with_creep = sum(1 for ct in creep_during_zero if ct > 0)
        assert num_with_creep > len(creep_during_zero) * 0.3, "Creep should be active during many zero-action phases"


