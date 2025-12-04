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

        # With zero action: no drive torque, motor voltage = 0
        assert plant.V_cmd == 0.0, f"V_cmd should be 0 with zero action, got {plant.V_cmd}"
        assert plant.drive_torque == 0.0, f"Drive torque should be 0 with zero action, got {plant.drive_torque}"

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

        # Should decelerate
        for accel in accelerations:
            assert accel < 0, f"Should decelerate with brake, got {accel}"

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

    def test_wheel_speed_positive(self, plant: ExtendedPlant) -> None:
        """Test that wheel speed stays positive (not clamped to zero)."""
        dt = 0.1

        # Test with various actions
        actions = [0.0, 0.5, -0.5, 1.0, -1.0]
        for action in actions:
            plant.reset(speed=5.0)  # Reset for each test
            for _ in range(10):
                plant.step(action, dt)
                assert plant.wheel_speed > 0, f"Wheel speed became non-positive: {plant.wheel_speed}"

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
        assert plant.drive_torque == 0.0, f"Drive torque should be 0 during braking, got {plant.drive_torque}"

        # Motor voltage should also be 0
        assert plant.motor_voltage == 0.0, f"Motor voltage should be 0 during braking, got {plant.motor_voltage}"

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
