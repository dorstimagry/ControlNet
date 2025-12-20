"""Unit tests for the PID controller."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.pid_controller import PIDController


class TestPIDController:
    """Tests for PIDController class."""
    
    def test_initialization_default_gains(self) -> None:
        """Test initialization with default gains (all zero)."""
        pid = PIDController()
        assert pid.kp == 0.0
        assert pid.ki == 0.0
        assert pid.kd == 0.0
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0
    
    def test_initialization_custom_gains(self) -> None:
        """Test initialization with custom gains."""
        pid = PIDController(kp=1.0, ki=0.5, kd=0.2)
        assert pid.kp == 1.0
        assert pid.ki == 0.5
        assert pid.kd == 0.2
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0
    
    def test_reset_clears_state(self) -> None:
        """Test reset() clears integral and previous error."""
        pid = PIDController(kp=1.0, ki=0.5, kd=0.2)
        
        # Run some computations to accumulate state
        pid.compute(1.0, 0.1)
        pid.compute(0.5, 0.1)
        
        # Verify state is non-zero
        assert pid.integral != 0.0
        assert pid.prev_error != 0.0
        
        # Reset
        pid.reset()
        
        # Verify state is cleared
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0
    
    def test_proportional_term_only(self) -> None:
        """Test proportional term: output = kp * error (with ki=0, kd=0)."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0)
        
        # Test positive error
        output = pid.compute(1.0, 0.1)
        assert output == pytest.approx(2.0)  # kp * error = 2.0 * 1.0
        
        # Test negative error
        output = pid.compute(-0.5, 0.1)
        assert output == pytest.approx(-1.0)  # kp * error = 2.0 * -0.5
    
    def test_integral_term_accumulates(self) -> None:
        """Test integral term: verify integral accumulates over time."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        dt = 0.1
        
        # First step: error = 1.0, integral should be 1.0 * 0.1 = 0.1
        output1 = pid.compute(1.0, dt)
        assert pid.integral == pytest.approx(0.1)
        assert output1 == pytest.approx(0.1)
        
        # Second step: error = 1.0, integral should accumulate to 0.2
        output2 = pid.compute(1.0, dt)
        assert pid.integral == pytest.approx(0.2)
        assert output2 == pytest.approx(0.2)
        
        # Third step: error = -0.5, integral should be 0.2 + (-0.5 * 0.1) = 0.15
        output3 = pid.compute(-0.5, dt)
        assert pid.integral == pytest.approx(0.15)
        assert output3 == pytest.approx(0.15)
    
    def test_derivative_term(self) -> None:
        """Test derivative term: verify derivative responds to error changes."""
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
        dt = 0.1
        
        # First step: no previous error, derivative should be 0
        output1 = pid.compute(1.0, dt)
        assert output1 == pytest.approx(0.0)
        
        # Second step: error increases from 1.0 to 2.0, derivative = (2.0 - 1.0) / 0.1 = 10.0
        output2 = pid.compute(2.0, dt)
        assert output2 == pytest.approx(10.0)
        
        # Third step: error decreases from 2.0 to 0.5, derivative = (0.5 - 2.0) / 0.1 = -15.0
        output3 = pid.compute(0.5, dt)
        assert output3 == pytest.approx(-15.0)
    
    def test_output_clipping(self) -> None:
        """Test output clipping to [-1, 1]."""
        # Test positive clipping
        pid = PIDController(kp=10.0, ki=0.0, kd=0.0)
        output = pid.compute(1.0, 0.1)
        assert output == 1.0  # Should be clipped to 1.0
        
        # Test negative clipping
        output = pid.compute(-1.0, 0.1)
        assert output == -1.0  # Should be clipped to -1.0
        
        # Test within bounds
        pid = PIDController(kp=0.5, ki=0.0, kd=0.0)
        output = pid.compute(1.0, 0.1)
        assert output == pytest.approx(0.5)  # Should not be clipped
    
    def test_integral_windup_prevention(self) -> None:
        """Test integral windup prevention (integral term should be clamped)."""
        pid = PIDController(kp=0.0, ki=10.0, kd=0.0, integral_min=-0.1, integral_max=0.1)
        dt = 0.1
        
        # Accumulate error over many steps
        for _ in range(20):
            pid.compute(1.0, dt)
        
        # Integral should be clamped to configured limits
        assert pid.integral <= 0.1 + 1e-6  # Allow small numerical error
        assert pid.integral >= -0.1 - 1e-6  # Also check negative side
        
        # Output should be clamped accordingly
        output = pid.compute(1.0, dt)
        assert abs(output) <= 1.0 + 1e-6
    
    def test_integral_saturation_limits(self) -> None:
        """Test configurable integral saturation limits."""
        # Test with custom limits
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0, integral_min=-0.5, integral_max=0.5)
        dt = 0.1
        
        # Accumulate large error
        for _ in range(20):
            pid.compute(10.0, dt)
        
        # Integral should be clamped to custom limits
        assert pid.integral <= 0.5 + 1e-6
        assert pid.integral >= -0.5 - 1e-6
        
        # Test with default limits
        pid_default = PIDController(kp=0.0, ki=1.0, kd=0.0)
        for _ in range(20):
            pid_default.compute(10.0, dt)
        
        # Should use default limits of -0.1 to 0.1
        assert pid_default.integral <= 0.1 + 1e-6
        assert pid_default.integral >= -0.1 - 1e-6
    
    def test_combined_pid_terms(self) -> None:
        """Test combined PID terms produce expected output."""
        pid = PIDController(kp=1.0, ki=0.5, kd=0.2)
        dt = 0.1
        
        # First step: error = 1.0
        # P = 1.0 * 1.0 = 1.0
        # I = 0.5 * (1.0 * 0.1) = 0.05
        # D = 0.2 * (1.0 - 0.0) / 0.1 = 2.0
        # Total = 1.0 + 0.05 + 2.0 = 3.05 (clipped to 1.0)
        output1 = pid.compute(1.0, dt)
        assert output1 == 1.0  # Clipped
        
        # Second step: error = 0.5
        # P = 1.0 * 0.5 = 0.5
        # I = 0.5 * (0.1 + 0.5 * 0.1) = 0.5 * 0.15 = 0.075
        # D = 0.2 * (0.5 - 1.0) / 0.1 = -1.0
        # Total = 0.5 + 0.075 - 1.0 = -0.425
        output2 = pid.compute(0.5, dt)
        assert output2 == pytest.approx(-0.425, abs=1e-3)
    
    def test_reset_between_episodes(self) -> None:
        """Test reset between episodes."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        dt = 0.1
        
        # Simulate first episode
        for _ in range(10):
            pid.compute(1.0, dt)
        
        # Verify state accumulated
        assert pid.integral != 0.0
        assert pid.prev_error != 0.0
        
        # Reset for new episode
        pid.reset()
        
        # Verify state is cleared
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0
        
        # Simulate second episode - should start fresh
        output = pid.compute(1.0, dt)
        assert pid.integral == pytest.approx(0.1)  # Fresh start
        assert output == pytest.approx(0.1)
    
    def test_zero_dt_handling(self) -> None:
        """Test handling of zero or very small dt."""
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0)
        
        # With dt = 0, derivative should be 0 (avoid division by zero)
        output = pid.compute(1.0, 0.0)
        # P = 1.0, I = 0.0 (no accumulation), D = 0.0 (dt=0)
        assert output == pytest.approx(1.0)
        
        # With very small dt, should still work
        output = pid.compute(1.0, 1e-10)
        assert not np.isnan(output)
        assert not np.isinf(output)

