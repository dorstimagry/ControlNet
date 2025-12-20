"""PID controller for closed-loop evaluation."""

from __future__ import annotations


class PIDController:
    """PID controller that outputs actions in [-1, 1].
    
    The controller computes:
    - Proportional term: kp * error
    - Integral term: ki * integral(error)
    - Derivative term: kd * d(error)/dt
    
    The output is clipped to [-1, 1] and integral windup is prevented
    by clamping the integral term.
    """
    
    def __init__(
        self, 
        kp: float = 0.0, 
        ki: float = 0.0, 
        kd: float = 0.0,
        integral_min: float = -0.1,
        integral_max: float = 0.1,
    ):
        """Initialize PID controller.
        
        Args:
            kp: Proportional gain (default: 0.0)
            ki: Integral gain (default: 0.0)
            kd: Derivative gain (default: 0.0)
            integral_min: Minimum saturation limit for integral term (default: -0.1)
            integral_max: Maximum saturation limit for integral term (default: 0.1)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_min = integral_min
        self.integral_max = integral_max
        self.reset()
    
    def reset(self) -> None:
        """Reset controller state (integral and previous error)."""
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute(self, error: float, dt: float) -> float:
        """Compute PID output.
        
        Args:
            error: Current error (reference - current)
            dt: Time step (seconds)
            
        Returns:
            PID output clipped to [-1, 1]
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with windup prevention)
        self.integral += error * dt
        # Clamp integral to configured saturation limits
        self.integral = max(self.integral_min, min(self.integral_max, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            error_derivative = (error - self.prev_error) / dt
        else:
            error_derivative = 0.0
        d_term = self.kd * error_derivative
        
        # Update previous error
        self.prev_error = error
        
        # Compute output and clip
        output = p_term + i_term + d_term
        return max(-1.0, min(1.0, output))

