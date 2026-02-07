"""Torch implementation of vectorized SecondOrderLPF with rate and acceleration limits."""

import torch


def second_order_lpf_update(
    y: torch.Tensor,  # [B] current filtered value (m/s)
    r: torch.Tensor,  # [B] current rate (m/s²)
    u: torch.Tensor,  # [B] input command (target value)
    dt: float,        # policy step time
    wc: torch.Tensor, # [B] cutoff frequency (rad/s)
    zeta: float,      # damping ratio
    rate_max: torch.Tensor,  # [B] max positive rate (m/s²)
    rate_neg_max: torch.Tensor,  # [B] max negative rate magnitude (positive value, m/s²)
    jerk_max: torch.Tensor,  # [B] max jerk (m/s³)
    device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update second-order low-pass filter with rate and jerk limits.

    Args:
        y: Current filtered value tensor [B]
        r: Current rate tensor [B] (m/s²)
        u: Input command tensor [B] (target values)
        dt: Time step (seconds)
        wc: Cutoff frequency tensor [B] (rad/s)
        zeta: Damping ratio (scalar)
        rate_max: Maximum positive rate (m/s²)
        rate_neg_max: Maximum negative rate magnitude (positive value, m/s²)
        jerk_max: Maximum jerk (m/s³)
        device: Torch device

    Returns:
        Updated y and r tensors
    """
    if device is None:
        device = y.device

    # Store previous rate for jerk limiting
    r_prev = r.clone()
    
    # Maximum allowed rate change in this timestep
    max_rate_change = jerk_max * dt

    # Compute LPF parameters
    wc_LPO1 = 2 * zeta * wc  # [B]
    K = wc**2 / wc_LPO1      # [B]

    # Input to first-order LPF
    input_LPO1 = K * (u - y)  # [B]

    # Compute candidate rate derivative (jerk)
    d_r_dt = wc_LPO1 * (input_LPO1 - r)  # [B] (m/s³)

    # Limit jerk
    d_r_dt_clamped = torch.clamp(d_r_dt, -jerk_max, jerk_max)

    # Update rate
    r_new = r + d_r_dt_clamped * dt

    # Limit rate to bounds, but respect jerk constraint
    # If clamping would cause excessive jerk, limit the clamp
    r_clamped = torch.clamp(r_new, -rate_neg_max, rate_max)
    
    # Compute the rate change that would result from clamping
    rate_change = r_clamped - r_prev
    
    # Ensure rate change doesn't exceed jerk limit
    rate_change_limited = torch.clamp(rate_change, -max_rate_change, max_rate_change)
    r = r_prev + rate_change_limited
    
    # Final rate clamp (should be redundant but ensures safety)
    r = torch.clamp(r, -rate_neg_max, rate_max)

    # Integrate to get new value
    y = y + r * dt

    return y, r


class SecondOrderLPF:
    """Vectorized second-order low-pass filter with rate and jerk limits."""

    def __init__(
        self,
        batch_size: int,
        freq_cutoff: float,  # Hz
        zeta: float = 0.9,
        dt: float = 0.02,  # policy step time (50 Hz default)
        rate_max: torch.Tensor = None,      # m/s² - now tensor for per-batch limits
        rate_neg_max: torch.Tensor = None,  # m/s² - now tensor for per-batch limits
        jerk_max: torch.Tensor = None,      # m/s³ - now tensor for per-batch limits
        device: torch.device = None
    ):
        """Initialize the LPF.

        Args:
            batch_size: Number of parallel filters
            freq_cutoff: Cutoff frequency in Hz
            zeta: Damping ratio
            dt: Time step in seconds
            rate_max: Maximum positive acceleration (m/s²)
            rate_neg_max: Maximum deceleration magnitude (m/s²)
            jerk_max: Maximum jerk (m/s³)
            device: Torch device
        """
        self.batch_size = batch_size
        self.freq_cutoff = freq_cutoff
        self.zeta = zeta
        self.dt = dt
        self.rate_max = rate_max
        self.rate_neg_max = rate_neg_max
        self.jerk_max = jerk_max
        self.device = device or torch.device('cpu')

        # Convert frequency to rad/s
        self.wc = 2 * torch.pi * freq_cutoff

        # Handle rate limit parameters (tensor or scalar)
        if rate_max is None:
            self.rate_max = torch.full((batch_size,), 4.0, device=device)
        else:
            self.rate_max = rate_max.to(device)

        if rate_neg_max is None:
            self.rate_neg_max = torch.full((batch_size,), 8.0, device=device)
        else:
            self.rate_neg_max = rate_neg_max.to(device)

        if jerk_max is None:
            self.jerk_max = torch.full((batch_size,), 12.0, device=device)
        else:
            self.jerk_max = jerk_max.to(device)

        # Initialize states
        self.reset()

    def reset(self, initial_y: torch.Tensor = None, initial_r: torch.Tensor = None):
        """Reset filter states.

        Args:
            initial_y: Initial values [B], defaults to zeros
            initial_r: Initial rates [B], defaults to zeros
        """
        if initial_y is None:
            self.y = torch.zeros(self.batch_size, device=self.device)
        else:
            self.y = initial_y.clone()

        if initial_r is None:
            self.r = torch.zeros(self.batch_size, device=self.device)
        else:
            self.r = initial_r.clone()

    def update(self, u: torch.Tensor) -> torch.Tensor:
        """Update filter with new input command.

        Args:
            u: Input commands [B]

        Returns:
            Updated filtered values [B]
        """
        self.y, self.r = second_order_lpf_update(
            self.y, self.r, u, self.dt, self.wc, self.zeta,
            self.rate_max, self.rate_neg_max, self.jerk_max, self.device
        )
        return self.y.clone()
