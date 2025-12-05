"""Samplers for event-driven target speed generation."""

from dataclasses import dataclass
import torch


@dataclass
class GeneratorParams:
    """Parameters for the target speed generator."""
    p_change: float = 0.03  # Probability of changing target per step
    p_zero_stop: float = 0.08  # Probability of sampling zero speed target
    v_min: float = 0.0  # Minimum speed (m/s)
    v_max_sample: float = 30.0  # Maximum speed for sampling (m/s)
    t_min: float = 2.0  # Minimum arrival time (s)
    t_max: float = 12.0  # Maximum arrival time (s)
    stop_hold_min: float = 1.0  # Minimum dwell time for stops (s)
    stop_hold_max: float = 5.0  # Maximum dwell time for stops (s)
    allow_negative_speeds: bool = False  # Whether to allow negative speeds


def sample_eventual_targets(
    num_samples: int,
    params: GeneratorParams,
    device: torch.device = None
) -> torch.Tensor:
    """Sample eventual target speeds with zero-stop probability.

    Args:
        num_samples: Number of targets to sample
        params: Generator parameters
        device: Torch device

    Returns:
        Target speeds tensor [num_samples]
    """
    if device is None:
        device = torch.device('cpu')

    # Sample whether each target should be zero
    is_zero = torch.rand(num_samples, device=device) < params.p_zero_stop

    # Sample non-zero targets
    non_zero_targets = torch.rand(num_samples, device=device) * (
        params.v_max_sample - params.v_min
    ) + params.v_min

    # If allowing negative speeds, randomly flip sign
    if params.allow_negative_speeds:
        sign_flip = torch.rand(num_samples, device=device) < 0.5
        non_zero_targets = torch.where(sign_flip, -non_zero_targets, non_zero_targets)

    # Set zero targets
    targets = torch.where(is_zero, torch.zeros_like(non_zero_targets), non_zero_targets)

    return targets


def sample_arrival_times(
    num_samples: int,
    params: GeneratorParams,
    device: torch.device = None
) -> torch.Tensor:
    """Sample arrival times uniformly between t_min and t_max.

    Args:
        num_samples: Number of times to sample
        params: Generator parameters
        device: Torch device

    Returns:
        Arrival times tensor [num_samples]
    """
    if device is None:
        device = torch.device('cpu')

    return torch.rand(num_samples, device=device) * (
        params.t_max - params.t_min
    ) + params.t_min


def sample_initial_targets(
    batch_size: int,
    params: GeneratorParams,
    device: torch.device = None
) -> torch.Tensor:
    """Sample initial eventual targets for a batch.

    Args:
        batch_size: Number of episodes in batch
        params: Generator parameters
        device: Torch device

    Returns:
        Initial targets tensor [batch_size]
    """
    return sample_eventual_targets(batch_size, params, device)


def sample_initial_arrival_deadlines(
    batch_size: int,
    params: GeneratorParams,
    device: torch.device = None
) -> torch.Tensor:
    """Sample initial arrival deadlines for a batch.

    Args:
        batch_size: Number of episodes in batch
        params: Generator parameters
        device: Torch device

    Returns:
        Initial arrival deadlines tensor [batch_size] (in seconds)
    """
    return sample_arrival_times(batch_size, params, device)


def sample_grades_for_episode(
    episode_length: int,
    device: torch.device = None
) -> torch.Tensor:
    """Sample grade profile for an episode (currently flat road).

    Args:
        episode_length: Length of episode in steps
        device: Torch device

    Returns:
        Grade profile tensor [episode_length] (radians)
    """
    if device is None:
        device = torch.device('cpu')

    # For now, return flat road (zero grade)
    # TODO: Implement varied grade sampling if needed
    return torch.zeros(episode_length, device=device)


def generate_ou_spatial_grids(
    batch_size: int,
    L_total: float,
    ds: float = 1.0,
    mu: float = 0.0,
    theta: float = 0.02,  # 1/l_corr, l_corr=50m
    sigma: float = 0.0028,  # derived from sigma_g_stat=0.002, theta=0.02
    g_min: float = -0.08,
    g_max: float = 0.08,
    device: torch.device = None
) -> torch.Tensor:
    """Generate OU spatial grade grids for a batch of episodes.

    Args:
        batch_size: Number of episodes in batch
        L_total: Total spatial length (m)
        ds: Spatial grid step (m)
        mu: OU mean (radians)
        theta: OU reversion rate (1/m)
        sigma: OU noise intensity (radians/sqrt(m))
        g_min: Minimum grade (radians)
        g_max: Maximum grade (radians)
        device: Torch device

    Returns:
        Grade grids tensor [batch_size, N] where N = ceil(L_total/ds) + 1
    """
    if device is None:
        device = torch.device('cpu')

    N = int(torch.ceil(torch.tensor(L_total / ds, device=device)).item()) + 1

    # Initialize grade grid [B, N]
    x = torch.empty(batch_size, N, device=device)

    # Initial values from stationary distribution
    var = (sigma**2) / (2 * theta)  # Stationary variance
    x[:, 0] = mu + torch.sqrt(var) * torch.randn(batch_size, device=device)

    # Generate OU process along spatial dimension
    sqrt_ds = torch.sqrt(torch.tensor(ds, device=device))
    eps = torch.randn(batch_size, N-1, device=device)

    for n in range(N-1):
        x[:, n+1] = x[:, n] + theta * (mu - x[:, n]) * ds + sigma * sqrt_ds * eps[:, n]

    # Clip to allowed grade bounds
    x = torch.clamp(x, g_min, g_max)

    return x  # [B, N] in radians


def sample_grade_at_positions(
    g_grid: torch.Tensor,
    ds: float,
    s: torch.Tensor,
    device: torch.device = None
) -> torch.Tensor:
    """Sample grades at given spatial positions using linear interpolation.

    Args:
        g_grid: Grade grid [B, N]
        ds: Spatial grid step (m)
        s: Positions [B, T] (m)
        device: Torch device

    Returns:
        Grades at positions [B, T] (radians)
    """
    if device is None:
        device = torch.device('cpu')

    B, N = g_grid.shape
    T = s.shape[1]

    # Compute grid indices and interpolation weights
    idx = torch.floor(s / ds).long().clamp(0, N - 2)  # [B, T]

    # Interpolation weights
    s0 = idx.float() * ds
    alpha = ((s - s0) / ds).clamp(0.0, 1.0)  # [B, T]

    # Gather values at grid points
    g_i = g_grid.gather(1, idx)          # [B, T]
    g_ip1 = g_grid.gather(1, idx + 1)    # [B, T]

    # Linear interpolation
    g_time = (1 - alpha) * g_i + alpha * g_ip1

    return g_time  # [B, T]


def positions_from_speed(
    v: torch.Tensor,
    dt: float,
    device: torch.device = None
) -> torch.Tensor:
    """Compute cumulative positions from speed time series.

    Args:
        v: Speed time series [B, T] (m/s)
        dt: Time step (s)
        device: Torch device

    Returns:
        Positions [B, T] (m)
    """
    if device is None:
        device = torch.device('cpu')

    # Cumulative sum of distances traveled
    s = torch.cumsum(v * dt, dim=1)  # [B, T]

    # Adjust so position starts at 0 (optional, depending on convention)
    # s = s - v[:, 0:1] * dt  # Start at 0

    return s


def sample_grades_for_batch(
    batch_size: int,
    episode_length: int,
    dt: float = 0.1,
    v_max_sample: float = 30.0,
    ds: float = 1.0,
    l_corr: float = 50.0,
    sigma_g_stat: float = 0.002,
    g_min: float = -0.08,
    g_max: float = 0.08,
    device: torch.device = None
) -> torch.Tensor:
    """Sample grade profiles for a batch of episodes using OU spatial process.

    Args:
        batch_size: Number of episodes in batch
        episode_length: Length of each episode in steps
        dt: Time step (s)
        v_max_sample: Maximum speed for distance calculation (m/s)
        ds: Spatial grid step (m)
        l_corr: Correlation length (m)
        sigma_g_stat: Stationary grade standard deviation (radians)
        g_min: Minimum grade (radians)
        g_max: Maximum grade (radians)
        device: Torch device

    Returns:
        Grade profiles tensor [batch_size, episode_length] (radians)
    """
    if device is None:
        device = torch.device('cpu')

    # OU parameters
    mu = 0.0  # Mean grade (flat road)
    theta = 1.0 / l_corr  # Reversion rate
    sigma = torch.sqrt(torch.tensor(2 * theta, device=device)) * sigma_g_stat  # Noise intensity

    # Total spatial length (conservative estimate)
    episode_time = episode_length * dt
    L_total = v_max_sample * episode_time * 1.2  # 20% safety margin

    # Generate spatial grade grids
    g_grid = generate_ou_spatial_grids(
        batch_size=batch_size,
        L_total=L_total,
        ds=ds,
        mu=mu,
        theta=theta,
        sigma=sigma,
        g_min=g_min,
        g_max=g_max,
        device=device
    )

    # For now, assume constant speed for position calculation
    # In practice, this would use the actual target speeds
    v_const = torch.full((batch_size, episode_length), v_max_sample * 0.5, device=device)
    s = positions_from_speed(v_const, dt, device)

    # Sample grades at temporal positions
    g_time = sample_grade_at_positions(g_grid, ds, s, device)

    return g_time  # [B, T]
