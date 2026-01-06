"""Main batch target speed generator with event-driven behavior and feasibility constraints."""

from dataclasses import dataclass
import torch
from typing import Optional

from generator.lpf import SecondOrderLPF, second_order_lpf_update
from generator.samplers import (
    GeneratorParams,
    sample_initial_targets,
    sample_initial_arrival_deadlines,
    sample_eventual_targets,
    sample_arrival_times,
    sample_grades_for_batch,
    generate_ou_spatial_grids,
    sample_grade_at_positions,
    positions_from_speed
)
from generator.feasibility import (
    VehicleCapabilities,
    FeasibilityParams,
    project_window_to_feasible,
    compute_feasible_accelerations,
)


@dataclass
class GeneratorConfig:
    """Configuration for the target speed generator."""
    # Internal parameter (set by environment, not in config.yaml)
    prediction_horizon: int = 20  # H

    # LPF parameters (rate limits now computed from vehicle feasibility)
    freq_cutoff: float = 0.6  # Hz
    zeta: float = 0.9
    dt: float = 0.02  # 50 Hz policy step
    jerk_max: float = 12.0  # m/s³ max jerk

    # Generator parameters
    p_change: float = 0.03
    p_zero_stop: float = 0.08
    v_min: float = 0.0
    v_max_sample: float = 30.0
    t_min: float = 2.0
    t_max: float = 12.0
    stop_hold_min: float = 1.0
    stop_hold_max: float = 5.0

    # Stochastic acceleration & jerk parameters
    p_change_acc: float = 0.04  # Probability of changing acceleration scale per step
    p_change_jerk: float = 0.03  # Probability of changing jerk scale per step
    p_zero_accel: float = 0.15  # Probability of sampling zero acceleration when acceleration changes
    accel_beta: float = 2.0  # Beta distribution parameter for acceleration sampling (higher = more small accelerations)

    # Road grade parameters
    ds: float = 1.0  # Spatial grid step (m)
    l_corr: float = 50.0  # Correlation length (m)
    sigma_g_stat: float = 0.002  # Stationary grade std dev (radians)
    g_min: float = -0.08  # Min grade (radians, ~ -4.6°)
    g_max: float = 0.08  # Max grade (radians, ~ +4.6°)

    # Speed limit safety factor
    speed_limit_safety_factor: float = 0.75  # Multiplier for max feasible speed (0.75 = 25% reduction)


class BatchTargetGenerator:
    """Event-driven batch target speed generator with vehicle feasibility constraints."""

    def _compute_vehicle_lpf_limits(self, vehicle: VehicleCapabilities) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute conservative LPF rate limits based on vehicle feasibility constraints.

        Uses representative operating conditions to compute conservative bounds that ensure
        feasibility across typical driving scenarios.
        """
        # Representative conditions for conservative limit calculation
        v_rep = torch.full_like(vehicle.m, 15.0)  # Representative speed (m/s) - highway cruising
        grade_rep = torch.zeros_like(vehicle.m)    # Flat road assumption

        # Compute resistive forces at representative conditions
        F_drag = 0.5 * vehicle.C_dA * v_rep**2  # Aerodynamic drag
        F_roll = vehicle.C_r * vehicle.m * 9.81  # Rolling resistance
        F_resist = F_drag + F_roll  # Total resistance (flat road)

        # Maximum drive acceleration (torque-limited, conservative)
        T_wheel_max = vehicle.i_max * vehicle.K_t * vehicle.eta_gb * vehicle.gear_ratio
        a_drive_max_raw = (T_wheel_max / vehicle.r_w - F_resist) / vehicle.m

        # Maximum braking acceleration (regenerative + mechanical braking)
        # Use mechanical braking as conservative limit (regenerative may vary)
        a_brake_min_raw = -vehicle.T_brake_max / vehicle.r_w / vehicle.m

        # Apply safety margins for conservative operation
        safety_factor = 0.85  # Conservative safety margin

        # Ensure reasonable bounds and positive values
        rate_max = torch.clamp(a_drive_max_raw * safety_factor, min=1.0, max=15.0)
        rate_neg_max = torch.clamp(-a_brake_min_raw * safety_factor, min=1.0, max=20.0)

        return rate_max, rate_neg_max

    def _create_dummy_vehicle_for_init(self) -> VehicleCapabilities:
        """Create a dummy vehicle for initialization purposes."""
        return VehicleCapabilities(
            m=torch.full((self.batch_size,), 1500.0, device=self.device),
            r_w=torch.full((self.batch_size,), 0.3, device=self.device),
            gear_ratio=torch.full((self.batch_size,), 10.0, device=self.device),
            eta_gb=torch.full((self.batch_size,), 0.9, device=self.device),
            K_t=torch.full((self.batch_size,), 0.5, device=self.device),
            K_e=torch.full((self.batch_size,), 0.5, device=self.device),
            R=torch.full((self.batch_size,), 0.1, device=self.device),
            i_max=torch.full((self.batch_size,), 200.0, device=self.device),
            V_max=torch.full((self.batch_size,), 400.0, device=self.device),
            b=torch.full((self.batch_size,), 1e-3, device=self.device),  # Motor viscous damping (Nm·s/rad)
            T_brake_max=torch.full((self.batch_size,), 2000.0, device=self.device),
            mu=torch.full((self.batch_size,), 0.8, device=self.device),
            C_dA=torch.full((self.batch_size,), 0.6, device=self.device),
            C_r=torch.full((self.batch_size,), 0.01, device=self.device)
        )

    def _compute_feasibility_bounds(self, v: torch.Tensor, grade: torch.Tensor, vehicle: VehicleCapabilities) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute current acceleration feasibility bounds at given speed and grade.

        Args:
            v: Speed tensor [B]
            grade: Grade tensor [B] (radians)
            vehicle: Vehicle capabilities

        Returns:
            Tuple of (a_drive_max, a_brake_max) tensors [B]
        """
        # Compute resistive forces
        F_drag = 0.5 * vehicle.C_dA * v**2  # Aerodynamic drag
        F_roll = vehicle.C_r * vehicle.m * 9.81  # Rolling resistance
        F_grade = vehicle.m * 9.81 * torch.sin(grade)  # Grade force
        F_resist = F_drag + F_roll + F_grade

        # Maximum drive acceleration (torque-limited)
        T_wheel_max = vehicle.i_max * vehicle.K_t * vehicle.eta_gb * vehicle.gear_ratio
        a_drive_max = (T_wheel_max / vehicle.r_w - F_resist) / vehicle.m

        # Maximum braking acceleration (magnitude, positive value)
        # Use mechanical braking limit (conservative)
        a_brake_max = vehicle.T_brake_max / vehicle.r_w / vehicle.m

        # Ensure positive values and reasonable bounds
        a_drive_max = torch.clamp(a_drive_max, min=0.0, max=20.0)
        a_brake_max = torch.clamp(a_brake_max, min=0.1, max=15.0)

        return a_drive_max, a_brake_max

    def __init__(self, config: GeneratorConfig, batch_size: int = 1, episode_length: int = 200, device: Optional[torch.device] = None):
        """Initialize the generator.

        Args:
            config: Generator configuration
            batch_size: Number of parallel episodes to generate
            episode_length: Length of each episode in steps
            device: Torch device (CPU/GPU)
        """
        self.config = config
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.device = device or torch.device('cpu')

        # Initialize persistent stochastic state with maximum feasible values
        # This allows full acceleration initially, constrained only when coin flips occur
        dummy_vehicle = self._create_dummy_vehicle_for_init()
        a_drive_max_init, a_brake_max_init = self._compute_feasibility_bounds(
            torch.zeros(batch_size, device=self.device),  # At zero speed
            torch.zeros(batch_size, device=self.device),  # Flat grade
            dummy_vehicle
        )
        self.a_chosen_prev = a_drive_max_init  # Initialize to max feasible acceleration
        self.jerk_bound_prev = torch.full((batch_size,), config.jerk_max, device=self.device)  # Max jerk

        # Create parameter objects
        self.generator_params = GeneratorParams(
            p_change=config.p_change,
            p_zero_stop=config.p_zero_stop,
            v_min=config.v_min,
            v_max_sample=config.v_max_sample,
            t_min=config.t_min,
            t_max=config.t_max,
            stop_hold_min=config.stop_hold_min,
            stop_hold_max=config.stop_hold_max
        )


    def generate_batch(
        self,
        vehicle: VehicleCapabilities
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of target speed sequences.

        Args:
            vehicle: Vehicle capabilities [B] tensors

        Returns:
            Tuple of (targets, grades, raw_targets) tensors:
            - targets: [B, T, H] filtered target speeds for each episode step and horizon
            - grades: [B, T, H] corresponding grade profiles
            - raw_targets: [B, T, H] raw (unfiltered) target speeds for each episode step and horizon
        """
        B, T, H = self.batch_size, self.episode_length, self.config.prediction_horizon

        # Compute vehicle-specific LPF rate limits based on feasibility constraints
        rate_max, rate_neg_max = self._compute_vehicle_lpf_limits(vehicle)
        
        # Pre-compute max feasible speed once (cached for efficiency)
        # Apply configurable safety margin for conservative operation
        from generator.feasibility import compute_max_feasible_speed
        v_max_theoretical = compute_max_feasible_speed(vehicle)  # [B]
        v_max_feasible = self.config.speed_limit_safety_factor * v_max_theoretical
        zero_tensor = torch.tensor(0.0, device=self.device)  # Reusable zero tensor

        # Initialize LPF states for each batch element with vehicle-specific limits
        # Convert scalar jerk_max to tensor
        jerk_max_tensor = torch.full((B,), self.config.jerk_max, device=self.device)

        lpf = SecondOrderLPF(
            batch_size=B,
            freq_cutoff=self.config.freq_cutoff,
            zeta=self.config.zeta,
            dt=self.config.dt,
            rate_max=rate_max,          # Now vehicle-specific
            rate_neg_max=rate_neg_max,  # Now vehicle-specific
            jerk_max=jerk_max_tensor,   # Convert to tensor
            device=self.device
        )

        # Reuse feasibility parameters across steps
        feasibility_params = FeasibilityParams()

        # Initialize event states
        # Constrain initial targets to max feasible speed
        eventual_target_raw = sample_initial_targets(B, self.generator_params, self.device)
        eventual_target = torch.clamp(eventual_target_raw, min=zero_tensor, max=v_max_feasible)
        arrival_deadline = sample_initial_arrival_deadlines(B, self.generator_params, self.device)
        current_event_start = lpf.y.clone()
        start_time = torch.zeros(B, device=self.device)

        # Sample grade profiles for all episodes using OU spatial process
        grades = sample_grades_for_batch(
            batch_size=B,
            episode_length=T + H,  # Extra H for lookahead
            dt=self.config.dt,
            v_max_sample=self.config.v_max_sample,
            ds=self.config.ds,
            l_corr=self.config.l_corr,
            sigma_g_stat=self.config.sigma_g_stat,
            g_min=self.config.g_min,
            g_max=self.config.g_max,
            device=self.device
        )

        # Initialize output tensors
        targets = torch.zeros(B, T, H, device=self.device)
        raw_targets = torch.zeros(B, T, H, device=self.device)  # Track raw (unfiltered) targets

        # Simulation time
        t_now = 0.0

        # Simulate episode generation
        for t in range(T):
            # Coin flip: decide which episodes change target
            coin = (torch.rand(B, device=self.device) < self.generator_params.p_change)

            # Sample new targets and deadlines for changing episodes
            if coin.any():
                num_changes = coin.sum().item()
                new_targets_raw = sample_eventual_targets(num_changes, self.generator_params, self.device)
                # Constrain new targets to max feasible speed (per batch element)
                v_max_for_changes = v_max_feasible[coin]
                new_targets = torch.clamp(new_targets_raw, min=zero_tensor, max=v_max_for_changes)
                new_deadlines = t_now + sample_arrival_times(num_changes, self.generator_params, self.device)

                # Update only the episodes that change
                eventual_target[coin] = new_targets
                arrival_deadline[coin] = new_deadlines
                current_event_start[coin] = lpf.y[coin]
                start_time[coin] = t_now

            # Compute u_target input to LPF
            time_left = arrival_deadline - t_now
            tolerance = 0.1 * torch.abs(eventual_target - current_event_start)
            reached_mask = torch.abs(lpf.y - eventual_target) <= tolerance

            # For reached targets, keep current target (ignore timer)
            # Constrain u_target to max feasible speed to ensure LPF naturally stays below limit
            u_target = torch.clamp(eventual_target.clone(), min=zero_tensor, max=v_max_feasible)

            # Stochastic acceleration and jerk sampling
            # Compute current feasibility bounds at current LPF output y
            grade_current = grades[:, t] if t < grades.shape[1] else torch.zeros(B, device=self.device)
            a_drive_max, a_brake_max = self._compute_feasibility_bounds(lpf.y, grade_current, vehicle)

            # Coin flips for acceleration and jerk changes
            rand_acc = torch.rand(B, device=self.device)
            rand_jerk = torch.rand(B, device=self.device)
            C_acc = (rand_acc < self.config.p_change_acc)
            C_jerk = (rand_jerk < self.config.p_change_jerk)

            # Sample new scalars where coin is True
            # For acceleration: first check if we should sample zero, otherwise use beta distribution
            num_acc_changes = C_acc.sum()
            s_acc_new = torch.zeros(num_acc_changes, device=self.device)
            if num_acc_changes > 0:
                # Check if we should sample zero acceleration
                zero_accel_rand = torch.rand(num_acc_changes, device=self.device)
                zero_accel_mask = (zero_accel_rand < self.config.p_zero_accel)
                
                # For non-zero accelerations, sample from beta distribution (favors smaller values)
                non_zero_mask = ~zero_accel_mask
                if non_zero_mask.any():
                    beta_dist = torch.distributions.Beta(1.0, self.config.accel_beta)
                    s_acc_new[non_zero_mask] = beta_dist.sample([non_zero_mask.sum()])
                # Zero accelerations remain at 0.0
            
            s_jerk_new = torch.rand(C_jerk.sum(), device=self.device)  # [0,1]

            # Initialize with previous values
            a_chosen = self.a_chosen_prev.clone()
            jerk_bound = self.jerk_bound_prev.clone()

            # Determine desired direction based on eventual_target - current_y
            desired_delta = eventual_target - lpf.y
            sign_dir = torch.sign(desired_delta)
            # For near-zero deltas, default to forward direction
            small_delta_mask = (torch.abs(desired_delta) < 1e-3)
            sign_dir = torch.where(small_delta_mask, torch.ones_like(sign_dir), sign_dir)

            # Apply sampled scalars for acceleration (where coin flipped)
            if C_acc.any():
                idx = C_acc.nonzero(as_tuple=True)[0]
                s_acc = s_acc_new

                # Choose acceleration direction and magnitude
                forward_mask = (sign_dir[idx] >= 0)
                a_choice_idx = torch.zeros_like(s_acc)

                # Forward acceleration
                if forward_mask.any():
                    a_choice_idx[forward_mask] = s_acc[forward_mask] * a_drive_max[idx][forward_mask]

                # Braking/deceleration
                if (~forward_mask).any():
                    brake_mask = ~forward_mask
                    a_choice_idx[brake_mask] = -s_acc[brake_mask] * a_brake_max[idx][brake_mask]

                # Clip to ensure feasibility
                a_choice_idx = torch.clamp(a_choice_idx,
                                         min=-a_brake_max[idx],
                                         max=a_drive_max[idx])

                a_chosen[idx] = a_choice_idx

            # Apply sampled scalars for jerk (where coin flipped)
            if C_jerk.any():
                idx_jerk = C_jerk.nonzero(as_tuple=True)[0]
                s_jerk = s_jerk_new
                jerk_bound[idx_jerk] = s_jerk * self.config.jerk_max

            # Store for next iteration
            self.a_chosen_prev = a_chosen
            self.jerk_bound_prev = jerk_bound

            # Update LPF with stochastic acceleration and jerk bounds
            # Use a_chosen as the maximum allowed acceleration in the desired direction
            rate_max = torch.where(sign_dir >= 0, a_chosen, a_drive_max)
            rate_neg_max = a_brake_max  # Always use braking limit for negative rates

            # Temporarily update LPF bounds for this step
            original_rate_max = lpf.rate_max.clone()
            original_rate_neg_max = lpf.rate_neg_max.clone()
            original_jerk_max = lpf.jerk_max

            lpf.rate_max = rate_max
            lpf.rate_neg_max = rate_neg_max
            lpf.jerk_max = jerk_bound

            # Update LPF by one policy step
            lpf.update(u_target)

            # Ensure speeds stay within feasible bounds (non-negative and below max feasible speed)
            lpf.y = torch.clamp(lpf.y, min=zero_tensor, max=v_max_feasible)

            # Restore original bounds (they'll be set again next step)
            lpf.rate_max = original_rate_max
            lpf.rate_neg_max = original_rate_neg_max
            lpf.jerk_max = original_jerk_max

            # Store raw target (u_target) for all horizon steps (for violent profile mode)
            raw_target_window = u_target.unsqueeze(-1).expand(-1, H)  # [B, H] - repeat u_target for all H steps
            
            # Generate raw H-step lookahead by simulating LPF for H steps
            # Use the same stochastic bounds for the entire lookahead window
            y_h = lpf.y.clone()
            r_h = lpf.r.clone()
            raw_window = torch.zeros(B, H, device=self.device)

            for h in range(H):
                # Use same u_target and stochastic bounds for lookahead
                y_h, r_h = second_order_lpf_update(
                    y_h, r_h, u_target, lpf.dt, lpf.wc, lpf.zeta,
                    rate_max, rate_neg_max, jerk_bound, lpf.device
                )
                # Ensure speeds stay within feasible bounds during lookahead
                y_h = torch.clamp(y_h, min=zero_tensor, max=v_max_feasible)
                raw_window[:, h] = y_h

            # Since we've constrained targets, LPF input, and LPF output during generation,
            # the raw_window should already respect the max feasible speed limit.
            # We now add a fast path: if raw_window already satisfies acceleration bounds,
            # we skip the expensive projection. Otherwise, we project to the feasible set.
            grade_window = grades[:, t:t+H] if t + H <= grades.shape[1] else grades[:, t:]
            if grade_window.shape[1] < H:
                # Pad with last grade value if needed
                padding = torch.zeros(B, H - grade_window.shape[1], device=self.device)
                padding[:] = grade_window[:, -1:].expand(-1, H - grade_window.shape[1])
                grade_window = torch.cat([grade_window, padding], dim=1)

            # Fast feasibility check: compute accelerations and see if they are within bounds
            # Raw accelerations from current LPF state and window
            v_with_current = torch.cat([lpf.y.unsqueeze(-1), raw_window], dim=-1)  # [B, H+1]
            a_raw = (v_with_current[:, 1:] - v_with_current[:, :-1]) / self.config.dt  # [B, H]

            # Feasible acceleration bounds over the window
            a_brake_min, a_drive_max = compute_feasible_accelerations(
                raw_window, grade_window, vehicle, feasibility_params
            )

            # Check if all accelerations are within feasible bounds (with small tolerance)
            tol = 1e-5
            within_lower = a_raw >= (a_brake_min - tol)
            within_upper = a_raw <= (a_drive_max + tol)
            all_feasible = bool(torch.all(within_lower & within_upper))

            if all_feasible:
                feasible_window = raw_window
            else:
                # Apply feasibility projection only when needed
                feasible_window = project_window_to_feasible(
                    raw_window, grade_window, vehicle, self.config.dt, feasibility_params
                )

            targets[:, t, :] = feasible_window
            raw_targets[:, t, :] = raw_target_window  # Store raw targets

            # Advance time
            t_now += self.config.dt

        return targets, grades[:, :T], raw_targets  # Return filtered targets, grades, and raw targets


def generate_batch_targets(
    vehicle: VehicleCapabilities,
    config: Optional[GeneratorConfig] = None,
    device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience function to generate batch targets with default config.

    Args:
        vehicle: Vehicle capabilities [B] tensors
        config: Generator configuration (uses defaults if None)
        device: Torch device

    Returns:
        Tuple of (targets, grades) tensors [B, T, H]
    """
    if config is None:
        config = GeneratorConfig()

    generator = BatchTargetGenerator(config, device)
    return generator.generate_batch(vehicle)
