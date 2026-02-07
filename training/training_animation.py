#!/usr/bin/env python3
"""Online training animation module.

Creates animated visualizations of policy performance on a fixed episode,
showing how reward components evolve during training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
# Try to use interactive backend, fall back to Agg if not available
_backend_set = False
for backend in ['TkAgg', 'Qt5Agg', 'Qt4Agg']:
    try:
        matplotlib.use(backend)
        _backend_set = True
        break
    except Exception:
        continue
if not _backend_set:
    matplotlib.use('Agg')  # Fall back to non-interactive

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from env.longitudinal_env import LongitudinalEnv, LongitudinalEnvConfig
from training.train_sac import SACTrainer
from utils.dynamics import RandomizationConfig

# Global figure handle for updating the same window
_animation_figure = None


class RewardComponentTracker:
    """Tracks individual reward components during an episode."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset tracking for new episode."""
        self.tracking_error = []
        self.bias_penalty = []
        self.jerk_penalty = []
        self.action_penalty = []
        self.smooth_action_penalty = []
        self.horizon_penalty = []
        self.oscillation_penalty = []
        self.overshoot_penalty = []
        self.max_acc_penalty = []  # Maximum acceleration penalty (single value at episode end)
        self.max_jerk_penalty = []  # Maximum jerk penalty (single value at episode end)
        self.lqr_error_penalty = []  # LQR error (tracking) penalty
        self.lqr_accel_penalty = []  # LQR acceleration penalty
        self.lqr_jerk_penalty = []  # LQR jerk penalty
        self.negative_speed_penalty = []  # Negative speed penalty
        self.zero_speed_error_penalty = []  # Zero speed error penalty
        self.zero_speed_throttle_penalty = []  # Zero speed throttle penalty
        self.total_reward = []
        self.speeds = []
        self.references = []
        self.filtered_references = []  # Filtered reference for reward calculation
        self.actions = []
        self.time_steps = []
        self.annealing_multiplier = []  # Track annealing multiplier
    
    def add_step(
        self,
        reward: float,
        speed: float,
        reference: float,
        action: float,
        time_step: float,
        reward_components: Optional[Dict[str, float]] = None,
        annealing_multiplier: float = 1.0,
        filtered_reference: Optional[float] = None
    ):
        """Add a step's data."""
        self.total_reward.append(reward)
        self.speeds.append(speed)
        self.references.append(reference)
        self.filtered_references.append(filtered_reference if filtered_reference is not None else reference)
        self.actions.append(action)
        self.time_steps.append(time_step)
        self.annealing_multiplier.append(annealing_multiplier)
        
        if reward_components:
            # reward_components already contain weighted values (with annealing applied and normalized)
            self.tracking_error.append(reward_components.get('tracking_error', 0.0))
            self.bias_penalty.append(reward_components.get('bias_penalty', 0.0))
            self.jerk_penalty.append(reward_components.get('jerk_penalty', 0.0))
            self.action_penalty.append(reward_components.get('action_penalty', 0.0))
            self.smooth_action_penalty.append(reward_components.get('smooth_action_penalty', 0.0))
            self.horizon_penalty.append(reward_components.get('horizon_penalty', 0.0))
            self.oscillation_penalty.append(reward_components.get('oscillation_penalty', 0.0))
            self.overshoot_penalty.append(reward_components.get('overshoot_penalty', 0.0))
            self.max_acc_penalty.append(reward_components.get('max_acc_penalty', 0.0))
            self.max_jerk_penalty.append(reward_components.get('max_jerk_penalty', 0.0))
            self.lqr_error_penalty.append(reward_components.get('lqr_error_penalty', 0.0))
            self.lqr_accel_penalty.append(reward_components.get('lqr_accel_penalty', 0.0))
            self.lqr_jerk_penalty.append(reward_components.get('lqr_jerk_penalty', 0.0))
            self.negative_speed_penalty.append(reward_components.get('negative_speed_penalty', 0.0))
            self.zero_speed_error_penalty.append(reward_components.get('zero_speed_error_penalty', 0.0))
            self.zero_speed_throttle_penalty.append(reward_components.get('zero_speed_throttle_penalty', 0.0))


def get_active_reward_components(env_config: LongitudinalEnvConfig) -> Dict[str, bool]:
    """Determine which reward components are active based on config weights.
    
    Args:
        env_config: Environment configuration
        
    Returns:
        Dictionary mapping component names to whether they're active (weight > 0)
    """
    return {
        'tracking_error': env_config.track_weight > 0.0,
        'bias_penalty': env_config.bias_penalty_enabled and env_config.bias_penalty_weight > 0.0,
        'jerk_penalty': env_config.jerk_weight > 0.0,
        'action_penalty': env_config.action_weight > 0.0,
        'smooth_action_penalty': env_config.smooth_action_weight > 0.0,
        'horizon_penalty': env_config.horizon_penalty_weight > 0.0,
        'oscillation_penalty': env_config.oscillation_weight > 0.0,
        'overshoot_penalty': env_config.overshoot_weight > 0.0,
        'max_acc_penalty': env_config.max_acc_penalty_weight > 0.0,
        'max_jerk_penalty': env_config.max_jerk_penalty_weight > 0.0,
        'lqr_error_penalty': env_config.lqr_penalty_enabled and env_config.lqr_error_weight > 0.0,
        'lqr_accel_penalty': env_config.lqr_penalty_enabled and env_config.lqr_accel_weight > 0.0,
        'lqr_jerk_penalty': env_config.lqr_penalty_enabled and env_config.lqr_jerk_weight > 0.0,
        'negative_speed_penalty': env_config.negative_speed_weight > 0.0,
        'zero_speed_error_penalty': env_config.zero_speed_error_weight > 0.0,
        'zero_speed_throttle_penalty': env_config.zero_speed_throttle_weight > 0.0,
    }


def create_training_animation(
    trainer: SACTrainer,
    eval_env: LongitudinalEnv,
    fixed_reference: np.ndarray,
    output_dir: Path,
    step: int,
    animation_history: List[Dict],
    max_history: int = 50,
    initial_speed_errors: List[float] | None = None,
    initial_actions: List[float] | None = None,
    num_episodes: int = 3
) -> Optional[Path]:
    """Create an animated visualization of policy performance.
    
    Args:
        trainer: SACTrainer instance with current policy
        eval_env: Evaluation environment
        fixed_reference: Fixed reference speed profile to evaluate on
        output_dir: Directory to save animation
        step: Current training step
        animation_history: List of previous animation data (for reward component trends)
        max_history: Maximum number of historical steps to keep
        initial_speed_errors: List of initial speed errors (m/s) for each episode.
                             If None, generates errors covering small, medium, and large ranges.
        initial_actions: List of initial actions for each episode (constant across training).
                        If None, generates random but fixed actions.
        num_episodes: Number of example episodes to show (default: 3)
    
    Returns:
        Path to saved animation file, or None if failed
    """
    try:
        # Determine initial speed errors if not provided
        reference_initial_speed = float(fixed_reference[0])
        if initial_speed_errors is None:
            # Default to -2, 0, 2 m/s errors for 3 episodes
            if num_episodes == 1:
                initial_speed_errors = [0.0]
            elif num_episodes == 2:
                initial_speed_errors = [-1.0, 1.0]
            elif num_episodes == 3:
                initial_speed_errors = [-2.0, 0.0, 2.0]
            else:
                # For more episodes, distribute evenly from -2 to 2
                initial_speed_errors = np.linspace(-2.0, 2.0, num_episodes).tolist()
        
        # Ensure we have the right number of errors
        if len(initial_speed_errors) != num_episodes:
            raise ValueError(f"Number of initial_speed_errors ({len(initial_speed_errors)}) "
                           f"must match num_episodes ({num_episodes})")
        
        # Generate fixed initial actions if not provided
        if initial_actions is None:
            # Generate random but fixed initial actions (using a fixed seed for reproducibility)
            import random
            rng = random.Random(42)  # Fixed seed for reproducibility
            initial_actions = [rng.uniform(eval_env.config.action_low, eval_env.config.action_high) 
                              for _ in range(num_episodes)]
        
        # Ensure we have the right number of actions
        if len(initial_actions) != num_episodes:
            raise ValueError(f"Number of initial_actions ({len(initial_actions)}) "
                           f"must match num_episodes ({num_episodes})")
        
        # Run multiple episodes and collect data
        episode_trackers = []
        for episode_idx in range(num_episodes):
            # Calculate initial speed: reference speed + error
            initial_speed_error = initial_speed_errors[episode_idx]
            initial_speed = reference_initial_speed + initial_speed_error
            initial_action = initial_actions[episode_idx]
            
            # Reset environment with fixed reference, episode-specific initial speed, and fixed initial action
            reset_options = {
                "reference_profile": fixed_reference,
                "initial_speed": initial_speed,
                "initial_action": initial_action
            }
            obs, _ = eval_env.reset(options=reset_options)
            
            # Set global step in eval_env for correct annealing multiplier
            eval_env.set_global_step(step)
        
            # Reset context state if enabled
            if trainer.context_mode == "sysid":
                from src.sysid.encoder import FeatureBuilder
                from src.sysid.integration import compute_z_online
                encoder_hidden = trainer.encoder.reset(batch_size=1, device=trainer.device).squeeze(0)
                feature_builder = FeatureBuilder(dt=eval_env.config.dt)
                prev_speed = obs[0]
                prev_action = 0.0
            elif trainer.context_mode == "dynamics_map":
                # Reset dynamics map context for evaluation
                trainer.dynamics_map_context.reset()
                encoder_hidden = None
                feature_builder = None
                prev_speed = None
                prev_action = None
            else:
                encoder_hidden = None
                feature_builder = None
                prev_speed = None
                prev_action = None
            
            # Run episode and collect data
            tracker = RewardComponentTracker()
            done = False
            step_count = 0
            
            while not done:
                # Compute context based on mode
                if trainer.context_mode == "sysid":
                    current_speed = obs[0]
                    encoder_hidden, z_t = compute_z_online(
                        encoder=trainer.encoder,
                        feature_builder=feature_builder,
                        encoder_norm=trainer.encoder_norm,
                        h_prev=encoder_hidden,
                        v_t=current_speed,
                        u_t=prev_action,
                        device=trainer.device
                    )
                    z_t_np = z_t.cpu().numpy()
                    obs_aug = np.concatenate([obs, z_t_np])
                elif trainer.context_mode == "dynamics_map":
                    # Get context from dynamics map
                    z_t = trainer.dynamics_map_context.get_context()
                    z_t_np = z_t.cpu().numpy()
                    obs_aug = np.concatenate([obs, z_t_np])
                else:
                    obs_aug = obs
                
                # Get action
                _, action_scalar = trainer.act(obs_aug, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = eval_env.step(action_scalar)
                
                # Extract reward components from info if available
                reward_components = info.get('reward_components', None)
                
                # Get current annealing multiplier
                annealing_mult = eval_env.get_comfort_anneal_multiplier()
                
                # Track data
                current_speed = float(obs[0])
                current_ref = float(fixed_reference[min(step_count, len(fixed_reference) - 1)])
                # Get filtered reference for reward calculation (if available)
                filtered_ref = None
                if eval_env.filtered_reference is not None and eval_env._ref_idx < len(eval_env.filtered_reference):
                    filtered_ref = float(eval_env.filtered_reference[eval_env._ref_idx])
                tracker.add_step(
                    reward=reward,
                    speed=current_speed,
                    reference=current_ref,
                    action=action_scalar,
                    time_step=step_count * eval_env.config.dt,
                    reward_components=reward_components if reward_components else None,
                    annealing_multiplier=annealing_mult,
                    filtered_reference=filtered_ref
                )
                
                # Update context state for SysID
                if trainer.context_mode == "sysid":
                    prev_speed = current_speed
                    prev_action = action_scalar
                
                done = terminated or truncated
                step_count += 1
            
            # Add tracker after episode completes
            if len(tracker.time_steps) > 0:
                episode_trackers.append(tracker)
        
        if len(episode_trackers) == 0:
            print(f"[animation] Warning: No data collected at step {step}")
            return None
        
        # Find maximum time across all episodes for shared time axis
        max_time = max(max(t.time_steps) if t.time_steps else 0 for t in episode_trackers)
        
        # Reuse existing figure or create new one
        global _animation_figure
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if figure exists and is valid
        if _backend_set and _animation_figure is not None:
            try:
                if plt.fignum_exists(_animation_figure.number):
                    # Reuse existing figure - clear all axes
                    fig = _animation_figure
                    fig.clear()  # Clear all axes
                    # Recreate gridspec
                    gs = fig.add_gridspec(4, num_episodes, hspace=0.4, wspace=0.3, 
                                         height_ratios=[1, 1, 1, 1.2])
                else:
                    # Figure was closed externally, create new one
                    fig = plt.figure(figsize=(20, 14))
                    gs = fig.add_gridspec(4, num_episodes, hspace=0.4, wspace=0.3, 
                                         height_ratios=[1, 1, 1, 1.2])
                    _animation_figure = fig
            except Exception:
                # Error accessing figure, create new one
                fig = plt.figure(figsize=(20, 14))
                gs = fig.add_gridspec(4, num_episodes, hspace=0.4, wspace=0.3, 
                                     height_ratios=[1, 1, 1, 1.2])
                _animation_figure = fig
        else:
            # Create new figure with layout for multiple episodes
            # Layout: 4 rows x num_episodes cols
            # Row 0: Speed tracking for all episodes (shared x-axis)
            # Row 1: Actions for all episodes (shared x-axis)
            # Row 2: Reward breakdown for all episodes (shared x-axis)
            # Row 3: Training trends (spans first 2 cols) + Annealing multiplier (last col)
            fig = plt.figure(figsize=(20, 14))
            gs = fig.add_gridspec(4, num_episodes, hspace=0.4, wspace=0.3, 
                                 height_ratios=[1, 1, 1, 1.2])
            if _backend_set:
                _animation_figure = fig
        
        # Colors for episodes (cycle through if more than 3)
        base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        episode_colors = [base_colors[i % len(base_colors)] for i in range(num_episodes)]
        
        # Plot speed tracking for each episode (shared time axis)
        for ep_idx, tracker in enumerate(episode_trackers):
            ax = fig.add_subplot(gs[0, ep_idx])
            # Show reference on all plots
            ax.plot(tracker.time_steps, tracker.references, 'k--', label='Reference', linewidth=2, alpha=0.7)
            # Plot filtered reference only if filter_reward_mode or violent_profile_mode is enabled
            # AND if it's different from regular reference
            should_show_filtered = (
                eval_env.config.filter_reward_mode or 
                eval_env.config.violent_profile_mode
            )
            if should_show_filtered and tracker.filtered_references and any(fr != ref for fr, ref in zip(tracker.filtered_references, tracker.references)):
                ax.plot(tracker.time_steps, tracker.filtered_references, 'r', label='Filtered Reference (reward)', linewidth=2, alpha=0.6, linestyle='--')
            error_str = f" (err: {initial_speed_errors[ep_idx]:+.1f} m/s)"
            ax.plot(tracker.time_steps, tracker.speeds, color=episode_colors[ep_idx], 
                   label=f'Actual Speed{error_str}', linewidth=2)
            
            # Mark initial speed with a star
            if len(tracker.speeds) > 0:
                initial_speed = tracker.speeds[0]
                ax.scatter([0.0], [initial_speed], color='orange', s=150, marker='*', 
                          zorder=5, label='Initial Speed', edgecolors='black', linewidths=0.5)
            
            ax.fill_between(tracker.time_steps, tracker.references, tracker.speeds, 
                           where=np.array(tracker.speeds) < np.array(tracker.references),
                           alpha=0.2, color='red', label='Error')
            ax.fill_between(tracker.time_steps, tracker.references, tracker.speeds,
                           where=np.array(tracker.speeds) >= np.array(tracker.references),
                           alpha=0.2, color='green', label='Overshoot')
            ax.set_xlim([0, max_time])
            if ep_idx == 0:
                ax.set_ylabel('Speed (m/s)')
            ax.set_title(f'Episode {ep_idx+1}{error_str}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot actions for each episode (shared time axis)
        for ep_idx, tracker in enumerate(episode_trackers):
            ax = fig.add_subplot(gs[1, ep_idx])
            ax.plot(tracker.time_steps, tracker.actions, color=episode_colors[ep_idx], 
                   linewidth=1.5, label='Action')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.fill_between(tracker.time_steps, tracker.actions, 0,
                           where=np.array(tracker.actions) > 0,
                           alpha=0.3, color='green', label='Throttle')
            ax.fill_between(tracker.time_steps, tracker.actions, 0,
                           where=np.array(tracker.actions) < 0,
                           alpha=0.3, color='red', label='Brake')
            
            # Mark initial actuation (previous action at episode start)
            if ep_idx < len(initial_actions):
                initial_action = initial_actions[ep_idx]
                ax.scatter([0.0], [initial_action], color='orange', s=100, marker='*', 
                          zorder=5, label='Initial Actuation', edgecolors='black', linewidths=0.5)
            
            ax.set_xlim([0, max_time])
            ax.set_ylim([-1.1, 1.1])
            if ep_idx == 0:
                ax.set_ylabel('Action')
            ax.set_title('Control Actions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            if ep_idx == num_episodes - 1:
                ax.set_xlabel('Time (s)')
        
        # Determine which components are active based on config
        active_components = get_active_reward_components(eval_env.config)
        
        # Plot reward breakdown for each episode (shared time axis) - use line plots instead of stacked
        for ep_idx, tracker in enumerate(episode_trackers):
            ax = fig.add_subplot(gs[2, ep_idx])
            if tracker.tracking_error:
                # Line plots instead of stacked area to avoid overlap
                # These are already weighted (including annealing) and normalized from reward_components
                # Only plot active components
                if active_components['tracking_error']:
                    ax.plot(tracker.time_steps, tracker.tracking_error, 'b-', label='Tracking', linewidth=1.5, alpha=0.8)
                if active_components['bias_penalty']:
                    ax.plot(tracker.time_steps, tracker.bias_penalty, 'c-', label='Bias', linewidth=1.5, alpha=0.8)
                if active_components['jerk_penalty']:
                    ax.plot(tracker.time_steps, tracker.jerk_penalty, 'r-', label='Jerk', linewidth=1.5, alpha=0.8)
                if active_components['action_penalty']:
                    ax.plot(tracker.time_steps, tracker.action_penalty, 'g-', label='Action', linewidth=1.5, alpha=0.8)
                if active_components['smooth_action_penalty']:
                    ax.plot(tracker.time_steps, tracker.smooth_action_penalty, 'm-', label='Smooth', linewidth=1.5, alpha=0.8)
                if active_components['horizon_penalty']:
                    ax.plot(tracker.time_steps, tracker.horizon_penalty, 'y-', label='Horizon', linewidth=1.5, alpha=0.8)
                if active_components['oscillation_penalty']:
                    ax.plot(tracker.time_steps, tracker.oscillation_penalty, 'orange', label='Oscillation', linewidth=1.5, alpha=0.8, linestyle=':')
                if active_components['overshoot_penalty']:
                    ax.plot(tracker.time_steps, tracker.overshoot_penalty, 'brown', label='Overshoot', linewidth=1.5, alpha=0.8, linestyle=':')
                if active_components['max_acc_penalty']:
                    ax.plot(tracker.time_steps, tracker.max_acc_penalty, 'pink', label='Max Acc', linewidth=1.5, alpha=0.8, linestyle='--')
                if active_components['max_jerk_penalty']:
                    ax.plot(tracker.time_steps, tracker.max_jerk_penalty, 'gray', label='Max Jerk', linewidth=1.5, alpha=0.8, linestyle='--')
                if active_components['lqr_error_penalty']:
                    ax.plot(tracker.time_steps, tracker.lqr_error_penalty, 'darkblue', label='LQR Error', linewidth=1.5, alpha=0.8, linestyle='-.')
                if active_components['lqr_accel_penalty']:
                    ax.plot(tracker.time_steps, tracker.lqr_accel_penalty, 'magenta', label='LQR Accel', linewidth=1.5, alpha=0.8, linestyle='-.')
                if active_components['lqr_jerk_penalty']:
                    ax.plot(tracker.time_steps, tracker.lqr_jerk_penalty, 'teal', label='LQR Jerk', linewidth=1.5, alpha=0.8, linestyle='-.')
                if active_components['negative_speed_penalty']:
                    ax.plot(tracker.time_steps, tracker.negative_speed_penalty, 'navy', label='Neg Speed', linewidth=1.5, alpha=0.8, linestyle=':')
                if active_components['zero_speed_error_penalty']:
                    ax.plot(tracker.time_steps, tracker.zero_speed_error_penalty, 'maroon', label='Zero Speed Err', linewidth=1.5, alpha=0.8, linestyle=':')
                if active_components['zero_speed_throttle_penalty']:
                    ax.plot(tracker.time_steps, tracker.zero_speed_throttle_penalty, 'crimson', label='Zero Speed Throttle', linewidth=1.5, alpha=0.8, linestyle=':')
            else:
                # Fallback: total reward
                ax.plot(tracker.time_steps, tracker.total_reward, 'b-', label='Total Reward', linewidth=1.5)
            ax.set_xlim([0, max_time])
            if ep_idx == 0:
                ax.set_ylabel('Reward')
            ax.set_title('Reward Components')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            if ep_idx == num_episodes - 1:
                ax.set_xlabel('Time (s)')
        
        # Plot 4: Reward component trends over training (weighted components) - spans first 2 columns
        ax4 = fig.add_subplot(gs[3, :num_episodes-1])
        if len(animation_history) > 0:
            # Extract steps and ensure they're correct (not offset by animation_interval)
            steps = [h['step'] for h in animation_history]
            # These are already weighted (including annealing) and normalized from reward_components
            # Only plot active components
            if active_components['tracking_error']:
                tracking_errors = [h.get('avg_tracking_error', 0) for h in animation_history]
                ax4.plot(steps, tracking_errors, 'o-', label='Tracking', linewidth=2, markersize=6)
            if active_components['bias_penalty']:
                bias_penalties = [h.get('avg_bias_penalty', 0) for h in animation_history]
                ax4.plot(steps, bias_penalties, 'o-', label='Bias', linewidth=2, markersize=6, color='cyan')
            if active_components['jerk_penalty']:
                jerk_penalties = [h.get('avg_jerk_penalty', 0) for h in animation_history]
                ax4.plot(steps, jerk_penalties, 's-', label='Jerk', linewidth=2, markersize=6)
            if active_components['action_penalty']:
                action_penalties = [h.get('avg_action_penalty', 0) for h in animation_history]
                ax4.plot(steps, action_penalties, '^-', label='Action', linewidth=2, markersize=6)
            if active_components['smooth_action_penalty']:
                smooth_action_penalties = [h.get('avg_smooth_action_penalty', 0) for h in animation_history]
                ax4.plot(steps, smooth_action_penalties, 'd-', label='Smooth', linewidth=2, markersize=6)
            if active_components['horizon_penalty']:
                horizon_penalties = [h.get('avg_horizon_penalty', 0) for h in animation_history]
                ax4.plot(steps, horizon_penalties, 'o-', label='Horizon', linewidth=2, markersize=6, color='yellow')
            if active_components['oscillation_penalty']:
                oscillation_penalties = [h.get('avg_oscillation_penalty', 0) for h in animation_history]
                ax4.plot(steps, oscillation_penalties, 'o', label='Oscillation', linewidth=2, markersize=6, color='orange', linestyle=':')
            if active_components['overshoot_penalty']:
                overshoot_penalties = [h.get('avg_overshoot_penalty', 0) for h in animation_history]
                ax4.plot(steps, overshoot_penalties, 'o', label='Overshoot', linewidth=2, markersize=6, color='brown', linestyle=':')
            if active_components['max_acc_penalty']:
                max_acc_penalties = [h.get('avg_max_acc_penalty', 0) for h in animation_history]
                ax4.plot(steps, max_acc_penalties, 'y^', label='Max Acc', linewidth=2, markersize=6, linestyle='--')
            if active_components['max_jerk_penalty']:
                max_jerk_penalties = [h.get('avg_max_jerk_penalty', 0) for h in animation_history]
                ax4.plot(steps, max_jerk_penalties, 'k^', label='Max Jerk', linewidth=2, markersize=6, linestyle='--')
            if active_components['lqr_error_penalty']:
                lqr_error_penalties = [h.get('avg_lqr_error_penalty', 0) for h in animation_history]
                ax4.plot(steps, lqr_error_penalties, 'o', label='LQR Error', linewidth=2, markersize=6, color='darkblue', linestyle='-.')
            if active_components['lqr_accel_penalty']:
                lqr_accel_penalties = [h.get('avg_lqr_accel_penalty', 0) for h in animation_history]
                ax4.plot(steps, lqr_accel_penalties, 'o', label='LQR Accel', linewidth=2, markersize=6, color='magenta', linestyle='-.')
            if active_components['lqr_jerk_penalty']:
                lqr_jerk_penalties = [h.get('avg_lqr_jerk_penalty', 0) for h in animation_history]
                ax4.plot(steps, lqr_jerk_penalties, 'o', label='LQR Jerk', linewidth=2, markersize=6, color='teal', linestyle='-.')
            if active_components['negative_speed_penalty']:
                negative_speed_penalties = [h.get('avg_negative_speed_penalty', 0) for h in animation_history]
                ax4.plot(steps, negative_speed_penalties, 'o', label='Neg Speed', linewidth=2, markersize=6, color='navy', linestyle=':')
            if active_components['zero_speed_error_penalty']:
                zero_speed_error_penalties = [h.get('avg_zero_speed_error_penalty', 0) for h in animation_history]
                ax4.plot(steps, zero_speed_error_penalties, 'o', label='Zero Speed Err', linewidth=2, markersize=6, color='maroon', linestyle=':')
            if active_components['zero_speed_throttle_penalty']:
                zero_speed_throttle_penalties = [h.get('avg_zero_speed_throttle_penalty', 0) for h in animation_history]
                ax4.plot(steps, zero_speed_throttle_penalties, 'o', label='Zero Speed Throttle', linewidth=2, markersize=6, color='crimson', linestyle=':')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Reward Component')
            ax4.set_title('Reward Components Over Training')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No training history yet', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Reward Components Over Training')
        
        # Plot 5: Annealing multiplier over training - right side (last column)
        ax5 = fig.add_subplot(gs[3, -1])  # Last column
        if len(animation_history) > 0:
            steps = [h['step'] for h in animation_history]
            annealing_multipliers = [h.get('avg_annealing_multiplier', 1.0) for h in animation_history]
            ax5.plot(steps, annealing_multipliers, 'o-', color='purple', label='Annealing', linewidth=2, markersize=4)
            ax5.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Full (1.0)')
            ax5.set_xlabel('Step')
            ax5.set_ylabel('Multiplier')
            ax5.set_title('Annealing', fontsize=10)
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim([0, 1.1])
        else:
            ax5.text(0.5, 0.5, 'No history', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=8)
            ax5.set_title('Annealing', fontsize=10)
        
        plt.suptitle(f'Step {step}', fontsize=16, y=0.995)
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
        
        # Display interactively
        if _backend_set:
            # Interactive backend available - display the plot
            plt.ion()  # Turn on interactive mode
            # Make sure this figure is the current figure
            plt.figure(fig.number)
            # Show if this is the first time (figure was just created)
            if _animation_figure == fig and not hasattr(fig, '_shown'):
                plt.show(block=False)
                fig._shown = True  # Mark as shown
            # Force canvas update without bringing to front
            fig.canvas.draw_idle()  # Use draw_idle for better performance
            fig.canvas.flush_events()
            plt.pause(0.01)  # Brief pause to update display (reduced from 0.1 for less interruption)
        else:
            # Non-interactive backend - save as fallback
            static_path = output_dir / f"training_animation_step_{step}.png"
            fig.savefig(static_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # Store episode summary for history (average across all episodes)
        avg_tracking_error = np.mean([np.mean(np.abs(np.array(t.speeds) - np.array(t.references))) 
                                      for t in episode_trackers])
        avg_bias_penalty = np.mean([np.mean(np.abs(t.bias_penalty)) if t.bias_penalty else 0.0 
                                    for t in episode_trackers])
        avg_jerk_penalty = np.mean([np.mean(np.abs(t.jerk_penalty)) if t.jerk_penalty else 0.0 
                                    for t in episode_trackers])
        avg_action_penalty = np.mean([np.mean(np.abs(t.action_penalty)) if t.action_penalty else 0.0 
                                     for t in episode_trackers])
        avg_smooth_action_penalty = np.mean([np.mean(np.abs(t.smooth_action_penalty)) if t.smooth_action_penalty else 0.0 
                                             for t in episode_trackers])
        avg_oscillation_penalty = np.mean([np.mean(np.abs(t.oscillation_penalty)) if t.oscillation_penalty else 0.0 
                                           for t in episode_trackers])
        avg_overshoot_penalty = np.mean([np.mean(np.abs(t.overshoot_penalty)) if t.overshoot_penalty else 0.0 
                                         for t in episode_trackers])
        avg_max_acc_penalty = np.mean([np.mean(np.abs(t.max_acc_penalty)) if t.max_acc_penalty else 0.0 
                                       for t in episode_trackers])
        avg_max_jerk_penalty = np.mean([np.mean(np.abs(t.max_jerk_penalty)) if t.max_jerk_penalty else 0.0 
                                        for t in episode_trackers])
        avg_lqr_error_penalty = np.mean([np.mean(np.abs(t.lqr_error_penalty)) if t.lqr_error_penalty else 0.0 
                                        for t in episode_trackers])
        avg_lqr_accel_penalty = np.mean([np.mean(np.abs(t.lqr_accel_penalty)) if t.lqr_accel_penalty else 0.0 
                                        for t in episode_trackers])
        avg_lqr_jerk_penalty = np.mean([np.mean(np.abs(t.lqr_jerk_penalty)) if t.lqr_jerk_penalty else 0.0 
                                       for t in episode_trackers])
        avg_negative_speed_penalty = np.mean([np.mean(np.abs(t.negative_speed_penalty)) if t.negative_speed_penalty else 0.0 
                                             for t in episode_trackers])
        avg_zero_speed_error_penalty = np.mean([np.mean(np.abs(t.zero_speed_error_penalty)) if t.zero_speed_error_penalty else 0.0 
                                               for t in episode_trackers])
        avg_zero_speed_throttle_penalty = np.mean([np.mean(np.abs(t.zero_speed_throttle_penalty)) if t.zero_speed_throttle_penalty else 0.0 
                                                  for t in episode_trackers])
        avg_horizon_penalty = np.mean([np.mean(np.abs(t.horizon_penalty)) if t.horizon_penalty else 0.0 
                                      for t in episode_trackers])
        avg_annealing_multiplier = np.mean([np.mean(t.annealing_multiplier) if t.annealing_multiplier else 1.0 
                                            for t in episode_trackers])
        total_reward = np.sum([np.sum(t.total_reward) for t in episode_trackers])
        
        episode_summary = {
            'step': step,  # Store actual step, not offset
            'avg_tracking_error': avg_tracking_error,
            'avg_bias_penalty': avg_bias_penalty,
            'avg_jerk_penalty': avg_jerk_penalty,
            'avg_action_penalty': avg_action_penalty,
            'avg_smooth_action_penalty': avg_smooth_action_penalty,
            'avg_horizon_penalty': avg_horizon_penalty,
            'avg_oscillation_penalty': avg_oscillation_penalty,
            'avg_overshoot_penalty': avg_overshoot_penalty,
            'avg_max_acc_penalty': avg_max_acc_penalty,
            'avg_max_jerk_penalty': avg_max_jerk_penalty,
            'avg_lqr_error_penalty': avg_lqr_error_penalty,
            'avg_lqr_accel_penalty': avg_lqr_accel_penalty,
            'avg_lqr_jerk_penalty': avg_lqr_jerk_penalty,
            'avg_negative_speed_penalty': avg_negative_speed_penalty,
            'avg_zero_speed_error_penalty': avg_zero_speed_error_penalty,
            'avg_zero_speed_throttle_penalty': avg_zero_speed_throttle_penalty,
            'avg_annealing_multiplier': avg_annealing_multiplier,
            'total_reward': total_reward,
            'episode_length': np.mean([len(t.time_steps) for t in episode_trackers]),
        }
        
        # Update history (keep last max_history steps)
        animation_history.append(episode_summary)
        if len(animation_history) > max_history:
            animation_history.pop(0)
        
        # Save history to JSON
        history_path = output_dir / "animation_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(animation_history, f, indent=2)
        
        if _backend_set:
            return None  # No file path when displaying interactively
        else:
            return static_path  # Return path when saving as fallback
        
    except Exception as e:
        print(f"[animation] Warning: Failed to create animation at step {step}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_animation_history(output_dir: Path) -> List[Dict]:
    """Load animation history from JSON file."""
    history_path = output_dir / "animation_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)
    return []

