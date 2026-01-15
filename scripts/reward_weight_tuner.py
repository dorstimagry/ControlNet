#!/usr/bin/env python3
"""Interactive GUI for tuning reward penalty weights.

This script provides a GUI with sliders to adjust penalty weights and visualize
how different penalties scale with vehicle states (speed, target speed, actions, accelerations).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Reward weight configuration."""
    track_weight: float = 1.0
    jerk_weight: float = 0.1
    action_weight: float = 0.01
    smooth_action_weight: float = 0.05
    horizon_penalty_weight: float = 0.0
    oscillation_weight: float = 0.0
    overshoot_weight: float = 0.0


def compute_tracking_penalty(speed_error: float, track_weight: float) -> float:
    """Compute speed tracking penalty."""
    return -track_weight * (speed_error ** 2)


def compute_jerk_penalty(jerk: float, jerk_weight: float) -> float:
    """Compute jerk penalty."""
    return -jerk_weight * abs(jerk)


def compute_action_penalty(action: float, action_weight: float) -> float:
    """Compute action penalty."""
    return -action_weight * abs(action)


def compute_smooth_action_penalty(action_change: float, smooth_action_weight: float) -> float:
    """Compute smooth action penalty."""
    return -smooth_action_weight * abs(action_change)


def compute_horizon_penalty(
    current_speed: float,
    future_refs: list[float],
    horizon_penalty_weight: float,
    horizon_decay: float = 0.9
) -> float:
    """Compute horizon penalty."""
    if horizon_penalty_weight == 0.0:
        return 0.0
    
    penalty = 0.0
    decay_factor = 1.0
    for future_ref in future_refs:
        speed_deviation = abs(current_speed - future_ref)
        penalty -= decay_factor * speed_deviation
        decay_factor *= horizon_decay
    
    return horizon_penalty_weight * penalty


def compute_oscillation_penalty(
    action_sign_change: float,
    speed_error: float,
    ref_rate: float,
    oscillation_weight: float,
    error_scale: float = 0.3,
    ref_scale: float = 0.3,
    epsilon: float = 0.05
) -> float:
    """Compute oscillation penalty."""
    if oscillation_weight == 0.0:
        return 0.0
    
    switching_energy = action_sign_change ** 2
    proximity_gate = np.exp(-np.abs(speed_error) / error_scale)
    stationarity_gate = np.exp(-ref_rate / ref_scale)
    
    return -oscillation_weight * switching_energy * proximity_gate * stationarity_gate


def compute_overshoot_penalty(
    error_rate: float,
    speed_error: float,
    prev_error: float,
    overshoot_weight: float,
    error_scale: float = 0.3,
    crossing_scale: float = 0.02
) -> float:
    """Compute overshoot penalty."""
    if overshoot_weight == 0.0:
        return 0.0
    
    x = -(-speed_error * prev_error) / crossing_scale
    x_clipped = np.clip(x, -500.0, 500.0)
    crossing_gate = 1.0 / (1.0 + np.exp(-x_clipped))
    proximity_gate = np.exp(-np.abs(speed_error) / error_scale)
    
    return -overshoot_weight * (error_rate ** 2) * crossing_gate * proximity_gate


class RewardWeightTuner:
    """Interactive GUI for tuning reward weights."""
    
    def __init__(self):
        self.weights = RewardWeights()
        
        # Default vehicle states for visualization
        self.speed_range = np.linspace(0, 20, 100)  # m/s
        self.target_speed = 10.0  # m/s
        self.action_range = np.linspace(-1, 1, 100)
        self.jerk_range = np.linspace(-5, 5, 100)  # m/s³
        self.action_change_range = np.linspace(-0.5, 0.5, 100)
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Reward Weight Tuner - Adjust weights to match penalty scales', fontsize=14, fontweight='bold')
        
        # Create subplots: 3 rows x 2 cols
        # Left column: 2 subplots (tracking, action)
        # Right column: 3 subplots (jerk, smooth action, combined)
        gs = self.fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, left=0.1, right=0.95, top=0.92, bottom=0.15)
        
        self.ax_tracking = self.fig.add_subplot(gs[0, 0])  # Left column, top
        self.ax_action = self.fig.add_subplot(gs[1, 0])    # Left column, middle
        self.ax_jerk = self.fig.add_subplot(gs[0, 1])      # Right column, top
        self.ax_smooth_action = self.fig.add_subplot(gs[1, 1])  # Right column, middle
        self.ax_total = self.fig.add_subplot(gs[2, 1])     # Right column, bottom
        
        # Create sliders
        self._create_sliders()
        
        # Initial plot
        self.update_plots()
        
        # Connect slider events
        for slider in self.sliders.values():
            slider.on_changed(self.update_plots)
    
    def _create_sliders(self):
        """Create sliders for weight adjustment."""
        slider_height = 0.02
        slider_spacing_vertical = 0.05  # Increased vertical spacing
        slider_spacing_horizontal = 0.15  # Horizontal spacing between columns
        slider_width = 0.15
        start_y = 0.08
        start_x_left = 0.05  # Left column start
        start_x_right = start_x_left + slider_width + slider_spacing_horizontal  # Right column start
        
        self.sliders = {}
        
        # Track weight slider
        ax_track = plt.axes([start_x_left, start_y + 4 * slider_spacing_vertical, slider_width, slider_height])
        self.sliders['track'] = Slider(ax_track, 'Track', 0.0, 5.0, valinit=self.weights.track_weight, valstep=0.1)
        
        # Jerk weight slider
        ax_jerk = plt.axes([start_x_left, start_y + 3 * slider_spacing_vertical, slider_width, slider_height])
        self.sliders['jerk'] = Slider(ax_jerk, 'Jerk', 0.0, 2.0, valinit=self.weights.jerk_weight, valstep=0.01)
        
        # Action weight slider
        ax_action = plt.axes([start_x_left, start_y + 2 * slider_spacing_vertical, slider_width, slider_height])
        self.sliders['action'] = Slider(ax_action, 'Action', 0.0, 20.0, valinit=self.weights.action_weight, valstep=0.1)
        
        # Smooth action weight slider
        ax_smooth = plt.axes([start_x_left, start_y + 1 * slider_spacing_vertical, slider_width, slider_height])
        self.sliders['smooth_action'] = Slider(ax_smooth, 'Smooth', 0.0, 40.0, valinit=self.weights.smooth_action_weight, valstep=0.1)
        
        # Horizon penalty weight slider
        ax_horizon = plt.axes([start_x_right, start_y + 4 * slider_spacing_vertical, slider_width, slider_height])
        self.sliders['horizon'] = Slider(ax_horizon, 'Horizon', 0.0, 1.0, valinit=self.weights.horizon_penalty_weight, valstep=0.01)
        
        # Oscillation weight slider
        ax_osc = plt.axes([start_x_right, start_y + 3 * slider_spacing_vertical, slider_width, slider_height])
        self.sliders['oscillation'] = Slider(ax_osc, 'Oscillation', 0.0, 1.0, valinit=self.weights.oscillation_weight, valstep=0.01)
        
        # Overshoot weight slider
        ax_over = plt.axes([start_x_right, start_y + 2 * slider_spacing_vertical, slider_width, slider_height])
        self.sliders['overshoot'] = Slider(ax_over, 'Overshoot', 0.0, 1.0, valinit=self.weights.overshoot_weight, valstep=0.01)
        
        # Reset button
        ax_reset = plt.axes([start_x_right, start_y + 0.5 * slider_spacing_vertical, slider_width * 0.5, slider_height * 2])
        self.reset_button = Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(self.reset_weights)
        
        # Export button
        ax_export = plt.axes([start_x_right + slider_width * 0.5 + 0.02, start_y + 0.5 * slider_spacing_vertical, slider_width * 0.5, slider_height * 2])
        self.export_button = Button(ax_export, 'Export')
        self.export_button.on_clicked(self.export_weights)
    
    def reset_weights(self, event):
        """Reset weights to defaults."""
        self.weights = RewardWeights()
        self.sliders['track'].set_val(self.weights.track_weight)
        self.sliders['jerk'].set_val(self.weights.jerk_weight)
        self.sliders['action'].set_val(self.weights.action_weight)
        self.sliders['smooth_action'].set_val(self.weights.smooth_action_weight)
        self.sliders['horizon'].set_val(self.weights.horizon_penalty_weight)
        self.sliders['oscillation'].set_val(self.weights.oscillation_weight)
        self.sliders['overshoot'].set_val(self.weights.overshoot_weight)
        self.update_plots()
    
    def export_weights(self, event):
        """Export current weights to console and clipboard-friendly format."""
        print("\n" + "="*60)
        print("Current Reward Weights (YAML format):")
        print("="*60)
        print(f"  track_weight: {self.weights.track_weight:.6f}")
        print(f"  jerk_weight: {self.weights.jerk_weight:.6f}")
        print(f"  action_weight: {self.weights.action_weight:.6f}")
        print(f"  smooth_action_weight: {self.weights.smooth_action_weight:.6f}")
        print(f"  horizon_penalty_weight: {self.weights.horizon_penalty_weight:.6f}")
        print(f"  oscillation_weight: {self.weights.oscillation_weight:.6f}")
        print(f"  overshoot_weight: {self.weights.overshoot_weight:.6f}")
        print("="*60)
        print("\nCopy the above values to your config file under 'env:' section")
        print("="*60 + "\n")
    
    def update_plots(self, val=None):
        """Update all plots based on current weight values."""
        # Update weights from sliders
        self.weights.track_weight = self.sliders['track'].val
        self.weights.jerk_weight = self.sliders['jerk'].val
        self.weights.action_weight = self.sliders['action'].val
        self.weights.smooth_action_weight = self.sliders['smooth_action'].val
        self.weights.horizon_penalty_weight = self.sliders['horizon'].val
        self.weights.oscillation_weight = self.sliders['oscillation'].val
        self.weights.overshoot_weight = self.sliders['overshoot'].val
        
        # Clear axes
        for ax in [self.ax_tracking, self.ax_jerk, self.ax_action, self.ax_smooth_action, self.ax_total]:
            ax.clear()
        
        # Plot 1: Tracking penalty vs speed error
        speed_errors = self.speed_range - self.target_speed
        tracking_penalties = [compute_tracking_penalty(err, self.weights.track_weight) for err in speed_errors]
        self.ax_tracking.plot(speed_errors, tracking_penalties, 'b-', linewidth=2, label='Tracking Penalty')
        self.ax_tracking.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.ax_tracking.set_xlabel('Speed Error (m/s)')
        self.ax_tracking.set_ylabel('Penalty')
        self.ax_tracking.set_title(f'Tracking Penalty (weight={self.weights.track_weight:.2f})')
        self.ax_tracking.legend()
        self.ax_tracking.grid(True, alpha=0.3)
        
        # Plot 2: Jerk penalty vs jerk
        jerk_penalties = [compute_jerk_penalty(j, self.weights.jerk_weight) for j in self.jerk_range]
        self.ax_jerk.plot(self.jerk_range, jerk_penalties, 'r-', linewidth=2, label='Jerk Penalty')
        self.ax_jerk.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.ax_jerk.set_xlabel('Jerk (m/s³)')
        self.ax_jerk.set_ylabel('Penalty')
        self.ax_jerk.set_title(f'Jerk Penalty (weight={self.weights.jerk_weight:.3f})')
        self.ax_jerk.legend()
        self.ax_jerk.grid(True, alpha=0.3)
        
        # Plot 3: Action penalty vs action
        action_penalties = [compute_action_penalty(a, self.weights.action_weight) for a in self.action_range]
        self.ax_action.plot(self.action_range, action_penalties, 'g-', linewidth=2, label='Action Penalty')
        self.ax_action.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.ax_action.set_xlabel('Action')
        self.ax_action.set_ylabel('Penalty')
        self.ax_action.set_title(f'Action Penalty (weight={self.weights.action_weight:.4f})')
        self.ax_action.legend()
        self.ax_action.grid(True, alpha=0.3)
        
        # Plot 4: Smooth action penalty vs action change
        smooth_penalties = [compute_smooth_action_penalty(ac, self.weights.smooth_action_weight) 
                          for ac in self.action_change_range]
        self.ax_smooth_action.plot(self.action_change_range, smooth_penalties, 'm-', linewidth=2, 
                                  label='Smooth Action Penalty')
        self.ax_smooth_action.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.ax_smooth_action.set_xlabel('Action Change')
        self.ax_smooth_action.set_ylabel('Penalty')
        self.ax_smooth_action.set_title(f'Smooth Action Penalty (weight={self.weights.smooth_action_weight:.3f})')
        self.ax_smooth_action.legend()
        self.ax_smooth_action.grid(True, alpha=0.3)
        
        # Plot 5: Combined penalties for typical scenarios
        # Scenario 1: Speed error from -5 to +5 m/s (typical tracking scenario)
        scenario_speed_errors = np.linspace(-5, 5, 50)
        scenario_jerk = 2.0  # Typical jerk value (m/s³)
        scenario_action = 0.5  # Typical action value
        scenario_action_change = 0.2  # Typical action change
        
        combined_penalties = []
        tracking_vals = []
        jerk_vals = []
        action_vals = []
        smooth_vals = []
        
        for speed_error in scenario_speed_errors:
            track_pen = compute_tracking_penalty(speed_error, self.weights.track_weight)
            jerk_pen = compute_jerk_penalty(scenario_jerk, self.weights.jerk_weight)
            action_pen = compute_action_penalty(scenario_action, self.weights.action_weight)
            smooth_pen = compute_smooth_action_penalty(scenario_action_change, self.weights.smooth_action_weight)
            
            total = track_pen + jerk_pen + action_pen + smooth_pen
            
            combined_penalties.append(total)
            tracking_vals.append(track_pen)
            jerk_vals.append(jerk_pen)
            action_vals.append(action_pen)
            smooth_vals.append(smooth_pen)
        
        self.ax_total.plot(scenario_speed_errors, tracking_vals, 'b-', linewidth=2, label='Tracking', alpha=0.8)
        self.ax_total.plot(scenario_speed_errors, jerk_vals, 'r-', linewidth=2, label='Jerk', alpha=0.8)
        self.ax_total.plot(scenario_speed_errors, action_vals, 'g-', linewidth=2, label='Action', alpha=0.8)
        self.ax_total.plot(scenario_speed_errors, smooth_vals, 'm-', linewidth=2, label='Smooth', alpha=0.8)
        self.ax_total.plot(scenario_speed_errors, combined_penalties, 'k--', linewidth=2, label='Total', alpha=0.9)
        self.ax_total.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        self.ax_total.set_xlabel('Speed Error (m/s)')
        self.ax_total.set_ylabel('Penalty')
        self.ax_total.set_title('Combined Penalties (Typical Scenario: jerk=2.0, action=0.5, action_change=0.2)')
        self.ax_total.legend()
        self.ax_total.grid(True, alpha=0.3)
        
        # Add text box with current weights
        weight_text = (f"Weights: Track={self.weights.track_weight:.2f}, "
                      f"Jerk={self.weights.jerk_weight:.3f}, "
                      f"Action={self.weights.action_weight:.4f}, "
                      f"Smooth={self.weights.smooth_action_weight:.3f}")
        self.ax_total.text(0.02, 0.98, weight_text, transform=self.ax_total.transAxes,
                          fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Show the GUI."""
        plt.show()


def main():
    """Main entry point."""
    tuner = RewardWeightTuner()
    tuner.show()


if __name__ == "__main__":
    main()

