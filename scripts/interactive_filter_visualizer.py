#!/usr/bin/env python3
"""Interactive GUI for visualizing how 2nd order LPF parameters affect step function filtering."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch

from generator.lpf import SecondOrderLPF


def generate_step_function(start_speed: float, end_speed: float, length: int, dt: float) -> np.ndarray:
    """Generate a step function profile.
    
    Args:
        start_speed: Initial speed (m/s)
        end_speed: Target speed (m/s)
        length: Profile length (steps)
        dt: Time step (seconds)
        
    Returns:
        Step function profile [length]
    """
    profile = np.full(length, start_speed, dtype=np.float32)
    # Step change at midpoint
    step_idx = length // 2
    profile[step_idx:] = end_speed
    return profile


def create_interactive_visualizer():
    """Create an interactive GUI for visualizing filter effects."""
    
    # Default filter parameters
    default_freq_cutoff = 0.6
    default_zeta = 0.9
    default_dt = 0.1
    default_rate_max = 15.0
    default_rate_neg_max = 20.0
    default_jerk_max = 12.0
    
    # Generate step functions
    dt = 0.1
    profile_length = 200
    time_axis = np.arange(profile_length) * dt
    
    # Create multiple step functions with different transitions
    step_functions = [
        ("0→1 m/s", generate_step_function(0.0, 1.0, profile_length, dt)),
        ("0→5 m/s", generate_step_function(0.0, 5.0, profile_length, dt)),
        ("0→10 m/s", generate_step_function(0.0, 10.0, profile_length, dt)),
        ("1→0 m/s", generate_step_function(1.0, 0.0, profile_length, dt)),
        ("5→0 m/s", generate_step_function(5.0, 0.0, profile_length, dt)),
        ("10→0 m/s", generate_step_function(10.0, 0.0, profile_length, dt)),
    ]
    
    # Create figure with more space
    fig = plt.figure(figsize=(18, 14))
    
    # Calculate grid layout: 2 columns for plots, with space for sliders at bottom
    num_plots = len(step_functions)
    num_rows = (num_plots + 1) // 2  # Round up for 2 columns
    num_cols = 2
    
    # Create grid: plots take most space, sliders at bottom
    gs = fig.add_gridspec(num_rows + 1, num_cols, 
                          height_ratios=[1] * num_rows + [0.4],
                          hspace=0.4, wspace=0.3,
                          left=0.08, right=0.95, top=0.95, bottom=0.25)
    
    axes = []
    lines_raw = []
    lines_filtered = []
    
    # Create subplots for each step function in 2 columns
    for i, (label, raw_profile) in enumerate(step_functions):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
        
        # Plot raw step function
        line_raw, = ax.plot(time_axis, raw_profile, 'r-', linewidth=2, label='Raw Step Function', alpha=0.7)
        lines_raw.append(line_raw)
        
        # Placeholder for filtered signal (will be updated)
        line_filtered, = ax.plot(time_axis, raw_profile, 'b--', linewidth=2, label='Filtered Signal', alpha=0.9)
        lines_filtered.append(line_filtered)
        
        ax.set_ylabel('Speed (m/s)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-2, 25])
    
    # Create sliders panel (spanning both columns at bottom)
    slider_ax = fig.add_subplot(gs[-1, :])
    slider_ax.axis('off')
    
    # Create sliders with more spacing
    slider_width = 0.12
    slider_height = 0.04
    slider_spacing = 0.12  # Increased spacing between sliders
    start_x = 0.05
    slider_y_top = 0.18
    slider_y_bottom = 0.10
    
    # Frequency cutoff slider
    ax_freq = fig.add_axes([start_x, slider_y_top, slider_width, slider_height])
    slider_freq = Slider(ax_freq, 'freq_cutoff\n(Hz)', 0.1, 5.0, valinit=default_freq_cutoff, valstep=0.1)
    
    # Zeta slider
    ax_zeta = fig.add_axes([start_x + slider_width + slider_spacing, slider_y_top, slider_width, slider_height])
    slider_zeta = Slider(ax_zeta, 'zeta\n(damping)', 0.1, 10.0, valinit=default_zeta, valstep=0.1)
    
    # dt slider
    ax_dt = fig.add_axes([start_x + 2*(slider_width + slider_spacing), slider_y_top, slider_width, slider_height])
    slider_dt = Slider(ax_dt, 'dt\n(seconds)', 0.01, 0.2, valinit=default_dt, valstep=0.01)
    
    # Rate max slider
    ax_rate_max = fig.add_axes([start_x, slider_y_bottom, slider_width, slider_height])
    slider_rate_max = Slider(ax_rate_max, 'rate_max\n(m/s²)', 0.5, 30.0, valinit=default_rate_max, valstep=0.5)
    
    # Rate neg max slider
    ax_rate_neg_max = fig.add_axes([start_x + slider_width + slider_spacing, slider_y_bottom, slider_width, slider_height])
    slider_rate_neg_max = Slider(ax_rate_neg_max, 'rate_neg_max\n(m/s²)', 0.5, 30.0, valinit=default_rate_neg_max, valstep=0.5)
    
    # Jerk max slider
    ax_jerk_max = fig.add_axes([start_x + 2*(slider_width + slider_spacing), slider_y_bottom, slider_width, slider_height])
    slider_jerk_max = Slider(ax_jerk_max, 'jerk_max\n(m/s³)', 0.5, 30.0, valinit=default_jerk_max, valstep=0.5)
    
    # Store filter state for each step function
    filters = []
    
    def update_filtered_signals(val=None):
        """Update all filtered signals based on current slider values."""
        freq_cutoff = slider_freq.val
        zeta = slider_zeta.val
        dt_filter = slider_dt.val
        rate_max_val = slider_rate_max.val
        rate_neg_max_val = slider_rate_neg_max.val
        jerk_max_val = slider_jerk_max.val
        
        # Create rate and jerk limit tensors
        rate_max_tensor = torch.tensor([rate_max_val], device=torch.device('cpu'))
        rate_neg_max_tensor = torch.tensor([rate_neg_max_val], device=torch.device('cpu'))
        jerk_max_tensor = torch.tensor([jerk_max_val], device=torch.device('cpu'))
        
        # Update each step function plot
        for i, (label, raw_profile) in enumerate(step_functions):
            # Create or recreate filter with new parameters
            filter_lpf = SecondOrderLPF(
                batch_size=1,
                freq_cutoff=freq_cutoff,
                zeta=zeta,
                dt=dt_filter,
                rate_max=rate_max_tensor,
                rate_neg_max=rate_neg_max_tensor,
                jerk_max=jerk_max_tensor,
                device=torch.device('cpu')
            )
            
            # Reset filter with initial value
            initial_y = torch.tensor([[raw_profile[0]]], device=torch.device('cpu'), dtype=torch.float32)
            filter_lpf.reset(initial_y=initial_y)
            
            # Filter the raw profile
            raw_tensor = torch.from_numpy(raw_profile).unsqueeze(0)  # [1, T]
            filtered_tensor = torch.zeros_like(raw_tensor)
            
            for t in range(len(raw_profile)):
                u_t = raw_tensor[:, t:t+1]  # [1, 1]
                filtered_y = filter_lpf.update(u_t)
                filtered_tensor[:, t] = filtered_y.squeeze(0)
            
            filtered_profile = filtered_tensor.squeeze(0).cpu().numpy()
            
            # Update plot
            lines_filtered[i].set_ydata(filtered_profile)
        
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    slider_freq.on_changed(update_filtered_signals)
    slider_zeta.on_changed(update_filtered_signals)
    slider_dt.on_changed(update_filtered_signals)
    slider_rate_max.on_changed(update_filtered_signals)
    slider_rate_neg_max.on_changed(update_filtered_signals)
    slider_jerk_max.on_changed(update_filtered_signals)
    
    # Initial update
    update_filtered_signals()
    
    plt.suptitle('Interactive 2nd Order LPF Filter Visualization', fontsize=16, fontweight='bold', y=0.99)
    plt.show()


if __name__ == "__main__":
    create_interactive_visualizer()

