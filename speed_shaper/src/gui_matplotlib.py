"""Interactive GUI for speed profile shaping with matplotlib widgets."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.shaper_math import shape_speed_profile


class SpeedShaperGUI:
    """Interactive GUI for tuning speed profile shaper weights."""
    
    def __init__(self, example_type='step'):
        """
        Args:
            example_type: Profile type - 'step', 'sine', 'random_quantized', 'random_smooth', etc.
        """
        # Profile parameters (will be controlled by sliders)
        self.profile_params = {
            'duration': 20.0,      # Profile duration in seconds
            'step_size': 3.0,      # For step profile
            'amplitude': 5.0,      # For sine profile
            'frequency': 0.2,      # For sine profile
            'num_steps': 5,        # For random_quantized
            'speed_range': 10.0,   # For random profiles
            'smoothness': 0.5,     # For random_smooth
        }
        
        # Flag to prevent recursive updates
        self._updating = False
        
        # Generate example raw profile
        self.dt = 0.1  # 100 ms
        self.example_type = example_type
        self.t, self.r = self._generate_example_profile(example_type, self.profile_params)
        self.N = len(self.r) - 1
        
        # Initial measurements
        self.meas = {
            'v_meas': self.r[0],
            'a_meas': 0.0,
            'j_meas': 0.0
        }
        
        # Initial shaped profile (will be computed)
        self.v = self.r.copy()
        
        # Setup figure and axes
        self._setup_figure()
        
        # Setup widgets
        self._setup_widgets()
        
        # Initial solve
        self._update_profile(None)
        
    def _generate_example_profile(self, example_type, params=None):
        """Generate example speed profile based on type.
        
        Args:
            example_type: Type of profile ('step', 'sine', 'ramp', etc.)
            params: Dictionary of parameters for the profile
        """
        if params is None:
            params = self.profile_params
            
        dt = 0.1
        duration = params.get('duration', 20.0)  # seconds
        t = np.arange(0, duration + dt, dt)
        
        if example_type == 'step':
            # Two-step profile: 0 → step_size → 0
            step_size = params.get('step_size', 3.0)
            v = np.zeros_like(t)
            
            # Step up at 25% of duration
            step_up_time = duration * 0.25
            v[t >= step_up_time] = step_size
            # Step down at 75% of duration
            step_down_time = duration * 0.75
            v[t >= step_down_time] = 0.0
            
            # Add small noise
            np.random.seed(42)
            v += np.random.normal(0, 0.02, len(v))
            
        elif example_type == 'sine':
            # Sinusoidal profile around a mean speed
            amplitude = params.get('amplitude', 5.0)
            frequency = params.get('frequency', 0.2)  # Hz
            mean_speed = params.get('mean_speed', 10.0)
            
            v = mean_speed + amplitude * np.sin(2 * np.pi * frequency * t)
            v = np.clip(v, 0, None)  # No negative speeds
            
        elif example_type == 'ramp':
            # Ramp up and down
            ramp_rate = params.get('ramp_rate', 1.0)  # m/s²
            max_speed = params.get('max_speed', 15.0)
            
            v = np.zeros_like(t)
            # Ramp up starting at 10% of duration
            ramp_start = duration * 0.1
            ramp_up_time = max_speed / ramp_rate
            mask_up = (t >= ramp_start) & (t < ramp_start + ramp_up_time)
            v[mask_up] = ramp_rate * (t[mask_up] - ramp_start)
            
            # Hold until 75% of duration
            hold_end = duration * 0.75
            v[(t >= ramp_start + ramp_up_time) & (t < hold_end)] = max_speed
            
            # Ramp down
            ramp_down_start = hold_end
            mask_down = (t >= ramp_down_start) & (t < ramp_down_start + ramp_up_time)
            v[mask_down] = max_speed - ramp_rate * (t[mask_down] - ramp_down_start)
            
            # Final hold at zero
            v[t >= ramp_down_start + ramp_up_time] = 0.0
            v = np.clip(v, 0, None)
            
        elif example_type == 'sawtooth':
            # Sawtooth wave
            amplitude = params.get('amplitude', 8.0)
            frequency = params.get('frequency', 0.15)
            
            v = amplitude * (2 * ((frequency * t) % 1) - 1) + amplitude
            v = np.clip(v, 0, None)
            
        elif example_type == 'random_quantized':
            # Random steps with quantized levels
            num_steps = int(params.get('num_steps', 5))
            speed_range = params.get('speed_range', 10.0)
            
            # Generate random speed levels
            np.random.seed(42)
            segment_duration = duration / num_steps
            v = np.zeros_like(t)
            
            for i in range(num_steps):
                speed_level = np.random.uniform(0, speed_range)
                mask = (t >= i * segment_duration) & (t < (i + 1) * segment_duration)
                v[mask] = speed_level
            
            # Add small noise
            v += np.random.normal(0, 0.05, len(v))
            
        elif example_type == 'random_smooth':
            # Smooth random profile using interpolation
            speed_range = params.get('speed_range', 15.0)
            smoothness = params.get('smoothness', 0.5)  # 0=very smooth, 1=more variation
            
            # Generate random waypoints
            np.random.seed(42)
            num_waypoints = max(5, int(20 * smoothness))
            waypoint_times = np.linspace(0, duration, num_waypoints)
            waypoint_speeds = np.random.uniform(0, speed_range, num_waypoints)
            
            # Interpolate smoothly
            v = np.interp(t, waypoint_times, waypoint_speeds)
            
        else:  # 'smooth' - legacy smooth profile
            # Smooth example with high acceleration and jerk
            # Build segments but ensure final length matches duration
            t_smooth = []
            v_smooth = []
            
            # Aggressive acceleration from 0 to 20 m/s
            t1 = np.arange(0, 2.0, dt)
            v1 = 10.0 * t1**2
            v1 = np.clip(v1, 0, 20)
            t_smooth.append(t1)
            v_smooth.append(v1)
            
            # Sinusoidal oscillation
            t2 = np.arange(t1[-1] + dt, t1[-1] + 3.0, dt)
            v2 = 20.0 + 4.0 * np.sin(2 * np.pi * (t2 - t1[-1]) * 2.0)
            t_smooth.append(t2)
            v_smooth.append(v2)
            
            # Aggressive deceleration
            t3 = np.arange(t2[-1] + dt, t2[-1] + 3.0, dt)
            t3_local = t3 - t2[-1]
            v3 = v2[-1] - 3.33 * t3_local**2
            v3 = np.clip(v3, 2, None)
            t_smooth.append(t3)
            v_smooth.append(v3)
            
            # Sharp S-curve
            t4 = np.arange(t3[-1] + dt, t3[-1] + 2.5, dt)
            t4_local = t4 - t3[-1]
            v4 = v3[-1] + 10.0 * (1 + np.tanh(4*(t4_local - 1.25))) / 2
            t_smooth.append(t4)
            v_smooth.append(v4)
            
            # High-frequency oscillation
            t5 = np.arange(t4[-1] + dt, t4[-1] + 2.0, dt)
            v5 = v4[-1] + 3.0 * np.sin(2 * np.pi * (t5 - t4[-1]) * 4.0)
            t_smooth.append(t5)
            v_smooth.append(v5)
            
            # Concatenate
            t_concat = np.concatenate(t_smooth)
            v_concat = np.concatenate(v_smooth)
            
            # Pad to exactly duration (20 seconds)
            if t_concat[-1] < duration:
                t_pad = np.arange(t_concat[-1] + dt, duration + dt, dt)
                v_pad = np.ones(len(t_pad)) * v_concat[-1]
                v = np.concatenate([v_concat, v_pad])
            else:
                v = v_concat
            
            # Ensure t and v have same length as standard profiles
            # Truncate or pad to match expected length
            expected_length = len(t)
            if len(v) > expected_length:
                v = v[:expected_length]
            elif len(v) < expected_length:
                v = np.pad(v, (0, expected_length - len(v)), mode='edge')
        
        return t, v
    
    def _setup_figure(self):
        """Setup matplotlib figure with GridSpec layout."""
        # Step 1: Increase figure size to accommodate profile controls
        self.fig = plt.figure(figsize=(18, 14))  # Larger to fit new section
        
        # Step 2: Create top-level GridSpec with 4 rows
        # plots / weights / constraints / profile_controls
        gs = GridSpec(
            nrows=4,
            ncols=1,
            height_ratios=[5.5, 1.2, 1.8, 1.5],  # Adjusted: more space for weights and profile controls
            hspace=0.40,  # Reduced to bring everything closer
            figure=self.fig
        )
        
        # Store gs for widget setup
        self.gs = gs
        
        # Step 3: Setup plots area (top section)
        gs_plots = gs[0].subgridspec(
            nrows=2,
            ncols=1,
            hspace=0.25
        )
        
        # Speed profile plot
        self.ax_speed = self.fig.add_subplot(gs_plots[0])
        title_text = f'Speed Profile Shaping - {self.example_type.capitalize()} Example'
        self.ax_speed.set_title(title_text, fontsize=13, fontweight='bold', pad=10)
        self.ax_speed.set_xlabel('Time (s)', labelpad=8)
        self.ax_speed.set_ylabel('Speed (m/s)', labelpad=8)
        self.ax_speed.grid(True, alpha=0.3)
        
        # Plot raw and shaped profiles
        self.line_raw, = self.ax_speed.plot(self.t, self.r, 'k--', linewidth=1.5, 
                                             label='Raw target', alpha=0.6)
        self.line_shaped, = self.ax_speed.plot(self.t, self.v, 'b-', linewidth=2.5, 
                                                label='Shaped profile')
        self.ax_speed.legend(loc='upper right')
        
        # Acceleration and jerk plot
        self.ax_accel = self.fig.add_subplot(gs_plots[1])
        self.ax_accel.set_xlabel('Time (s)', labelpad=8)
        self.ax_accel.set_ylabel('Acceleration (m/s²)', color='g', labelpad=8)
        self.ax_accel.tick_params(axis='y', labelcolor='g', pad=6)
        self.ax_accel.grid(True, alpha=0.3)
        
        self.ax_jerk = self.ax_accel.twinx()
        self.ax_jerk.set_ylabel('Jerk (m/s³)', color='r', labelpad=8)
        self.ax_jerk.tick_params(axis='y', labelcolor='r', pad=6)
        
        # Will be populated after first solve
        self.line_accel = None
        self.line_jerk = None
        self.line_a_min = None
        self.line_a_max = None
        self.line_j_min = None
        self.line_j_max = None
        
        # Step 9: Final spacing cleanup (MANDATORY)
        # Slightly narrower plots horizontally (increased left/right margins)
        plt.subplots_adjust(
            left=0.08,   # Increased from 0.06 (narrower plots)
            right=0.95,  # Decreased from 0.97 (narrower plots)
            top=0.95,
            bottom=0.05
        )
    
    def _setup_widgets(self):
        """Setup sliders and checkbox widgets using GridSpec."""
        # Step 4: Fix the weight sliders layout (middle section)
        # 3 columns × 3 rows: Error | Accel | Jerk
        # Increased hspace for more vertical spacing between rows
        gs_weights = self.gs[1].subgridspec(
            nrows=3,
            ncols=3,
            wspace=0.35,
            hspace=1.2  # Much more spacing between rows (was 0.6)
        )
        
        # Error weight sliders (column 0)
        ax_wE_start = self.fig.add_subplot(gs_weights[0, 0])
        ax_wE_end = self.fig.add_subplot(gs_weights[1, 0])
        ax_lamE = self.fig.add_subplot(gs_weights[2, 0])
        
        ax_wE_start.set_facecolor("#f5f5f5")
        ax_wE_end.set_facecolor("#f5f5f5")
        ax_lamE.set_facecolor("#f5f5f5")
        
        self.slider_wE_start = Slider(ax_wE_start, 'wE start', 0.0, 100.0, valinit=30.0, valstep=1.0)
        self.slider_wE_end = Slider(ax_wE_end, 'wE end', 0.0, 100.0, valinit=10.0, valstep=1.0)
        self.slider_lamE = Slider(ax_lamE, 'λE', -5.0, 5.0, valinit=1.0, valstep=0.1)
        
        # Make slider fonts larger
        self.slider_wE_start.label.set_fontsize(12)
        self.slider_wE_start.valtext.set_fontsize(11)
        self.slider_wE_end.label.set_fontsize(12)
        self.slider_wE_end.valtext.set_fontsize(11)
        self.slider_lamE.label.set_fontsize(12)
        self.slider_lamE.valtext.set_fontsize(11)
        
        # Acceleration weight sliders (column 1)
        ax_wA_start = self.fig.add_subplot(gs_weights[0, 1])
        ax_wA_end = self.fig.add_subplot(gs_weights[1, 1])
        ax_lamA = self.fig.add_subplot(gs_weights[2, 1])
        
        ax_wA_start.set_facecolor("#f5f5f5")
        ax_wA_end.set_facecolor("#f5f5f5")
        ax_lamA.set_facecolor("#f5f5f5")
        
        self.slider_wA_start = Slider(ax_wA_start, 'wA start', 0.0, 50.0, valinit=2.0, valstep=0.5)
        self.slider_wA_end = Slider(ax_wA_end, 'wA end', 0.0, 50.0, valinit=10.0, valstep=0.5)
        self.slider_lamA = Slider(ax_lamA, 'λA', -5.0, 5.0, valinit=-0.5, valstep=0.1)
        
        # Make slider fonts larger
        self.slider_wA_start.label.set_fontsize(12)
        self.slider_wA_start.valtext.set_fontsize(11)
        self.slider_wA_end.label.set_fontsize(12)
        self.slider_wA_end.valtext.set_fontsize(11)
        self.slider_lamA.label.set_fontsize(12)
        self.slider_lamA.valtext.set_fontsize(11)
        
        # Jerk weight sliders (column 2)
        ax_wJ_start = self.fig.add_subplot(gs_weights[0, 2])
        ax_wJ_end = self.fig.add_subplot(gs_weights[1, 2])
        ax_lamJ = self.fig.add_subplot(gs_weights[2, 2])
        
        ax_wJ_start.set_facecolor("#f5f5f5")
        ax_wJ_end.set_facecolor("#f5f5f5")
        ax_lamJ.set_facecolor("#f5f5f5")
        
        self.slider_wJ_start = Slider(ax_wJ_start, 'wJ start', 0.0, 50.0, valinit=2.0, valstep=0.5)
        self.slider_wJ_end = Slider(ax_wJ_end, 'wJ end', 0.0, 50.0, valinit=10.0, valstep=0.5)
        self.slider_lamJ = Slider(ax_lamJ, 'λJ', -5.0, 5.0, valinit=-0.5, valstep=0.1)
        
        # Make slider fonts larger
        self.slider_wJ_start.label.set_fontsize(12)
        self.slider_wJ_start.valtext.set_fontsize(11)
        self.slider_wJ_end.label.set_fontsize(12)
        self.slider_wJ_end.valtext.set_fontsize(11)
        self.slider_lamJ.label.set_fontsize(12)
        self.slider_lamJ.valtext.set_fontsize(11)
        
        # Connect sliders to update callback (auto-solve on change)
        self.slider_wE_start.on_changed(self._update_profile)
        self.slider_wE_end.on_changed(self._update_profile)
        self.slider_lamE.on_changed(self._update_profile)
        self.slider_wA_start.on_changed(self._update_profile)
        self.slider_wA_end.on_changed(self._update_profile)
        self.slider_lamA.on_changed(self._update_profile)
        self.slider_wJ_start.on_changed(self._update_profile)
        self.slider_wJ_end.on_changed(self._update_profile)
        self.slider_lamJ.on_changed(self._update_profile)
        
        # Add section labels for weights (positioned lower to avoid overlap with sliders)
        # Larger fonts for better visibility
        self.fig.text(0.17, 0.38, 'Error Weights', ha='center', fontsize=13, fontweight='bold')
        self.fig.text(0.50, 0.38, 'Acceleration Weights', ha='center', fontsize=13, fontweight='bold')
        self.fig.text(0.83, 0.38, 'Jerk Weights', ha='center', fontsize=13, fontweight='bold')
        
        # Step 5: Fix constraint controls (bottom section)
        # 3 columns: Accel constraints | Jerk constraints | Global controls
        gs_constraints = self.gs[2].subgridspec(
            nrows=1,
            ncols=3,
            wspace=0.4
        )
        
        # Step 6: Acceleration constraints column
        gs_accel = gs_constraints[0].subgridspec(
            nrows=4,
            ncols=1,
            hspace=0.6
        )
        
        # Enable checkbox
        ax_check_accel = self.fig.add_subplot(gs_accel[0])
        ax_check_accel.axis("off")
        self.check_accel = CheckButtons(ax_check_accel, ['Enable accel bounds'], [False])
        self.check_accel.on_clicked(self._update_profile)
        
        # a_min slider
        ax_a_min = self.fig.add_subplot(gs_accel[1])
        ax_a_min.set_facecolor("#f5f5f5")
        self.slider_a_min = Slider(ax_a_min, 'a_min', -10.0, 0.0, valinit=-3.0, valstep=0.5)
        self.slider_a_min.on_changed(self._update_profile)
        
        # a_max slider
        ax_a_max = self.fig.add_subplot(gs_accel[2])
        ax_a_max.set_facecolor("#f5f5f5")
        self.slider_a_max = Slider(ax_a_max, 'a_max', 0.0, 10.0, valinit=3.0, valstep=0.5)
        self.slider_a_max.on_changed(self._update_profile)
        
        # Spacer (empty axis)
        ax_spacer_accel = self.fig.add_subplot(gs_accel[3])
        ax_spacer_accel.axis("off")
        
        # Step 7: Jerk constraints column
        gs_jerk = gs_constraints[1].subgridspec(
            nrows=4,
            ncols=1,
            hspace=0.6
        )
        
        # Enable checkbox
        ax_check_jerk = self.fig.add_subplot(gs_jerk[0])
        ax_check_jerk.axis("off")
        self.check_jerk = CheckButtons(ax_check_jerk, ['Enable jerk bounds'], [False])
        self.check_jerk.on_clicked(self._update_profile)
        
        # j_min slider
        ax_j_min = self.fig.add_subplot(gs_jerk[1])
        ax_j_min.set_facecolor("#f5f5f5")
        self.slider_j_min = Slider(ax_j_min, 'j_min', -20.0, 0.0, valinit=-8.0, valstep=1.0)
        self.slider_j_min.on_changed(self._update_profile)
        
        # j_max slider
        ax_j_max = self.fig.add_subplot(gs_jerk[2])
        ax_j_max.set_facecolor("#f5f5f5")
        self.slider_j_max = Slider(ax_j_max, 'j_max', 0.0, 20.0, valinit=8.0, valstep=1.0)
        self.slider_j_max.on_changed(self._update_profile)
        
        # Spacer (empty axis)
        ax_spacer_jerk = self.fig.add_subplot(gs_jerk[3])
        ax_spacer_jerk.axis("off")
        
        # Step 8: Global controls column (right)
        gs_global = gs_constraints[2].subgridspec(
            nrows=5,
            ncols=1,
            hspace=0.8
        )
        
        # Terminal constraint checkbox
        ax_check_term = self.fig.add_subplot(gs_global[0])
        ax_check_term.axis("off")
        self.check_terminal = CheckButtons(ax_check_term, ['Enforce v[N]=r[N]'], [False])
        self.check_terminal.on_clicked(self._update_profile)
        
        # Duration slider
        ax_duration = self.fig.add_subplot(gs_global[1])
        ax_duration.set_facecolor("#f5f5f5")
        self.slider_duration = Slider(ax_duration, 'Duration (s)', 5.0, 60.0, valinit=20.0, valstep=1.0)
        self.slider_duration.label.set_fontsize(11)
        self.slider_duration.valtext.set_fontsize(10)
        self.slider_duration.on_changed(self._on_profile_param_changed)
        
        # Save Config button
        ax_save_btn = self.fig.add_subplot(gs_global[2])
        self.btn_save = Button(ax_save_btn, 'Save Config', color='lightgreen', hovercolor='green')
        self.btn_save.on_clicked(self._on_save_config)
        
        # Instructions text
        ax_instructions = self.fig.add_subplot(gs_global[3])
        ax_instructions.axis("off")
        instruction_text = "Move sliders to adjust weights and constraints - plot updates automatically"
        ax_instructions.text(0.5, 0.5, instruction_text, ha='center', va='center', 
                           fontsize=9, style='italic', color='blue', transform=ax_instructions.transAxes)
        
        # Bottom spacer
        ax_spacer2 = self.fig.add_subplot(gs_global[4])
        ax_spacer2.axis("off")
        
        # Step 7: Profile Controls (bottom section - row 3)
        gs_profile = self.gs[3].subgridspec(
            nrows=1,
            ncols=2,
            wspace=0.5,
            width_ratios=[1, 3]
        )
        
        # Profile type selector (left side)
        ax_profile_type = self.fig.add_subplot(gs_profile[0])
        ax_profile_type.set_title('Profile Type', fontsize=11, fontweight='bold')
        
        profile_types = ['step', 'sine', 'ramp', 'sawtooth', 'random_quantized', 'random_smooth', 'smooth']
        self.profile_radio = RadioButtons(ax_profile_type, profile_types, active=0)
        self.profile_radio.on_clicked(self._on_profile_type_changed)
        
        # Profile parameters (right side) - create nested grid for sliders
        gs_params = gs_profile[1].subgridspec(
            nrows=2,
            ncols=3,
            hspace=1.0,  # Increased from 0.8 for more vertical spacing
            wspace=0.5   # Increased from 0.4 for more horizontal spacing
        )
        
        # Create all parameter sliders (will show/hide based on profile type)
        # Row 0
        ax_step_size = self.fig.add_subplot(gs_params[0, 0])
        ax_amplitude = self.fig.add_subplot(gs_params[0, 1])
        ax_frequency = self.fig.add_subplot(gs_params[0, 2])
        
        # Row 1
        ax_mean_speed = self.fig.add_subplot(gs_params[1, 0])
        ax_max_speed = self.fig.add_subplot(gs_params[1, 1])
        ax_ramp_rate = self.fig.add_subplot(gs_params[1, 2])
        
        # Create sliders
        self.slider_step_size = Slider(ax_step_size, 'Step Size', 0.0, 20.0, valinit=3.0, valstep=0.5)
        self.slider_amplitude = Slider(ax_amplitude, 'Amplitude', 0.0, 10.0, valinit=5.0, valstep=0.5)
        self.slider_frequency = Slider(ax_frequency, 'Frequency', 0.05, 1.0, valinit=0.2, valstep=0.05)
        self.slider_mean_speed = Slider(ax_mean_speed, 'Mean Speed', 0.0, 20.0, valinit=10.0, valstep=1.0)
        self.slider_max_speed = Slider(ax_max_speed, 'Max Speed', 0.0, 25.0, valinit=15.0, valstep=1.0)
        self.slider_ramp_rate = Slider(ax_ramp_rate, 'Ramp Rate', 0.1, 5.0, valinit=1.0, valstep=0.1)
        
        # Store slider axes for visibility control
        self.param_sliders = {
            'step_size': (self.slider_step_size, ax_step_size),
            'amplitude': (self.slider_amplitude, ax_amplitude),
            'frequency': (self.slider_frequency, ax_frequency),
            'mean_speed': (self.slider_mean_speed, ax_mean_speed),
            'max_speed': (self.slider_max_speed, ax_max_speed),
            'ramp_rate': (self.slider_ramp_rate, ax_ramp_rate),
        }
        
        # Connect parameter sliders to regeneration callback
        for slider, _ in self.param_sliders.values():
            slider.on_changed(self._on_profile_param_changed)
        
        # Update visibility for initial profile type
        self._update_param_visibility()
    
    def _get_weight_params(self):
        """Read current slider values into weight parameters dictionary."""
        return {
            'wE_start': self.slider_wE_start.val,
            'wE_end': self.slider_wE_end.val,
            'lamE': self.slider_lamE.val,
            'wA_start': self.slider_wA_start.val,
            'wA_end': self.slider_wA_end.val,
            'lamA': self.slider_lamA.val,
            'wJ_start': self.slider_wJ_start.val,
            'wJ_end': self.slider_wJ_end.val,
            'lamJ': self.slider_lamJ.val,
        }
    
    def _on_save_config(self, event):
        """Callback for Save Config button - save current configuration to JSON."""
        from tkinter import Tk, filedialog
        from pathlib import Path
        from src.config_schema import ShaperConfig
        
        # Create Tk root (hidden) for file dialog
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Get default directory
        default_dir = Path(__file__).parent.parent / "configs"
        default_dir.mkdir(parents=True, exist_ok=True)
        
        # Open file save dialog
        filepath = filedialog.asksaveasfilename(
            initialdir=str(default_dir),
            title="Save Speed Shaper Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        root.destroy()
        
        if not filepath:
            # User cancelled
            return
        
        # Collect current configuration
        enable_accel = self.check_accel.get_status()[0]
        enable_jerk = self.check_jerk.get_status()[0]
        enable_terminal = self.check_terminal.get_status()[0]
        
        config = ShaperConfig(
            dt=self.dt,
            wE_start=self.slider_wE_start.val,
            wE_end=self.slider_wE_end.val,
            lamE=self.slider_lamE.val,
            wA_start=self.slider_wA_start.val,
            wA_end=self.slider_wA_end.val,
            lamA=self.slider_lamA.val,
            wJ_start=self.slider_wJ_start.val,
            wJ_end=self.slider_wJ_end.val,
            lamJ=self.slider_lamJ.val,
            a_min=self.slider_a_min.val if enable_accel else None,
            a_max=self.slider_a_max.val if enable_accel else None,
            j_min=self.slider_j_min.val if enable_jerk else None,
            j_max=self.slider_j_max.val if enable_jerk else None,
            enable_accel_bounds=enable_accel,
            enable_jerk_bounds=enable_jerk,
            enable_terminal_constraint=enable_terminal,
            metadata={
                'profile_type': self.example_type,
                'description': f'Speed shaper config for {self.example_type} profile'
            }
        )
        
        # Save to file
        try:
            config.to_json(Path(filepath))
            print(f"✓ Configuration saved to: {filepath}")
            
            # Show success message in GUI (update instructions temporarily)
            # This is a simple way to provide feedback without modal dialogs
            print(f"Configuration:\n{config}")
        except Exception as e:
            print(f"✗ Error saving configuration: {e}")
    
    def _update_param_visibility(self):
        """Show/hide parameter sliders based on selected profile type."""
        # Define which parameters are visible for each profile type
        visibility_map = {
            'step': ['step_size'],
            'sine': ['amplitude', 'frequency', 'mean_speed'],
            'ramp': ['max_speed', 'ramp_rate'],
            'sawtooth': ['amplitude', 'frequency'],
            'random_quantized': ['max_speed'],  # Uses max_speed as speed_range
            'random_smooth': ['max_speed', 'ramp_rate'],  # ramp_rate as smoothness
            'smooth': [],  # No parameters
        }
        
        visible_params = visibility_map.get(self.example_type, [])
        
        # Show/hide sliders
        for param_name, (slider, ax) in self.param_sliders.items():
            if param_name in visible_params:
                ax.set_visible(True)
            else:
                ax.set_visible(False)
        
        self.fig.canvas.draw_idle()
    
    def _on_profile_type_changed(self, label):
        """Callback when profile type is changed."""
        if self._updating:
            return
        self.example_type = label
        self._update_param_visibility()
        self._regenerate_profile()
    
    def _on_profile_param_changed(self, val):
        """Callback when any profile parameter is changed."""
        if self._updating:
            return
            
        # Update profile_params dictionary
        self.profile_params['duration'] = self.slider_duration.val
        self.profile_params['step_size'] = self.slider_step_size.val
        self.profile_params['amplitude'] = self.slider_amplitude.val
        self.profile_params['frequency'] = self.slider_frequency.val
        self.profile_params['mean_speed'] = self.slider_mean_speed.val
        self.profile_params['max_speed'] = self.slider_max_speed.val
        self.profile_params['ramp_rate'] = self.slider_ramp_rate.val
        
        # Map for different profile types
        self.profile_params['speed_range'] = self.slider_max_speed.val
        self.profile_params['smoothness'] = self.slider_ramp_rate.val / 5.0  # Normalize to 0-1
        
        self._regenerate_profile()
    
    def _regenerate_profile(self):
        """Regenerate reference profile and resolve."""
        if self._updating:
            return
            
        self._updating = True
        try:
            # Generate new profile
            self.t, self.r = self._generate_example_profile(self.example_type, self.profile_params)
            self.N = len(self.r) - 1
            
            # Update measurements to match new profile
            self.meas['v_meas'] = self.r[0]
            
            # Update plot title
            self.ax_speed.set_title(f'Speed Profile Shaping - {self.example_type.capitalize()} Example',
                                   fontsize=13, fontweight='bold', pad=10)
            
            # Clear existing plots
            self.ax_speed.clear()
            self.ax_accel.clear()
            
            # Remove the old twin axis if it exists
            if hasattr(self, 'ax_jerk') and self.ax_jerk is not None:
                self.ax_jerk.remove()
            
            # Recreate speed plot
            self.ax_speed.set_title(f'Speed Profile Shaping - {self.example_type.capitalize()} Example',
                                   fontsize=13, fontweight='bold', pad=10)
            self.ax_speed.set_xlabel('Time (s)', labelpad=8)
            self.ax_speed.set_ylabel('Speed (m/s)', labelpad=8)
            self.ax_speed.grid(True, alpha=0.3)
            
            self.line_raw, = self.ax_speed.plot(self.t, self.r, 'r--', linewidth=1.5, 
                                               label='Raw Target', alpha=0.6)
            self.line_shaped, = self.ax_speed.plot(self.t, self.r, 'b-', linewidth=2, 
                                                  label='Shaped', alpha=0.9)
            self.ax_speed.legend(loc='upper right', fontsize=10)
            
            # Recreate accel/jerk plot
            self.ax_accel.set_xlabel('Time (s)', labelpad=8)
            self.ax_accel.set_ylabel('Acceleration (m/s²)', color='green', labelpad=8)
            self.ax_accel.tick_params(axis='y', labelcolor='green')
            self.ax_accel.grid(True, alpha=0.3, axis='x')
            
            # Compute initial accel/jerk
            a_init = np.diff(self.r) / self.dt
            j_init = np.diff(a_init) / self.dt
            t_a = self.t[:-1]
            t_j = self.t[:-2]
            
            self.line_accel, = self.ax_accel.plot(t_a, a_init, 'g-', linewidth=1.5, 
                                                  label='Accel', alpha=0.7)
            
            # Create new twin axis for jerk
            self.ax_jerk = self.ax_accel.twinx()
            self.ax_jerk.set_ylabel('Jerk (m/s³)', color='orange', labelpad=8)
            self.ax_jerk.tick_params(axis='y', labelcolor='orange')
            
            self.line_jerk, = self.ax_jerk.plot(t_j, j_init, 'orange', linewidth=1.5, 
                                               linestyle='--', label='Jerk', alpha=0.7)
            
            # Add bound lines (initially invisible)
            self.line_a_min, = self.ax_accel.plot([], [], 'r--', linewidth=1, alpha=0.5)
            self.line_a_max, = self.ax_accel.plot([], [], 'r--', linewidth=1, alpha=0.5)
            self.line_j_min, = self.ax_jerk.plot([], [], 'm--', linewidth=1, alpha=0.5)
            self.line_j_max, = self.ax_jerk.plot([], [], 'm--', linewidth=1, alpha=0.5)
            
            # Force redraw
            self.fig.canvas.draw_idle()
            
            # Re-solve and update
            self._update_profile(None)
        finally:
            self._updating = False
    
    def _update_profile(self, event):
        """Callback for solve button: recompute and update plots."""
        # Get current parameters
        weight_params = self._get_weight_params()
        enforce_terminal = self.check_terminal.get_status()[0]
        
        # Get constraint parameters
        enable_accel = self.check_accel.get_status()[0]
        enable_jerk = self.check_jerk.get_status()[0]
        a_min = self.slider_a_min.val
        a_max = self.slider_a_max.val
        j_min = self.slider_j_min.val
        j_max = self.slider_j_max.val
        
        # Validate bounds
        if enable_accel and a_min >= a_max:
            print(f"Warning: Invalid acceleration bounds: a_min={a_min} >= a_max={a_max}. Skipping solve.")
            return
        
        if enable_jerk and j_min >= j_max:
            print(f"Warning: Invalid jerk bounds: j_min={j_min} >= j_max={j_max}. Skipping solve.")
            return
        
        # Pre-check feasibility with initial conditions
        if enable_accel:
            a_meas = self.meas['a_meas']
            if not (a_min <= a_meas <= a_max):
                print(f"Warning: Initial acceleration a_meas={a_meas:.2f} violates bounds [{a_min:.2f}, {a_max:.2f}]")
                print("  Cannot solve. Adjust bounds or initial conditions.")
                return
        
        if enable_jerk:
            j_meas = self.meas['j_meas']
            if not (j_min <= j_meas <= j_max):
                print(f"Warning: Initial jerk j_meas={j_meas:.2f} violates bounds [{j_min:.2f}, {j_max:.2f}]")
                print("  Cannot solve. Adjust bounds or initial conditions.")
                return
        
        # Solve QP
        try:
            self.v = shape_speed_profile(
                self.r, self.dt, self.meas, weight_params, enforce_terminal,
                enable_accel_bounds=enable_accel, a_min=a_min, a_max=a_max,
                enable_jerk_bounds=enable_jerk, j_min=j_min, j_max=j_max
            )
        except Exception as e:
            print(f"Error solving QP: {e}")
            return
        
        # Update speed plot
        self.line_shaped.set_ydata(self.v)
        
        # Compute acceleration and jerk
        t_accel = self.t[:-1] + self.dt / 2  # Midpoints
        accel = np.diff(self.v) / self.dt
        
        t_jerk = t_accel[:-1] + self.dt / 2
        jerk = np.diff(accel) / self.dt
        
        # Update acceleration and jerk plots
        if self.line_accel is None:
            self.line_accel, = self.ax_accel.plot(t_accel, accel, 'g-', linewidth=2, label='Acceleration')
            self.ax_accel.legend(loc='upper left')
        else:
            self.line_accel.set_data(t_accel, accel)
        
        if self.line_jerk is None:
            self.line_jerk, = self.ax_jerk.plot(t_jerk, jerk, 'r-', linewidth=2, label='Jerk', alpha=0.7)
            self.ax_jerk.legend(loc='upper right')
        else:
            self.line_jerk.set_data(t_jerk, jerk)
        
        # Update constraint bound lines
        t_min, t_max = self.t.min(), self.t.max()
        
        # Acceleration bounds
        if enable_accel:
            if self.line_a_min is None:
                self.line_a_min, = self.ax_accel.plot([t_min, t_max], [a_min, a_min], 'g--', 
                                                       linewidth=1.5, alpha=0.6, label=f'a_min={a_min}')
                self.line_a_max, = self.ax_accel.plot([t_min, t_max], [a_max, a_max], 'g--', 
                                                       linewidth=1.5, alpha=0.6, label=f'a_max={a_max}')
            else:
                self.line_a_min.set_data([t_min, t_max], [a_min, a_min])
                self.line_a_max.set_data([t_min, t_max], [a_max, a_max])
                self.line_a_min.set_label(f'a_min={a_min}')
                self.line_a_max.set_label(f'a_max={a_max}')
            self.line_a_min.set_visible(True)
            self.line_a_max.set_visible(True)
        else:
            if self.line_a_min is not None:
                self.line_a_min.set_visible(False)
                self.line_a_max.set_visible(False)
        
        # Jerk bounds
        if enable_jerk:
            if self.line_j_min is None:
                self.line_j_min, = self.ax_jerk.plot([t_min, t_max], [j_min, j_min], 'r--', 
                                                      linewidth=1.5, alpha=0.6, label=f'j_min={j_min}')
                self.line_j_max, = self.ax_jerk.plot([t_min, t_max], [j_max, j_max], 'r--', 
                                                      linewidth=1.5, alpha=0.6, label=f'j_max={j_max}')
            else:
                self.line_j_min.set_data([t_min, t_max], [j_min, j_min])
                self.line_j_max.set_data([t_min, t_max], [j_max, j_max])
                self.line_j_min.set_label(f'j_min={j_min}')
                self.line_j_max.set_label(f'j_max={j_max}')
            self.line_j_min.set_visible(True)
            self.line_j_max.set_visible(True)
        else:
            if self.line_j_min is not None:
                self.line_j_min.set_visible(False)
                self.line_j_max.set_visible(False)
        
        # Update legends
        self.ax_accel.legend(loc='upper left')
        self.ax_jerk.legend(loc='upper right')
        
        # Auto-scale acceleration and jerk axes
        self.ax_accel.relim()
        self.ax_accel.autoscale_view()
        self.ax_jerk.relim()
        self.ax_jerk.autoscale_view()
        
        # Redraw
        self.fig.canvas.draw_idle()
        
        # Print some stats
        print(f"\n{'='*60}")
        print(f"Profile updated:")
        print(f"  Error weights:  start={weight_params['wE_start']:.1f}, end={weight_params['wE_end']:.1f}, λ={weight_params['lamE']:.2f}")
        print(f"  Accel weights:  start={weight_params['wA_start']:.1f}, end={weight_params['wA_end']:.1f}, λ={weight_params['lamA']:.2f}")
        print(f"  Jerk weights:   start={weight_params['wJ_start']:.1f}, end={weight_params['wJ_end']:.1f}, λ={weight_params['lamJ']:.2f}")
        print(f"  Terminal constraint: {enforce_terminal}")
        if enable_accel:
            print(f"  Accel bounds:   [{a_min:.1f}, {a_max:.1f}] m/s²  (actual: [{accel.min():.2f}, {accel.max():.2f}])")
        if enable_jerk:
            print(f"  Jerk bounds:    [{j_min:.1f}, {j_max:.1f}] m/s³  (actual: [{jerk.min():.2f}, {jerk.max():.2f}])")
        print(f"  Max |accel|: {np.abs(accel).max():.2f} m/s²")
        print(f"  Max |jerk|:  {np.abs(jerk).max():.2f} m/s³")
        print(f"  RMS tracking error: {np.sqrt(np.mean((self.v - self.r)**2)):.3f} m/s")
        print(f"{'='*60}")
    
    def show(self):
        """Display the GUI."""
        plt.show()


def main():
    """Main entry point for the GUI."""
    import sys
    
    # Check for command-line argument to select example type
    example_type = 'coarse'  # default
    if len(sys.argv) > 1:
        if sys.argv[1] in ['coarse', 'smooth']:
            example_type = sys.argv[1]
        else:
            print("Usage: python -m src.gui_matplotlib [coarse|smooth]")
            print("  coarse - Discontinuous steps with large jumps (default)")
            print("  smooth - Smooth profile with high acceleration/jerk")
            sys.exit(1)
    
    print("=" * 60)
    print("Speed Profile Shaper - Interactive GUI")
    print("=" * 60)
    print(f"\nExample type: {example_type}")
    if example_type == 'coarse':
        print("  - Discontinuous steps with large instantaneous jumps")
        print("  - Great for testing jerk smoothing on sharp transitions")
    else:
        print("  - Smooth profile with high acceleration and jerk")
        print("  - Great for testing dynamic smoothing on aggressive maneuvers")
    print("\nInstructions:")
    print("  - Adjust sliders to tune weight schedules")
    print("  - Plot updates automatically (no button needed)")
    print("  - wX_start: weight at t=0")
    print("  - wX_end: weight at t=∞")
    print("  - λX: exponential decay rate")
    print("  - Check box to enforce terminal constraint v[N]=r[N]")
    print("\nTips:")
    print("  - High error weight → tight tracking")
    print("  - High accel weight → smooth slopes")
    print("  - High jerk weight → no sharp corners")
    print("=" * 60)
    print()
    
    gui = SpeedShaperGUI(example_type=example_type)
    gui.show()


if __name__ == '__main__':
    main()
