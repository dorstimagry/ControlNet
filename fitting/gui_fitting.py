"""GUI application for vehicle parameter fitting configuration and execution."""

from __future__ import annotations

import json
import logging
import threading
import time
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from fitting.vehicle_fitter import FitterConfig, VehicleParamFitter

matplotlib.use("TkAgg")

LOGGER = logging.getLogger(__name__)

# Parameter groups for DC motor model
PARAM_GROUPS_DC = {
    "Body": ["mass", "drag_area", "rolling_coeff"],
    "Motor": ["motor_V_max", "motor_R", "motor_L", "motor_K", "motor_b", "motor_J"],
    "Drivetrain": ["gear_ratio", "eta_gb"],
    "Brake": ["brake_T_max", "brake_tau", "brake_p", "brake_kappa", "mu"],
    "Wheel": ["wheel_radius", "wheel_inertia"],
}

# Parameter groups for polynomial motor model
PARAM_GROUPS_POLY = {
    "Body": ["mass", "drag_area", "rolling_coeff"],
    "Motor": ["motor_V_max"],
    "Polynomial Coefficients": [
        "poly_c_00", "poly_c_10", "poly_c_01", "poly_c_20", "poly_c_11", "poly_c_02",
        "poly_c_30", "poly_c_21", "poly_c_12", "poly_c_03"
    ],
    "Drivetrain": ["gear_ratio", "eta_gb"],
    "Brake": ["brake_T_max", "brake_tau", "brake_p", "brake_kappa", "mu"],
    "Wheel": ["wheel_radius", "wheel_inertia"],
}

# Parameter display names and units
PARAM_DISPLAY = {
    "mass": ("Mass", "kg"),
    "drag_area": ("Drag Area", "m²"),
    "rolling_coeff": ("Rolling Coefficient", ""),
    "motor_V_max": ("Motor V_max", "V"),
    "motor_R": ("Motor R", "Ω"),
    "motor_L": ("Motor L", "H"),
    "motor_K": ("Motor K", "Nm/A"),
    "motor_b": ("Motor b", "Nm·s/rad"),
    "motor_J": ("Motor J", "kg·m²"),
    "poly_c_00": ("c₀₀ (constant)", ""),
    "poly_c_10": ("c₁₀ (V)", ""),
    "poly_c_01": ("c₀₁ (ω)", ""),
    "poly_c_20": ("c₂₀ (V²)", ""),
    "poly_c_11": ("c₁₁ (V·ω)", ""),
    "poly_c_02": ("c₀₂ (ω²)", ""),
    "poly_c_30": ("c₃₀ (V³)", ""),
    "poly_c_21": ("c₂₁ (V²·ω)", ""),
    "poly_c_12": ("c₁₂ (V·ω²)", ""),
    "poly_c_03": ("c₀₃ (ω³)", ""),
    "gear_ratio": ("Gear Ratio", ""),
    "eta_gb": ("Gearbox Efficiency", ""),
    "brake_T_max": ("Brake T_max", "Nm"),
    "brake_tau": ("Brake τ", "s"),
    "brake_p": ("Brake p", ""),
    "brake_kappa": ("Brake κ", ""),
    "mu": ("Friction μ", ""),
    "wheel_radius": ("Wheel Radius", "m"),
    "wheel_inertia": ("Wheel Inertia", "kg·m²"),
}


class FittingGUI:
    """Main GUI application for vehicle parameter fitting."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Vehicle Parameter Fitting")
        self.root.geometry("1400x900")

        # Default config for initial values
        self.default_config = FitterConfig()

        # Data paths
        self.data_dir = Path(__file__).parent.parent / "data" / "processed"
        self.results_dir = Path(__file__).parent / "results"

        # Fitting state
        self.fitting_thread: Optional[threading.Thread] = None
        self.is_fitting = False
        self.current_fitter: Optional[VehicleParamFitter] = None  # Store fitter for param names

        # Update throttling
        self.last_update_time = 0.0
        self.update_interval = 0.5  # Minimum seconds between plot updates
        self.pending_params: Optional[np.ndarray] = None
        self.pending_loss: Optional[float] = None
        self.update_pending = False

        # Plot line objects for efficient updates
        self.throttle_lines = {}
        self.brake_lines = {}

        # Create main layout
        self._create_widgets()

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with paned windows for resizing
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Configuration
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)

        # Right panel: Simulation preview
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # Left panel: Top section (dataset and name)
        top_left = ttk.LabelFrame(left_frame, text="Fitting Configuration", padding=10)
        top_left.pack(fill=tk.X, padx=5, pady=5)

        # Dataset selection
        ttk.Label(top_left, text="Dataset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(
            top_left, textvariable=self.dataset_var, state="readonly", width=50
        )
        self.dataset_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=5, padx=5)
        self._populate_datasets()

        # Fitting name
        ttk.Label(top_left, text="Fitting Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(top_left, textvariable=self.name_var, width=50)
        self.name_entry.grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=5, padx=5)

        # Barrier function controls
        self.use_barrier_var = tk.BooleanVar(value=False)
        barrier_check = ttk.Checkbutton(
            top_left,
            text="Use Barrier Functions",
            variable=self.use_barrier_var,
        )
        barrier_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)
        
        ttk.Label(top_left, text="Barrier μ:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.barrier_mu_var = tk.StringVar(value="0.01")
        barrier_mu_entry = ttk.Entry(top_left, textvariable=self.barrier_mu_var, width=15)
        barrier_mu_entry.grid(row=3, column=1, sticky=tk.W, pady=5, padx=5)
        
        ttk.Label(
            top_left,
            text="(keeps parameters away from boundaries)",
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).grid(row=3, column=2, sticky=tk.W, padx=5)

        # Motor model type selector
        ttk.Label(top_left, text="Motor Model:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.motor_model_var = tk.StringVar(value="dc")
        motor_model_frame = ttk.Frame(top_left)
        motor_model_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=5, padx=5)
        ttk.Radiobutton(
            motor_model_frame, text="DC Motor", variable=self.motor_model_var, value="dc",
            command=self._on_motor_model_changed
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            motor_model_frame, text="Polynomial Map", variable=self.motor_model_var, value="polynomial",
            command=self._on_motor_model_changed
        ).pack(side=tk.LEFT, padx=5)
        
        # Fit DC from map option (only shown for polynomial model)
        self.fit_dc_from_map_var = tk.BooleanVar(value=False)
        self.fit_dc_check = ttk.Checkbutton(
            top_left,
            text="Fit DC parameters from map after optimization",
            variable=self.fit_dc_from_map_var,
        )
        self.fit_dc_check.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=5, padx=5)
        self.fit_dc_check.grid_remove()  # Hidden by default

        top_left.columnconfigure(1, weight=1)

        # Left panel: Parameters (scrollable)
        params_frame = ttk.LabelFrame(left_frame, text="Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create scrollable canvas for parameters
        canvas = tk.Canvas(params_frame)
        scrollbar = ttk.Scrollbar(params_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Store parameter entry widgets and scrollable frame reference
        self.param_entries: Dict[str, Dict[str, tk.StringVar]] = {}
        self.scrollable_frame = scrollable_frame
        self.param_canvas = canvas

        # Create parameter input fields (will be updated when motor model changes)
        self._create_parameter_fields()

        # Left panel: Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.update_sim_btn = ttk.Button(
            button_frame, text="Update Simulation", command=self._update_simulation
        )
        self.update_sim_btn.pack(side=tk.LEFT, padx=5)

        self.start_fitting_btn = ttk.Button(
            button_frame, text="Start Fitting", command=self._start_fitting
        )
        self.start_fitting_btn.pack(side=tk.LEFT, padx=5)

        # Progress bar and status
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(button_frame, textvariable=self.progress_var)
        self.progress_label.pack(side=tk.LEFT, padx=10)

        self.progress_bar = ttk.Progressbar(
            button_frame, mode="indeterminate", length=200
        )
        self.progress_bar.pack(side=tk.LEFT, padx=5)

        # Right panel: Simulation plots
        plot_frame = ttk.LabelFrame(right_frame, text="Simulation Preview", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 10), dpi=100)
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)

        self.ax1.set_title("Throttle Dynamics (from 0 m/s)")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Speed (m/s)")
        self.ax1.grid(True, alpha=0.3)

        self.ax2.set_title("Braking Dynamics (from 20 m/s)")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Speed (m/s)")
        self.ax2.grid(True, alpha=0.3)

        self.fig.tight_layout()

        # Embed in tkinter
        self.canvas_plot = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas_plot.draw()
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _on_motor_model_changed(self):
        """Called when motor model type changes - update parameter fields."""
        # Clear existing parameter fields
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.param_entries.clear()
        
        # Recreate parameter fields for new model
        self._create_parameter_fields()
        
        # Show/hide fit DC checkbox
        if self.motor_model_var.get() == "polynomial":
            self.fit_dc_check.grid()
        else:
            self.fit_dc_check.grid_remove()
    
    def _create_parameter_fields(self):
        """Create parameter input fields based on current motor model type."""
        # Get appropriate parameter groups
        if self.motor_model_var.get() == "polynomial":
            param_groups = PARAM_GROUPS_POLY
        else:
            param_groups = PARAM_GROUPS_DC
        
        row = 0
        for group_name, param_names in param_groups.items():
            # Group header
            ttk.Label(
                self.scrollable_frame, text=group_name, font=("TkDefaultFont", 10, "bold")
            ).grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(10, 5))
            row += 1

            # Column headers (only show once)
            if row == 1:
                ttk.Label(self.scrollable_frame, text="Parameter").grid(
                    row=row, column=0, sticky=tk.W, padx=5
                )
                ttk.Label(self.scrollable_frame, text="Initial").grid(
                    row=row, column=1, sticky=tk.W, padx=5
                )
                ttk.Label(self.scrollable_frame, text="Min").grid(
                    row=row, column=2, sticky=tk.W, padx=5
                )
                ttk.Label(self.scrollable_frame, text="Max").grid(
                    row=row, column=3, sticky=tk.W, padx=5
                )
                row += 1

            # Parameter rows
            for param_name in param_names:
                display_name, unit = PARAM_DISPLAY.get(param_name, (param_name, ""))
                label_text = f"{display_name}"
                if unit:
                    label_text += f" ({unit})"

                ttk.Label(self.scrollable_frame, text=label_text).grid(
                    row=row, column=0, sticky=tk.W, padx=5, pady=2
                )

                # Get default values
                try:
                    init_val = getattr(self.default_config, f"{param_name}_init")
                    bounds = getattr(self.default_config, f"{param_name}_bounds")
                    min_val, max_val = bounds
                except AttributeError:
                    # Use defaults if not found
                    init_val = 0.0
                    min_val, max_val = -100.0, 100.0

                # Entry variables
                init_var = tk.StringVar(value=str(init_val))
                min_var = tk.StringVar(value=str(min_val))
                max_var = tk.StringVar(value=str(max_val))

                # Entry widgets
                ttk.Entry(self.scrollable_frame, textvariable=init_var, width=12).grid(
                    row=row, column=1, padx=5, pady=2
                )
                ttk.Entry(self.scrollable_frame, textvariable=min_var, width=12).grid(
                    row=row, column=2, padx=5, pady=2
                )
                ttk.Entry(self.scrollable_frame, textvariable=max_var, width=12).grid(
                    row=row, column=3, padx=5, pady=2
                )

                self.param_entries[param_name] = {
                    "init": init_var,
                    "min": min_var,
                    "max": max_var,
                }

                row += 1
        
        # Update canvas scroll region
        self.scrollable_frame.update_idletasks()
        self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all"))

    def _populate_datasets(self):
        """Scan data/processed for .pt files and populate dropdown."""
        datasets = []
        if self.data_dir.exists():
            for pt_file in self.data_dir.rglob("*.pt"):
                # Get relative path from data/processed
                try:
                    rel_path = pt_file.relative_to(self.data_dir.parent.parent)
                    datasets.append(str(rel_path))
                except ValueError:
                    # Skip if path resolution fails
                    continue

        datasets.sort()
        self.dataset_combo["values"] = datasets
        if datasets:
            self.dataset_combo.current(0)
        else:
            # Show warning if no datasets found
            self.dataset_combo.set("No datasets found")

    def _get_params_from_gui(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Extract parameter values from GUI entries."""
        params = {}
        for param_name, entries in self.param_entries.items():
            try:
                init_val = float(entries["init"].get())
                min_val = float(entries["min"].get())
                max_val = float(entries["max"].get())

                if min_val > max_val:
                    messagebox.showerror(
                        "Validation Error",
                        f"Parameter {param_name}: min ({min_val}) must be <= max ({max_val})",
                    )
                    return None
                
                # If min == max, parameter is fixed (not optimized)
                # Ensure initial value matches the fixed value
                if min_val == max_val:
                    # Update initial value to match fixed value
                    entries["init"].set(str(min_val))
                    init_val = min_val

                params[param_name] = {
                    "init": init_val,
                    "min": min_val,
                    "max": max_val,
                }
            except ValueError:
                messagebox.showerror(
                    "Validation Error", f"Invalid number for parameter {param_name}"
                )
                return None

        return params

    def _update_simulation(self):
        """Update simulation plots with current parameter values."""
        params_dict = self._get_params_from_gui()
        if params_dict is None:
            return

        # Convert to parameter array for simulation
        motor_model_type = self.motor_model_var.get()
        param_array = self._params_dict_to_array(params_dict, use_init=True, motor_model_type=motor_model_type)

        # Clear existing lines to force recreation
        self.throttle_lines.clear()
        self.brake_lines.clear()
        self.ax1.clear()
        self.ax2.clear()

        # Update plots (will recreate lines)
        self._update_simulation_plots_callback(param_array)

    def _simulate_throttle_response(
        self, params: np.ndarray, dt: float = 0.1, duration: float = 40.0
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """Simulate throttle response from 0 m/s."""
        # Create fitter with current motor model type
        motor_model_type = self.motor_model_var.get()
        config = FitterConfig(motor_model_type=motor_model_type)
        fitter = VehicleParamFitter(config)
        results = {}

        throttle_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        n_steps = int(duration / dt)
        time = np.arange(n_steps) * dt

        for throttle_pct in throttle_values:
            speed = np.zeros(n_steps)
            throttle_cmd = throttle_pct * 100.0  # Convert to 0-100 scale
            brake_cmd = 0.0
            grade = 0.0  # Flat road

            for t in range(n_steps - 1):
                accel = fitter._compute_acceleration(
                    params, speed[t], throttle_cmd, brake_cmd, grade
                )
                speed[t + 1] = max(speed[t] + accel * dt, 0.0)

            results[throttle_pct] = (time, speed)

        return results

    def _simulate_brake_response(
        self,
        params: np.ndarray,
        dt: float = 0.1,
        duration: float = 40.0,
        initial_speed: float = 20.0,
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """Simulate brake response from initial speed."""
        # Create fitter with current motor model type
        motor_model_type = self.motor_model_var.get()
        config = FitterConfig(motor_model_type=motor_model_type)
        fitter = VehicleParamFitter(config)
        results = {}

        brake_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        n_steps = int(duration / dt)
        time = np.arange(n_steps) * dt

        for brake_pct in brake_values:
            speed = np.zeros(n_steps)
            speed[0] = initial_speed
            throttle_cmd = 0.0
            brake_cmd = brake_pct * 100.0  # Convert to 0-100 scale
            grade = 0.0  # Flat road

            for t in range(n_steps - 1):
                accel = fitter._compute_acceleration(
                    params, speed[t], throttle_cmd, brake_cmd, grade
                )
                speed[t + 1] = max(speed[t] + accel * dt, 0.0)

            results[brake_pct] = (time, speed)

        return results

    def _params_dict_to_array(
        self, params_dict: Dict[str, Dict[str, float]], use_init: bool = True, motor_model_type: str = "dc"
    ) -> np.ndarray:
        """Convert parameter dictionary to array in correct order."""
        # Create temporary config to get param names
        temp_config = FitterConfig(motor_model_type=motor_model_type)
        temp_fitter = VehicleParamFitter(temp_config)
        param_names = temp_fitter.PARAM_NAMES
        param_array = np.zeros(len(param_names))

        for i, param_name in enumerate(param_names):
            if param_name in params_dict:
                if use_init:
                    param_array[i] = params_dict[param_name]["init"]
                else:
                    # Use midpoint of bounds
                    min_val = params_dict[param_name]["min"]
                    max_val = params_dict[param_name]["max"]
                    param_array[i] = (min_val + max_val) / 2.0

        return param_array

    def _validate_inputs(self) -> bool:
        """Validate all inputs before starting fitting."""
        # Check dataset
        dataset_path = self.dataset_var.get()
        if not dataset_path:
            messagebox.showerror("Validation Error", "Please select a dataset")
            return False

        full_path = Path(__file__).parent.parent / dataset_path
        if not full_path.exists():
            messagebox.showerror("Validation Error", f"Dataset not found: {full_path}")
            return False

        # Check name
        fitting_name = self.name_var.get().strip()
        if not fitting_name:
            messagebox.showerror("Validation Error", "Please enter a fitting name")
            return False

        # Validate parameters
        params_dict = self._get_params_from_gui()
        if params_dict is None:
            return False

        return True

    def _create_fitter_config(self) -> FitterConfig:
        """Create FitterConfig from GUI inputs."""
        params_dict = self._get_params_from_gui()
        if params_dict is None:
            raise ValueError("Invalid parameters")

        config_kwargs = {}

        # Add all parameter initial values and bounds
        for param_name, values in params_dict.items():
            config_kwargs[f"{param_name}_init"] = values["init"]
            config_kwargs[f"{param_name}_bounds"] = (values["min"], values["max"])

        # Add barrier function settings
        config_kwargs["use_barrier"] = self.use_barrier_var.get()
        try:
            barrier_mu = float(self.barrier_mu_var.get())
            if barrier_mu <= 0:
                raise ValueError("Barrier μ must be positive")
            config_kwargs["barrier_mu"] = barrier_mu
        except ValueError as e:
            raise ValueError(f"Invalid barrier μ value: {e}")

        # Add motor model settings
        config_kwargs["motor_model_type"] = self.motor_model_var.get()
        config_kwargs["fit_dc_from_map"] = self.fit_dc_from_map_var.get()

        return FitterConfig(**config_kwargs)

    def _start_fitting(self):
        """Start the fitting process in a background thread."""
        if not self._validate_inputs():
            return

        if self.is_fitting:
            messagebox.showwarning("Warning", "Fitting already in progress")
            return

        # Disable button and show progress
        self.start_fitting_btn.config(state=tk.DISABLED)
        self.is_fitting = True
        self.progress_var.set("Starting fitting...")
        self.progress_bar.start()

        # Start background thread
        self.fitting_thread = threading.Thread(target=self._run_fitting, daemon=True)
        self.fitting_thread.start()

    def _run_fitting(self):
        """Run fitting in background thread."""
        try:
            # Get inputs
            dataset_path = Path(__file__).parent.parent / self.dataset_var.get()
            fitting_name = self.name_var.get().strip()
            output_dir = self.results_dir / fitting_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create config
            config = self._create_fitter_config()

            # Save config
            config_path = output_dir / "config.json"
            config_dict = {
                "dataset": str(dataset_path),
                "fitting_name": fitting_name,
                **asdict(config),
            }
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            # Update status
            self.root.after(0, lambda: self.progress_var.set("Loading data..."))

            # Create fitter and run
            fitter = VehicleParamFitter(config)
            self.current_fitter = fitter  # Store for param names access
            output_params_path = output_dir / "fitted_params.json"

            self.root.after(0, lambda: self.progress_var.set("Fitting parameters..."))

            fitted = fitter.fit(
                dataset_path,
                verbose=True,
                log_path=output_dir / "fitting_checkpoint.json",
                progress_callback=self._progress_callback
            )
            
            self.current_fitter = None  # Clear after fitting

            # Save results
            fitted.save(output_params_path)

            # Save simulation preview plots
            # Extract parameters based on model type
            if config.motor_model_type == "polynomial":
                # Polynomial model - extract from dict
                param_dict = fitted.to_dict()
                motor_model_type = param_dict.get("motor_model_type", "polynomial")
                temp_config = FitterConfig(motor_model_type=motor_model_type)
                temp_fitter = VehicleParamFitter(temp_config)
                param_names = temp_fitter.PARAM_NAMES
                param_array = np.array([param_dict.get(name, 0.0) for name in param_names])
            else:
                # DC model - use FittedVehicleParams structure
                param_array = np.array([
                    fitted.mass, fitted.drag_area, fitted.rolling_coeff,
                    fitted.motor_V_max, fitted.motor_R, fitted.motor_L,
                    fitted.motor_K, fitted.motor_b, fitted.motor_J,
                    fitted.gear_ratio, fitted.eta_gb,
                    fitted.brake_T_max, fitted.brake_tau, fitted.brake_p,
                    fitted.brake_kappa, fitted.mu,
                    fitted.wheel_radius, fitted.wheel_inertia,
                ])

            # Use the model type from config (what was actually fitted)
            # Temporarily set motor model type for simulations
            original_model_type = self.motor_model_var.get()
            self.motor_model_var.set(config.motor_model_type)
            
            throttle_data = self._simulate_throttle_response(param_array)
            brake_data = self._simulate_brake_response(param_array)
            
            # Restore original model type
            self.motor_model_var.set(original_model_type)

            # Create and save preview plots
            preview_fig = Figure(figsize=(10, 8), dpi=100)
            ax1 = preview_fig.add_subplot(2, 1, 1)
            ax2 = preview_fig.add_subplot(2, 1, 2)

            for throttle, (time, speed) in throttle_data.items():
                ax1.plot(time, speed, label=f"Throttle {throttle:.1f}")
            ax1.set_title("Throttle Dynamics (from 0 m/s)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Speed (m/s)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            for brake, (time, speed) in brake_data.items():
                ax2.plot(time, speed, label=f"Brake {brake:.1f}")
            ax2.set_title("Braking Dynamics (from 20 m/s)")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Speed (m/s)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            preview_fig.tight_layout()
            preview_fig.savefig(output_dir / "simulation_preview.png", dpi=150, bbox_inches="tight")

            # Success
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Success",
                    f"Fitting completed!\nResults saved to:\n{output_dir}",
                ),
            )

        except Exception as e:
            LOGGER.exception("Fitting failed")
            error_msg = str(e)
            self.root.after(
                0, lambda msg=error_msg: messagebox.showerror("Fitting Error", f"Fitting failed:\n{msg}")
            )
        finally:
            # Re-enable button
            self.root.after(0, self._fitting_complete)

    def _fitting_complete(self):
        """Called when fitting completes (in main thread)."""
        self.is_fitting = False
        self.start_fitting_btn.config(state=tk.NORMAL)
        self.progress_bar.stop()
        self.progress_var.set("Fitting completed - parameters updated")

    def _progress_callback(self, best_params: np.ndarray, best_loss: float):
        """Callback called during fitting when new best parameters are found."""
        # Store pending update
        self.pending_params = best_params.copy()
        self.pending_loss = best_loss

        # Throttle updates - only schedule if enough time has passed
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            if not self.update_pending:
                self.update_pending = True
                self.root.after(0, self._process_pending_update)
                self.last_update_time = current_time

    def _process_pending_update(self):
        """Process pending parameter update (called in main thread)."""
        if self.pending_params is None or self.pending_loss is None:
            self.update_pending = False
            return

        params = self.pending_params
        loss = self.pending_loss

        # Update parameter display with current best
        # Get param names from current fitter if available
        if self.current_fitter is not None:
            param_names = self.current_fitter.PARAM_NAMES
            for i, param_name in enumerate(param_names):
                if param_name in self.param_entries:
                    # Update the initial value field to show current best
                    self.param_entries[param_name]["init"].set(f"{params[i]:.6f}")

        # Update progress text
        rmse = np.sqrt(loss)
        self.progress_var.set(f"New best found - RMSE: {rmse:.4f} m/s")

        # Update simulation plots
        self._update_simulation_plots_callback(params)

        # Clear pending
        self.pending_params = None
        self.pending_loss = None
        self.update_pending = False

    def _update_simulation_plots_callback(self, params: np.ndarray):
        """Update simulation plots with given parameters (called in main thread)."""
        # Run simulations
        throttle_data = self._simulate_throttle_response(params)
        brake_data = self._simulate_brake_response(params)

        # Throttle plot - reuse lines if they exist
        if not self.throttle_lines:
            # First time: create lines and legend
            for throttle, (time_arr, speed) in throttle_data.items():
                line, = self.ax1.plot(time_arr, speed, label=f"Throttle {throttle:.1f}")
                self.throttle_lines[throttle] = line
            self.ax1.set_title("Throttle Dynamics (from 0 m/s)")
            self.ax1.set_xlabel("Time (s)")
            self.ax1.set_ylabel("Speed (m/s)")
            self.ax1.legend(ncol=2, fontsize=8)
            self.ax1.grid(True, alpha=0.3)
        else:
            # Update existing lines
            for throttle, (time_arr, speed) in throttle_data.items():
                if throttle in self.throttle_lines:
                    self.throttle_lines[throttle].set_data(time_arr, speed)

        # Brake plot - reuse lines if they exist
        if not self.brake_lines:
            # First time: create lines and legend
            for brake, (time_arr, speed) in brake_data.items():
                line, = self.ax2.plot(time_arr, speed, label=f"Brake {brake:.1f}")
                self.brake_lines[brake] = line
            self.ax2.set_title("Braking Dynamics (from 20 m/s)")
            self.ax2.set_xlabel("Time (s)")
            self.ax2.set_ylabel("Speed (m/s)")
            self.ax2.legend(ncol=2, fontsize=8)
            self.ax2.grid(True, alpha=0.3)
        else:
            # Update existing lines
            for brake, (time_arr, speed) in brake_data.items():
                if brake in self.brake_lines:
                    self.brake_lines[brake].set_data(time_arr, speed)

        # Update axis limits
        self.ax1.relim()
        self.ax1.autoscale()
        self.ax2.relim()
        self.ax2.autoscale()

        # Only call tight_layout once initially, then just draw
        if not self.throttle_lines or not self.brake_lines:
            # First time setup - need tight_layout
            self.fig.tight_layout()
        
        self.canvas_plot.draw_idle()  # Use draw_idle for better performance


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = FittingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

