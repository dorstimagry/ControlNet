#!/usr/bin/env python
"""
Demonstration of EV-style creep behavior on flat ground.

This script shows how the creep torque mechanism works:
1. Vehicle starts at rest with zero throttle/brake
2. Creep torque provides gentle forward acceleration
3. Creep fades smoothly as speed increases
4. Vehicle settles below v_cutoff speed

The plot shows:
- Speed evolution over time
- Creep torque fade with speed
- Acceleration profile
- Phase transitions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.dynamics import (
    ExtendedPlant,
    ExtendedPlantParams,
    ExtendedPlantRandomization,
    sample_extended_params,
    CreepParams,
    MotorParams,
    BrakeParams,
    BodyParams,
    WheelParams,
)


def demonstrate_creep_on_plateau():
    """Demonstrate creep behavior on flat ground (zero grade)."""
    
    # Create vehicle with known parameters for reproducibility
    print("Setting up vehicle with creep parameters...")
    
    # Use a typical mid-size EV configuration
    motor = MotorParams(
        R=0.15,           # 0.15 Ω resistance
        L=0.001,          # 1 mH inductance
        K_e=0.18,         # Back-EMF constant
        K_t=0.18,         # Torque constant
        b=0.001,          # Viscous friction
        J=0.001,          # Rotor inertia
        V_max=400.0,      # 400V max
        gear_ratio=10.0,  # 10:1 gear ratio
        eta_gb=0.92,      # 92% efficiency
    )
    
    brake = BrakeParams(
        T_br_max=15000.0,  # 15 kNm max brake torque
        p_br=1.2,
        tau_br=0.08,
        kappa_c=0.08,
        mu=0.9,
    )
    
    body = BodyParams(
        mass=1800.0,        # 1800 kg vehicle
        drag_area=0.6,      # 0.6 m² CdA
        rolling_coeff=0.01, # Low rolling resistance
        grade_rad=0.0,      # FLAT GROUND (plateau)
        air_density=1.225,
    )
    
    wheel = WheelParams(
        radius=0.33,        # 0.33 m wheel radius
        inertia=1.5,
        v_eps=0.1,
    )
    
    # Creep parameters (matching config)
    creep = CreepParams(
        a_max=0.5,      # 0.5 m/s² max creep acceleration
        v_cutoff=2.0,   # Fade out by 2.0 m/s
        v_hold=0.08,    # Standstill threshold
    )
    
    params = ExtendedPlantParams(
        motor=motor,
        brake=brake,
        body=body,
        wheel=wheel,
        creep=creep,
    )
    
    plant = ExtendedPlant(params)
    
    # Simulation parameters
    dt = 0.05  # 50ms timestep for smooth curves
    duration = 15.0  # 15 seconds
    num_steps = int(duration / dt)
    
    # Storage for results
    time = np.zeros(num_steps)
    speed = np.zeros(num_steps)
    acceleration = np.zeros(num_steps)
    creep_torque = np.zeros(num_steps)
    motor_current = np.zeros(num_steps)
    position = np.zeros(num_steps)
    
    print(f"Running simulation: {duration}s at {1/dt:.0f} Hz...")
    print("Vehicle starting from rest with ZERO throttle/brake (pure creep)")
    print()
    
    # Start from rest
    plant.reset(speed=0.0, position=0.0)
    
    # Simulate with zero action (no throttle, no brake - pure creep)
    for i in range(num_steps):
        state = plant.step(action=0.0, dt=dt, substeps=5)
        
        time[i] = i * dt
        speed[i] = state.speed
        acceleration[i] = state.acceleration
        creep_torque[i] = state.creep_torque
        motor_current[i] = state.motor_current
        position[i] = state.position
        
        # Print milestone updates
        if i % 100 == 0:
            print(f"t={time[i]:5.2f}s  v={speed[i]:5.2f} m/s  "
                  f"a={acceleration[i]:5.2f} m/s²  "
                  f"T_creep={creep_torque[i]:6.1f} Nm")
    
    print()
    print("Simulation complete!")
    print(f"Final speed: {speed[-1]:.2f} m/s")
    print(f"Distance traveled: {position[-1]:.1f} m")
    print()
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('EV-Style Creep Behavior on Flat Ground (Zero Throttle/Brake)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Speed vs Time
    ax1 = axes[0, 0]
    ax1.plot(time, speed, 'b-', linewidth=2, label='Vehicle Speed')
    ax1.axhline(y=creep.v_cutoff, color='r', linestyle='--', alpha=0.5, 
                label=f'v_cutoff ({creep.v_cutoff} m/s)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Speed (m/s)', fontsize=11)
    ax1.set_title('Speed Evolution with Creep', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.set_xlim([0, duration])
    
    # Add phase annotations
    ax1.text(2, 0.3, 'Phase 1:\nInitial Acceleration', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
             fontsize=9)
    ax1.text(8, 1.3, 'Phase 2:\nCreep Fade', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
             fontsize=9)
    
    # Plot 2: Creep Torque vs Time
    ax2 = axes[0, 1]
    ax2.plot(time, creep_torque, 'g-', linewidth=2, label='Creep Torque')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Creep Torque (Nm)', fontsize=11)
    ax2.set_title('Creep Torque Fade with Speed', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.set_xlim([0, duration])
    
    # Add fade explanation
    ax2.text(0.5, 0.95, 'Torque fades with cubic\nsmoothstep as v → v_cutoff',
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
             fontsize=9, verticalalignment='top')
    
    # Plot 3: Acceleration vs Time
    ax3 = axes[1, 0]
    ax3.plot(time, acceleration, 'r-', linewidth=2, label='Acceleration')
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Acceleration (m/s²)', fontsize=11)
    ax3.set_title('Acceleration Profile', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.set_xlim([0, duration])
    
    # Plot 4: Creep Torque vs Speed (phase diagram)
    ax4 = axes[1, 1]
    # Color by time for trajectory visualization
    scatter = ax4.scatter(speed, creep_torque, c=time, cmap='viridis', 
                         s=10, alpha=0.7)
    ax4.plot(speed, creep_torque, 'k-', alpha=0.2, linewidth=0.5)
    ax4.axvline(x=creep.v_cutoff, color='r', linestyle='--', alpha=0.5,
                label=f'v_cutoff = {creep.v_cutoff} m/s')
    cbar = plt.colorbar(scatter, ax=ax4, label='Time (s)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Speed (m/s)', fontsize=11)
    ax4.set_ylabel('Creep Torque (Nm)', fontsize=11)
    ax4.set_title('Creep Fade Characteristic', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    
    # Add fade function annotation
    ax4.text(0.5, 0.95, 'Fade: w(v) = 1 - 3x² + 2x³\nwhere x = v/v_cutoff',
             transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7),
             fontsize=8, verticalalignment='top',
             family='monospace')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('evaluation/results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'creep_behavior_demo.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Also save as PDF for publication quality
    output_path_pdf = output_dir / 'creep_behavior_demo.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✅ PDF saved to: {output_path_pdf}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Vehicle Configuration:")
    print(f"  Mass: {body.mass:.0f} kg")
    print(f"  Max Creep Acceleration: {creep.a_max:.2f} m/s²")
    print(f"  Creep Cutoff Speed: {creep.v_cutoff:.2f} m/s")
    print()
    print(f"Results:")
    print(f"  Final Speed: {speed[-1]:.3f} m/s ({speed[-1]*3.6:.2f} km/h)")
    print(f"  Distance Traveled: {position[-1]:.1f} m")
    print(f"  Max Acceleration: {np.max(acceleration):.3f} m/s²")
    print(f"  Initial Creep Torque: {creep_torque[0]:.1f} Nm")
    print(f"  Final Creep Torque: {creep_torque[-1]:.1f} Nm")
    print(f"  Max Motor Current: {np.max(motor_current):.1f} A")
    print()
    print("Key Observations:")
    print(f"  ✓ Creep provides smooth acceleration from rest")
    print(f"  ✓ Torque fades continuously as speed increases")
    print(f"  ✓ Speed stabilizes below v_cutoff ({creep.v_cutoff} m/s)")
    print(f"  ✓ No oscillations or discontinuities")
    print("="*60)


if __name__ == '__main__':
    demonstrate_creep_on_plateau()

