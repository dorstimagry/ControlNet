#!/usr/bin/env python3
"""Example script to visualize raw vs filtered speed profiles for violent profile mode."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

from generator.lpf import SecondOrderLPF
from generator.adapter import create_reference_generator
from utils.dynamics import sample_extended_params, ExtendedPlantRandomization


def main():
    # Load config to get parameters
    config_path = Path("training/config_sac_pretrained_violent.yaml")
    with config_path.open() as f:
        config = yaml.safe_load(f)
    
    env_config = config["env"]
    generator_config = config.get("generator", {})
    
    # Create a sample vehicle (as would be done during training)
    extended_random = ExtendedPlantRandomization.from_config(config)
    rng = np.random.default_rng(42)
    extended_params = sample_extended_params(rng, extended_random)
    
    # Create reference generator (as would be done during training)
    dt = env_config["dt"]
    preview_steps = max(int(round(env_config["preview_horizon_s"] / dt)), 1)
    generator = create_reference_generator(
        dt=dt,
        prediction_horizon=preview_steps,
        generator_config=generator_config,
        device=torch.device('cpu')
    )
    
    # Generate profiles as would happen during training initialization
    # The generator returns: (filtered_profile, grade_profile, raw_profile)
    profile_length = 200
    filtered_profile, grade_profile, raw_profile = generator.sample(
        profile_length, rng=rng, vehicle=None
    )
    
    # Apply reward filter to raw profile (as done during environment reset in violent mode)
    reward_filter_freq_cutoff = env_config["reward_filter_freq_cutoff"]
    reward_filter_zeta = env_config["reward_filter_zeta"]
    reward_filter_dt = env_config["reward_filter_dt"]
    
    # Create reward filter (as done in LongitudinalEnv.__init__)
    rate_max = torch.tensor([15.0], device=torch.device('cpu'))
    rate_neg_max = torch.tensor([20.0], device=torch.device('cpu'))
    jerk_max = torch.tensor([12.0], device=torch.device('cpu'))
    
    reward_filter = SecondOrderLPF(
        batch_size=1,
        freq_cutoff=reward_filter_freq_cutoff,
        zeta=reward_filter_zeta,
        dt=reward_filter_dt,
        rate_max=rate_max,
        rate_neg_max=rate_neg_max,
        jerk_max=jerk_max,
        device=torch.device('cpu')
    )
    
    # Filter the raw profile (as done in LongitudinalEnv.reset() in violent mode)
    raw_tensor = torch.from_numpy(raw_profile).unsqueeze(0)  # [1, T]
    filtered_by_reward_filter = torch.zeros_like(raw_tensor)
    
    # Reset filter state with initial value from raw profile
    initial_y = torch.tensor([[raw_profile[0]]], device=torch.device('cpu'), dtype=torch.float32)
    reward_filter.reset(initial_y=initial_y)
    
    # Process each timestep through the filter
    for t in range(len(raw_profile)):
        u_t = raw_tensor[:, t:t+1]  # [1, 1]
        filtered_y = reward_filter.update(u_t)
        filtered_by_reward_filter[:, t] = filtered_y.squeeze(0)
    
    reward_filtered_profile = filtered_by_reward_filter.squeeze(0).cpu().numpy().astype(np.float32)
    
    # Create time axis
    time = np.arange(profile_length) * dt
    
    # Clamp filtered profile to non-negative for visualization (realistic constraint)
    reward_filtered_profile_clamped = np.maximum(reward_filtered_profile, 0.0)
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top plot: Raw profile vs Reward filter profile (main comparison)
    axes[0].plot(time, raw_profile, label="Raw Profile (RL observations in violent mode)", 
                 color="#d62728", linewidth=2.5, linestyle="-", alpha=0.9)
    axes[0].plot(time, reward_filtered_profile_clamped, label="Reward Filtered Profile (for reward computation)", 
                 color="#2ca02c", linewidth=2, linestyle="--", alpha=0.9)
    axes[0].plot(time, filtered_profile, label="Generator Filtered Profile (normal mode)", 
                 color="#1f77b4", linewidth=1.5, linestyle=":", alpha=0.7)
    axes[0].set_ylabel("Speed (m/s)", fontsize=12)
    axes[0].set_title("Training Initialization: Profiles Generated During Reset (Violent Mode)", fontsize=14, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].grid(alpha=0.3, linestyle=":")
    axes[0].set_ylim([-1, max(25, raw_profile.max() * 1.1)])
    
    # Bottom plot: Show the difference/smoothing effect
    speed_diff = reward_filtered_profile_clamped - raw_profile
    axes[1].plot(time, speed_diff, label="Reward Filtered - Raw (smoothing effect)", 
                 color="#9467bd", linewidth=2, linestyle="-")
    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Speed Difference (m/s)", fontsize=12)
    axes[1].set_title("Smoothing Effect: How the Reward Filter Smooths the Raw Profile", fontsize=14, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(alpha=0.3, linestyle=":")
    
    # Add text annotation explaining what happens during training
    fig.text(0.5, 0.02, 
             "During training initialization: Generator creates raw_profile (discontinuous). "
             "In violent mode: RL model receives raw_profile in observations, "
             "but reward is computed against reward-filtered (smooth) version of raw_profile.",
             ha="center", fontsize=10, style="italic")
    
    fig.suptitle("Violent Profile Mode: Training Initialization Example", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    
    # Save plot
    output_path = Path("evaluation/results/violent_profile_example.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Example plot saved to {output_path}")
    print(f"\nProfile statistics (as generated during training initialization):")
    print(f"  Raw profile (RL observations): min={raw_profile.min():.2f} m/s, max={raw_profile.max():.2f} m/s, std={raw_profile.std():.2f} m/s")
    print(f"  Generator filtered (normal mode): min={filtered_profile.min():.2f} m/s, max={filtered_profile.max():.2f} m/s, std={filtered_profile.std():.2f} m/s")
    print(f"  Reward filtered (for reward): min={reward_filtered_profile.min():.2f} m/s, max={reward_filtered_profile.max():.2f} m/s, std={reward_filtered_profile.std():.2f} m/s")
    print(f"\nIn violent mode:")
    print(f"  - RL model receives: raw_profile (discontinuous)")
    print(f"  - Reward computed against: reward_filtered_profile (smooth)")


if __name__ == "__main__":
    main()

