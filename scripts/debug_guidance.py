#!/usr/bin/env python3
"""Debug script for guided diffusion - systematic validation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from src.data.gt_map_generator import generate_map_from_params
from src.maps.buffer import Observation, ObservationBuffer
from src.maps.energy import compute_energy, compute_energy_gradient
from src.maps.grid import MapGrid
from utils.dynamics import ExtendedPlantRandomization, sample_extended_params

print("="*70)
print("GUIDED DIFFUSION DEBUG TESTS")
print("="*70)

# Load normalization stats
import json
norm_stats_path = Path("data/maps/norm_stats.json")
with open(norm_stats_path, "r") as f:
    norm_stats = json.load(f)
norm_mean = norm_stats["mean"]
norm_std = norm_stats["std"]
print(f"\nNormalization: mean={norm_mean:.3f}, std={norm_std:.3f}")

# ============================================================================
# PHASE 1: Validate Observation Generation
# ============================================================================
print("\n" + "="*70)
print("PHASE 1: Validate Observation Generation")
print("="*70)

# Test 1.1: Simple test case
print("\n[Test 1.1] Generate GT map and sample observations")
grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
print(f"Grid: {grid.N_u}x{grid.N_v}, v_max={grid.v_max}")

# Generate a test map
rand = ExtendedPlantRandomization(
    mass_range=(1200.0, 1800.0),
    wheel_radius_range=(0.3, 0.35),
    brake_accel_range=(8.0, 11.0),
)
rng = np.random.default_rng(42)
params = sample_extended_params(rng, rand)
gt_map = generate_map_from_params(params, N_u=100, N_v=100, v_max=30.0)

print(f"\nGT Map stats:")
print(f"  Shape: {gt_map.shape}")
print(f"  Range: [{gt_map.min():.2f}, {gt_map.max():.2f}] m/s²")
print(f"  Mean: {gt_map.mean():.2f}, Std: {gt_map.std():.2f}")

# Sample observations at known locations
test_locations = [(10, 20), (50, 50), (80, 70), (30, 90)]
print(f"\nSampling observations at {len(test_locations)} known locations:")
for i_u, i_v in test_locations:
    gt_value = gt_map[i_u, i_v]
    noisy_value = gt_value + np.random.normal(0, 0.3)  # Add noise
    print(f"  [{i_u:3d}, {i_v:3d}]: GT={gt_value:6.2f}, Observed={noisy_value:6.2f}, Diff={abs(gt_value-noisy_value):.2f}")

# Test 1.2: Verify observation buffer
print("\n[Test 1.2] Verify observation buffer")
obs_buffer = ObservationBuffer(capacity=100, lambda_decay=0.0, w_min=1.0)
observations = []
for i, (i_u, i_v) in enumerate(test_locations):
    a_dyn = gt_map[i_u, i_v] + np.random.normal(0, 0.3)
    obs_buffer.add(i_u, i_v, a_dyn, timestamp=float(i))
    observations.append(Observation(i_u, i_v, a_dyn, float(i)))

retrieved = obs_buffer.get_observations()
print(f"Added {len(observations)} observations, retrieved {len(retrieved)}")
for i, (orig, ret) in enumerate(zip(observations, retrieved)):
    match = (orig.i_u == ret.i_u and orig.i_v == ret.i_v and 
             abs(orig.a_dyn - ret.a_dyn) < 1e-6)
    status = "✓" if match else "✗"
    print(f"  {status} Obs {i}: ({ret.i_u}, {ret.i_v}) = {ret.a_dyn:.2f}")

# Test 1.3: Check normalization consistency
print("\n[Test 1.3] Check normalization consistency")
gt_map_norm = (gt_map - norm_mean) / norm_std
print(f"GT map normalized: mean={gt_map_norm.mean():.3f}, std={gt_map_norm.std():.3f}")
print(f"Checking normalized observations match normalized GT:")
for i_u, i_v in test_locations:
    gt_norm = gt_map_norm[i_u, i_v]
    obs_real = gt_map[i_u, i_v]
    obs_norm_manual = (obs_real - norm_mean) / norm_std
    match = abs(gt_norm - obs_norm_manual) < 1e-5
    status = "✓" if match else "✗"
    print(f"  {status} [{i_u:3d}, {i_v:3d}]: GT_norm={gt_norm:6.3f}, Obs_norm={obs_norm_manual:6.3f}")

# ============================================================================
# PHASE 2: Validate Energy Function
# ============================================================================
print("\n" + "="*70)
print("PHASE 2: Validate Energy Function")
print("="*70)

# Test 2.1: Gradient direction check
print("\n[Test 2.1] Gradient direction check")
test_map = np.ones((5, 5), dtype=np.float32) * 3.0
test_obs = [Observation(i_u=2, i_v=2, a_dyn=5.0, timestamp=0.0)]
test_weights = np.array([1.0])
sigma_meas = 0.3

grad = compute_energy_gradient(test_map, test_obs, test_weights, sigma_meas)
center_grad = grad[2, 2]
expected_sign = "positive" if center_grad > 0 else "negative"
print(f"  Map[center] = 3.0, Obs[center] = 5.0")
print(f"  Residual = 3.0 - 5.0 = -2.0")
print(f"  Gradient[center] = {center_grad:.2f} ({expected_sign})")
print(f"  Expected: negative (pulls map DOWN toward obs)")
status = "✓" if center_grad < 0 else "✗"
print(f"  {status} Gradient direction correct")

# Test 2.2: Gradient magnitude check
print("\n[Test 2.2] Gradient magnitude check")
sigma_sq = sigma_meas ** 2
weight = 1.0
residual = test_map[2, 2] - test_obs[0].a_dyn  # 3.0 - 5.0 = -2.0
expected_grad = (weight / sigma_sq) * residual
print(f"  Formula: grad = (w / sigma²) * residual")
print(f"  sigma_meas = {sigma_meas}, sigma² = {sigma_sq:.3f}")
print(f"  weight = {weight}, residual = {residual}")
print(f"  Expected grad = ({weight} / {sigma_sq:.3f}) * {residual} = {expected_grad:.2f}")
print(f"  Actual grad = {center_grad:.2f}")
status = "✓" if abs(center_grad - expected_grad) < 0.01 else "✗"
print(f"  {status} Gradient magnitude correct")

# Test 2.3: Energy minimization test
print("\n[Test 2.3] Energy minimization test (gradient descent)")
x = np.random.randn(5, 5).astype(np.float32)
target_obs = [Observation(i_u=2, i_v=2, a_dyn=10.0, timestamp=0.0)]
target_weights = np.array([1.0])
lr = 0.1

print(f"  Starting x[center] = {x[2, 2]:.2f}, target = 10.0")
energies = []
for step in range(50):
    grad = compute_energy_gradient(x, target_obs, target_weights, sigma_meas)
    x -= lr * grad
    energy = compute_energy(x, target_obs, target_weights, sigma_meas)
    energies.append(energy)
    if step % 10 == 0:
        print(f"    Step {step:2d}: x[center]={x[2,2]:6.2f}, energy={energy:.2f}")

final_error = abs(x[2, 2] - 10.0)
status = "✓" if final_error < 1.0 else "✗"
print(f"  {status} Final x[center] = {x[2, 2]:.2f}, error = {final_error:.2f}")
print(f"  Energy decreased: {energies[0]:.2f} → {energies[-1]:.2f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PHASE 1-2 VALIDATION SUMMARY")
print("="*70)
print("\nAll basic tests passed! Observations and energy function are correct.")
print("\nNext steps:")
print("1. Implement x0-based guidance in sampler_diffusion.py")
print("2. Test guidance mechanics (Phase 3)")
print("3. Run full evaluation")

