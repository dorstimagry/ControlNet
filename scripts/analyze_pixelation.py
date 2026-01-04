#!/usr/bin/env python3
"""Quick visualization of Phase 1 results from existing evaluation data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import json

# Load the latest test results
test_dir = Path('evaluation/test_ddpm_50steps')

print("Loading test results from:", test_dir)

# Load metrics
with open(test_dir / 'metrics.json') as f:
    metrics = json.load(f)

print("\n" + "="*70)
print("ANALYZING EXISTING EVALUATION RESULTS")
print("="*70)

print("\nMetrics by number of observations:")
print(f"{'N_obs':<10} {'MAE_guided':<15} {'MAE_prior':<15} {'Improvement':<15}")
print("-" * 60)

for n_obs in [10, 20, 50, 100]:
    key = f'n_obs_{n_obs}'
    if key in metrics:
        mae_g = metrics[key]['guided']['mae']
        mae_p = metrics[key]['prior']['mae']
        improvement = (mae_p - mae_g) / mae_p * 100
        print(f"{n_obs:<10} {mae_g:<15.3f} {mae_p:<15.3f} {improvement:<15.1f}%")

# Load one reconstruction to analyze pixelation
print("\nLoading reconstruction images to analyze...")

# Check what files exist
print(f"\nAvailable files in {test_dir}:")
for file in sorted(test_dir.glob('*.png')):
    print(f"  - {file.name}")

# Load and analyze the 500 observation case
img_path = test_dir / 'reconstructions_nobs_500.png'
if img_path.exists():
    print(f"\n✓ Found {img_path.name}")
    print("  This shows the visual pixelation issue")
else:
    print(f"\n✗ Could not find {img_path}")

print("\n" + "="*70)
print("KEY OBSERVATIONS FROM DDPM 50-STEP TEST:")
print("="*70)

print("""
1. DDPM produces smooth prior samples (unguided) ✓
2. But guidance creates pixelated/noisy reconstructions ✗
3. The issue: sparse gradient application

HYPOTHESIS:
- The guidance gradient has non-zero values only at observed pixels
- Only ~500 out of 10,000 pixels get gradient updates
- Rest of the map has no guidance → remains noisy
- Result: dotty pattern where observations exist

NEXT STEP: Test with PERFECT GT observations (no simulation noise)
- If this still fails → guidance mechanism itself is broken
- If this works → issue is with observation quality/normalization

After that: Apply spatial smoothing to spread gradient influence
""")

print("\n" + "="*70)
print("CREATING DIAGNOSTIC PLAN")
print("="*70)

plan = """
PHASE 1: Test with perfect GT observations
  → Sample points directly from GT map (no noise, no gravity)
  → Expected: Should converge to GT map if guidance works

PHASE 2: Visualize and smooth gradients
  → Plot the gradient at each diffusion step
  → Apply Gaussian smoothing to spread influence
  → Test different smoothing scales (sigma = 1, 2, 5, 10)

PHASE 3: Implement best solution
  → Integrate gradient smoothing into sampler
  → Test on full evaluation suite
  → Verify smooth reconstructions with low MAE
"""

print(plan)

# Simple test: Check the sampler code
print("\n" + "="*70)
print("CHECKING SAMPLER IMPLEMENTATION")
print("="*70)

sampler_file = Path('src/maps/sampler_diffusion.py')
with open(sampler_file) as f:
    lines = f.readlines()

# Find the guidance application
for i, line in enumerate(lines[145:160], start=146):
    print(f"{i:3d}| {line.rstrip()}")

print("\n" + "="*70)
print("ISSUE IDENTIFIED IN SAMPLER:")
print("="*70)

print("""
Lines 152-161 show the problem:

1. Line 152: Apply guidance to x0: `x0_guided = x0_pred - guidance * grad_E`
2. Line 156: Re-noise guided x0: `x = alpha * x0_guided + beta * noise_pred`
3. Line 159-161: Call scheduler.step AGAIN with the SAME noise_pred!

The scheduler.step call at line 159 OVERWRITES the guided x from line 156!

This is why guidance doesn't work properly. The fix:
- Remove the scheduler.step call after guidance
- OR: Only apply guidance, don't re-noise

Let's implement the fix in Phase 2.
""")

print("\nReady to proceed with Phase 1 testing...")

