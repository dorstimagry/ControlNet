#!/usr/bin/env python3
"""Analyze the guidance debug results."""

import json
from pathlib import Path

results_file = Path('evaluation/guidance_debug/results.json')

with open(results_file) as f:
    results = json.load(f)

print("="*80)
print("GUIDANCE DEBUG RESULTS ANALYSIS")
print("="*80)

print("\n### PHASE 1: Perfect GT Observations ###\n")
print(f"{'Map':<6} {'Guidance':<12} {'MAE':<12} {'MSE':<12} {'Smoothness':<12} {'MAE@Obs':<12}")
print("-" * 80)

for r in results['phase1']:
    print(f"{r['map_idx']:<6} {r['guidance_scale']:<12.1f} {r['mae']:<12.3f} {r['mse']:<12.3f} {r['smoothness']:<12.3f} {r['mae_at_obs']:<12.3f}")

# Average by guidance scale
print("\n### Average across maps ###\n")
print(f"{'Guidance':<12} {'MAE':<12} {'Smoothness':<12}")
print("-" * 40)

guidance_scales = sorted(set(r['guidance_scale'] for r in results['phase1']))
for scale in guidance_scales:
    scale_results = [r for r in results['phase1'] if r['guidance_scale'] == scale]
    avg_mae = sum(r['mae'] for r in scale_results) / len(scale_results)
    avg_smooth = sum(r['smoothness'] for r in scale_results) / len(scale_results)
    print(f"{scale:<12.1f} {avg_mae:<12.3f} {avg_smooth:<12.3f}")

print("\n### PHASE 2: Gradient Smoothing (guidance=1.0) ###\n")
print(f"{'Smoothing σ':<15} {'MAE':<12} {'Smoothness':<12}")
print("-" * 45)

for r in results['phase2']:
    print(f"{r['smoothing_sigma']:<15.1f} {r['mae']:<12.3f} {r['smoothness']:<12.3f}")

print(f"\n### BEST CONFIGURATION ###")
print(f"Smoothing sigma: {results['best_smoothing_sigma']}")

# Analysis
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

prior_mae = [r['mae'] for r in results['phase1'] if r['guidance_scale'] == 0.0]
guided_mae = [r['mae'] for r in results['phase1'] if r['guidance_scale'] == 1.0]

avg_prior = sum(prior_mae) / len(prior_mae)
avg_guided = sum(guided_mae) / len(guided_mae)
improvement = (avg_prior - avg_guided) / avg_prior * 100

print(f"\n1. GUIDANCE EFFECTIVENESS (100 perfect GT observations):")
print(f"   - Prior only (no guidance): MAE = {avg_prior:.3f} m/s²")
print(f"   - With guidance (scale=1.0): MAE = {avg_guided:.3f} m/s²")
print(f"   - Improvement: {improvement:.1f}%")

if improvement > 0:
    print("   ✓ Guidance WORKS! Scheduler bug is fixed.")
else:
    print("   ✗ Guidance still not working properly")

# Smoothness analysis
prior_smooth = [r['smoothness'] for r in results['phase1'] if r['guidance_scale'] == 0.0]
guided_smooth = [r['smoothness'] for r in results['phase1'] if r['guidance_scale'] == 1.0]

avg_prior_smooth = sum(prior_smooth) / len(prior_smooth)
avg_guided_smooth = sum(guided_smooth) / len(guided_smooth)
smooth_increase = (avg_guided_smooth / avg_prior_smooth - 1) * 100

print(f"\n2. PIXELATION/SMOOTHNESS (without gradient smoothing):")
print(f"   - Prior smoothness: {avg_prior_smooth:.3f}")
print(f"   - Guided smoothness: {avg_guided_smooth:.3f}")
print(f"   - Increase: {smooth_increase:.1f}%")

if smooth_increase > 50:
    print("   ✗ Guidance creates significant pixelation")
    print("   → Gradient smoothing is NEEDED")
else:
    print("   ✓ Guidance preserves smoothness reasonably well")

# Gradient smoothing effectiveness
no_smooth = [r for r in results['phase2'] if r['smoothing_sigma'] == 0.0][0]
best = [r for r in results['phase2'] if r['smoothing_sigma'] == results['best_smoothing_sigma']][0]

print(f"\n3. GRADIENT SMOOTHING EFFECT:")
print(f"   - No smoothing: MAE={no_smooth['mae']:.3f}, Smoothness={no_smooth['smoothness']:.3f}")
print(f"   - With σ={best['smoothing_sigma']}: MAE={best['mae']:.3f}, Smoothness={best['smoothness']:.3f}")

mae_change = (best['mae'] / no_smooth['mae'] - 1) * 100
smooth_change = (best['smoothness'] / no_smooth['smoothness'] - 1) * 100

print(f"   - MAE change: {mae_change:+.1f}%")
print(f"   - Smoothness change: {smooth_change:+.1f}%")

if abs(mae_change) < 20 and smooth_change < -10:
    print("   ✓ Gradient smoothing improves smoothness without hurting accuracy!")
elif mae_change > 20:
    print("   ⚠ Gradient smoothing hurts accuracy significantly")
else:
    print("   ~ Gradient smoothing has mixed effects")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print(f"""
Based on the results:

1. **Scheduler Bug Fix**: {"SUCCESSFUL ✓" if improvement > 0 else "NEEDS MORE WORK ✗"}
   
2. **Recommended Configuration**:
   - guidance_scale: 1.0
   - gradient_smoothing_sigma: {results['best_smoothing_sigma']}
   - num_inference_steps: 50 (DDPM)
   
3. **Expected Performance** (with 100 perfect GT observations):
   - MAE: ~{best['mae']:.2f} m/s²
   - Smooth, non-pixelated reconstructions
   
4. **Next Steps**:
   - Update config files with optimal smoothing parameter
   - Run full evaluation with real (noisy) observations
   - Integrate into SAC training pipeline
""")

print("Visualizations saved to:")
print("  - evaluation/guidance_debug/phase1_guidance_scales.png")
print("  - evaluation/guidance_debug/phase2_gradient_smoothing.png")

