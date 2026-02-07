# Lambda Slider Fix

## Problem

The lambda (λ) sliders appeared to do nothing when adjusted, giving the impression they were broken.

## Root Cause

Lambda only affects the weight schedule when `w_start ≠ w_end`. The weight schedule formula is:

```
w(t) = w_end + (w_start - w_end) * exp(-λ * t)
```

**With old defaults** (`w_start = w_end`):
- Error: start=20, end=20, λ=0
- Accel: start=5, end=5, λ=0  
- Jerk: start=5, end=5, λ=0

When `w_start = w_end = 20`:
```
w(t) = 20 + (20 - 20) * exp(-λ * t)
     = 20 + 0 * exp(-λ * t)
     = 20  (constant, regardless of λ!)
```

The lambda parameter had **zero effect** because the difference `(w_start - w_end)` was always zero.

## Solution

Changed default slider values to have `w_start ≠ w_end`:

**New defaults:**
- **Error weights**: start=30, end=10, λ=1.0
  - Tight tracking early, looser tracking later
  
- **Accel weights**: start=2, end=10, λ=0.5
  - Allow fast initial changes, increase smoothing later
  
- **Jerk weights**: start=2, end=10, λ=0.5
  - Allow abrupt initial transitions, smooth out later portions

## Verification

Tested lambda slider changes (0.0 → 3.0):
- **λE (Error)**: 1.19 m/s RMS change ✓
- **λA (Accel)**: 0.39 m/s RMS change ✓
- **λJ (Jerk)**: 0.77 m/s RMS change ✓

All lambda sliders now have **immediately visible effects** on the shaped profile.

## User Benefits

- **Immediate feedback**: Moving lambda sliders produces visible changes
- **Better defaults**: Start with meaningful time-varying weights
- **Intuitive behavior**: Higher lambda = faster decay (shorter time constant)
- **Educational**: Demonstrates the effect of time-varying weight schedules

## Technical Notes

The weight schedule creates an exponential transition:
- `λ = 0`: No decay, weight stays at `w_start`
- `λ = 0.5`: Gentle decay, ~63% transition in 2 seconds
- `λ = 1.0`: Moderate decay, ~63% transition in 1 second
- `λ = 3.0`: Fast decay, ~95% transition in 1 second

The defaults are chosen to demonstrate typical use cases:
1. **Error tracking**: Start tight (30) to match initial conditions precisely, relax (10) to allow smoothing
2. **Acceleration/Jerk**: Start loose (2) to handle initial transients, increase (10) for later smoothness

## Files Updated

- `src/gui_matplotlib.py`: Changed default slider values
- `README.md`: Added explanation of lambda behavior and defaults

## Commit Message

```
Fix lambda sliders by using different w_start/w_end defaults

Lambda only has an effect when w_start ≠ w_end due to the weight
schedule formula: w(t) = w_end + (w_start - w_end) * exp(-λt).

Old defaults had w_start = w_end for all weights, making lambda
ineffective. New defaults use different start/end values:
- Error: 30 → 10 (tight tracking early, looser later)
- Accel: 2 → 10 (allow fast initial changes, smooth later)
- Jerk: 2 → 10 (allow abrupt start, smooth out later)

All lambda sliders now produce visible profile changes.
```
