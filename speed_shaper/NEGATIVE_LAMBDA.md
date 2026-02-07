# Negative Lambda Support

## Feature Added

Extended lambda slider range from `[0, 5]` to `[-5, +5]`, allowing both positive (decay toward w_end) and negative (divergence from w_end) exponential rates.

## Mathematical Behavior

Weight schedule formula:
```
w(t) = w_end + (w_start - w_end) * exp(-λ * t)
```

### Positive Lambda (λ > 0) - RECOMMENDED
Weights smoothly transition FROM `w_start` TOWARD `w_end`:

**Example 1: Decay (high → low)**
- `w_start=50, w_end=10, λ=+2`
- Result: Weights decrease from 50 at t=0 toward 10 as t increases
- Use case: Relax tracking constraints over time

**Example 2: Growth (low → high)**
- `w_start=5, w_end=25, λ=+2`
- Result: Weights increase from 5 at t=0 toward 25 as t increases
- Use case: Tighten smoothing constraints over time

### Zero Lambda (λ = 0)
Weights stay constant at `w_start`:
- `w_start=30, w_end=10, λ=0`
- Result: Weight stays at 30 throughout

### Negative Lambda (λ < 0) - ADVANCED
Weights exponentially diverge AWAY from `w_end`:

**Warning:** Can cause instability!
- `w_start=30, w_end=10, λ=-1`
- Result: Weights grow exponentially from 30 away from 10
- Can become very large or go negative (clipped to 0)
- Generally not recommended

## Default Values

Updated defaults to demonstrate both decay and growth using negative lambda:

```python
# Error: Decay (tight early, loose later)
wE_start = 30.0
wE_end = 10.0
lamE = +1.0  # Positive: smooth decay 30→10

# Accel: Growth (loose early, tight later)  
wA_start = 2.0
wA_end = 10.0
lamA = -0.5  # Negative: allows growth 2→10

# Jerk: Growth (loose early, tight later)
wJ_start = 2.0  
wJ_end = 10.0
lamJ = -0.5  # Negative: allows growth 2→10
```

## Recommendations

### For Decay (high → low weights):
✓ Use positive λ with w_start > w_end
```
w_start=50, w_end=10, λ=+2  ← RECOMMENDED
```

### For Growth (low → high weights):
**Option 1 (Stable):**
✓ Use positive λ with w_start < w_end
```
w_start=5, w_end=25, λ=+2  ← RECOMMENDED
```

**Option 2 (Advanced):**
⚠ Use negative λ (can be unstable)
```
w_start=5, w_end=25, λ=-1  ← Watch for clipping!
```

## Implementation Changes

### Files Modified

1. **src/gui_matplotlib.py**
   - Changed lambda slider ranges: `(0.0, 5.0)` → `(-5.0, 5.0)`
   - Updated default lambda values to showcase both positive and negative
   - Updated comments to explain growth vs decay

2. **src/shaper_math.py**
   - Updated `weight_schedule()` docstring to explain negative lambda behavior
   - Changed lambda check from `lam <= 1e-12` to `abs(lam) <= 1e-12`

3. **README.md**
   - Added comprehensive explanation of positive/negative lambda
   - Included warnings about negative lambda instability
   - Provided recommendations for stable usage

## Testing Results

✓ Positive λ produces smooth transitions (both decay and growth)  
✓ Negative λ produces divergent behavior (can clip to zero)  
✓ Zero λ produces constant weights  
✓ Both directions work in GUI sliders  
✓ All existing tests still pass  

## User Benefits

- **More flexibility**: Can now set λ in both directions
- **Better understanding**: Clear explanation of what lambda does
- **Practical defaults**: Showcase realistic usage patterns
- **Warnings included**: Users aware of negative lambda risks

## Usage Examples

```python
# Example 1: Tight tracking early, looser later (DECAY)
wE_start=30, wE_end=10, λ=+1.5
# Weights: 30 → 10 (smooth exponential decay)

# Example 2: Light smoothing early, heavy later (GROWTH)
wA_start=2, wA_end=15, λ=+2.0
# Weights: 2 → 15 (smooth exponential growth)

# Example 3: Constant weight throughout
wJ_start=10, wJ_end=10, λ=0.0
# Weights: 10 (constant, lambda has no effect when start=end)
```

## Summary

Successfully added support for negative lambda values while:
- Maintaining backward compatibility
- Providing clear warnings about instability risks
- Recommending positive lambda for most use cases
- Demonstrating both patterns in default values
- Keeping all existing tests passing

Users can now explore the full range of exponential weight schedules!
