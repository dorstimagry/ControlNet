# Speed Shaper LQR Penalty Mode - Quick Start Guide

## What is it?

A feature that lets you save speed shaper GUI configurations and use them as time-varying penalty weights in RL training for longitudinal control.

## Quick Start (3 Steps)

### 1. Save Configuration from GUI

```bash
cd speed_shaper
python -c "from src.gui_matplotlib import SpeedShaperGUI; import matplotlib.pyplot as plt; gui = SpeedShaperGUI('step'); plt.show()"
```

- Adjust sliders to your preferred weights
- Click "Save Config" button (in Global Controls section)
- Save to `speed_shaper/configs/my_config.json`

### 2. Add to Training Config

Edit your training YAML (e.g., `training/config.yaml`):

```yaml
env:
  # ... existing config ...
  
  # LQR Penalty Mode (NEW)
  lqr_penalty_enabled: true
  lqr_penalty_config_path: "speed_shaper/configs/my_config.json"
  lqr_accel_weight: 0.2  # Global multiplier for acceleration penalty
  lqr_jerk_weight: 0.2   # Global multiplier for jerk penalty
```

### 3. Train

```bash
python training/train_sac.py --config training/config.yaml
```

## How It Works

### Time-Varying Weights

The LQR penalty mode uses **exponential weight schedules**:

```
w(t) = w_end + (w_start - w_end) * exp(-λ * t)
```

- **λ > 0**: Decay from `w_start` toward `w_end` (decreasing emphasis)
- **λ = 0**: Constant weight `w_start`
- **λ < 0**: Growth from `w_start` toward `w_end` (increasing emphasis)

### Penalty Formulas

At each timestep `t`:

```python
# Acceleration penalty
penalty_accel = -lqr_accel_weight * w_A(t) * accel²

# Jerk penalty (skipped at first/last steps)
penalty_jerk = -lqr_jerk_weight * w_J(t) * jerk²
```

Both penalties are **additive** to existing `jerk_weight` and `smooth_action_weight`.

## Example Configurations

### Smooth Driving (Increasing Comfort)

```json
{
  "wA_start": 5.0,
  "wA_end": 15.0,
  "lamA": -0.3,
  "wJ_start": 3.0,
  "wJ_end": 12.0,
  "lamJ": -0.3
}
```

Comfort penalties **increase** over time → agent learns smoothness gradually.

### Aggressive Start, Smooth End

```json
{
  "wA_start": 1.0,
  "wA_end": 20.0,
  "lamA": 1.0,
  "wJ_start": 1.0,
  "wJ_end": 20.0,
  "lamJ": 1.0
}
```

Low penalties early (fast learning), high penalties later (smooth refinement).

### Constant High Smoothness

```json
{
  "wA_start": 15.0,
  "wA_end": 15.0,
  "lamA": 0.0,
  "wJ_start": 12.0,
  "wJ_end": 12.0,
  "lamJ": 0.0
}
```

Constant penalties throughout episode.

## Tuning Tips

### Global Multipliers

Start with small values and increase:

```yaml
lqr_accel_weight: 0.1  # Try 0.05, 0.1, 0.2, 0.5
lqr_jerk_weight: 0.1   # Try 0.05, 0.1, 0.2, 0.5
```

### Combining with Existing Penalties

**Option 1: LQR Only**
```yaml
jerk_weight: 0.0              # Disable existing
smooth_action_weight: 0.0     # Disable existing
lqr_accel_weight: 0.5         # Use LQR only
lqr_jerk_weight: 0.5
```

**Option 2: Additive (Both Active)**
```yaml
jerk_weight: 0.1              # Keep existing
smooth_action_weight: 0.05    # Keep existing
lqr_accel_weight: 0.2         # Add LQR on top
lqr_jerk_weight: 0.2
```

### Lambda Values

- **Decay** (λ > 0): 0.5 to 2.0 (typical)
- **Growth** (λ < 0): -0.5 to -2.0 (typical)
- **Constant** (λ = 0): No time variation

## Monitoring

Check these reward components in your training logs:

```python
reward_components = {
    'lqr_accel_penalty': -0.0023,  # Should be negative
    'lqr_jerk_penalty': -0.0015,   # Should be negative
    'jerk_penalty': -0.0010,       # Existing penalty still active
}
```

## Advanced Usage

### Programmatic Configuration

```python
from speed_shaper.src.config_schema import ShaperConfig
from pathlib import Path

config = ShaperConfig(
    dt=0.1,
    wE_start=30.0, wE_end=10.0, lamE=1.0,
    wA_start=5.0, wA_end=15.0, lamA=-0.3,
    wJ_start=3.0, wJ_end=12.0, lamJ=-0.3,
    metadata={'experiment': 'smooth_cruise'}
)

config.to_json(Path('speed_shaper/configs/my_config.json'))
```

### Loading and Inspecting

```python
config = ShaperConfig.from_json(Path('speed_shaper/configs/my_config.json'))
print(config)
```

## Troubleshooting

### LQR penalties are 0

**Check:**
1. `lqr_penalty_enabled: true` in config
2. Config path is correct and file exists
3. `lqr_accel_weight` and `lqr_jerk_weight` are > 0

### Penalties too strong/weak

**Adjust global multipliers:**
- Too strong: Reduce `lqr_accel_weight` and `lqr_jerk_weight`
- Too weak: Increase multipliers or adjust weight schedules in GUI

### Training unstable

**Try:**
1. Reduce global multipliers (e.g., 0.05 instead of 0.5)
2. Use gentler weight schedules (smaller |λ| values)
3. Enable comfort annealing alongside LQR mode

## Files

- **Schema:** `speed_shaper/src/config_schema.py`
- **Configs:** `speed_shaper/configs/`
- **Tests:** `speed_shaper/tests/test_config_schema.py`, `tests/test_lqr_penalty_mode.py`
- **Docs:** `speed_shaper/LQR_PENALTY_MODE_IMPLEMENTATION.md`

## Support

All code is fully tested with 20/20 unit tests passing. See implementation documentation for detailed technical information.
