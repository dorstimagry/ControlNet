# Speed Shaper Configuration Files

This directory stores saved speed shaper configurations in JSON format.

## Configuration Format

Each configuration file contains the following parameters:

### Required Fields

- **`dt`**: Time step size in seconds
- **`wE_start`, `wE_end`, `lamE`**: Error weight parameters
  - `wE_start`: Initial error weight
  - `wE_end`: Final error weight  
  - `lamE`: Decay/growth rate (1/seconds)
- **`wA_start`, `wA_end`, `lamA`**: Acceleration weight parameters
- **`wJ_start`, `wJ_end`, `lamJ`**: Jerk weight parameters

### Optional Fields

- **`a_min`, `a_max`**: Acceleration bounds (m/s²), `null` if disabled
- **`j_min`, `j_max`**: Jerk bounds (m/s³), `null` if disabled
- **`enable_accel_bounds`**: Boolean flag for acceleration bounds
- **`enable_jerk_bounds`**: Boolean flag for jerk bounds
- **`enable_terminal_constraint`**: Boolean flag for terminal velocity constraint
- **`metadata`**: Dictionary of additional metadata

## Example Configuration

```json
{
  "dt": 0.1,
  "wE_start": 30.0,
  "wE_end": 10.0,
  "lamE": 1.0,
  "wA_start": 2.0,
  "wA_end": 10.0,
  "lamA": -0.5,
  "wJ_start": 2.0,
  "wJ_end": 10.0,
  "lamJ": -0.5,
  "a_min": null,
  "a_max": null,
  "j_min": null,
  "j_max": null,
  "enable_accel_bounds": false,
  "enable_jerk_bounds": false,
  "enable_terminal_constraint": false,
  "metadata": {
    "created_at": "2026-01-25T12:30:00",
    "description": "Default smooth profile configuration"
  }
}
```

## Usage

### From GUI

1. Adjust sliders to desired configuration
2. Click "Save Config" button
3. Choose filename and location
4. Configuration is saved as JSON

### In RL Training

Reference the configuration file in your training YAML:

```yaml
env:
  lqr_penalty_enabled: true
  lqr_penalty_config_path: "speed_shaper/configs/my_config.json"
  lqr_accel_weight: 0.1
  lqr_jerk_weight: 0.1
```

### Programmatically

```python
from pathlib import Path
from speed_shaper.src.config_schema import ShaperConfig

# Load configuration
config = ShaperConfig.from_json(Path("configs/my_config.json"))
print(config)

# Create and save configuration
config = ShaperConfig(
    dt=0.1,
    wE_start=30.0, wE_end=10.0, lamE=1.0,
    wA_start=2.0, wA_end=10.0, lamA=-0.5,
    wJ_start=2.0, wJ_end=10.0, lamJ=-0.5,
)
config.to_json(Path("configs/my_config.json"))
```
