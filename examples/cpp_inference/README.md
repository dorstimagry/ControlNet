# C++ Inference Example for SAC Policy

This example demonstrates how to run inference on an exported SAC policy ONNX model using ONNX Runtime in C++.

## Prerequisites

1. **ONNX Runtime**: Download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
   
   For Linux x64:
   ```bash
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
   tar -xzf onnxruntime-linux-x64-1.16.3.tgz
   export ONNXRUNTIME_ROOT=$(pwd)/onnxruntime-linux-x64-1.16.3
   ```

2. **CMake** >= 3.14

3. **C++17 compatible compiler**

## Exporting the Model

First, export your trained PyTorch checkpoint to ONNX format:

```bash
cd /path/to/DiffDynamics

# Export the latest checkpoint
python scripts/export_onnx.py \
    --checkpoint training/checkpoints/latest.pt \
    --output training/checkpoints/latest.onnx

# Or export a specific checkpoint
python scripts/export_onnx.py \
    --checkpoint training/checkpoints/sac_step_10000000.pt \
    --output models/policy.onnx
```

This generates:
- `policy.onnx` - The ONNX model
- `policy.json` - Metadata with dimensions and configuration

## Building

```bash
cd examples/cpp_inference

mkdir build && cd build

# Configure with ONNX Runtime path
cmake -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT ..

# Build
cmake --build .
```

## Running

```bash
# Run inference on the exported model
./sac_inference ../../training/checkpoints/latest.onnx

# Or with explicit observation dimension
./sac_inference /path/to/policy.onnx 34
```

## Expected Output

```
[SACPolicy] Loaded model: /path/to/policy.onnx
[SACPolicy] obs_dim: 34, action_dim: 10

[Test] Running single inference...
[Test] Input observation (first 8 elements): 10 9.8 9.6 0.5 15 15 15 15 ...
[Test] Output action: 0.234 0.231 0.228 ...
[Test] Immediate action (throttle/brake): 0.234

[Benchmark] Running 1000 inferences...
[Benchmark] Average inference time: 45 us
[Benchmark] Throughput: 22000 inferences/sec

[Done] Inference test completed successfully!
```

## Integration Guide

### Observation Format

The observation vector has the format:
```
[current_speed, prev_speed, prev_prev_speed, prev_action, ref_speed_0, ref_speed_1, ..., ref_speed_N]
```

Where:
- `current_speed`: Current vehicle speed in m/s
- `prev_speed`: Speed from previous timestep
- `prev_prev_speed`: Speed from two timesteps ago
- `prev_action`: Previous action output
- `ref_speed_*`: Target speed profile for the preview horizon

The number of reference speeds is `obs_dim - 4`. With the default configuration (3s preview at 0.1s dt), this is 30 values.

### Action Output

The output is an action sequence of length `action_dim` (typically 10 with `action_horizon_steps=10`).

- **Values range**: [-1, 1]
- **Interpretation**:
  - Negative values = braking
  - Positive values = throttle
  - 0 = coast

For immediate control, use only the first action (`action[0]`). The remaining actions are the planned future actions.

### Using in Your Project

```cpp
#include "onnxruntime_cxx_api.h"

// Initialize once
SACPolicyInference policy("path/to/policy.onnx");

// Each control loop iteration:
std::vector<float> obs = buildObservation(current_speed, speed_history, target_profile);
std::vector<float> action = policy.infer(obs);
float throttle_brake = action[0];  // Apply this to your vehicle
```

## Metadata JSON

The export script also generates a JSON file with useful configuration:

```json
{
  "obs_dim": 34,
  "action_dim": 10,
  "env_action_dim": 1,
  "horizon": 10,
  "action_low": -1.0,
  "action_high": 1.0,
  "dt": 0.1,
  "preview_horizon_s": 3.0,
  "action_scale": [...],
  "action_bias": [...]
}
```

You can parse this JSON in C++ to dynamically configure your observation builder.

## Troubleshooting

### ONNX Runtime Not Found

Ensure `ONNXRUNTIME_ROOT` points to the extracted ONNX Runtime directory:
```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.16.3
cmake -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT ..
```

### Model Load Failure

- Verify the ONNX file exists and is not corrupted
- Check that the model was exported with a compatible opset version (default: 17)
- Try re-exporting with `--opset 14` for older ONNX Runtime versions

### Dimension Mismatch

If you see "Observation size mismatch", check the `obs_dim` in the metadata JSON and ensure your observation vector has the correct size.

