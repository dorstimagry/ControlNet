# ControlNet: SAC for Autonomous Vehicle Longitudinal Control

This repository implements a Soft Actor-Critic (SAC) reinforcement learning agent for longitudinal control of autonomous vehicles. The system learns to track reference speed profiles while optimizing for safety, comfort, and efficiency.

## 📁 Project Structure

```
DiffDynamics/
├── data/                          # Data directory
│   ├── raw/                       # Raw data (not included in repo)
│   │   └── trips/                 # Real vehicle trip data (CSV/JSON)
│   └── processed/                 # Processed datasets
│       ├── NiroEV/                # Processed real vehicle data
│       └── synthetic/             # Generated synthetic datasets
├── env/                           # Environment implementation
│   ├── longitudinal_env.py        # Gym-compatible longitudinal control env
│   └── __init__.py
├── evaluation/                    # Evaluation and visualization
│   ├── eval_closed_loop.py        # Closed-loop evaluation script
│   ├── eval_offline.py            # Offline evaluation script
│   ├── plotting.py                # Plotting utilities
│   ├── policy_loader.py           # Policy loading utilities
│   ├── pid_controller.py          # PID controller for closed-loop evaluation
│   ├── cpp_onnx_policy.py         # Python wrapper for C++ ONNX inference
│   ├── cpp_inference/              # C++ ONNX inference module (PyBind11)
│   │   ├── onnx_policy.h/cpp      # C++ ONNX Runtime wrapper
│   │   ├── pybind_module.cpp      # PyBind11 bindings
│   │   └── CMakeLists.txt         # Build configuration
│   └── results/                   # Evaluation outputs (ignored)
├── fitting/                       # Vehicle parameter fitting
│   ├── vehicle_fitter.py          # Parameter fitting from trip data
│   ├── randomization_config.py    # Centered randomization around fitted params
│   └── __init__.py
├── scripts/                       # Utility scripts
│   ├── generate_synthetic_dataset.py    # Generate synthetic training data
│   ├── analyze_dataset_coverage.py      # Dataset analysis tools
│   ├── build_segment_metadata.py        # Metadata processing
│   ├── fetch_trips.py                   # Data fetching utilities
│   ├── parse_trips.py                   # Trip parsing tools
│   ├── fit_vehicle_params.py            # Fit vehicle params from real data
│   ├── export_onnx.py                   # Export trained policy to ONNX
│   ├── eval_sysid.py                    # Evaluate SysID quality (NEW!)
│   └── tune_objective_weights.py        # Hyperparameter tuning
├── src/                          # Additional source code
│   ├── sysid/                     # Vehicle dynamics encoding (NEW!)
│   │   ├── encoder.py             # GRU context encoder
│   │   ├── predictor.py           # Dynamics predictor MLP
│   │   ├── normalization.py       # Running normalization
│   │   ├── sysid_trainer.py       # Multi-step rollout training
│   │   ├── dataset.py             # Sequence sampling utilities
│   │   ├── integration.py         # SAC integration helpers
│   │   └── __init__.py
│   └── data/                     # Data processing modules
├── tests/                        # Unit tests
│   ├── sysid/                     # SysID tests (NEW!)
│   │   ├── test_encoder.py        # Encoder tests
│   │   ├── test_predictor.py      # Predictor tests
│   │   ├── test_normalization.py  # Normalization tests
│   │   ├── test_dataset.py        # Sequence sampling tests
│   │   └── test_actor_critic_conditioning.py  # z conditioning tests
│   ├── test_env.py               # Environment tests
│   ├── test_dynamics.py          # Dynamics model tests
│   ├── test_evaluation.py        # Evaluation tests
│   ├── test_training.py          # Training tests
│   ├── test_data.py              # Data processing tests
│   ├── test_fitting.py           # Parameter fitting tests
│   └── test_pid_controller.py    # PID controller tests
├── training/                     # Training code and configuration
│   ├── train_sac.py              # Main training script (updated for SysID)
│   ├── config.yaml               # Training configuration
│   ├── config_sysid.yaml         # SysID-enabled configuration (NEW!)
│   └── checkpoints/              # Model checkpoints (ignored)
├── utils/                        # Core utilities
│   ├── dynamics.py               # Vehicle dynamics models (ExtendedPlant)
│   ├── data_utils.py             # Data processing utilities
│   └── __init__.py
├── examples/                     # Example code
│   └── cpp_inference/            # C++ ONNX inference example
├── .gitignore                    # Git ignore rules
├── tuning_results.yaml           # Hyperparameter tuning results
└── README.md                     # This file
```

## 🚗 System Overview

### 🔬 Vehicle-Conditioned SAC with Online Dynamics Encoding (NEW!)

The system now supports **vehicle-conditioned SAC** with online system identification (SysID). This enables the agent to:
- **Adapt to different vehicle dynamics** in real-time
- **Generalize better** to unseen vehicles
- **Learn interpretable vehicle-specific latents** z_t

#### Architecture

```
Environment → Encoder (GRU) → z_t → Actor/Critic → Action
     ↓                          ↓
Speed/Action History    SysID Predictor (Multi-step Rollout)
```

**Key Components:**
1. **Context Encoder**: GRU that processes speed and action history to produce dynamics latent z_t
2. **Dynamics Predictor**: MLP that predicts speed changes from (v, u, z)
3. **Multi-step Rollout Training**: Self-supervised objective trains encoder to capture vehicle-specific dynamics
4. **SAC Conditioning**: Actor and critic receive augmented observations [obs, z_t]

**How it Works:**
1. During environment interaction, the encoder maintains a hidden state h_t
2. At each step, features o_t = [v_t, u_{t-1}, dv_t, du_{t-1}] are normalized and fed to the GRU
3. The GRU outputs z_t, a low-dimensional latent capturing vehicle dynamics
4. z_t is concatenated with the observation and fed to SAC actor/critic
5. The encoder is trained with a separate multi-step rollout loss:
   - **Burn-in**: Compute z_t from recent history
   - **Rollout**: Predict speeds H steps into the future using true actions
   - **Loss**: MSE between predicted and actual speeds + regularization

**Benefits:**
- **Better generalization** to vehicles with different mass, drag, motor characteristics
- **Faster adaptation** during evaluation (burn-in period)
- **Interpretable**: z captures vehicle-specific properties
- **Stop-gradient**: Encoder trained only by SysID loss (stable training)

### Vehicle Model: ExtendedPlant

The system uses a detailed **DC Motor + Nonlinear Brake** vehicle dynamics model with 18 fitted parameters:

#### Core Vehicle Parameters
- **mass**: Vehicle mass (kg)
- **drag_area**: Drag coefficient × frontal area (m²)
- **rolling_coeff**: Rolling resistance coefficient
- **wheel_radius**: Wheel radius (m)
- **wheel_inertia**: Wheel rotational inertia (kg·m²)

#### Motor Parameters (DC Motor Model)
- **motor_V_max**: Maximum motor voltage (V)
- **motor_R**: Armature resistance (Ω)
- **motor_L**: Armature inductance (H)
- **motor_K**: Torque constant = back-EMF constant (Nm/A, V·s/rad)
- **motor_b**: Viscous friction coefficient (Nm·s/rad)
- **motor_J**: Rotor inertia (kg·m²)
- **gear_ratio**: Gear reduction ratio
- **eta_gb**: Gearbox efficiency

#### Brake Parameters
- **brake_T_max**: Maximum brake torque at wheel (Nm)
- **brake_tau**: Brake time constant (s)
- **brake_p**: Brake exponent (nonlinearity)
- **brake_kappa**: Brake slip constant
- **mu**: Tire-road friction coefficient

The model includes:
- **DC motor dynamics**: First-order electrical dynamics with back-EMF
- **Nonlinear braking**: Slip-dependent brake torque with time constant
- **Wheel dynamics**: Rotational inertia and tire forces
- **Aerodynamic drag**: Speed-squared drag force
- **Rolling resistance**: Speed-dependent rolling friction
- **Road grade**: Gravitational force component

### Control Architecture
- **Algorithm**: Soft Actor-Critic (SAC) with automatic entropy tuning
- **Action Space**: [-1, 1] (throttle/brake command)
- **Observation Space**: 34 elements
  - Current speed, previous speeds (2), previous action
  - 30 future reference speeds (3-second preview at 10Hz)
- **Horizon**: 3 seconds preview (30 timesteps at 10Hz)
- **Reward**: Speed tracking + jerk penalty + action regularization + voltage constraints

### Key Features
- **Domain Randomization**: Vehicle parameters randomized per episode
- **Parameter Fitting**: Fit all 18 parameters from real-world trip data
- **Centered Randomization**: Train on distributions centered around fitted parameters
- **PID Controller**: Optional PID controller for closed-loop evaluation
- **C++ ONNX Inference**: Deploy trained policies in C++ via ONNX Runtime
- **Acceleration Filtering**: Smoothed acceleration for jerk computation
- **Preview-Based Control**: Agent sees future reference trajectory

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Python 3.10+
python --version

# Required packages
pip install torch gymnasium numpy matplotlib scipy pyyaml tqdm

# Optional: For ONNX export and C++ inference
pip install onnx onnxruntime
```

### C++ ONNX Inference Setup (Optional)

To use C++ ONNX inference, you need to build the PyBind11 module:

```bash
cd evaluation/cpp_inference
mkdir -p build && cd build
cmake ..
make
```

This creates `_cpp_onnx_policy.so` (or `.pyd` on Windows) that can be imported in Python.

**Dependencies:**
- CMake >= 3.12
- C++17 compiler
- ONNX Runtime (included in `onnxruntime-linux-x64-1.16.3/`)
- PyBind11 (install via `pip install pybind11`)

### Environment Setup
```bash
# Clone and enter repository
cd DiffDynamics

# Install in development mode
pip install -e .
```

## 🔧 Vehicle Parameter Fitting

Fit all 18 vehicle parameters from real-world trip data to train RL models on distributions close to specific vehicles.

### Basic Usage

```bash
# Fit parameters from trip data
python scripts/fit_vehicle_params.py \
    --data data/processed/NiroEV/NIROEV_HA_02/all_trips_data.pt \
    --output fitted_params.json
```

### Advanced Options

```bash
# Custom optimization settings
python scripts/fit_vehicle_params.py \
    --data data/processed/NiroEV/NIROEV_HA_02/all_trips_data.pt \
    --epochs 5 \
    --batch-size 20 \
    --max-iter 50 \
    --val-fraction 0.2

# Fix specific parameters (use same min/max bounds)
python scripts/fit_vehicle_params.py \
    --data data/processed/NiroEV/NIROEV_HA_02/all_trips_data.pt \
    --mass-bounds 1885 1885  # Fix mass to 1885 kg

# Data filtering options
python scripts/fit_vehicle_params.py \
    --data data/processed/NiroEV/NIROEV_HA_02/all_trips_data.pt \
    --min-speed 0.5 \
    --max-speed 30.0 \
    --max-accel 5.0 \
    --downsample 2  # Use every 2nd sample

# Warmup: try random parameter sets before optimization
python scripts/fit_vehicle_params.py \
    --data data/processed/NiroEV/NIROEV_HA_02/all_trips_data.pt \
    --warmup \
    --warmup-samples 10
```

### Fitting Process

1. **Data Loading**: Loads trip data from `.pt` files (speed, throttle, brake, time)
2. **Segment Creation**: Splits trips into contiguous segments
3. **Filtering**: Filters by speed, acceleration, segment length, zero-speed fraction
4. **Downsampling**: Optional downsampling to reduce computation
5. **Train/Validation Split**: Holds out validation set for early stopping
6. **Warmup** (optional): Evaluates random parameter sets, picks best as initial guess
7. **Optimization**: L-BFGS-B optimization minimizing velocity MSE over trajectories
8. **Checkpointing**: Saves best parameters (by validation loss) to JSON

### Output

The fitting process generates:
- `fitted_params.json`: Best-fit parameters (all 18 parameters)
- Console output: Progress bars, RMSE metrics, validation plots
- Checkpoint file: Saved when validation loss improves

## 🎯 Training with Fitted Parameters

Train RL models on parameter distributions centered around fitted values.

### Using Fitted Parameters

1. **Fit parameters** (see above)
2. **Create training config** with fitted params:

```yaml
# training/config_niro_fitted_10pct.yaml
vehicle_randomization:
  fitted_params_path: "data/processed/NiroEV/NIROEV_HA_02/fitted_params.json"
  spread_pct: 0.1  # ±10% spread around fitted mean
```

3. **Train with config**:

```bash
python training/train_sac.py \
    --config training/config_niro_fitted_10pct.yaml \
    --num-train-timesteps 1000000
```

### Centered Randomization

The system creates parameter ranges centered on fitted values:
- **Uniform parameters**: `[mean * (1 - spread), mean * (1 + spread)]`
- **Log-uniform parameters**: Geometric mean = fitted value, log spread
- **Positivity constraints**: Only enforces positivity (no hard bounds)
- **Feasibility checks**: Skipped when using fitted parameters

This enables training on distributions very close to the target vehicle while maintaining some diversity for robustness.

## 📊 Data Generation

### Synthetic Dataset Creation
Generate synthetic training datasets with configurable size and characteristics:

```bash
# Generate medium dataset (32 vehicles, 1024 timesteps each)
python scripts/generate_synthetic_dataset.py \
    --output data/processed/synthetic/synth_medium.pt \
    --num-segments 32 \
    --segment-length 1024 \
    --seed 42

# Generate large dataset (200 vehicles, 4096 timesteps each)
python scripts/generate_synthetic_dataset.py \
    --output data/processed/synthetic/synth_large.pt \
    --num-segments 200 \
    --segment-length 4096 \
    --seed 123
```

### Dataset Analysis
Analyze dataset coverage and characteristics:

```bash
python scripts/analyze_dataset_coverage.py
```

## 🎯 Training

### Two-Stage Training (Recommended): Pretrain SysID → Train SAC

For better results, pretrain the SysID encoder before training SAC:

#### Stage 1: Pretrain SysID

```bash
# Pretrain SysID encoder with on-the-fly episode generation (50k steps, ~15-20 min on GPU)
python scripts/pretrain_sysid.py \
    --config training/config_sysid.yaml \
    --num-steps 50000 \
    --min-buffer-size 5000 \
    --output-dir training/sysid_pretrained
```

This phase:
- Generates episodes on-the-fly during training (no long upfront collection)
- Continuously samples new vehicles from the randomization distribution
- Trains encoder/predictor on multi-step rollout prediction
- Saves best checkpoint to `training/sysid_pretrained/sysid_best.pt`

**Key arguments:**
- `--min-buffer-size`: Initial buffer size before training starts (default: 10k, use 5k for quick start)
- `--num-steps`: Number of training steps (not environment steps)
- Data collection happens continuously during training

#### Stage 2: Train SAC with Pretrained SysID

```bash
# Create config with pretrained SysID
# (Set pretrained_path and freeze_encoder in your config)

python training/train_sac.py \
    --config training/config_sac_pretrained.yaml \
    --num-train-timesteps 1000000
```

**Config for pretrained SysID:**
```yaml
sysid:
  enabled: true
  pretrained_path: "training/sysid_pretrained/sysid_best.pt"
  freeze_encoder: true    # Freeze (recommended) or false to fine-tune
  # ... rest of sysid config ...
```

**Benefits of Two-Stage Training:**
- ✅ Better SysID representation (no RL interference)
- ✅ Faster SAC convergence (starts with good features)
- ✅ Modular development (debug SysID separately)
- ✅ Transfer learning (reuse encoder across tasks)

See `docs/two_stage_training.md` for complete guide.

### Training with Vehicle-Conditioned SAC (Joint Training)

Train SAC with online dynamics encoding for better generalization:

```bash
# Train with SysID enabled
python training/train_sac.py --config training/config_sysid.yaml

# Or enable SysID in your config file:
# sysid:
#   enabled: true
#   dz: 12                    # Latent dimension
#   burn_in: 20              # Burn-in steps
#   horizon: 40              # Rollout horizon
```

**Configuration Options** (`config_sysid.yaml`):

```yaml
sysid:
  enabled: true               # Enable vehicle-conditioned SAC
  dz: 12                      # Dynamics latent dimension
  gru_hidden: 64              # GRU hidden size
  predictor_hidden: 128       # Predictor MLP hidden size
  burn_in: 20                 # Burn-in window (steps)
  horizon: 40                 # Rollout horizon (steps)
  lambda_slow: 0.005          # Slow latent regularization
  lambda_z: 0.0005            # L2 latent regularization
  learning_rate: 0.001        # SysID optimizer learning rate
  update_every: 1             # Update frequency
  updates_per_step: 1         # Updates per trigger
```

**Training Process:**
1. Environment step → Compute z_t from encoder
2. Augment observation with z_t
3. Sample action from policy([obs, z_t])
4. Store transition with z_t, z_{t+1}, speed
5. Update SAC (detached z)
6. Update SysID (multi-step rollout loss)

**Monitoring:**
- `sysid/pred_loss`: Multi-step prediction loss
- `sysid/slow_loss`: Hidden state smoothness
- `sysid/z_norm`: Latent magnitude
- `train/policy_loss`, `train/q_loss`: Standard SAC losses

### Basic Training (without SysID)
Train SAC agent with default configuration:

```bash
python training/train_sac.py
```

### Custom Configuration
Modify training parameters in `training/config.yaml`:

```yaml
env:
  max_episode_steps: 2048      # Episode length (204.8 seconds)
  horizon_penalty_weight: 0.1  # Preview-based penalty weight
  accel_filter_alpha: 0.2      # Acceleration smoothing factor

training:
  num_train_timesteps: 1000000 # Total training steps
  batch_size: 256             # Replay buffer batch size
  learning_rate: 0.0003       # SAC learning rate
  eval_interval: 5000         # Evaluation frequency
```

### Advanced Training Options

```bash
# Train with custom checkpoint saving
python training/train_sac.py --num-train-timesteps 500000

# Resume training from checkpoint
# (Modify config.yaml to point to existing checkpoint)

# Custom output directory
python training/train_sac.py --output-dir training/custom_run

# Use fitted parameters
python training/train_sac.py \
    --config training/config_niro_fitted_10pct.yaml
```

### Hyperparameter Tuning
Use the tuning script to optimize reward weights:

```bash
python scripts/tune_objective_weights.py
```

Results are saved to `tuning_results.yaml`.

## 📈 Evaluation

### SysID Quality Evaluation

Evaluate the quality of vehicle dynamics encoding:

```bash
# Evaluate SysID prediction accuracy
python scripts/eval_sysid.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --num-episodes 20 \
    --horizons 10 20 40 \
    --output evaluation/results/sysid_metrics.json
```

**Metrics:**
- `sysid_eval/rmse_h{H}`: RMSE for H-step rollout prediction
- `sysid_eval/rmse_h{H}_z0`: RMSE with z=0 (ablation)
- `sysid_eval/improvement_h{H}`: Improvement percentage vs z=0

**Interpretation:**
- Lower RMSE → Better dynamics prediction
- Higher improvement → z is capturing useful vehicle-specific information
- Stable across horizons → Good long-term modeling

### Closed-Loop Evaluation

Evaluate trained policy on reference tracking:

```bash
# Basic evaluation
python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --episodes 10 \
    --plot-dir evaluation/results/plots/test_run

# Custom evaluation parameters
python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints/sac_step_500000.pt \
    --episodes 20 \
    --output evaluation/results/custom_eval.json \
    --config training/config.yaml
```

### PID Controller Integration

Add a PID controller to assist the RL policy:

```bash
# With PID controller (default: feeds only RL action back to network)
python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --episodes 10 \
    --pid-kp 0.1 \
    --pid-ki 0.01 \
    --pid-kd 0.05 \
    --pid-integral-min -0.1 \
    --pid-integral-max 0.1

# Feed combined RL+PID action back to network
python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --episodes 10 \
    --pid-ki 0.01 \
    --pid-feedback-combined
```

**PID Controller Behavior:**
- **Default**: PID action is treated as "part of the plant" - network only sees its own action
- **Combined mode**: Network sees the combined (RL + PID) action
- **Output**: Final action = `clip(RL_action + PID_action, -1, 1)`
- **Plots**: Show final action, RL contribution, and PID contribution separately

### C++ ONNX Inference Comparison

Compare Python PyTorch inference with C++ ONNX Runtime:

```bash
# Export model to ONNX first
python scripts/export_onnx.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --output training/checkpoints/sac_step_1000000.onnx

# Run evaluation with C++ comparison
python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --onnx-model training/checkpoints/sac_step_1000000.onnx \
    --use-cpp-inference \
    --episodes 10
```

The evaluation reports:
- Maximum difference between Python and C++ outputs
- Mean difference
- Number of mismatches (if any)

### Offline Evaluation
Evaluate policy against recorded trajectories:

```bash
python evaluation/eval_offline.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --dataset data/processed/synthetic/synth_test.pt
```

### Evaluation Metrics
- **Speed Tracking Error**: RMS error between actual and reference speed
- **Acceleration Smoothness**: Variance in acceleration changes
- **Control Effort**: Total throttle/brake activity
- **Constraint Satisfaction**: Motor voltage limits, speed bounds
- **C++ Comparison**: Max/mean differences between Python and C++ inference

## 🔧 Configuration Details

### Environment Configuration (`training/config.yaml`)

#### Core Settings
```yaml
env:
  dt: 0.1                          # Simulation timestep (10Hz)
  max_episode_steps: 2048          # Episode length
  max_speed: 25.0                  # Maximum vehicle speed (m/s)
  max_position: 5000.0             # Maximum travel distance (m)
  preview_horizon_s: 3.0           # Reference preview horizon (s)
  use_extended_plant: true         # Use ExtendedPlant model (18 params)
  plant_substeps: 5                # Integration substeps per policy step
```

#### Reward Weights
```yaml
env:
  track_weight: 1.0                # Speed tracking penalty weight
  jerk_weight: 0.1                 # Acceleration jerk penalty
  action_weight: 0.01              # Control effort penalty
  smooth_action_weight: 0.1        # Action smoothness penalty
  horizon_penalty_weight: 0.1      # Future preview penalty
  voltage_weight: 0.0              # Motor voltage constraint
```

#### Vehicle Randomization
```yaml
vehicle_randomization:
  # Option 1: Standard randomization (ranges)
  mass_range: [1500.0, 6000.0]
  motor_Vmax_range: [200.0, 800.0]
  # ... (see ExtendedPlantRandomization for all parameters)

  # Option 2: Fitted parameters mode (centered randomization)
  fitted_params_path: "data/processed/NiroEV/NIROEV_HA_02/fitted_params.json"
  spread_pct: 0.1  # ±10% spread around fitted mean
```

### Training Configuration
```yaml
training:
  batch_size: 256                  # Replay buffer batch size
  replay_size: 1000000             # Maximum replay buffer size
  warmup_steps: 10000              # Initial random exploration steps
  learning_rate: 0.0003            # SAC optimizer learning rate
  tau: 0.005                       # Target network update rate
  gamma: 0.99                      # Discount factor
  target_entropy_scale: 0.98       # Entropy temperature tuning
```

## 🚀 Model Export to ONNX

Export trained policies to ONNX format for deployment:

```bash
# Basic export
python scripts/export_onnx.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --output policy.onnx

# Custom opset version
python scripts/export_onnx.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --output policy.onnx \
    --opset 17

# Skip validation
python scripts/export_onnx.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --output policy.onnx \
    --no-validate
```

**Output:**
- `policy.onnx`: ONNX model file
- `policy.json`: Metadata (obs_dim, action_dim, action_scale, etc.)

The exported model uses deterministic policy (mean action, no stochastic sampling).

## 🧪 Testing

### Run All Tests
```bash
# Run complete test suite
pytest tests/

# Run specific test categories
pytest tests/test_env.py          # Environment tests
pytest tests/test_dynamics.py     # Dynamics model tests
pytest tests/test_training.py     # Training pipeline tests
pytest tests/test_evaluation.py   # Evaluation tests
pytest tests/test_fitting.py      # Parameter fitting tests
pytest tests/test_pid_controller.py  # PID controller tests
```

### Test Coverage
- **Environment**: Reset/step functionality, observation spaces, termination
- **Dynamics**: Vehicle models, parameter sampling, acceleration capabilities
- **Training**: SAC implementation, replay buffer, policy updates
- **Evaluation**: Metric calculation, plotting, policy loading
- **Fitting**: Parameter optimization, trajectory simulation
- **PID Controller**: Proportional, integral, derivative terms, windup prevention

## 📋 Dependencies

### Core Dependencies
- `torch>=2.0.0`: PyTorch for neural networks
- `gymnasium`: Reinforcement learning environment interface
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `scipy`: Scientific computing (optimization, stats, signal processing)
- `pyyaml`: Configuration file parsing
- `tqdm`: Progress bars

### Optional Dependencies
- `wandb`: Experiment tracking
- `pytest`: Testing framework
- `onnx`: ONNX model format
- `onnxruntime`: ONNX Runtime for inference
- `pybind11`: C++ Python bindings (for C++ inference)
- `accelerate`: Multi-GPU training

## 🎯 Usage Examples

### Quick Start with SysID

```bash
# 1. Train SAC with vehicle-conditioned dynamics encoding
python training/train_sac.py \
    --config training/config_sysid.yaml \
    --num-train-timesteps 1000000

# 2. Evaluate SysID prediction quality
python scripts/eval_sysid.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --num-episodes 20 \
    --horizons 10 20 40 \
    --output evaluation/results/sysid_metrics.json

# 3. Evaluate control performance (closed-loop)
python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --episodes 50 \
    --plot-dir evaluation/results/plots/sysid_eval
```

### Baseline Comparison

```bash
# Train baseline SAC (no SysID)
python training/train_sac.py \
    --config training/config.yaml \
    --num-train-timesteps 1000000 \
    --output-dir training/checkpoints_baseline

# Train SAC + SysID
python training/train_sac.py \
    --config training/config_sysid.yaml \
    --num-train-timesteps 1000000 \
    --output-dir training/checkpoints_sysid

# Compare on held-out vehicles
python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints_baseline/sac_step_1000000.pt \
    --episodes 50 \
    --output evaluation/results/baseline.json

python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints_sysid/sac_step_1000000.pt \
    --episodes 50 \
    --output evaluation/results/sysid.json
```

### Quick Start
```bash
# 1. Generate small dataset
python scripts/generate_synthetic_dataset.py \
    --output data/processed/synthetic/quick_test.pt \
    --num-segments 10 --segment-length 512

# 2. Train for 100k steps
python training/train_sac.py --num-train-timesteps 100000

# 3. Evaluate performance
python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints/sac_step_100000.pt \
    --episodes 5 --plot-dir evaluation/results/quick_eval
```

### Production Training with Fitted Parameters
```bash
# 1. Fit parameters from real vehicle data
python scripts/fit_vehicle_params.py \
    --data data/processed/NiroEV/NIROEV_HA_02/all_trips_data.pt \
    --epochs 5 --batch-size 20

# 2. Create config with fitted parameters
# (Edit training/config_niro_fitted_10pct.yaml)

# 3. Train with fitted parameters (10% spread)
python training/train_sac.py \
    --config training/config_niro_fitted_10pct.yaml \
    --num-train-timesteps 2000000

# 4. Export to ONNX
python scripts/export_onnx.py \
    --checkpoint training/checkpoints/sac_step_2000000.pt \
    --output training/checkpoints/sac_step_2000000.onnx

# 5. Comprehensive evaluation with PID
python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints/sac_step_2000000.pt \
    --onnx-model training/checkpoints/sac_step_2000000.onnx \
    --use-cpp-inference \
    --episodes 50 \
    --pid-ki 0.01 \
    --plot-dir evaluation/results/production_eval
```

## 🔬 Research Features

### Vehicle-Conditioned SAC (SysID)

**Multi-Step Rollout Training:**
- Encoder learns from **self-supervised prediction** objective
- Predicts speeds H steps into future using true actions
- Forces z to capture long-term dynamics (not just single-step noise)

**Stop-Gradient Design:**
- z is **detached** when used by SAC (no gradient flow from RL to encoder)
- Encoder trained **only** by SysID loss
- Prevents representation collapse and training instability
- Allows independent tuning of RL and SysID objectives

**Ablation Studies:**
1. **Baseline SAC** (z_dim=0): Standard SAC without dynamics encoding
2. **SAC + z** (default): Full vehicle-conditioned SAC
3. **z=0 at test time**: Evaluate with zero latent (measures z importance)
4. **Different horizons**: H ∈ {10, 20, 40, 60} (longer = more long-term)
5. **Different burn-in**: B ∈ {10, 20, 30} (longer = more history)

**Recommended Experiments:**
- Train baseline SAC and SAC+z on same vehicle distribution
- Evaluate both on **held-out vehicles** (different seed)
- Compare tracking error, jerk, action smoothness
- Measure adaptation speed (performance vs burn-in length)

### Preview-Based Control
The agent receives 30 timesteps (3 seconds) of future reference speeds, enabling:
- **Anticipatory control**: Prepare for upcoming speed changes
- **Smooth transitions**: Avoid abrupt accelerations
- **Optimal planning**: Consider future constraints

### Domain Randomization
Vehicle parameters randomized per episode:
- **Standard mode**: Full parameter ranges (mass: 1500-6000 kg, motor: 200-800V, etc.)
- **Fitted mode**: Centered around fitted values with configurable spread (e.g., ±10%)
- **18 parameters**: All ExtendedPlant parameters can be randomized

### Parameter Fitting
- **Trajectory-based optimization**: Minimizes velocity MSE over full simulated trips
- **L-BFGS-B optimization**: Efficient gradient-based optimization
- **Validation-based early stopping**: Prevents overfitting
- **Warmup step**: Random parameter search for better initialization
- **Data filtering**: Speed, acceleration, segment length, zero-speed fraction

### PID Controller
- **Configurable gains**: kp, ki, kd (default: all 0.0)
- **Integral saturation**: Prevents windup with configurable limits
- **Two feedback modes**:
  - **RL-only** (default): Network sees only its own action (PID as "plant")
  - **Combined**: Network sees RL + PID action
- **Visualization**: Separate plots for RL and PID contributions

### C++ ONNX Deployment
- **ONNX export**: Convert PyTorch models to ONNX format
- **C++ inference**: PyBind11 module for ONNX Runtime inference
- **Verification**: Compare Python and C++ outputs during evaluation
- **Production-ready**: Deterministic policy export for deployment

### Acceleration Filtering
- **Exponential smoothing** prevents noisy jerk calculations
- **Configurable alpha** (0.0 = no smoothing, 1.0 = instant)
- **Applied during training** for consistent reward signals

## 🤝 Contributing

### Code Style
- Follow PEP 8 conventions
- Use type hints for function signatures
- Add docstrings to all functions and classes
- Write comprehensive unit tests

### Development Workflow
```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Make changes and add tests
# 3. Run tests
pytest tests/

# 4. Commit changes
git commit -m "Add new feature"

# 5. Push and create PR
git push origin feature/new-feature
```

## 📄 License

This project is released under the MIT License. See `LICENSE` file for details.

## 📚 References

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- [SAC Implementation Details](https://arxiv.org/abs/1812.05905)
- [Domain Randomization for Transfer Learning](https://arxiv.org/abs/1703.06907)

## 🆘 Troubleshooting

### Common Issues

**Training doesn't converge:**
- Check reward weights in `config.yaml`
- Verify dataset quality and diversity
- Ensure sufficient exploration (warmup_steps)
- Try different learning rates or batch sizes

**Parameter fitting fails:**
- Check data quality (speed, throttle, brake signals)
- Adjust filtering parameters (min_speed, max_speed, max_accel)
- Try warmup step for better initialization
- Increase batch size or reduce max_iter if optimization is unstable

**Evaluation shows poor performance:**
- Verify checkpoint path is correct
- Check if model was fully trained
- Compare evaluation vs training environments
- Ensure config file matches training config

**C++ inference doesn't work:**
- Verify ONNX model was exported correctly
- Check that C++ module is built (`_cpp_onnx_policy.so` exists)
- Ensure ONNX Runtime is available in `onnxruntime-linux-x64-1.16.3/`
- Check CMake and compiler versions

**Memory issues:**
- Reduce `replay_size` in training config
- Use smaller `batch_size`
- Reduce `batch_size` in parameter fitting
- Consider gradient accumulation

**Slow training:**
- Enable GPU acceleration if available
- Reduce model size or batch size
- Use mixed precision training
- Reduce `plant_substeps` (may affect accuracy)

### Support
For issues and questions:
1. Check existing issues on GitHub
2. Review documentation and examples
3. Create a new issue with detailed information

---

**Happy researching!** 🚗🧠✨
