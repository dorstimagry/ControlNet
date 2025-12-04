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
│   └── results/                   # Evaluation outputs (ignored)
├── scripts/                       # Utility scripts
│   ├── generate_synthetic_dataset.py    # Generate synthetic training data
│   ├── analyze_dataset_coverage.py      # Dataset analysis tools
│   ├── build_segment_metadata.py        # Metadata processing
│   ├── fetch_trips.py                   # Data fetching utilities
│   ├── parse_trips.py                   # Trip parsing tools
│   └── tune_objective_weights.py        # Hyperparameter tuning
├── src/                          # Additional source code
│   └── data/                     # Data processing modules
├── tests/                        # Unit tests
│   ├── test_env.py               # Environment tests
│   ├── test_dynamics.py          # Dynamics model tests
│   ├── test_evaluation.py        # Evaluation tests
│   ├── test_training.py          # Training tests
│   └── test_data.py              # Data processing tests
├── training/                     # Training code and configuration
│   ├── train_sac.py              # Main training script
│   ├── config.yaml               # Training configuration
│   └── checkpoints/              # Model checkpoints (ignored)
├── utils/                        # Core utilities
│   ├── dynamics.py               # Vehicle dynamics models
│   ├── data_utils.py             # Data processing utilities
│   └── __init__.py
├── .gitignore                    # Git ignore rules
├── tuning_results.yaml           # Hyperparameter tuning results
└── README.md                     # This file
```

## 🚗 System Overview

### Vehicle Model
- **Pure DC Motor**: Direct wheel drive with gear reduction
- **Mass Range**: 1500-6000 kg (light cars to heavy trucks)
- **Motor Power**: 200-800V, 0.05-0.5Ω, 0.1-0.4 Nm/A constants
- **Gear Ratio**: 4:1 to 20:1 reduction
- **Guaranteed Performance**: All vehicles can achieve 2.5-4.0 m/s² acceleration

### Control Architecture
- **Algorithm**: Soft Actor-Critic (SAC) with automatic entropy tuning
- **Action Space**: [-1, 1] (throttle/brake command)
- **Observation Space**: [speed] + 30 future reference speeds
- **Horizon**: 3 seconds preview (30 timesteps at 10Hz)
- **Reward**: Speed tracking + jerk penalty + action regularization + voltage constraints

### Key Features
- **Domain Randomization**: Vehicle parameters randomized per episode
- **Acceleration Filtering**: Smoothed acceleration for jerk computation
- **Preview-Based Control**: Agent sees future reference trajectory
- **Physical Constraints**: Realistic motor limits and vehicle dynamics

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Python 3.10+
python --version

# Required packages
pip install torch gymnasium numpy matplotlib scipy pyyaml
```

### Environment Setup
```bash
# Clone and enter repository
cd DiffDynamics

# Install in development mode
pip install -e .
```

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

### Basic Training
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
```

### Hyperparameter Tuning
Use the tuning script to optimize reward weights:

```bash
python scripts/tune_objective_weights.py
```

Results are saved to `tuning_results.yaml`.

## 📈 Evaluation

### Closed-Loop Evaluation
Evaluate trained policy on reference tracking:

```bash
# Evaluate latest checkpoint
python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints/sac_step_1000000.pt \
    --episodes 10 \
    --plot-dir evaluation/results/plots/test_run

# Custom evaluation parameters
python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints/sac_step_500000.pt \
    --episodes 20 \
    --output evaluation/results/custom_eval.json
```

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

#### Advanced Settings
```yaml
env:
  accel_filter_alpha: 0.2          # Acceleration filtering (0-1)
  horizon_penalty_decay: 0.95      # Future penalty exponential decay
  use_extended_plant: true         # Use DC motor dynamics
  plant_substeps: 5                # Integration substeps
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
```

### Test Coverage
- **Environment**: Reset/step functionality, observation spaces, termination
- **Dynamics**: Vehicle models, parameter sampling, acceleration capabilities
- **Training**: SAC implementation, replay buffer, policy updates
- **Evaluation**: Metric calculation, plotting, policy loading

## 📋 Dependencies

### Core Dependencies
- `torch>=2.0.0`: PyTorch for neural networks
- `gymnasium`: Reinforcement learning environment interface
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `scipy`: Scientific computing (stats, signal processing)
- `pyyaml`: Configuration file parsing

### Optional Dependencies
- `wandb`: Experiment tracking
- `pytest`: Testing framework
- `tqdm`: Progress bars
- `accelerate`: Multi-GPU training

## 🎯 Usage Examples

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

### Production Training
```bash
# Generate large dataset
python scripts/generate_synthetic_dataset.py \
    --output data/processed/synthetic/training_data.pt \
    --num-segments 500 --segment-length 2048 --seed 42

# Train with optimized settings
# (edit training/config.yaml for production settings)
python training/train_sac.py --num-train-timesteps 2000000

# Comprehensive evaluation
python evaluation/eval_closed_loop.py \
    --checkpoint training/checkpoints/sac_step_2000000.pt \
    --episodes 50 --plot-dir evaluation/results/production_eval
```

## 🔬 Research Features

### Preview-Based Control
The agent receives 30 timesteps (3 seconds) of future reference speeds, enabling:
- **Anticipatory control**: Prepare for upcoming speed changes
- **Smooth transitions**: Avoid abrupt accelerations
- **Optimal planning**: Consider future constraints

### Domain Randomization
Vehicle parameters randomized per episode:
- **Mass**: 1500-6000 kg
- **Motor voltage**: 200-800V
- **Resistance**: 0.05-0.5Ω
- **Gear ratio**: 4:1 to 20:1
- **Friction**: Variable road conditions

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

**Evaluation shows poor performance:**
- Verify checkpoint path is correct
- Check if model was fully trained
- Compare evaluation vs training environments

**Memory issues:**
- Reduce `replay_size` in training config
- Use smaller `batch_size`
- Consider gradient accumulation

**Slow training:**
- Enable GPU acceleration if available
- Reduce model size or batch size
- Use mixed precision training

### Support
For issues and questions:
1. Check existing issues on GitHub
2. Review documentation and examples
3. Create a new issue with detailed information

---

**Happy researching!** 🚗🧠✨
