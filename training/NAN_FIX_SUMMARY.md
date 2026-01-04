# NaN Issue Fix Summary

## Problem
Training crashed at step 10,000 with NaN values in the policy network's output (mu), preventing the Normal distribution from being created. The error showed:
```
ValueError: Expected parameter loc (Tensor of shape (1, 10)) of distribution Normal(loc: torch.Size([1, 10]), scale: torch.Size([1, 10])) to satisfy the constraint Real(), but found invalid values:
tensor([[nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]], device='cuda:0')
```

## Root Causes
1. **Exploding Q-values**: Extremely large reward values (-2.2M in evaluation) can cause Q-values to explode
2. **Gradient instability**: Even with clipping, gradients can spike under certain conditions
3. **Context issues**: Dynamics map context features may contain extreme values
4. **High learning rate**: Combined with large rewards, can destabilize training

## Implemented Fixes

### 1. NaN Detection System
**File**: `training/train_sac.py`
- Added `_check_for_nans()` method to `SACTrainer` class
- Checks policy, Q1, and Q2 networks for NaN/Inf values
- Reports which parameters are corrupted and their statistics
- Integrated into main training loop with automatic checkpoint tracking

### 2. Safety Checks in Policy Network
**File**: `training/train_sac.py` - `GaussianPolicy.forward()`
- Validates input observations for NaN/Inf
- Checks intermediate features for corruption
- Validates output mu and log_std
- Prints diagnostic information when issues detected
- Emergency fallback: replaces NaN inputs with zeros

### 3. Gradient Monitoring
**File**: `training/train_sac.py` - `SACTrainer.update()`
- Captures and logs gradient norms for both policy and Q-networks
- Added metrics:
  - `train/policy_grad_norm`: Policy network gradient magnitude
  - `train/q_grad_norm`: Q-network gradient magnitude
  - `train/q_mean`: Mean Q-value
  - `train/q_std`: Q-value standard deviation
- Helps identify when gradients are spiking before NaN occurs

### 4. Dynamics Map Context Validation
**File**: `training/train_sac.py` - `SACTrainer.collect_step()`
- Validates dynamics map context output for NaN/Inf
- Logs warning with statistics when invalid values detected
- Automatic fallback to zero context to prevent crash
- Allows training to continue while identifying context issues

### 5. Reward Scaling
**Files**: 
- `training/train_sac.py` - `TrainingParams` dataclass
- `training/train_sac.py` - `SACTrainer.collect_step()`
- `training/config_dynamics_map.yaml`

Added `reward_scale` parameter (default: 1.0, recommended: 0.001):
- Scales rewards before storing in replay buffer
- Prevents Q-value explosion
- Config updated to use `reward_scale: 0.001` (1/1000 scaling)
- This brings rewards from ~-2.2M to ~-2.2K range

### 6. Conservative Hyperparameters
**File**: `training/config_dynamics_map.yaml`
- **Learning rate**: Reduced from `0.0003` to `0.00003` (10x reduction)
- **Tau (target update)**: Reduced from `0.01` to `0.005` (slower target updates)
- Both changes improve training stability with dynamics map context

### 7. Training Loop Protection
**File**: `training/train_sac.py` - `train()` function
- NaN check after every SAC update
- Tracks last successful checkpoint
- On NaN detection:
  - Prints detailed error message with diagnostics
  - Suggests recovery steps (resume from last checkpoint, tune hyperparameters)
  - Saves emergency checkpoint for debugging
  - Gracefully exits with informative error
- Updates checkpoint tracker after successful saves

## Usage

### Starting Fresh Training
Use the updated config file which includes the fixes:
```bash
python training/train_sac.py --config training/config_dynamics_map.yaml
```

### Resuming After NaN Crash
If training crashes with NaN, resume from the last good checkpoint:
```bash
python training/train_sac.py \
    --config training/config_dynamics_map.yaml \
    --resume training/checkpoints_dynamics_map/sac_step_10000.pt
```

### Tuning for Your Setup
If you still encounter NaN issues, try:

1. **Increase reward scaling** (makes rewards smaller):
   ```yaml
   training:
     reward_scale: 0.0001  # Even more aggressive scaling
   ```

2. **Reduce learning rate further**:
   ```yaml
   training:
     learning_rate: 0.00001  # 10x smaller than current
   ```

3. **Slower target network updates**:
   ```yaml
   training:
     tau: 0.001  # Even slower updates
   ```

4. **Tighter gradient clipping**:
   ```yaml
   training:
     max_grad_norm: 1.0  # Reduced from 5.0
   ```

## Monitoring

Watch these new metrics in your logs/TensorBoard:
- `train/policy_grad_norm`: Should stay < 5.0 (clipping threshold)
- `train/q_grad_norm`: Should stay < 5.0 (clipping threshold)
- `train/q_mean`: Should be bounded, not grow to extreme values
- `train/q_std`: Should remain stable

**Warning signs before NaN**:
- Gradient norms consistently hitting the clipping limit (5.0)
- Q-values growing rapidly (mean > 10000)
- Large Q-value standard deviation (std > 5000)

## Testing the Fixes

The changes are backward compatible. Existing checkpoints can be loaded without issues. The new parameters have sensible defaults that won't break existing configs.

Key benefits:
1. **Early detection**: Catch NaN before it crashes training
2. **Better diagnostics**: Understand what went wrong
3. **Graceful recovery**: Save checkpoint and suggest solutions
4. **Preventive measures**: Reward scaling and conservative hyperparameters reduce likelihood of NaN

## Files Modified
1. `training/train_sac.py` - Core training script with all safety mechanisms
2. `training/config_dynamics_map.yaml` - Updated with conservative hyperparameters and reward scaling

