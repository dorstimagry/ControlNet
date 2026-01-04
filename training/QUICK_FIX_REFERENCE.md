# Quick Reference: NaN Training Issue

## What Happened?
Training crashed at step 10,000 with NaN in policy network output.

## Immediate Action
Resume from last checkpoint:
```bash
python training/train_sac.py \
    --config training/config_dynamics_map.yaml \
    --resume training/checkpoints_dynamics_map/sac_step_10000.pt
```

## What Was Fixed?

### ✅ Automatic NaN Detection
- Checks networks after every update
- Saves emergency checkpoint if NaN detected
- Points to last good checkpoint for recovery

### ✅ Reward Scaling
- **Before**: Rewards ~ -2.2 million
- **After**: Rewards ~ -2.2 thousand (scaled by 0.001)
- Prevents Q-value explosion

### ✅ Conservative Learning Rates
- **Before**: lr = 0.0003, tau = 0.01
- **After**: lr = 0.00003, tau = 0.005
- More stable training

### ✅ Input Validation
- Dynamics map context checked for NaN/Inf
- Falls back to zero context if invalid
- Policy network validates all inputs

### ✅ Gradient Monitoring
- Logs gradient norms: `train/policy_grad_norm`, `train/q_grad_norm`
- Logs Q-value statistics: `train/q_mean`, `train/q_std`
- Watch for warning signs before NaN

## Warning Signs

Monitor these in logs:
```
train/policy_grad_norm > 4.5  ← Getting close to clip limit!
train/q_mean > 10000          ← Q-values exploding!
train/q_std > 5000            ← High variance in Q-values!
```

## If NaN Occurs Again

1. **Resume from last checkpoint** (automatic tracking now)
2. **Increase reward scaling**:
   ```yaml
   reward_scale: 0.0001  # 10x smaller rewards
   ```
3. **Reduce learning rate**:
   ```yaml
   learning_rate: 0.00001  # Even more conservative
   ```
4. **Check dynamics map**: Look at context values in emergency checkpoint

## Files Changed
- ✅ `training/train_sac.py` - All safety mechanisms
- ✅ `training/config_dynamics_map.yaml` - Updated hyperparameters

## Next Steps
1. Start fresh training with new config
2. Monitor gradient norms and Q-values
3. If stable for 50K steps, consider increasing learning rate slightly

