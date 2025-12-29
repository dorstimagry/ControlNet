"""SysID trainer with multi-step rollout loss."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from .encoder import ContextEncoder, FeatureBuilder
from .predictor import DynamicsPredictor
from .normalization import RunningNorm
from .dataset import SequenceBatch


class SysIDTrainer:
    """Trains encoder and predictor with multi-step rollout objective.
    
    Training procedure:
        1. Burn-in: compute z_t from history [t-B, t]
        2. Rollout: predict v[t+1:t+H+1] using true actions u[t:t+H]
        3. Loss: MSE + regularization (slow latent + L2)
    """
    
    def __init__(
        self,
        encoder: ContextEncoder,
        predictor: DynamicsPredictor,
        encoder_norm: RunningNorm,
        predictor_v_norm: RunningNorm,
        predictor_u_norm: RunningNorm,
        learning_rate: float = 1e-3,
        lambda_slow: float = 5e-3,
        lambda_z: float = 5e-4,
        dt: float = 0.1,
        device: torch.device | None = None
    ):
        """Initialize SysID trainer.
        
        Args:
            encoder: Context encoder
            predictor: Dynamics predictor
            encoder_norm: Normalization for encoder features
            predictor_v_norm: Normalization for predictor speed input
            predictor_u_norm: Normalization for predictor action input
            learning_rate: Optimizer learning rate
            lambda_slow: Weight for slow latent regularization
            lambda_z: Weight for L2 latent regularization
            dt: Environment timestep
            device: Device for computation
        """
        self.encoder = encoder
        self.predictor = predictor
        self.encoder_norm = encoder_norm
        self.predictor_v_norm = predictor_v_norm
        self.predictor_u_norm = predictor_u_norm
        self.feature_builder = FeatureBuilder(dt=dt)
        
        self.lambda_slow = lambda_slow
        self.lambda_z = lambda_z
        self.device = device or torch.device("cpu")
        
        # Optimizer for encoder + predictor
        params = list(encoder.parameters()) + list(predictor.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)
    
    def train_step(self, batch: SequenceBatch) -> Dict[str, float]:
        """Single training step with multi-step rollout.
        
        Args:
            batch: Sequence batch with burn-in and rollout windows
        
        Returns:
            Dictionary of losses for logging
        """
        self.encoder.train()
        self.predictor.train()
        
        batch = batch.to(self.device)
        batch_size = batch.v_seq.shape[0]
        burn_in = batch.burn_in
        horizon = batch.horizon
        
        # Validate batch size
        assert batch_size > 0, f"Batch size must be > 0, got {batch_size}"
        assert batch.v_seq.shape == (batch_size, burn_in + horizon + 1), \
            f"Expected v_seq shape ({batch_size}, {burn_in + horizon + 1}), got {batch.v_seq.shape}"
        assert batch.u_seq.shape == (batch_size, burn_in + horizon), \
            f"Expected u_seq shape ({batch_size}, {burn_in + horizon}), got {batch.u_seq.shape}"
        
        # Extract burn-in and rollout windows
        # v_seq: [t-B, ..., t, t+1, ..., t+H]
        # u_seq: [t-B, ..., t-1, t, ..., t+H-1]
        
        v_burnin = batch.v_seq[:, :burn_in+1]  # [t-B, ..., t]
        u_burnin = batch.u_seq[:, :burn_in]    # [t-B, ..., t-1]
        
        v_anchor = batch.v_seq[:, burn_in]     # v[t]
        u_rollout = batch.u_seq[:, burn_in:]   # [t, ..., t+H-1]
        v_rollout_true = batch.v_seq[:, burn_in+1:]  # [t+1, ..., t+H]
        
        # Build features for burn-in
        features_burnin = self.feature_builder.build_features_batch(v_burnin, u_burnin)
        
        # Normalize features
        features_burnin_norm = self.encoder_norm(
            features_burnin.reshape(-1, 4),
            update_stats=True
        ).reshape(batch_size, burn_in+1, 4)
        
        # Burn-in phase: compute z_t and collect hidden states for slow loss
        h_seq, z_seq, h_final = self.encoder(features_burnin_norm)
        z_t = z_seq[:, -1, :]  # z at anchor time t
        
        # Rollout phase: predict v[t+1:t+H+1]
        v_hat = v_anchor.clone()
        rollout_losses = []
        
        for k in range(horizon):
            # Normalize inputs for predictor
            # v_hat and u_k have shape (batch,), need (batch, 1) for normalizer
            v_hat_norm = self.predictor_v_norm(v_hat.unsqueeze(-1), update_stats=True).squeeze(-1)
            u_k_norm = self.predictor_u_norm(u_rollout[:, k].unsqueeze(-1), update_stats=True).squeeze(-1)
            
            # Predict Δv
            dv_hat = self.predictor(v_hat_norm, u_k_norm, z_t).squeeze(-1)
            
            # Update v_hat
            v_hat = v_hat + dv_hat
            
            # Compute prediction error
            v_true = v_rollout_true[:, k]
            pred_error = (v_hat - v_true).pow(2)
            rollout_losses.append(pred_error)
        
        # Prediction loss: mean over horizon and batch
        rollout_losses = torch.stack(rollout_losses, dim=1)  # (batch, horizon)
        pred_loss = rollout_losses.mean()
        
        # Regularization: slow latent (encourage smooth hidden state evolution)
        slow_loss = 0.0
        if self.lambda_slow > 0:
            # Compute ||h_i - h_{i-1}||^2 over burn-in window
            h_diff = h_seq[:, 1:, :] - h_seq[:, :-1, :]
            slow_loss = h_diff.pow(2).mean()
        
        # Regularization: L2 on latent
        z_l2_loss = z_t.pow(2).mean()
        
        # Total loss
        total_loss = pred_loss + self.lambda_slow * slow_loss + self.lambda_z * z_l2_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Return metrics
        return {
            "sysid/pred_loss": float(pred_loss.item()),
            "sysid/slow_loss": float(slow_loss.item() if isinstance(slow_loss, torch.Tensor) else slow_loss),
            "sysid/z_l2_loss": float(z_l2_loss.item()),
            "sysid/total_loss": float(total_loss.item()),
            "sysid/z_norm": float(z_t.norm(dim=1).mean().item()),
            "sysid/batch_size": float(batch_size),  # Log batch size for verification
        }

