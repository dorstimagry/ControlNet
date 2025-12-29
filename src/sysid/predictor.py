"""Dynamics predictor for multi-step rollout."""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicsPredictor(nn.Module):
    """MLP dynamics predictor for vehicle speed changes.
    
    Predicts Δv_hat from inputs [v_hat, u, z_t].
    """
    
    def __init__(
        self,
        z_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "relu"
    ):
        """Initialize dynamics predictor.
        
        Args:
            z_dim: Dynamics latent dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers (minimum 2)
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        
        # Input: [v, u, z] -> 1 + 1 + z_dim
        input_dim = 2 + z_dim
        
        # Build MLP
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            in_dim = hidden_dim
        
        # Output layer: predict Δv (scalar)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        v: torch.Tensor,
        u: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """Predict speed change.
        
        Args:
            v: Current speed of shape (batch,) or (batch, 1)
            u: Action of shape (batch,) or (batch, 1)
            z: Dynamics latent of shape (batch, z_dim)
        
        Returns:
            Predicted speed change Δv of shape (batch, 1)
        """
        # Ensure correct shapes
        if v.dim() == 1:
            v = v.unsqueeze(1)
        if u.dim() == 1:
            u = u.unsqueeze(1)
        
        # Concatenate inputs
        x = torch.cat([v, u, z], dim=1)
        
        # Predict Δv
        dv = self.net(x)
        
        return dv

