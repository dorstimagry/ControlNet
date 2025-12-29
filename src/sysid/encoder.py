"""Context encoder for vehicle dynamics identification."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class FeatureBuilder:
    """Builds encoder features from raw speed and action observations.
    
    Features: o_t = [v_t, u_{t-1}, dv_t, du_{t-1}]
    where:
        - v_t: current speed
        - u_{t-1}: previous action
        - dv_t: (v_t - v_{t-1}) / dt (acceleration proxy)
        - du_{t-1}: u_{t-1} - u_{t-2} (action change)
    """
    
    def __init__(self, dt: float = 0.1):
        """Initialize feature builder.
        
        Args:
            dt: Environment timestep for computing derivatives
        """
        self.dt = dt
        self.v_prev = 0.0
        self.u_prev = 0.0
        self.u_prev_prev = 0.0
    
    def reset(self) -> None:
        """Reset history to initial state."""
        self.v_prev = 0.0
        self.u_prev = 0.0
        self.u_prev_prev = 0.0
    
    def build_features(self, v_t: float, u_t: float) -> torch.Tensor:
        """Build feature vector for current step.
        
        Args:
            v_t: Current speed
            u_t: Current action (action taken this step)
        
        Returns:
            Feature tensor of shape (4,): [v_t, u_{t-1}, dv_t, du_{t-1}]
        """
        # Compute derivatives
        dv_t = (v_t - self.v_prev) / self.dt
        du_prev = self.u_prev - self.u_prev_prev
        
        # Build feature vector (use u_prev, not u_t, since u_t produces v_t)
        features = torch.tensor(
            [v_t, self.u_prev, dv_t, du_prev],
            dtype=torch.float32
        )
        
        # Update history for next step
        self.v_prev = v_t
        self.u_prev_prev = self.u_prev
        self.u_prev = u_t
        
        return features
    
    def build_features_batch(
        self,
        v_seq: torch.Tensor,
        u_seq: torch.Tensor
    ) -> torch.Tensor:
        """Build features for a batch of sequences.
        
        Args:
            v_seq: Speed sequence of shape (batch, seq_len) 
            u_seq: Action sequence of shape (batch, seq_len-1) or (batch, seq_len)
                   If seq_len-1, u[i] is the action at step i (produces v[i+1])
        
        Returns:
            Feature tensor of shape (batch, seq_len, 4)
        """
        batch_size, v_len = v_seq.shape
        
        # Handle case where u_seq is one element shorter (u[t] produces v[t+1])
        if u_seq.shape[1] == v_len - 1:
            # Pad u with zero at the end for alignment
            u_seq = torch.cat([u_seq, torch.zeros(batch_size, 1, device=u_seq.device)], dim=1)
        
        seq_len = v_len
        
        # Compute dv (need v at t-1, so pad with zeros at start)
        v_prev = torch.cat([torch.zeros(batch_size, 1, device=v_seq.device), v_seq[:, :-1]], dim=1)
        dv = (v_seq - v_prev) / self.dt
        
        # For first step, assume v_prev = v_0 (dv=0)
        dv[:, 0] = 0.0
        
        # Compute du (need u at t-1 and t-2)
        u_prev = torch.cat([torch.zeros(batch_size, 1, device=u_seq.device), u_seq[:, :-1]], dim=1)
        u_prev_prev = torch.cat([torch.zeros(batch_size, 1, device=u_seq.device), u_prev[:, :-1]], dim=1)
        du_prev = u_prev - u_prev_prev
        
        # Stack features: [v_t, u_{t-1}, dv_t, du_{t-1}]
        features = torch.stack([v_seq, u_prev, dv, du_prev], dim=2)
        
        return features


class ContextEncoder(nn.Module):
    """GRU-based context encoder for vehicle dynamics.
    
    Takes features o_t = [v_t, u_{t-1}, dv_t, du_{t-1}] and produces:
        - h_t: GRU hidden state
        - z_t: Dynamics latent (linear projection from h_t)
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, z_dim: int = 12):
        """Initialize context encoder.
        
        Args:
            input_dim: Input feature dimension (default 4)
            hidden_dim: GRU hidden dimension
            z_dim: Output latent dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.z_proj = nn.Linear(hidden_dim, z_dim)
    
    def reset(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        """Reset hidden state to zeros.
        
        Args:
            batch_size: Batch size
            device: Device for tensor allocation
        
        Returns:
            Initial hidden state of shape (batch_size, hidden_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_dim, device=device)
    
    def step(self, o_t: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step update.
        
        Args:
            o_t: Input features of shape (batch, input_dim) or (input_dim,)
            h_prev: Previous hidden state of shape (batch, hidden_dim) or (hidden_dim,)
        
        Returns:
            Tuple of (h_t, z_t):
                - h_t: New hidden state of shape (batch, hidden_dim)
                - z_t: Dynamics latent of shape (batch, z_dim)
        """
        # Ensure batch dimension
        if o_t.dim() == 1:
            o_t = o_t.unsqueeze(0)
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)
        
        # GRU update
        h_t = self.gru_cell(o_t, h_prev)
        
        # Project to latent
        z_t = self.z_proj(h_t)
        
        return h_t, z_t
    
    def forward(
        self,
        o_seq: torch.Tensor,
        h_0: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process sequence of features.
        
        Args:
            o_seq: Feature sequence of shape (batch, seq_len, input_dim)
            h_0: Initial hidden state of shape (batch, hidden_dim), or None for zeros
        
        Returns:
            Tuple of (h_seq, z_seq, h_final):
                - h_seq: Hidden state sequence of shape (batch, seq_len, hidden_dim)
                - z_seq: Latent sequence of shape (batch, seq_len, z_dim)
                - h_final: Final hidden state of shape (batch, hidden_dim)
        """
        batch_size, seq_len, _ = o_seq.shape
        
        if h_0 is None:
            h_0 = self.reset(batch_size, device=o_seq.device)
        
        h_seq = []
        z_seq = []
        h_t = h_0
        
        for t in range(seq_len):
            h_t, z_t = self.step(o_seq[:, t], h_t)
            h_seq.append(h_t)
            z_seq.append(z_t)
        
        h_seq = torch.stack(h_seq, dim=1)  # (batch, seq_len, hidden_dim)
        z_seq = torch.stack(z_seq, dim=1)  # (batch, seq_len, z_dim)
        
        return h_seq, z_seq, h_t

