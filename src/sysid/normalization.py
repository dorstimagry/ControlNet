"""Running normalization for SysID features."""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """Running mean/std normalization with Welford's algorithm.
    
    Maintains online statistics and normalizes inputs to zero mean, unit variance.
    Clamps normalized values to prevent extreme outliers.
    """

    def __init__(self, dim: int, eps: float = 1e-6, clip: float = 10.0):
        """Initialize running normalization.
        
        Args:
            dim: Feature dimension
            eps: Small constant for numerical stability
            clip: Clamp normalized values to [-clip, clip]
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.clip = clip
        
        # Running statistics (not model parameters, but persistent buffers)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(1))
    
    def update(self, x: torch.Tensor) -> None:
        """Update running statistics with new batch of data.
        
        Uses Welford's online algorithm for numerical stability.
        
        Args:
            x: Tensor of shape (batch, dim) or (dim,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        
        # Welford's algorithm for combining statistics
        new_count = self.count + batch_size
        delta = batch_mean - self.mean
        
        # Update mean
        self.mean = self.mean + delta * batch_size / new_count
        
        # Update variance
        m_a = self.var * self.count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_size / new_count
        self.var = M2 / new_count
        
        self.count = new_count
    
    def normalize(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """Normalize input using running statistics.
        
        Args:
            x: Input tensor of shape (batch, dim) or (dim,)
            update_stats: If True, update running statistics with this batch
        
        Returns:
            Normalized tensor, clamped to [-clip, clip]
        """
        if update_stats:
            with torch.no_grad():
                self.update(x)
        
        # Normalize
        std = torch.sqrt(self.var + self.eps)
        x_norm = (x - self.mean) / std
        
        # Clamp to prevent extreme values
        x_norm = torch.clamp(x_norm, -self.clip, self.clip)
        
        return x_norm
    
    def forward(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """Forward pass (alias for normalize)."""
        return self.normalize(x, update_stats=update_stats)
    
    def reset(self) -> None:
        """Reset statistics to initial state."""
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.zero_()

