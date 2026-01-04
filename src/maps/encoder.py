"""Map context encoder for RL policy."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class MapEncoder(nn.Module):
    """Encode acceleration map to compact context vector.
    
    Simple CNN-based encoder that compresses a 2D map into a fixed-size vector.
    """
    
    def __init__(
        self,
        input_size: tuple[int, int] = (100, 100),
        context_dim: int = 32,
        hidden_channels: tuple[int, ...] = (16, 32),
    ):
        """Initialize map encoder.
        
        Args:
            input_size: Input map size (N_u, N_v)
            context_dim: Output context dimension
            hidden_channels: Hidden channel dimensions for conv layers
        """
        super().__init__()
        
        self.input_size = input_size
        self.context_dim = context_dim
        
        # Conv layers
        layers = []
        in_ch = 1
        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
            ])
            in_ch = out_ch
        
        self.conv_net = nn.Sequential(*layers)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output projection
        self.fc = nn.Linear(hidden_channels[-1], context_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode map to context vector.
        
        Args:
            x: Map tensor, shape (B, 1, H, W)
        
        Returns:
            Context vector, shape (B, context_dim)
        """
        # Conv layers
        features = self.conv_net(x)  # (B, C, H, W)
        
        # Global pooling
        pooled = self.pool(features)  # (B, C, 1, 1)
        pooled = pooled.squeeze(-1).squeeze(-1)  # (B, C)
        
        # Project to context
        context = self.fc(pooled)  # (B, context_dim)
        
        return context
    
    def save(self, path: Path) -> None:
        """Save encoder checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "input_size": self.input_size,
                "context_dim": self.context_dim,
                "hidden_channels": [m.out_channels for m in self.conv_net if isinstance(m, nn.Conv2d)],
            },
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: Path, device: torch.device = torch.device("cpu")) -> "MapEncoder":
        """Load encoder checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load to
        
        Returns:
            Loaded encoder
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        
        encoder = cls(
            input_size=tuple(config["input_size"]),
            context_dim=config["context_dim"],
            hidden_channels=tuple(config["hidden_channels"]),
        )
        
        encoder.load_state_dict(checkpoint["state_dict"])
        encoder.to(device)
        
        return encoder

