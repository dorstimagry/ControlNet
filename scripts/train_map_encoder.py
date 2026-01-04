#!/usr/bin/env python3
"""Train map encoder as an autoencoder on the GT dynamics maps dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.datasets import MapDataset
from src.maps.encoder import MapEncoder


class MapDecoder(nn.Module):
    """Decoder network to reconstruct maps from encoded representations.
    
    Mirrors the encoder architecture in reverse.
    """
    
    def __init__(
        self,
        context_dim: int = 32,
        hidden_channels: tuple[int, ...] = (32, 16),
        output_size: tuple[int, int] = (100, 100),
    ):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_channels = hidden_channels
        self.output_size = output_size
        
        # Calculate initial spatial size after encoder
        # Encoder does 2 downsample operations (2x2 each) -> 25x25
        h_init = output_size[0] // 4
        w_init = output_size[1] // 4
        
        # Linear projection from context to spatial features
        self.fc = nn.Linear(context_dim, hidden_channels[0] * h_init * w_init)
        
        # Upsampling blocks (reverse of encoder)
        layers = []
        in_channels = hidden_channels[0]
        
        for out_channels in hidden_channels[1:]:
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        
        # Final upsampling to original size
        layers.append(
            nn.ConvTranspose2d(in_channels, 1, kernel_size=4, stride=2, padding=1)
        )
        
        self.decoder = nn.Sequential(*layers)
        self.h_init = h_init
        self.w_init = w_init
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode context vector to map.
        
        Args:
            z: Context vector, shape (B, context_dim)
            
        Returns:
            Reconstructed map, shape (B, 1, H, W)
        """
        # Project to spatial features
        x = self.fc(z)  # (B, hidden_channels[0] * h * w)
        x = x.view(-1, self.hidden_channels[0], self.h_init, self.w_init)
        
        # Decode
        x = self.decoder(x)  # (B, 1, H, W)
        
        return x


class MapAutoencoder(nn.Module):
    """Autoencoder for dynamics maps."""
    
    def __init__(
        self,
        input_size: tuple[int, int] = (100, 100),
        context_dim: int = 32,
        hidden_channels: tuple[int, ...] = (16, 32),
    ):
        super().__init__()
        
        self.encoder = MapEncoder(
            input_size=input_size,
            context_dim=context_dim,
            hidden_channels=hidden_channels,
        )
        
        self.decoder = MapDecoder(
            context_dim=context_dim,
            hidden_channels=hidden_channels[::-1],  # Reverse order
            output_size=input_size,
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input map, shape (B, 1, H, W)
            
        Returns:
            z: Encoded representation, shape (B, context_dim)
            x_recon: Reconstructed map, shape (B, 1, H, W)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


def train_autoencoder(
    data_dir: Path,
    output_dir: Path,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    context_dim: int = 32,
    hidden_channels: tuple[int, ...] = (16, 32),
    val_split: float = 0.1,
    seed: int = 42,
) -> None:
    """Train map autoencoder.
    
    Args:
        data_dir: Directory containing GT maps
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        context_dim: Dimension of encoded representation
        hidden_channels: Hidden channel dimensions
        val_split: Fraction of data for validation
        seed: Random seed
    """
    # Setup
    accelerator = Accelerator()
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator.print(f"[setup] Loading dataset from {data_dir}")
    
    # Load dataset
    dataset = MapDataset(data_dir, split="train", load_norm_stats=True)
    
    # Get normalization stats
    norm_stats = {
        "mean": float(dataset.norm_mean),
        "std": float(dataset.norm_std),
    }
    accelerator.print(f"[setup] Dataset: {len(dataset)} maps")
    accelerator.print(f"[setup] Norm stats: mean={norm_stats['mean']:.3f}, std={norm_stats['std']:.3f}")
    
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    
    accelerator.print(f"[setup] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    model = MapAutoencoder(
        input_size=(100, 100),
        context_dim=context_dim,
        hidden_channels=hidden_channels,
    )
    
    accelerator.print(f"[model] Created autoencoder")
    accelerator.print(f"[model]   context_dim: {context_dim}")
    accelerator.print(f"[model]   hidden_channels: {hidden_channels}")
    
    # Count parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"[model]   encoder params: {encoder_params:,}")
    accelerator.print(f"[model]   decoder params: {decoder_params:,}")
    accelerator.print(f"[model]   total params: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Prepare with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
        )
        
        for batch in progress:
            # Unpack batch (MapDataset returns (tensor, metadata))
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                maps, _ = batch  # Discard metadata
            else:
                maps = batch
            
            # Forward
            z, maps_recon = model(maps)
            
            # Reconstruction loss (MSE)
            loss = F.mse_loss(maps_recon, maps)
            
            # Backward
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            progress.set_postfix({"loss": loss.item()})
        
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Unpack batch
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    maps, _ = batch
                else:
                    maps = batch
                
                z, maps_recon = model(maps)
                loss = F.mse_loss(maps_recon, maps)
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        
        # Log
        accelerator.print(
            f"[epoch {epoch+1}] train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            if accelerator.is_main_process:
                # Save encoder only (what we need for SAC)
                encoder = accelerator.unwrap_model(model).encoder
                encoder_path = output_dir / "best_encoder.pt"
                encoder.save(encoder_path)
                accelerator.print(f"[checkpoint] Saved best encoder: {encoder_path}")
                
                # Save full autoencoder (for inspection/analysis)
                full_model = accelerator.unwrap_model(model)
                full_path = output_dir / "best_autoencoder.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "encoder": full_model.encoder.state_dict(),
                    "decoder": full_model.decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": {
                        "context_dim": context_dim,
                        "hidden_channels": hidden_channels,
                        "input_size": (100, 100),
                    },
                    "norm_stats": norm_stats,
                }, full_path)
    
    # Save final encoder
    if accelerator.is_main_process:
        encoder = accelerator.unwrap_model(model).encoder
        final_path = output_dir / "final_encoder.pt"
        encoder.save(final_path)
        accelerator.print(f"[checkpoint] Saved final encoder: {final_path}")
        
        # Save norm stats
        norm_path = output_dir / "norm_stats.json"
        with open(norm_path, "w") as f:
            json.dump(norm_stats, f, indent=2)
        accelerator.print(f"[checkpoint] Saved norm stats: {norm_path}")
    
    accelerator.print(f"[done] Best val loss: {best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train map encoder as autoencoder")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/maps",
        help="Directory containing GT maps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/map_encoder",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--context-dim",
        type=int,
        default=32,
        help="Dimension of encoded representation",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        nargs="+",
        default=[16, 32],
        help="Hidden channel dimensions",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split fraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    train_autoencoder(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        context_dim=args.context_dim,
        hidden_channels=tuple(args.hidden_channels),
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

