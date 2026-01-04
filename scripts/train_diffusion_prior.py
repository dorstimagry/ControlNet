#!/usr/bin/env python3
"""Train diffusion prior model on GT acceleration maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import MapDataset
from src.models.diffusion_prior import DiffusionPrior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diffusion prior")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Data directory with train/val splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (overrides config)",
    )
    return parser.parse_args()


def train(config: dict, data_dir: Path, output_dir: Path):
    """Train diffusion prior model.
    
    Args:
        config: Training configuration
        data_dir: Data directory
        output_dir: Output directory for checkpoints
    """
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load datasets
    train_dataset = MapDataset(data_dir, split="train", load_norm_stats=True)
    val_dataset = MapDataset(data_dir, split="val", load_norm_stats=True)
    
    accelerator.print(f"Train: {len(train_dataset)} maps")
    accelerator.print(f"Val:   {len(val_dataset)} maps")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    
    # Create model
    model = DiffusionPrior(
        in_channels=1,
        out_channels=1,
        sample_size=tuple(config["sample_size"]),
        block_out_channels=tuple(config["block_out_channels"]),
        num_train_timesteps=config["num_train_timesteps"],
    )
    
    accelerator.print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
    )
    
    # Prepare with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    
    for epoch in range(config["num_epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config['num_epochs']}",
            disable=not accelerator.is_main_process,
        )
        
        for batch_idx, (maps, _) in enumerate(pbar):
            # Compute loss
            loss = model.compute_loss(maps, device=accelerator.device)
            
            # Backward
            accelerator.backward(loss)
            
            # Clip gradients
            if config.get("max_grad_norm") is not None:
                accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            
            # Step
            optimizer.step()
            optimizer.zero_grad()
            
            # Log
            train_loss += loss.item()
            global_step += 1
            
            if batch_idx % config.get("log_interval", 10) == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Save checkpoint
            if global_step % config.get("checkpoint_interval", 1000) == 0:
                if accelerator.is_main_process:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save(output_dir / f"checkpoint_step_{global_step}.pt")
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for maps, _ in tqdm(
                val_loader,
                desc="Validation",
                disable=not accelerator.is_main_process,
            ):
                loss = model.compute_loss(maps, device=accelerator.device)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        accelerator.print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )
        
        # Save best model
        if val_loss < best_val_loss and accelerator.is_main_process:
            best_val_loss = val_loss
            output_dir.mkdir(parents=True, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save(output_dir / "best.pt")
            accelerator.print(f"  Saved best model (val_loss={val_loss:.4f})")
    
    # Save final model
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save(output_dir / "final.pt")
        
        # Save config
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Save normalization stats
        with open(data_dir / "norm_stats.json", "r") as f:
            norm_stats = json.load(f)
        with open(output_dir / "norm_stats.json", "w") as f:
            json.dump(norm_stats, f)
        
        accelerator.print(f"\nTraining complete! Models saved to {output_dir}")


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override output dir if specified
    output_dir = args.output_dir if args.output_dir else Path(config["output_dir"])
    
    # Train
    train(config, args.data_dir, output_dir)


if __name__ == "__main__":
    main()

