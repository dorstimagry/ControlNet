"""PyTorch dataset for loading and normalizing acceleration maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Import existing classes for backward compatibility
try:
    from src.data.ev_sequence_dataset import EVSequenceDataset, SequenceWindowConfig
except ImportError:
    # These may not exist yet, that's ok
    pass


class MapDataset(Dataset):
    """Dataset for loading pre-generated acceleration maps.
    
    Maps are loaded from .npz files and optionally normalized.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        norm_stats: Optional[Dict[str, float]] = None,
        load_norm_stats: bool = True,
    ):
        """Initialize map dataset.
        
        Args:
            data_dir: Root directory containing train/val subdirectories
            split: Dataset split ("train" or "val")
            norm_stats: Optional normalization stats dict with "mean" and "std"
            load_norm_stats: If True, load norm_stats.json from data_dir
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")
        
        # Find all .npz files
        self.map_files = sorted(self.split_dir.glob("map_*.npz"))
        if len(self.map_files) == 0:
            raise ValueError(f"No map files found in {self.split_dir}")
        
        # Load normalization stats
        if norm_stats is not None:
            self.norm_mean = norm_stats["mean"]
            self.norm_std = norm_stats["std"]
        elif load_norm_stats:
            norm_path = self.data_dir / "norm_stats.json"
            if norm_path.exists():
                with open(norm_path, "r") as f:
                    stats = json.load(f)
                self.norm_mean = stats["mean"]
                self.norm_std = stats["std"]
            else:
                # No normalization
                self.norm_mean = 0.0
                self.norm_std = 1.0
        else:
            # No normalization
            self.norm_mean = 0.0
            self.norm_std = 1.0

    def __len__(self) -> int:
        """Return number of maps in dataset."""
        return len(self.map_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Load and return a map.
        
        Args:
            idx: Index
        
        Returns:
            Tuple of:
                - Map tensor with shape (1, N_u, N_v) (normalized)
                - Metadata dict
        """
        # Load map from .npz
        data = np.load(self.map_files[idx])
        X = data["map"]  # shape: (N_u, N_v)
        metadata = {k: data[k].item() if data[k].ndim == 0 else data[k] 
                    for k in data.keys() if k != "map"}
        
        # Normalize
        X_norm = (X - self.norm_mean) / self.norm_std
        
        # Add channel dimension: (1, N_u, N_v)
        X_norm = X_norm[np.newaxis, :, :].astype(np.float32)
        
        return torch.from_numpy(X_norm), metadata


def compute_normalization_stats(data_dir: Path, split: str = "train") -> Dict[str, float]:
    """Compute global mean and std for normalization.
    
    Args:
        data_dir: Root directory
        split: Split to compute stats from
    
    Returns:
        Dictionary with "mean" and "std"
    """
    split_dir = data_dir / split
    map_files = sorted(split_dir.glob("map_*.npz"))
    
    if len(map_files) == 0:
        raise ValueError(f"No map files found in {split_dir}")
    
    # Accumulate statistics
    all_values = []
    
    for map_file in map_files:
        data = np.load(map_file)
        X = data["map"]
        all_values.append(X.flatten())
    
    # Concatenate all values
    all_values = np.concatenate(all_values)
    
    # Compute global stats
    mean = float(np.mean(all_values))
    std = float(np.std(all_values))
    
    return {"mean": mean, "std": std}


def save_normalization_stats(stats: Dict[str, float], output_path: Path) -> None:
    """Save normalization stats to JSON.
    
    Args:
        stats: Stats dictionary
        output_path: Output path for JSON file
    """
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
