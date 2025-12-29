"""System identification module for vehicle-conditioned SAC.

This module provides online vehicle dynamics encoding through:
- ContextEncoder: GRU-based encoder that learns latent z_t from speed/action history
- DynamicsPredictor: MLP that predicts speed changes from (v, u, z)
- SysIDTrainer: Multi-step rollout training objective
- RunningNorm: Online normalization for features
"""

from .encoder import ContextEncoder, FeatureBuilder
from .predictor import DynamicsPredictor
from .normalization import RunningNorm
from .sysid_trainer import SysIDTrainer
from .dataset import SequenceBatch, sample_sequences
from .integration import compute_z_online, batch_compute_z

__all__ = [
    "ContextEncoder",
    "FeatureBuilder",
    "DynamicsPredictor",
    "RunningNorm",
    "SysIDTrainer",
    "SequenceBatch",
    "sample_sequences",
    "compute_z_online",
    "batch_compute_z",
]

