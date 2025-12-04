"""Data-generation helpers that rely on Hugging Face datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch

try:  # pragma: no cover - import guard exercised indirectly
    from datasets import Dataset
except ImportError as exc:  # pragma: no cover - explicit guidance for users
    raise ImportError(
        "The 'datasets' package is required. Install via `pip install datasets`."
    ) from exc

from src.data.datasets import EVSequenceDataset, SequenceWindowConfig


@dataclass(slots=True)
class SyntheticTrajectoryConfig:
    """Configuration for synthetic reference-speed generation."""

    num_sequences: int = 64
    sequence_length: int = 256
    dt: float = 0.1
    min_speed: float = 0.0
    max_speed: float = 30.0
    noise_std: float = 0.05
    max_accel: float = 2.5
    target_update_range: tuple[int, int] = (30, 120)
    smooth_noise_std: float = 0.05


class ReferenceTrajectoryGenerator:
    """Sample diverse reference-speed profiles for training/evaluation."""

    def __init__(
        self,
        dt: float = 0.1,
        min_speed: float = 0.0,
        max_speed: float = 30.0,
        max_accel: float = 2.5,
        target_update_range: tuple[int, int] = (30, 120),
        smooth_noise_std: float = 0.05,
    ) -> None:
        self.dt = dt
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.target_update_range = target_update_range
        self.smooth_noise_std = smooth_noise_std

    def sample(self, length: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return a single reference-speed profile of ``length`` samples."""

        rng = rng or np.random.default_rng()
        speeds = np.zeros(length, dtype=np.float32)
        current_speed = rng.uniform(self.min_speed, self.max_speed)
        target_speed = current_speed
        steps_remaining = 0
        accel_state = 0.0
        accel_gain = 0.5

        for idx in range(length):
            if steps_remaining <= 0:
                target_speed = rng.uniform(self.min_speed, self.max_speed)
                steps_remaining = int(
                    rng.integers(self.target_update_range[0], self.target_update_range[1])
                )
            steps_remaining -= 1

            desired_accel = np.clip(
                (target_speed - current_speed) * accel_gain,
                -self.max_accel,
                self.max_accel,
            )
            accel_state = 0.8 * accel_state + 0.2 * desired_accel
            accel_state = float(np.clip(accel_state, -self.max_accel, self.max_accel))
            current_speed = float(
                np.clip(current_speed + accel_state * self.dt, self.min_speed, self.max_speed)
            )
            speeds[idx] = current_speed

        if self.smooth_noise_std > 0:
            speeds += rng.normal(0.0, self.smooth_noise_std, size=length).astype(np.float32)
            kernel = max(3, int(round(0.5 / self.dt)))
            kernel += 1 - kernel % 2
            window = np.ones(kernel, dtype=np.float32) / kernel

            # Use edge padding instead of zero padding to avoid boundary artifacts
            pad = kernel // 2
            padded = np.pad(speeds, pad_width=pad, mode="edge")
            speeds = np.convolve(padded, window, mode="valid")

        return np.clip(speeds, self.min_speed, self.max_speed)


def build_synthetic_reference_dataset(
    config: SyntheticTrajectoryConfig,
    rng: np.random.Generator | None = None,
) -> Dataset:
    """Return a Hugging Face dataset containing synthetic reference trajectories."""

    rng = rng or np.random.default_rng()
    generator = ReferenceTrajectoryGenerator(
        dt=config.dt,
        min_speed=config.min_speed,
        max_speed=config.max_speed,
        max_accel=config.max_accel,
        target_update_range=config.target_update_range,
        smooth_noise_std=config.smooth_noise_std,
    )

    sequences: List[List[float]] = []
    metadata: List[dict[str, float]] = []
    for _ in range(config.num_sequences):
        profile = generator.sample(config.sequence_length, rng)
        if config.noise_std > 0:
            profile += rng.normal(0.0, config.noise_std, size=profile.shape).astype(np.float32)
        sequences.append(profile.tolist())
        metadata.append({"dt": config.dt, "length": config.sequence_length})

    return Dataset.from_dict(
        {
            "reference_speed": sequences,
            "metadata": metadata,
        }
    )


def ev_windows_to_dataset(
    data_path: Path,
    window: SequenceWindowConfig | None = None,
    state_features: Sequence[str] | None = None,
    max_samples: int | None = None,
) -> Dataset:
    """Convert :class:`EVSequenceDataset` windows into a Hugging Face dataset."""

    dataset = EVSequenceDataset(
        data_path=Path(data_path),
        window=window,
        state_features=state_features,
    )
    total = len(dataset)
    if total == 0:
        raise ValueError("EVSequenceDataset is empty; cannot create huggingface dataset")
    take = total if max_samples is None else min(total, max_samples)

    records = {
        "history_states": [],
        "history_actions": [],
        "future_states": [],
        "future_actions": [],
        "future_actuation": [],
        "loss_weight": [],
        "is_stationary": [],
    }

    for idx in range(take):
        sample = dataset[idx]
        records["history_states"].append(sample["history_states"].numpy().tolist())
        records["history_actions"].append(sample["history_actions"].numpy().tolist())
        records["future_states"].append(sample["future_states"].numpy().tolist())
        records["future_actions"].append(sample["future_actions"].numpy().tolist())
        records["future_actuation"].append(sample["future_actuation"].numpy().tolist())
        records["loss_weight"].append(float(sample["loss_weight"].item()))
        records["is_stationary"].append(int(sample["is_stationary"].item()))

    return Dataset.from_dict(records)


__all__ = [
    "ReferenceTrajectoryGenerator",
    "SyntheticTrajectoryConfig",
    "build_synthetic_reference_dataset",
    "ev_windows_to_dataset",
]


