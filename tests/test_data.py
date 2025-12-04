"""Unit tests for dataset helpers built on Hugging Face datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.datasets import SequenceWindowConfig
from utils.data_utils import (
    SyntheticTrajectoryConfig,
    build_synthetic_reference_dataset,
    ev_windows_to_dataset,
)


def test_build_synthetic_reference_dataset() -> None:
    """Synthetic generator should produce the requested number of sequences."""

    config = SyntheticTrajectoryConfig(num_sequences=5, sequence_length=40, dt=0.1)
    dataset = build_synthetic_reference_dataset(config, np.random.default_rng(0))
    assert len(dataset) == config.num_sequences
    first = dataset[0]
    assert len(first["reference_speed"]) == config.sequence_length
    assert first["metadata"]["dt"] == config.dt


def test_ev_windows_to_dataset_round_trip(tmp_path: Path) -> None:
    """EVSequenceDataset windows are converted into huggingface records."""

    data_file = tmp_path / "toy.pt"
    _write_toy_trip(data_file)
    window = SequenceWindowConfig(history=4, horizon=4, stride=2, allow_overlap_actuation=True)
    dataset = ev_windows_to_dataset(data_file, window=window, max_samples=2)
    assert len(dataset) == 2

    sample = dataset[0]
    assert "history_states" in sample
    assert len(sample["history_states"]) == window.history
    assert len(sample["future_states"]) == window.horizon


def _write_toy_trip(path: Path) -> None:
    num_steps = 32
    times = np.arange(num_steps, dtype=np.float32) * 0.1
    throttle = np.linspace(0.0, 50.0, num_steps, dtype=np.float32)
    brake = np.zeros(num_steps, dtype=np.float32)
    speed = np.linspace(0.0, 15.0, num_steps, dtype=np.float32)
    accel = np.gradient(speed, 0.1)
    angle = np.zeros(num_steps, dtype=np.float32)

    payload = {
        "metadata": {"dt": 0.1, "num_valid_trips": 1},
        "trip_000": {
            "time": times,
            "speed": speed,
            "throttle": throttle,
            "brake": brake,
            "angle": angle,
            "acceleration": accel.astype(np.float32),
        },
    }
    torch.save(payload, path)


