"""Dataset utilities for training transformer-based controllers."""

from __future__ import annotations

import json

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class SequenceWindowConfig:
    """Configuration for extracting rolling windows from trip segments."""

    history: int = 50
    horizon: int = 50
    stride: int = 5
    max_throttle: float = 100.0
    max_brake: float = 100.0
    allow_overlap_actuation: bool = False


STATIONARY_SPEED_EPS: float = 0.15  # m/s
STATIONARY_ACCEL_EPS: float = 0.1  # m/s^2


class EVSequenceDataset(Dataset):
    """PyTorch dataset that serves history/horizon windows from saved trips."""

    def __init__(
        self,
        data_path: Path,
        window: SequenceWindowConfig | None = None,
        state_features: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.window = window or SequenceWindowConfig()
        self.state_features = list(state_features) if state_features else ["speed", "grade", "acceleration"]

        raw = torch.load(data_path)
        self._metadata = raw.get("metadata", {})
        self._segments: List[dict[str, np.ndarray]] = []
        for key, value in raw.items():
            if key == "metadata":
                continue
            self._segments.append({name: np.asarray(arr) for name, arr in value.items()})

        self._sample_time = self._infer_sample_time(self._metadata)
        self._accelerations: List[np.ndarray] = []
        for seg in self._segments:
            speed = seg["speed"]
            if speed.size < 3:
                accel = np.gradient(speed, self._sample_time, edge_order=1)
            else:
                accel = np.gradient(speed, self._sample_time, edge_order=2)
            self._accelerations.append(accel)
        for seg, accel in zip(self._segments, self._accelerations):
            if "acceleration" not in seg:
                seg["acceleration"] = accel

        self._index: List[tuple[int, int]]
        self._stationary_flags: np.ndarray
        self._index, self._stationary_flags = self._build_index()

        self.stationary_fraction: float = float(self._stationary_flags.mean()) if len(self._stationary_flags) else 0.0
        percentage = max(self.stationary_fraction * 100.0, 1e-6)
        self.stationary_weight: float = 1.0 / percentage

    # ------------------------------------------------------------------
    def _infer_sample_time(self, metadata: dict) -> float:
        for key in ("sample_time", "dt", "delta_t", "time_step"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    continue
        return 0.1

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seg_idx, anchor = self._index[idx]
        seg = self._segments[seg_idx]
        w = self.window

        history_slice = slice(anchor - w.history, anchor)
        future_slice = slice(anchor, anchor + w.horizon)

        throttle = seg["throttle"]
        brake = seg["brake"]
        angle = seg.get("angle")

        def _states(slice_) -> np.ndarray:
            feats: List[np.ndarray] = []
            for name in self.state_features:
                if name == "speed":
                    feats.append(seg["speed"][slice_])
                elif name == "grade":
                    if angle is None:
                        feats.append(np.zeros_like(seg["speed"][slice_]))
                    else:
                        feats.append(np.sin(angle[slice_]))
                elif name == "angle":
                    feats.append(angle[slice_] if angle is not None else np.zeros_like(seg["speed"][slice_]))
                elif name == "acceleration":
                    feats.append(seg["acceleration"][slice_])
                else:
                    raise KeyError(f"Unsupported state feature: {name}")
            return np.stack(feats, axis=-1)

        throttle_hist = throttle[history_slice]
        brake_hist = brake[history_slice]
        throttle_future = throttle[future_slice]
        brake_future = brake[future_slice]

        history_states = torch.from_numpy(_states(history_slice)).float()
        history_actions = torch.from_numpy(
            np.stack(
                [
                    throttle_hist / self.window.max_throttle,
                    brake_hist / self.window.max_brake,
                ],
                axis=-1,
            )
        ).float()

        future_states = torch.from_numpy(_states(future_slice)).float()
        future_actions = torch.from_numpy(
            np.stack(
                [
                    throttle_future / self.window.max_throttle,
                    brake_future / self.window.max_brake,
                ],
                axis=-1,
            )
        ).float()

        # Composite actuation signal: positive -> throttle, negative -> brake
        actuation = self._compute_actuation(throttle_future, brake_future)

        loss_weight = self.stationary_weight if self._stationary_flags[idx] else 1.0

        return {
            "history_states": history_states,
            "history_actions": history_actions,
            "future_states": future_states,
            "future_actions": future_actions,
            "future_actuation": torch.from_numpy(actuation).float(),
            "loss_weight": torch.tensor(loss_weight, dtype=torch.float32),
            "is_stationary": torch.tensor(bool(self._stationary_flags[idx])),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_index(self) -> Tuple[List[tuple[int, int]], np.ndarray]:
        index: List[tuple[int, int]] = []
        stationary: List[bool] = []
        w = self.window
        for seg_idx, seg in enumerate(self._segments):
            length = len(seg["speed"])
            min_length = w.history + w.horizon
            if length < min_length:
                continue
            for anchor in range(w.history, length - w.horizon + 1, w.stride):
                if self._window_valid(seg, anchor):
                    index.append((seg_idx, anchor))
                    stationary.append(self._window_stationary(seg_idx, anchor))
        return index, np.asarray(stationary, dtype=bool)

    def _window_valid(self, seg: dict[str, np.ndarray], anchor: int) -> bool:
        w = self.window
        history_slice = slice(anchor - w.history, anchor)
        future_slice = slice(anchor, anchor + w.horizon)

        throttle_hist = seg["throttle"][history_slice]
        brake_hist = seg["brake"][history_slice]
        throttle_future = seg["throttle"][future_slice]
        brake_future = seg["brake"][future_slice]

        arrays: Iterable[np.ndarray] = (
            throttle_hist,
            brake_hist,
            throttle_future,
            brake_future,
            seg["speed"][history_slice],
            seg["speed"][future_slice],
        )

        for arr in arrays:
            if not np.all(np.isfinite(arr)):
                return False

        if np.any(throttle_hist < 0) or np.any(throttle_future < 0):
            return False
        if np.any(brake_hist < 0) or np.any(brake_future < 0):
            return False
        if np.any(throttle_hist > w.max_throttle) or np.any(throttle_future > w.max_throttle):
            return False
        if np.any(brake_hist > w.max_brake) or np.any(brake_future > w.max_brake):
            return False

        if not w.allow_overlap_actuation:
            if np.any((throttle_future > 1e-3) & (brake_future > 1e-3)):
                return False
        return True

    def _window_stationary(self, seg_idx: int, anchor: int) -> bool:
        w = self.window
        future_slice = slice(anchor, anchor + w.horizon)
        speed_future = self._segments[seg_idx]["speed"][future_slice]
        accel_future = self._accelerations[seg_idx][future_slice]
        return bool(
            np.all(np.abs(speed_future) <= STATIONARY_SPEED_EPS)
            and np.all(np.abs(accel_future) <= STATIONARY_ACCEL_EPS)
        )

    def _compute_actuation(self, throttle: np.ndarray, brake: np.ndarray) -> np.ndarray:
        throttle_component = throttle / self.window.max_throttle
        brake_component = brake / self.window.max_brake
        actuation = np.where(throttle_component > 0, throttle_component, 0.0)
        actuation -= np.where(brake_component > 0, brake_component, 0.0)
        return actuation.astype(np.float32)


@dataclass(slots=True)
class ConditionalFlowDatasetConfig:
    """Configuration for Conditional Flow dataset windows."""

    data_path: Path = Path("data/processed/synthetic/conditional_flow_large.pt")
    window: SequenceWindowConfig = field(default_factory=SequenceWindowConfig)
    state_features: Sequence[str] = ("speed",)
    max_samples: Optional[int] = None


class ConditionalFlowDataset(Dataset):
    """Wrapper producing tensors tailored for the conditional flow model."""

    def __init__(self, config: ConditionalFlowDatasetConfig) -> None:
        super().__init__()
        self.config = config
        self._base = EVSequenceDataset(
            data_path=config.data_path,
            window=config.window,
            state_features=config.state_features,
        )
        self.state_dim = len(self._base.state_features)
        self.action_dim = 2  # throttle, brake
        self.history = config.window.history
        self.horizon = config.window.horizon

    def __len__(self) -> int:
        if self.config.max_samples is not None:
            return min(len(self._base), self.config.max_samples)
        return len(self._base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self._base[idx]
        future_states = sample["future_states"]
        future_actions = sample["future_actions"]
        future = torch.cat([future_states, future_actions], dim=-1)
        history = torch.cat(
            [sample["history_states"], sample["history_actions"]], dim=-1
        )
        future_flat = torch.cat(
            [
                future_states.reshape(-1),
                future_actions.reshape(-1),
            ],
            dim=0,
        )

        return {
            "history_states": sample["history_states"],
            "history_actions": sample["history_actions"],
            "future_states": sample["future_states"],
            "future_actions": sample["future_actions"],
            "future_actuation": sample["future_actuation"],
            "history_combined": history,
            "future_combined": future,
            "future_flat": future_flat,
            "loss_weight": sample["loss_weight"],
            "is_stationary": sample["is_stationary"],
        }


@dataclass(slots=True)
class ConditionalFlowNormalizationStats:
    """Per-channel normalization constants for conditional flow data."""

    history_state_mean: torch.Tensor
    history_state_std: torch.Tensor
    history_action_mean: torch.Tensor
    history_action_std: torch.Tensor
    future_state_mean: torch.Tensor
    future_state_std: torch.Tensor
    future_action_mean: torch.Tensor
    future_action_std: torch.Tensor
    future_flat_mean: torch.Tensor
    future_flat_std: torch.Tensor
    eps: float = 1e-6

    @staticmethod
    def _reduce_stats(stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = stack.reshape(-1, stack.shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
        return mean, std

    @classmethod
    def compute(
        cls,
        dataset: ConditionalFlowDataset,
        max_samples: int = 512,
    ) -> "ConditionalFlowNormalizationStats":
        if len(dataset) == 0:
            raise ValueError("Cannot compute stats on empty dataset")
        take = min(len(dataset), max_samples)

        history_states: List[torch.Tensor] = []
        history_actions: List[torch.Tensor] = []
        future_states: List[torch.Tensor] = []
        future_actions: List[torch.Tensor] = []
        future_flat: List[torch.Tensor] = []

        for idx in range(take):
            item = dataset[idx]
            history_states.append(item["history_states"])
            history_actions.append(item["history_actions"])
            future_states.append(item["future_states"])
            future_actions.append(item["future_actions"])
            future_flat.append(item["future_flat"])

        hs = torch.stack(history_states)
        ha = torch.stack(history_actions)
        fs = torch.stack(future_states)
        fa = torch.stack(future_actions)
        ff = torch.stack(future_flat)

        hs_mean, hs_std = cls._reduce_stats(hs)
        ha_mean, ha_std = cls._reduce_stats(ha)
        fs_mean, fs_std = cls._reduce_stats(fs)
        fa_mean, fa_std = cls._reduce_stats(fa)
        ff_mean = ff.mean(dim=0)
        ff_std = ff.std(dim=0, unbiased=False).clamp_min(1e-6)

        return cls(
            history_state_mean=hs_mean,
            history_state_std=hs_std,
            history_action_mean=ha_mean,
            history_action_std=ha_std,
            future_state_mean=fs_mean,
            future_state_std=fs_std,
            future_action_mean=fa_mean,
            future_action_std=fa_std,
            future_flat_mean=ff_mean,
            future_flat_std=ff_std,
        )

    def _norm_time_series(
        self,
        tensor: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        return (tensor - mean.view(1, 1, -1)) / std.view(1, 1, -1)

    def _norm_vector(
        self,
        tensor: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        return (tensor - mean.view(1, -1)) / std.view(1, -1)

    def normalize_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            "history_states": self._norm_time_series(
                batch["history_states"], self.history_state_mean, self.history_state_std
            ),
            "history_actions": self._norm_time_series(
                batch["history_actions"], self.history_action_mean, self.history_action_std
            ),
            "future_states": self._norm_time_series(
                batch["future_states"], self.future_state_mean, self.future_state_std
            ),
            "future_actions": self._norm_time_series(
                batch["future_actions"], self.future_action_mean, self.future_action_std
            ),
            "future_flat": self._norm_vector(
                batch["future_flat"], self.future_flat_mean, self.future_flat_std
            ),
        }

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "history_state_mean": self.history_state_mean.tolist(),
            "history_state_std": self.history_state_std.tolist(),
            "history_action_mean": self.history_action_mean.tolist(),
            "history_action_std": self.history_action_std.tolist(),
            "future_state_mean": self.future_state_mean.tolist(),
            "future_state_std": self.future_state_std.tolist(),
            "future_action_mean": self.future_action_mean.tolist(),
            "future_action_std": self.future_action_std.tolist(),
            "future_flat_mean": self.future_flat_mean.tolist(),
            "future_flat_std": self.future_flat_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ConditionalFlowNormalizationStats":
        return cls(
            history_state_mean=torch.tensor(payload["history_state_mean"]),
            history_state_std=torch.tensor(payload["history_state_std"]),
            history_action_mean=torch.tensor(payload["history_action_mean"]),
            history_action_std=torch.tensor(payload["history_action_std"]),
            future_state_mean=torch.tensor(payload["future_state_mean"]),
            future_state_std=torch.tensor(payload["future_state_std"]),
            future_action_mean=torch.tensor(payload["future_action_mean"]),
            future_action_std=torch.tensor(payload["future_action_std"]),
            future_flat_mean=torch.tensor(payload["future_flat_mean"]),
            future_flat_std=torch.tensor(payload["future_flat_std"]),
        )


def collate_conditional_flow_batch(
    batch: Sequence[dict[str, torch.Tensor]],
    normalizer: Optional[ConditionalFlowNormalizationStats] = None,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    history_states = torch.stack([item["history_states"] for item in batch], dim=0)
    history_actions = torch.stack([item["history_actions"] for item in batch], dim=0)
    future_states = torch.stack([item["future_states"] for item in batch], dim=0)
    future_actions = torch.stack([item["future_actions"] for item in batch], dim=0)
    future_flat = torch.stack([item["future_flat"] for item in batch], dim=0)
    weights = torch.stack([item["loss_weight"] for item in batch], dim=0)

    result: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
        "history_states": history_states,
        "history_actions": history_actions,
        "future_states": future_states,
        "future_actions": future_actions,
        "future_flat": future_flat,
        "loss_weights": weights,
    }

    if normalizer is not None:
        result["normalized"] = normalizer.normalize_batch(
            {
                "history_states": history_states,
                "history_actions": history_actions,
                "future_states": future_states,
                "future_actions": future_actions,
                "future_flat": future_flat,
            }
        )

    return result


def summarize_conditional_flow_dataset(
    dataset: ConditionalFlowDataset,
) -> dict[str, object]:
    """Return high-level stats for documentation and manifests."""

    base = dataset._base
    summary: dict[str, object] = {
        "data_path": str(dataset.config.data_path),
        "num_segments": len(base._segments),
        "num_windows": len(dataset),
        "window": asdict(dataset.config.window),
        "state_features": list(dataset.config.state_features),
        "state_dim": dataset.state_dim,
        "action_dim": dataset.action_dim,
        "history": dataset.history,
        "horizon": dataset.horizon,
    }
    summary["stationary_fraction"] = float(base.stationary_fraction)
    return summary


def write_dataset_summary(
    dataset: ConditionalFlowDataset,
    output_path: Path,
) -> dict[str, object]:
    """Persist dataset summary to disk and return it."""

    summary = summarize_conditional_flow_dataset(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    return summary


__all__ = [
    "EVSequenceDataset",
    "SequenceWindowConfig",
    "STATIONARY_SPEED_EPS",
    "STATIONARY_ACCEL_EPS",
    "ConditionalFlowDataset",
    "ConditionalFlowDatasetConfig",
    "ConditionalFlowNormalizationStats",
    "collate_conditional_flow_batch",
    "summarize_conditional_flow_dataset",
    "write_dataset_summary",
]


