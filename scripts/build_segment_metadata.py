"""CLI to compute metadata for random segment diffusion dataset."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


def _ensure_float_array(arr: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


@dataclass
class RunningStats:
    """Numerically stable running mean/variance tracker."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_val: float = math.inf
    max_val: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        values = _ensure_float_array(values)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        chunk_count = values.size
        chunk_mean = float(values.mean())
        chunk_var = float(values.var())  # population variance
        chunk_min = float(values.min())
        chunk_max = float(values.max())

        if self.count == 0:
            self.mean = chunk_mean
            self.m2 = chunk_var * chunk_count
            self.count = chunk_count
            self.min_val = chunk_min
            self.max_val = chunk_max
            return

        total = self.count + chunk_count
        delta = chunk_mean - self.mean
        self.mean = (self.count * self.mean + chunk_count * chunk_mean) / total
        self.m2 = self.m2 + chunk_var * chunk_count + (delta ** 2) * self.count * chunk_count / total
        self.count = total
        self.min_val = min(self.min_val, chunk_min)
        self.max_val = max(self.max_val, chunk_max)

    def to_dict(self) -> dict:
        variance = self.m2 / self.count if self.count else 0.0
        return {
            "count": self.count,
            "mean": self.mean,
            "std": math.sqrt(variance),
            "min": self.min_val,
            "max": self.max_val,
        }


@dataclass
class HistogramAccumulator:
    """Accumulates histogram counts over multiple batches."""

    edges: np.ndarray

    def __post_init__(self) -> None:
        self.edges = _ensure_float_array(self.edges)
        self.counts = np.zeros(len(self.edges) - 1, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = _ensure_float_array(values)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        hist, _ = np.histogram(values, bins=self.edges)
        self.counts += hist

    def to_dict(self) -> dict:
        density = self.counts / self.counts.sum() if self.counts.sum() else self.counts
        centers = (self.edges[:-1] + self.edges[1:]) * 0.5
        return {
            "centers": centers.tolist(),
            "density": density.tolist(),
        }


def _collect_trip_files(source_root: Path) -> List[Path]:
    files: List[Path] = []
    for child in sorted(source_root.rglob("all_trips_data.pt")):
        if child.is_file():
            files.append(child)
    return files


def _reservoir_sample(buffer: List[np.ndarray], values: np.ndarray, cap: int) -> None:
    values = _ensure_float_array(values)
    values = values[np.isfinite(values)]
    if values.size == 0 or cap <= 0:
        return
    current = sum(arr.size for arr in buffer)
    remaining = cap - current
    if remaining <= 0:
        return
    if values.size <= remaining:
        buffer.append(values.astype(np.float32))
        return
    step = values.size / remaining
    idx = (np.arange(remaining) * step).astype(int)
    buffer.append(values[idx].astype(np.float32))


def _percentiles(buffer: List[np.ndarray], qs: Sequence[float]) -> Dict[str, float]:
    if not buffer:
        return {str(q): 0.0 for q in qs}
    values = np.concatenate(buffer, axis=0)
    perc = np.quantile(values, qs).tolist()
    return {f"p{int(q * 100):02d}": val for q, val in zip(qs, perc, strict=True)}


def build_metadata(
    source_root: Path,
    segment_length: int,
    stride: int,
    output_path: Path,
    max_samples: int = 200_000,
    hist_bins: int = 64,
) -> dict:
    files = _collect_trip_files(source_root)
    if not files:
        raise SystemExit(f"No trip files found under {source_root}")

    stats = {
        "speed_mps": RunningStats(),
        "grade_sin": RunningStats(),
        "throttle_pct": RunningStats(),
        "brake_pct": RunningStats(),
        "actuation_signed": RunningStats(),
    }

    hist_edges = {
        "speed_mps": np.linspace(0.0, 45.0, hist_bins + 1),
        "grade_sin": np.linspace(-0.4, 0.4, hist_bins + 1),
        "throttle_pct": np.linspace(-2.0, 102.0, hist_bins + 1),
        "brake_pct": np.linspace(-2.0, 102.0, hist_bins + 1),
        "actuation_signed": np.linspace(-1.2, 1.2, hist_bins + 1),
    }
    hists = {key: HistogramAccumulator(edges) for key, edges in hist_edges.items()}
    samples = {key: [] for key in stats.keys()}

    sample_dts: List[float] = []
    total_duration = 0.0
    raw_segments = 0
    usable_segments = 0
    window_count = 0
    per_file_counts: Dict[str, int] = {}

    for file_path in files:
        raw = torch.load(file_path, map_location="cpu")
        segment_ids = [k for k in raw.keys() if k != "metadata"]
        per_file_counts[str(file_path)] = len(segment_ids)
        for seg_id in segment_ids:
            seg = raw[seg_id]
            raw_segments += 1

            speed = np.asarray(seg["speed"], dtype=np.float32)
            throttle = np.asarray(seg["throttle"], dtype=np.float32)
            brake = np.asarray(seg["brake"], dtype=np.float32)
            angle = np.asarray(seg.get("angle", np.zeros_like(speed)), dtype=np.float32)
            grade = np.sin(angle)
            actuation = throttle / 100.0 - brake / 100.0

            for key, values in (
                ("speed_mps", speed),
                ("grade_sin", grade),
                ("throttle_pct", throttle),
                ("brake_pct", brake),
                ("actuation_signed", actuation),
            ):
                stats[key].update(values)
                hists[key].update(values)
                _reservoir_sample(samples[key], values, max_samples)

            time_arr = np.asarray(seg.get("time"), dtype=np.float32)
            if time_arr.size >= 2:
                dts = np.diff(time_arr)
                dts = dts[np.isfinite(dts)]
                if dts.size:
                    sample_dts.append(float(np.median(dts)))
                    total_duration += float(time_arr[-1] - time_arr[0])

            length = speed.shape[0]
            if length >= segment_length:
                usable_segments += 1
                steps = max(1, stride)
                window_count += 1 + max(0, (length - segment_length) // steps)

    if not sample_dts:
        raise SystemExit("Unable to infer sample time from provided trips.")
    sample_rate_hz = 1.0 / float(np.median(sample_dts))

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "source_files": per_file_counts,
        "segment_length": segment_length,
        "stride": stride,
        "sample_rate_hz": sample_rate_hz,
        "raw_segment_count": raw_segments,
        "usable_segment_count": usable_segments,
        "approx_window_count": window_count,
        "total_duration_hours": total_duration / 3600.0,
        "variables": {},
        "histograms": {},
    }

    percentiles = {key: _percentiles(samples[key], (0.01, 0.05, 0.5, 0.95, 0.99)) for key in samples}
    for key, tracker in stats.items():
        payload = tracker.to_dict()
        payload.update(percentiles[key])
        metadata["variables"][key] = payload
        metadata["histograms"][key] = hists[key].to_dict()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metadata for diffusion segment dataset.")
    parser.add_argument("--source-root", type=Path, required=True, help="Processed dataset root")
    parser.add_argument("--output", type=Path, default=Path("data/processed/segment_L/metadata.json"))
    parser.add_argument("--segment-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=32, help="Stride used to estimate total windows.")
    parser.add_argument("--max-samples", type=int, default=200_000, help="Reservoir sample size for percentiles.")
    parser.add_argument("--hist-bins", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_metadata(
        source_root=args.source_root,
        segment_length=args.segment_length,
        stride=args.stride,
        output_path=args.output,
        max_samples=args.max_samples,
        hist_bins=args.hist_bins,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()


