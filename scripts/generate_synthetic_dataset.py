#!/usr/bin/env python3
"""Generate a toy EV trajectory dataset with simple, controllable dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic dataset for diffusion sanity checks.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output .pt file.")
    parser.add_argument("--num-segments", type=int, default=16, help="Number of synthetic trips to generate.")
    parser.add_argument("--segment-length", type=int, default=2048, help="Number of samples per trip.")
    parser.add_argument("--sample-time", type=float, default=0.1, help="Sampling interval (seconds).")
    parser.add_argument("--noise-std", type=float, default=0.02, help="Std-dev for Gaussian noise on controls.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def synthesize_segment(length: int, dt: float, noise_std: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    time = np.arange(length, dtype=np.float32) * dt
    throttle = np.zeros(length, dtype=np.float32)
    brake = np.zeros(length, dtype=np.float32)

    # Piecewise-constant throttle profile with occasional braking
    window = length // 16
    for idx in range(16):
        start = idx * window
        end = length if idx == 15 else (idx + 1) * window
        base = 0.2 + 0.6 * np.sin(0.3 * idx)
        throttle[start:end] = base
        if idx % 5 == 4:
            brake[start:end] = 0.3 + 0.2 * rng.random()

    throttle += rng.normal(0.0, noise_std, size=length).astype(np.float32)
    brake += rng.normal(0.0, noise_std, size=length).astype(np.float32)
    throttle = np.clip(throttle, 0.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)

    speed = np.zeros(length, dtype=np.float32)
    accel = np.zeros(length, dtype=np.float32)
    damping = 0.05
    for t in range(1, length):
        input_acc = throttle[t - 1] - brake[t - 1]
        accel[t - 1] = input_acc - damping * speed[t - 1]
        speed[t] = speed[t - 1] + dt * accel[t - 1]
    accel[-1] = accel[-2]
    angle = np.zeros(length, dtype=np.float32)

    segment = {
        "time": time,
        "speed": speed,
        "throttle": throttle * 100.0,
        "brake": brake * 100.0,
        "angle": angle,
        "acceleration": accel,
        "driving_mode": np.zeros(length, dtype=np.float32),
    }
    return segment


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    payload: dict[str, dict[str, np.ndarray]] = {"metadata": {"dt": args.sample_time, "num_valid_trips": args.num_segments}}
    for idx in range(args.num_segments):
        key = f"synthetic_trip_{idx:03d}"
        payload[key] = synthesize_segment(args.segment_length, args.sample_time, args.noise_std, rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"[synthetic] wrote dataset to {args.output} ({args.num_segments} segments, length={args.segment_length})")


if __name__ == "__main__":
    main()

