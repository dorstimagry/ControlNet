#!/usr/bin/env python3
"""Parse raw trip folders into a tensor dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.parsing import TripDatasetParser, TripParserConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse raw trips into tensors")
    parser.add_argument("--root", type=Path, default=Path("data/raw/trips"))
    parser.add_argument("--car", required=True, help="Car model identifier")
    parser.add_argument("--vehicle-id")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out-file", default="all_trips_data.pt")
    parser.add_argument(
        "--smooth-speed",
        action="store_true",
        help="Apply a weak zero-phase filter to wheel speed before saving.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--allow-all-driving-modes",
        action="store_true",
        help="Parse full trips without filtering on driving mode",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    if out_dir is None:
        vehicle_part = args.vehicle_id or "unknown_vehicle"
        out_dir = Path("processed_data") / args.car / vehicle_part

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = TripParserConfig(
        root_folder=args.root,
        car_model=args.car,
        vehicle_id=args.vehicle_id,
        dt=args.dt,
        out_dir=out_dir,
        out_file=args.out_file,
        smooth_speed=args.smooth_speed,
    )

    if args.allow_all_driving_modes:
        config.require_driving_mode = False

    parser_obj = TripDatasetParser(config)
    parser_obj.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


