#!/usr/bin/env python3
"""Print compact projection diagnostics from a saved tensor payload."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.path, map_location="cpu", weights_only=True)
    print(f"diagnostics_file={args.path}")
    for key in (
        "erp_grid_min",
        "erp_grid_max",
        "erp_pixel_center_max_error",
        "perspective_round_trip_mean_error",
        "perspective_round_trip_max_error",
    ):
        if key in payload:
            print(f"{key}={payload[key].tolist()}")


if __name__ == "__main__":
    main()
