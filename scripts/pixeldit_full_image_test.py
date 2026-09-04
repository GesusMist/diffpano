#!/usr/bin/env python3
"""Run the complete PixelDiT schedule on one full image through ViewDenoiser."""

import argparse
import json

from scripts.pixeldit_single_view_test import run_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pixeldit_standard_average.yaml")
    parser.add_argument("--output", default="outputs/pixeldit-full-image")
    args = parser.parse_args()
    report = run_test(
        args.config,
        args.output,
        compare_order_one=False,
        mode="full_image_no_warp",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
