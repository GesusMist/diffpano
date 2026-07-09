#!/usr/bin/env python3
"""Download SphereDiff Hugging Face model snapshots into the configured cache."""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


MODEL_REPOS = {
    "SANA": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    "FLUX": "black-forest-labs/FLUX.1-dev",
    "LTX": "a-r-r-o-w/LTX-Video-0.9.1-diffusers",
    "HUNYUAN": "hunyuanvideo-community/HunyuanVideo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        default=["SANA"],
        help="Model aliases to download: SANA, FLUX, LTX, HUNYUAN, or ALL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = [model.upper() for model in args.models]
    if "ALL" in requested:
        requested = list(MODEL_REPOS)

    cache_home = os.environ.get("HF_HOME", "default Hugging Face cache")
    print(f"HF_HOME={cache_home}")

    for model in requested:
        if model not in MODEL_REPOS:
            valid = ", ".join([*MODEL_REPOS, "ALL"])
            raise SystemExit(f"Unknown model alias '{model}'. Valid choices: {valid}")

        repo_id = MODEL_REPOS[model]
        print(f"Downloading {model}: {repo_id}")
        path = snapshot_download(repo_id=repo_id, resume_download=True)
        print(f"Cached at: {path}")


if __name__ == "__main__":
    main()
