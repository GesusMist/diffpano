#!/usr/bin/env python3
"""Download/cache the model selected by an experiment config."""

import argparse
from pathlib import Path

from diffpano.config import load_experiment_config
from diffpano.pipelines.base import resolve_model_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the model configured for a DiffPano experiment.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if config.model.pipeline == "pixeldit":
        from diffpano.pipelines.pixeldit import _activate_official_repository

        _, modules = _activate_official_repository(
            config.pixeldit.repo_path, config.pixeldit.expected_commit
        )
        requested = config.pixeldit.model_path or config.pixeldit.checkpoint_name
        checkpoint = modules.resolve_checkpoint(requested)
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError(f"PixelDiT checkpoint not found: {checkpoint or requested}")
        print(f"PixelDiT checkpoint cached at: {Path(checkpoint).resolve()}")
        return
    source = resolve_model_source(config.model.path, config.model.id)

    if config.model.path:
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configured local model path does not exist: {path}")
        print(f"Using existing local model: {path.resolve()}")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download remote model IDs") from exc
    local_path = snapshot_download(repo_id=source, revision=config.model.revision, cache_dir=args.cache_dir)
    print(f"Model cached at: {local_path}")


if __name__ == "__main__":
    main()
