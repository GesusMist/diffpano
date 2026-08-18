"""Deprecated compatibility entrypoint; use ``python scripts/generate.py --config ...``."""

import argparse
import warnings

from tools_mpark.dictaction import DictAction

from scripts.legacy import run_legacy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_add", action=DictAction, default=dict(), nargs="*")
    args = parser.parse_args()
    warnings.warn(
        "--config_add is deprecated; migrate to scripts/generate.py --config config.yaml",
        DeprecationWarning,
        stacklevel=2,
    )
    run_legacy(args.config_add, video=True)


if __name__ == "__main__":
    main()
