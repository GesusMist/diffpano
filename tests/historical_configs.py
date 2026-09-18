"""Config-only fixtures after the user-requested G/H/I/J sidecar cleanup."""
from pathlib import Path


def retained_reference(spec):
    if Path(spec['reference']).exists():
        return spec
    # The retained generation metadata contains the actual resolved config.
    # This helper is for config tests only, never runtime/provenance replay.
    path=Path(spec['reference_generation'])/'metadata.json'
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(spec,reference=str(path))
