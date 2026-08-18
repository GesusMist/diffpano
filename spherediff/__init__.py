"""SphereDiff baseline compatibility namespace.

The current spherical adapters reproduce the baseline path when pixel fusion is disabled.
"""

from diffpano.geometry import SphericalFunctions
from diffpano.pipelines import (
    SphericalFluxPipeline,
    SphericalHunyuanVideoPipeline,
    SphericalLTXPipeline,
    SphericalSanaPipeline,
)

__all__ = [
    "SphericalFunctions",
    "SphericalFluxPipeline",
    "SphericalHunyuanVideoPipeline",
    "SphericalLTXPipeline",
    "SphericalSanaPipeline",
]
