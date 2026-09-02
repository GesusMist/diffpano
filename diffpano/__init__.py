"""DiffPano: synchronous panorama diffusion on a persistent ERP RGB canvas."""

from diffpano.config import ExperimentConfig, load_experiment_config
from diffpano.erp_pipeline import ERPRGBPipeline

__all__ = ["ERPRGBPipeline", "ExperimentConfig", "load_experiment_config"]
__version__ = "0.2.0"
