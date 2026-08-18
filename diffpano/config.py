"""Typed experiment and pixel-fusion configuration."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Set, Tuple, Union

import torch

FUSION_DTYPE = torch.float32


@dataclass
class ProjectionCache:
    """Small per-call cache for deterministic projection grids and weight maps."""

    grids: Dict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    weights: Dict[Tuple[Any, ...], torch.Tensor] = field(default_factory=dict)
    lod_maps: Dict[Tuple[Any, ...], torch.Tensor] = field(default_factory=dict)
    owner_maps: Dict[Tuple[Any, ...], "ExclusiveOwnerMap"] = field(default_factory=dict)
    saved_owner_map_keys: Set[Tuple[Any, ...]] = field(default_factory=set)


@dataclass
class PixelFusionConfig:
    pixel_fusion_enabled: bool = False
    random_seed: Optional[int] = None
    pixel_fusion_every_n_steps: int = 1
    pixel_fusion_start_ratio: float = 0.0
    pixel_fusion_end_ratio: float = 1.0

    warp_mode: str = "standard"
    aggregation_mode: str = "weighted_average"
    weight_mode: str = "distance_to_boundary"

    lpw_num_levels: int = 4
    lpw_lod_mode: str = "jacobian"
    lpw_lod_interpolation: str = "linear"
    erp_vertical_padding_mode: str = "reflect"
    erp_to_perspective_interpolation_mode: str = "bilinear"

    dpa_alpha: float = 1.0
    dpa_power: float = 1.0
    dpa_eps: float = 1e-6

    reinjection_mode: str = "noise_consistent"
    reinjection_strength: float = 1.0
    spherical_writeback_mode: str = "exclusive"
    spherical_owner_mode: str = "max_center_weight"
    exclusive_owner_map_static: bool = True
    exclusive_uncovered_mode: str = "error"
    save_owner_map: bool = False

    time_travel_enabled: bool = False
    time_travel_every_n_steps: int = 1
    time_travel_jump_length: int = 1
    time_travel_num_repeats: int = 1
    time_travel_strength: float = 1.0

    vae_chunk_size: int = 4
    save_intermediates: bool = False
    save_masks: bool = False
    save_diagnostics: bool = False
    measure_performance: bool = False
    diagnostics_dir: Optional[str] = None

    projection_chunk_size: int = 1
    vae_sample_posterior: bool = False

    # TEMPORARY DEBUG EXPORT: remove these fields with the temporary debug helpers below.
    temporary_save_fused_erp_per_step: bool = False
    temporary_fused_erp_dir: Optional[str] = None
    temporary_save_original_clean_erp_per_step: bool = False
    temporary_original_clean_erp_dir: Optional[str] = None
    projection_cache: ProjectionCache = field(default_factory=ProjectionCache)

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "projection_cache"
        }

    @classmethod
    def from_any(cls, value: Optional[Union["PixelFusionConfig", Dict[str, Any], str]]) -> "PixelFusionConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return load_pixel_fusion_config(value)
        if isinstance(value, dict):
            allowed = {field_name for field_name in cls.__dataclass_fields__ if field_name != "projection_cache"}
            return cls(**{key: _coerce_config_value(key, item) for key, item in value.items() if key in allowed})
        raise TypeError(f"Unsupported pixel fusion config type: {type(value)!r}")

    def validate(self) -> None:
        if self.warp_mode not in {"standard", "lpw"}:
            raise ValueError(f"Unsupported warp_mode={self.warp_mode!r}")
        if self.aggregation_mode not in {"average", "weighted_average", "detail_preserving_average"}:
            raise ValueError(f"Unsupported aggregation_mode={self.aggregation_mode!r}")
        if self.weight_mode not in {"uniform", "cosine", "gaussian", "distance_to_boundary"}:
            raise ValueError(f"Unsupported weight_mode={self.weight_mode!r}")
        if self.reinjection_mode not in {"noise_consistent", "replace", "weighted_replace", "residual"}:
            raise ValueError(f"Unsupported reinjection_mode={self.reinjection_mode!r}")
        if self.spherical_writeback_mode not in {"weighted_average", "exclusive"}:
            raise ValueError(f"Unsupported spherical_writeback_mode={self.spherical_writeback_mode!r}")
        if self.spherical_owner_mode != "max_center_weight":
            raise ValueError(f"Unsupported spherical_owner_mode={self.spherical_owner_mode!r}")
        if self.exclusive_uncovered_mode not in {"error", "weighted_average_fallback"}:
            raise ValueError(f"Unsupported exclusive_uncovered_mode={self.exclusive_uncovered_mode!r}")
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or not 0 <= self.random_seed <= 2**63 - 1
        ):
            raise ValueError("random_seed must be null or an integer from 0 through 2**63 - 1")
        if not 0 <= self.pixel_fusion_start_ratio <= self.pixel_fusion_end_ratio <= 1:
            raise ValueError("pixel fusion ratios must satisfy 0 <= start <= end <= 1")
        if self.pixel_fusion_every_n_steps < 1:
            raise ValueError("pixel_fusion_every_n_steps must be >= 1")
        if self.dpa_eps <= 0:
            raise ValueError("dpa_eps must be positive")
        if self.dpa_power < 0:
            raise ValueError("dpa_power must be nonnegative")
        if self.vae_chunk_size < 1:
            raise ValueError("vae_chunk_size must be >= 1")
        if self.projection_chunk_size < 1:
            raise ValueError("projection_chunk_size must be >= 1")
        if self.lpw_num_levels < 1:
            raise ValueError("lpw_num_levels must be >= 1")
        if self.lpw_lod_mode not in {"jacobian", "none"}:
            raise ValueError(f"Unsupported lpw_lod_mode={self.lpw_lod_mode!r}")
        if self.lpw_lod_interpolation not in {"linear", "nearest"}:
            raise ValueError(f"Unsupported lpw_lod_interpolation={self.lpw_lod_interpolation!r}")
        if self.erp_to_perspective_interpolation_mode not in {"bilinear", "nearest"}:
            raise ValueError(
                "Unsupported erp_to_perspective_interpolation_mode="
                f"{self.erp_to_perspective_interpolation_mode!r}"
            )

def load_pixel_fusion_config(path: str) -> PixelFusionConfig:
    """Load a YAML config through OmegaConf, matching the repository's existing config style."""

    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError("OmegaConf is required to load pixel fusion YAML configs") from exc

    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Pixel fusion config must contain a mapping: {path}")
    return PixelFusionConfig.from_any(data)


def _coerce_config_value(key: str, value: Any) -> Any:
    bool_fields = {
        "pixel_fusion_enabled",
        "time_travel_enabled",
        "save_intermediates",
        "save_masks",
        "save_diagnostics",
        "measure_performance",
        "vae_sample_posterior",
        "exclusive_owner_map_static",
        "save_owner_map",
        "temporary_save_fused_erp_per_step",
        "temporary_save_original_clean_erp_per_step",
    }
    int_fields = {
        "pixel_fusion_every_n_steps",
        "lpw_num_levels",
        "time_travel_every_n_steps",
        "time_travel_jump_length",
        "time_travel_num_repeats",
        "vae_chunk_size",
        "projection_chunk_size",
        "random_seed",
    }
    float_fields = {
        "pixel_fusion_start_ratio",
        "pixel_fusion_end_ratio",
        "dpa_alpha",
        "dpa_power",
        "dpa_eps",
        "reinjection_strength",
        "time_travel_strength",
    }
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"none", "null"}:
            return None
        if key in bool_fields:
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        if key in int_fields:
            return int(value)
        if key in float_fields:
            return float(value)
    return value


def build_pixel_fusion_config(
    pixel_fusion_config: Optional[Union[PixelFusionConfig, Dict[str, Any], str]] = None,
    pixel_fusion_config_path: Optional[str] = None,
    **overrides,
) -> PixelFusionConfig:
    config = PixelFusionConfig.from_any(pixel_fusion_config_path or pixel_fusion_config)
    valid_keys = {key for key in PixelFusionConfig.__dataclass_fields__ if key != "projection_cache"}
    for key, value in overrides.items():
        if key in valid_keys and value is not None:
            setattr(config, key, _coerce_config_value(key, value))
    config.validate()
    return config


@dataclass
class ExperimentSection:
    name: str = "diffpano-sana"
    seed: Optional[int] = 1


@dataclass
class ModelConfig:
    pipeline: str = "sana"
    id: str = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
    path: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = "bf16"
    precision: str = "bf16"
    cpu_offload: bool = False
    vae_tiling: bool = False
    vae_slicing: bool = False
    additional_pipeline_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptConfig:
    path: str = "prompts/ruins.txt"
    negative_path: Optional[str] = None


@dataclass
class OutputConfig:
    directory: str = "outputs"
    save_final: bool = True
    save_metadata: bool = True
    save_intermediate: bool = False
    run_id: Optional[str] = None
    fps: int = 24


@dataclass
class GenerationConfig:
    num_inference_steps: int = 20
    guidance_scale: float = 4.5
    height: int = 1024
    width: int = 1024
    num_frames: Optional[int] = None
    use_resolution_binning: bool = True
    additional_call_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SphereConfig:
    num_points: int = 2600
    fov: Optional[Tuple[float, float]] = None
    erp_height: int = 2048
    erp_width: int = 4096
    weighted_average_temperature: float = 0.1


@dataclass
class WarpConfig:
    mode: str = "lpw"


@dataclass
class LPWConfig:
    levels: int = 4
    lod_mode: str = "jacobian"
    lod_interpolation: str = "nearest"
    erp_vertical_padding_mode: str = "reflect"
    erp_to_perspective_interpolation_mode: str = "nearest"


@dataclass
class AggregationConfig:
    mode: str = "detail_preserving_average"
    weight_mode: str = "distance_to_boundary"
    alpha: float = 1.0
    power: float = 1.0
    epsilon: float = 1e-6


@dataclass
class TimeTravelConfig:
    enabled: bool = False
    every_n_steps: int = 1
    jump_length: int = 1
    num_repeats: int = 1
    strength: float = 1.0


@dataclass
class FusionConfig:
    enabled: bool = True
    start_ratio: float = 0.0
    end_ratio: float = 1.0
    every_n_steps: int = 1
    warp: WarpConfig = field(default_factory=WarpConfig)
    lpw: LPWConfig = field(default_factory=LPWConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    time_travel: TimeTravelConfig = field(default_factory=TimeTravelConfig)


@dataclass
class ReinjectionConfig:
    mode: str = "noise_consistent"
    strength: float = 1.0


@dataclass
class WritebackConfig:
    mode: str = "exclusive"
    owner_mode: str = "max_center_weight"
    owner_map_static: bool = True
    uncovered_mode: str = "error"


@dataclass
class PerformanceConfig:
    vae_chunk_size: int = 1
    projection_chunk_size: int = 1
    vae_sample_posterior: bool = False


@dataclass
class DebugConfig:
    enabled: bool = False
    save_predicted_x0: bool = False
    save_original_clean_erp: bool = False
    save_fused_erp: bool = False
    save_owner_map: bool = False
    save_projection_diagnostics: bool = False
    save_masks: bool = False
    measure_performance: bool = False


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    sphere: SphereConfig = field(default_factory=SphereConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    reinjection: ReinjectionConfig = field(default_factory=ReinjectionConfig)
    writeback: WritebackConfig = field(default_factory=WritebackConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    planar: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.model.pipeline not in {"sana", "flux", "hunyuan_video", "ltx_video", "planar_sana"}:
            raise ValueError(f"Unsupported model.pipeline={self.model.pipeline!r}")
        if not self.model.path and not self.model.id:
            raise ValueError("model.path or model.id must be configured")
        if self.model.precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError(f"Unsupported model.precision={self.model.precision!r}")
        if self.model.pipeline in {"hunyuan_video", "ltx_video"} and self.fusion.enabled:
            raise ValueError("fusion.enabled must be false for video adapters; pixel fusion is not implemented there")
        if self.experiment.seed is not None and (
            isinstance(self.experiment.seed, bool)
            or not isinstance(self.experiment.seed, int)
            or not 0 <= self.experiment.seed <= 2**63 - 1
        ):
            raise ValueError("experiment.seed must be null or an integer from 0 through 2**63 - 1")
        for name, value in (
            ("generation.num_inference_steps", self.generation.num_inference_steps),
            ("generation.height", self.generation.height),
            ("generation.width", self.generation.width),
            ("sphere.num_points", self.sphere.num_points),
            ("sphere.erp_height", self.sphere.erp_height),
            ("sphere.erp_width", self.sphere.erp_width),
            ("performance.vae_chunk_size", self.performance.vae_chunk_size),
            ("performance.projection_chunk_size", self.performance.projection_chunk_size),
            ("output.fps", self.output.fps),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.generation.num_frames is not None and self.generation.num_frames < 1:
            raise ValueError("generation.num_frames must be null or positive")
        if self.sphere.fov is not None:
            raise ValueError("sphere.fov is reserved; current view FOV is derived dynamically, so it must be null")
        if not 0 <= self.fusion.start_ratio <= self.fusion.end_ratio <= 1:
            raise ValueError("fusion start/end ratios must satisfy 0 <= start_ratio <= end_ratio <= 1")
        pixel_config = self.to_pixel_fusion_config()
        pixel_config.validate()

    @property
    def is_video(self) -> bool:
        return self.model.pipeline in {"hunyuan_video", "ltx_video"}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_pixel_fusion_config(self, diagnostics_dir: Optional[str] = None) -> PixelFusionConfig:
        return PixelFusionConfig(
            pixel_fusion_enabled=self.fusion.enabled,
            random_seed=self.experiment.seed,
            pixel_fusion_every_n_steps=self.fusion.every_n_steps,
            pixel_fusion_start_ratio=self.fusion.start_ratio,
            pixel_fusion_end_ratio=self.fusion.end_ratio,
            warp_mode=self.fusion.warp.mode,
            aggregation_mode=self.fusion.aggregation.mode,
            weight_mode=self.fusion.aggregation.weight_mode,
            lpw_num_levels=self.fusion.lpw.levels,
            lpw_lod_mode=self.fusion.lpw.lod_mode,
            lpw_lod_interpolation=self.fusion.lpw.lod_interpolation,
            erp_vertical_padding_mode=self.fusion.lpw.erp_vertical_padding_mode,
            erp_to_perspective_interpolation_mode=self.fusion.lpw.erp_to_perspective_interpolation_mode,
            dpa_alpha=self.fusion.aggregation.alpha,
            dpa_power=self.fusion.aggregation.power,
            dpa_eps=self.fusion.aggregation.epsilon,
            reinjection_mode=self.reinjection.mode,
            reinjection_strength=self.reinjection.strength,
            spherical_writeback_mode=self.writeback.mode,
            spherical_owner_mode=self.writeback.owner_mode,
            exclusive_owner_map_static=self.writeback.owner_map_static,
            exclusive_uncovered_mode=self.writeback.uncovered_mode,
            save_owner_map=self.debug.save_owner_map,
            time_travel_enabled=self.fusion.time_travel.enabled,
            time_travel_every_n_steps=self.fusion.time_travel.every_n_steps,
            time_travel_jump_length=self.fusion.time_travel.jump_length,
            time_travel_num_repeats=self.fusion.time_travel.num_repeats,
            time_travel_strength=self.fusion.time_travel.strength,
            vae_chunk_size=self.performance.vae_chunk_size,
            projection_chunk_size=self.performance.projection_chunk_size,
            vae_sample_posterior=self.performance.vae_sample_posterior,
            save_intermediates=self.output.save_intermediate,
            save_masks=self.debug.save_masks,
            save_diagnostics=self.debug.enabled or self.debug.save_projection_diagnostics,
            measure_performance=self.debug.measure_performance,
            diagnostics_dir=diagnostics_dir,
            temporary_save_fused_erp_per_step=self.debug.save_fused_erp,
            temporary_fused_erp_dir=diagnostics_dir,
            temporary_save_original_clean_erp_per_step=(self.debug.save_original_clean_erp or self.debug.save_predicted_x0),
            temporary_original_clean_erp_dir=diagnostics_dir,
        )


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _construct(cls: Any, value: Any, name: str, nested: Optional[Dict[str, Any]] = None) -> Any:
    data = _mapping(value, name)
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} fields: {unknown}")
    for key, nested_cls in (nested or {}).items():
        if key in data:
            data[key] = _construct(nested_cls, data[key], f"{name}.{key}")
    return cls(**data)


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load, type, and validate one complete experiment definition."""

    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError("OmegaConf is required to load experiment YAML") from exc
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    data = _mapping(raw, "config")
    allowed = set(ExperimentConfig.__dataclass_fields__)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown top-level config fields: {unknown}")
    config = ExperimentConfig(
        experiment=_construct(ExperimentSection, data.get("experiment"), "experiment"),
        model=_construct(ModelConfig, data.get("model"), "model"),
        prompt=_construct(PromptConfig, data.get("prompt"), "prompt"),
        output=_construct(OutputConfig, data.get("output"), "output"),
        generation=_construct(GenerationConfig, data.get("generation"), "generation"),
        sphere=_construct(SphereConfig, data.get("sphere"), "sphere"),
        fusion=_construct(
            FusionConfig,
            data.get("fusion"),
            "fusion",
            nested={
                "warp": WarpConfig,
                "lpw": LPWConfig,
                "aggregation": AggregationConfig,
                "time_travel": TimeTravelConfig,
            },
        ),
        reinjection=_construct(ReinjectionConfig, data.get("reinjection"), "reinjection"),
        writeback=_construct(WritebackConfig, data.get("writeback"), "writeback"),
        performance=_construct(PerformanceConfig, data.get("performance"), "performance"),
        debug=_construct(DebugConfig, data.get("debug"), "debug"),
        planar=_mapping(data.get("planar"), "planar"),
    )
    config.validate()
    return config
