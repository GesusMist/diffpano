"""Typed configuration for the ERP-RGB DiffPano architecture."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentSection:
    name: str = "erp-rgb-sana"
    seed: int = 0


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
class PixelDiTConfig:
    repo_path: str = "third_party/PixelDiT"
    config_path: str = (
        "third_party/PixelDiT/t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml"
    )
    model_path: Optional[str] = None
    checkpoint_name: str = "pixeldit_t2i_v1.pth"
    expected_commit: str = "41f73006ae532b0b41fee72b181dc22891a5a01a"
    cfg_scale: float = 2.75
    negative_prompt: str = (
        "low quality, worst quality, over-saturated, blurry, deformed, watermark"
    )
    flow_shift: Optional[float] = None
    interval_guidance: List[float] = field(default_factory=lambda: [0.0, 1.0])
    release_text_encoder: bool = True
    record_state_statistics: bool = True


@dataclass
class PromptConfig:
    path: str = "prompts/ruins.txt"
    negative_path: Optional[str] = None


@dataclass
class GenerationConfig:
    num_inference_steps: int = 20
    guidance_scale: float = 4.5
    true_cfg_scale: float = 1.0
    batch_size: int = 1


@dataclass
class ERPConfig:
    height: int = 2048
    width: int = 4096


@dataclass
class ViewConfig:
    height: int = 1024
    width: int = 1024
    fov_x: float = 80.0
    fov_y: float = 80.0


@dataclass
class RotationConfig:
    enabled: bool = False
    max_yaw_deg: float = 6.0
    max_pitch_deg: float = 3.0
    max_roll_deg: float = 2.0


@dataclass
class SamplingConfig:
    strategy: str = "spherediff_fixed"
    rotation: RotationConfig = field(default_factory=RotationConfig)


@dataclass
class InitializationConfig:
    mode: str = "erp_rgb_noise"
    distribution: str = "gaussian"
    mean: float = 0.0
    std: float = 1.0
    clamp: bool = False
    clamp_min: float = -1.0
    clamp_max: float = 1.0


@dataclass
class SamplingDirectionConfig:
    interpolation: str = "nearest"


@dataclass
class LPWConfig:
    levels: int = 4
    lod_mode: str = "jacobian"
    lod_interpolation: str = "nearest"
    vertical_padding_mode: str = "reflect"


@dataclass
class WarpConfig:
    mode: str = "lpw"
    erp_to_perspective: SamplingDirectionConfig = field(
        default_factory=lambda: SamplingDirectionConfig("nearest")
    )
    perspective_to_erp: SamplingDirectionConfig = field(
        default_factory=lambda: SamplingDirectionConfig("bilinear")
    )
    lpw: LPWConfig = field(default_factory=LPWConfig)


@dataclass
class FusionConfig:
    mode: str = "detail_preserving_average"
    weight_mode: str = "distance_to_boundary"
    spherediff_temperature: float = 0.1
    alpha: float = 1.0
    power: float = 1.0
    epsilon: float = 1.0e-6
    uncovered_mode: str = "keep_previous"


@dataclass
class PerformanceConfig:
    view_batch_size: int = 1
    vae_chunk_size: int = 1
    projection_cache_max_entries: Optional[int] = None
    projection_cache_cpu_fallback: bool = False


@dataclass
class OutputConfig:
    directory: str = "outputs"
    group: Optional[str] = None
    save_final: bool = True
    save_metadata: bool = True
    save_intermediate: bool = False
    run_id: Optional[str] = None


@dataclass
class DebugConfig:
    enabled: bool = False
    save_views_before_denoise: bool = False
    save_views_after_denoise: bool = False
    save_erp_each_step: bool = False
    save_masks: bool = False
    save_step_indices: List[int] = field(default_factory=list)
    save_view_indices: List[int] = field(default_factory=list)
    save_weights: bool = False
    save_lod_maps: bool = False
    measure_performance: bool = False


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    model: ModelConfig = field(default_factory=ModelConfig)
    pixeldit: PixelDiTConfig = field(default_factory=PixelDiTConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    erp: ERPConfig = field(default_factory=ERPConfig)
    view: ViewConfig = field(default_factory=ViewConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    warp: WarpConfig = field(default_factory=WarpConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def validate(self) -> None:
        if self.model.pipeline not in {"sana", "flux", "sd2", "pixeldit"}:
            raise ValueError("model.pipeline must be 'sana', 'flux', 'sd2', or 'pixeldit'")
        if self.model.pipeline != "pixeldit" and not self.model.path and not self.model.id:
            raise ValueError("model.path or model.id must be configured")
        if self.model.precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError("model.precision must be fp16, bf16, or fp32")
        if self.sampling.strategy not in {"spherediff_fixed", "spherediff_rotated"}:
            raise ValueError("sampling.strategy must be spherediff_fixed or spherediff_rotated")
        if self.initialization.mode not in {
            "erp_rgb_noise", "latent_native_bootstrap", "pixel_gaussian"
        }:
            raise ValueError("unsupported initialization.mode")
        if self.model.pipeline == "pixeldit" and self.initialization.mode != "pixel_gaussian":
            raise ValueError("PixelDiT requires initialization.mode=pixel_gaussian")
        if self.model.pipeline != "pixeldit" and self.initialization.mode == "pixel_gaussian":
            raise ValueError("initialization.mode=pixel_gaussian is reserved for PixelDiT")
        if self.initialization.mode == "pixel_gaussian" and self.initialization.clamp:
            raise ValueError("PixelDiT Gaussian state initialization must not be clamped")
        if self.initialization.distribution != "gaussian":
            raise ValueError("only gaussian initialization is currently supported")
        if self.initialization.std < 0:
            raise ValueError("initialization.std must be nonnegative")
        if self.initialization.clamp_min >= self.initialization.clamp_max:
            raise ValueError("initialization clamp_min must be less than clamp_max")
        if self.warp.mode not in {"standard", "lpw"}:
            raise ValueError("warp.mode must be standard or lpw")
        for name, direction in (
            ("warp.erp_to_perspective", self.warp.erp_to_perspective),
            ("warp.perspective_to_erp", self.warp.perspective_to_erp),
        ):
            if direction.interpolation not in {"nearest", "bilinear"}:
                raise ValueError(f"{name}.interpolation must be nearest or bilinear")
        if self.warp.lpw.lod_mode not in {"jacobian", "none"}:
            raise ValueError("warp.lpw.lod_mode must be jacobian or none")
        if self.warp.lpw.lod_interpolation not in {"nearest", "linear"}:
            raise ValueError("warp.lpw.lod_interpolation must be nearest or linear")
        if self.warp.lpw.vertical_padding_mode not in {"reflect", "replicate"}:
            raise ValueError("warp.lpw.vertical_padding_mode must be reflect or replicate")
        if self.fusion.mode not in {"average", "weighted_average", "detail_preserving_average"}:
            raise ValueError("unsupported fusion.mode")
        if self.fusion.weight_mode not in {
            "uniform", "cosine", "gaussian", "distance_to_boundary", "spherediff_center"
        }:
            raise ValueError("unsupported fusion.weight_mode")
        if self.fusion.uncovered_mode != "keep_previous":
            raise ValueError("fusion.uncovered_mode currently must be keep_previous")
        positive = {
            "generation.num_inference_steps": self.generation.num_inference_steps,
            "generation.batch_size": self.generation.batch_size,
            "erp.height": self.erp.height,
            "erp.width": self.erp.width,
            "view.height": self.view.height,
            "view.width": self.view.width,
            "performance.view_batch_size": self.performance.view_batch_size,
            "performance.vae_chunk_size": self.performance.vae_chunk_size,
            "warp.lpw.levels": self.warp.lpw.levels,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.erp.width % 2:
            raise ValueError("erp.width must be even for exact pole reflection")
        if not 0 < self.view.fov_x < 180 or not 0 < self.view.fov_y < 180:
            raise ValueError("view FOVs must be between 0 and 180 degrees")
        if self.fusion.epsilon <= 0 or self.fusion.power < 0:
            raise ValueError("fusion epsilon must be positive and power nonnegative")
        if self.fusion.spherediff_temperature <= 0:
            raise ValueError("fusion.spherediff_temperature must be positive")
        if self.pixeldit.cfg_scale <= 0:
            raise ValueError("pixeldit.cfg_scale must be positive")
        if self.pixeldit.flow_shift is not None and self.pixeldit.flow_shift <= 0:
            raise ValueError("pixeldit.flow_shift must be positive when overridden")
        if (
            len(self.pixeldit.interval_guidance) != 2
            or not 0 <= self.pixeldit.interval_guidance[0]
            <= self.pixeldit.interval_guidance[1] <= 1
        ):
            raise ValueError("pixeldit.interval_guidance must be [start, end] within [0,1]")
        if (
            self.performance.projection_cache_max_entries is not None
            and self.performance.projection_cache_max_entries < 0
        ):
            raise ValueError("performance.projection_cache_max_entries must be nonnegative or null")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _construct(cls: Any, value: Any, name: str, nested: Optional[Dict[str, Any]] = None) -> Any:
    data = _mapping(value, name)
    unknown = sorted(set(data) - set(cls.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unknown {name} fields: {unknown}")
    for key, nested_cls in (nested or {}).items():
        if key in data:
            data[key] = _construct(nested_cls, data[key], f"{name}.{key}")
    return cls(**data)


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load a complete YAML experiment and reject stale/unknown fields."""

    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError("OmegaConf is required to load experiment YAML") from exc
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    data = _mapping(raw, "config")
    unknown = sorted(set(data) - set(ExperimentConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unknown top-level config fields: {unknown}")
    config = ExperimentConfig(
        experiment=_construct(ExperimentSection, data.get("experiment"), "experiment"),
        model=_construct(ModelConfig, data.get("model"), "model"),
        pixeldit=_construct(PixelDiTConfig, data.get("pixeldit"), "pixeldit"),
        prompt=_construct(PromptConfig, data.get("prompt"), "prompt"),
        generation=_construct(GenerationConfig, data.get("generation"), "generation"),
        erp=_construct(ERPConfig, data.get("erp"), "erp"),
        view=_construct(ViewConfig, data.get("view"), "view"),
        sampling=_construct(
            SamplingConfig, data.get("sampling"), "sampling", nested={"rotation": RotationConfig}
        ),
        initialization=_construct(
            InitializationConfig, data.get("initialization"), "initialization"
        ),
        warp=_construct(
            WarpConfig,
            data.get("warp"),
            "warp",
            nested={
                "erp_to_perspective": SamplingDirectionConfig,
                "perspective_to_erp": SamplingDirectionConfig,
                "lpw": LPWConfig,
            },
        ),
        fusion=_construct(FusionConfig, data.get("fusion"), "fusion"),
        performance=_construct(PerformanceConfig, data.get("performance"), "performance"),
        output=_construct(OutputConfig, data.get("output"), "output"),
        debug=_construct(DebugConfig, data.get("debug"), "debug"),
    )
    config.validate()
    return config
