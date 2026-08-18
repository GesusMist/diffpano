"""Translation layer for the original ``--config_add`` entrypoints."""

from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from diffpano.config import ExperimentConfig


PIPELINE_NAMES = {
    "SanaPipeline": "sana",
    "SphericalSanaPipeline": "sana",
    "SphericalFluxPipeline": "flux",
    "LTXPipeline": "ltx_video",
    "SphericalLTXPipeline": "ltx_video",
    "SphericalHunyuanVideoPipeline": "hunyuan_video",
    "PlanarPatchSanaPipeline": "planar_sana",
}


def _optional(value: Any) -> Any:
    return None if value in {None, "None", "null"} else value


def _moved_prompt(path: str) -> str:
    candidate = Path(path)
    if not candidate.exists() and candidate.parent.as_posix().endswith("data/prompts"):
        moved = Path("prompts") / candidate.name
        if moved.exists():
            return str(moved)
    return path


def translate_legacy_config(values: Dict[str, Any], *, video: bool) -> ExperimentConfig:
    config = ExperimentConfig()
    pipeline_name = values.get("pipeline_cls", "LTXPipeline" if video else "SanaPipeline")
    try:
        config.model.pipeline = PIPELINE_NAMES[pipeline_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported legacy pipeline_cls={pipeline_name!r}") from exc
    config.experiment.seed = None
    config.experiment.name = "legacy-live" if video else "legacy-static"
    source = values.get(
        "pretrained_model_name_or_path",
        "a-r-r-o-w/LTX-Video-0.9.1-diffusers" if video else "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    )
    if Path(source).expanduser().exists():
        config.model.path = source
        config.model.id = ""
    else:
        config.model.id = source

    if config.model.pipeline == "flux":
        config.generation.num_inference_steps = 28
        config.generation.guidance_scale = 3.5
        config.sphere.num_points = 26500
    elif config.model.pipeline == "hunyuan_video":
        config.generation.height = 720
        config.generation.width = 720
        config.generation.guidance_scale = 6.0
        config.sphere.num_points = 14400
    elif config.model.pipeline == "ltx_video":
        config.generation.height = 512
        config.generation.width = 512
        config.sphere.erp_height = 1024
        config.sphere.erp_width = 2048
    config.model.revision = _optional(values.get("revision"))
    config.model.variant = _optional(values.get("variant", None if video else "bf16"))
    config.model.precision = values.get("mixed_precision", "bf16")
    config.model.cpu_offload = bool(values.get("enable_model_cpu_offload", False))
    config.model.vae_tiling = bool(values.get("enable_vae_tiling", video))
    config.model.vae_slicing = bool(values.get("enable_vae_slicing", video))
    config.model.additional_pipeline_kwargs = dict(values.get("additional_pipeline_kwargs") or {})
    config.output.fps = int(values.get("fps", 24))
    config.fusion.enabled = False

    save_path = Path(values.get("save_path", "./outputs/test"))
    config.output.directory = str(save_path.parent)
    config.experiment.name = save_path.name or config.experiment.name

    call = dict(values.get("call_kwargs") or {})
    if "prompt_txt_path" in call:
        config.prompt.path = _moved_prompt(str(call.pop("prompt_txt_path")))
    config.prompt.negative_path = _optional(call.pop("negative_prompt_txt_path", None))
    for legacy, target in (
        ("num_inference_steps", "num_inference_steps"),
        ("guidance_scale", "guidance_scale"),
        ("height", "height"),
        ("width", "width"),
        ("num_frames", "num_frames"),
        ("use_resolution_binning", "use_resolution_binning"),
    ):
        if legacy in call:
            setattr(config.generation, target, call.pop(legacy))
    for legacy, target in (
        ("n_spherical_points", "num_points"),
        ("weighted_average_temperature", "weighted_average_temperature"),
        ("erp_height", "erp_height"),
        ("erp_width", "erp_width"),
    ):
        if legacy in call:
            setattr(config.sphere, target, call.pop(legacy))

    if config.model.pipeline == "planar_sana":
        planar_value = call.pop("planar_fusion_config", None)
        planar_path = call.pop("planar_fusion_config_path", None)
        if planar_path:
            path = Path(planar_path)
            if not path.exists() and path.name == "planar_patch_test.yaml":
                path = Path("experiments/planar/config.yaml")
            loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            planar_value = loaded.get("planar", loaded)
        config.planar = dict(planar_value or {})
    config.generation.additional_call_kwargs = call
    config.validate()
    return config


def run_legacy(values: Dict[str, Any], *, video: bool):
    config = translate_legacy_config(values, video=video)
    if config.model.pipeline == "planar_sana":
        from scripts.planar_test import run
    else:
        from scripts.generate import run
    return run(config)
