"""Lazy adapter for NVIDIA PixelDiT text-to-image pixel-space flow inference."""

import gc
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence, Tuple

import torch

from diffpano.camera import PerspectiveCamera
from diffpano.conditioning import camera_prompt_indices, expand_directional_prompts
from diffpano.pipelines.base import ViewDenoiser
from diffpano.pipelines.pixeldit_solver import PixelDiTFirstOrderSolver


@dataclass
class PixelDiTPromptBank:
    prompt_directions: torch.Tensor
    positive: torch.Tensor
    positive_mask: torch.Tensor
    negative: torch.Tensor
    negative_mask: torch.Tensor


@dataclass
class PixelDiTConditioning:
    positive: torch.Tensor
    positive_mask: torch.Tensor
    negative: torch.Tensor
    negative_mask: torch.Tensor


def _activate_official_repository(repo_path: str, expected_commit: str) -> Tuple[Path, Any]:
    repository = Path(repo_path).expanduser().resolve()
    required = repository / "t2i" / "diffusion" / "model" / "trainer.py"
    if not required.is_file():
        raise FileNotFoundError(
            f"Official PixelDiT checkout not found at {repository}. "
            "Run scripts/setup_pixeldit.sh first."
        )
    if expected_commit:
        try:
            actual = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Cannot verify PixelDiT checkout at {repository}") from exc
        if actual != expected_commit:
            raise RuntimeError(
                f"PixelDiT checkout is {actual}, expected pinned commit {expected_commit}"
            )
    for directory in (repository, repository / "t2i"):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)
    try:
        from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
        from diffusion.model.flow_dpm import DPMS
        from diffusion.utils.config import PixDiTConfig, model_init_config
        from tools.download import resolve_checkpoint
    except ImportError as exc:
        raise ImportError(
            "PixelDiT dependencies are unavailable. Install requirements-pixeldit.txt."
        ) from exc
    modules = SimpleNamespace(
        build_model=build_model,
        get_tokenizer_and_text_encoder=get_tokenizer_and_text_encoder,
        DPMS=DPMS,
        PixDiTConfig=PixDiTConfig,
        model_init_config=model_init_config,
        resolve_checkpoint=resolve_checkpoint,
    )
    return repository, modules


def _load_official_config(modules: Any, config_path: str) -> Any:
    import pyrallis
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Official PixelDiT config not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return pyrallis.load(modules.PixDiTConfig, stream)


def _load_official_model(modules: Any, official_config: Any, backend_config: Any) -> Tuple[Any, str]:
    reference_size = int(official_config.model.image_size)
    model_kwargs = modules.model_init_config(official_config, reference_size)
    model = modules.build_model(
        official_config.model.model,
        use_fp32_attention=official_config.model.get("fp32_attention", False),
        **model_kwargs,
    )
    requested = backend_config.model_path or backend_config.checkpoint_name
    checkpoint_path = modules.resolve_checkpoint(requested)
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"PixelDiT checkpoint not found: {checkpoint_path or requested}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if str(checkpoint_path).endswith(".bin"):
        checkpoint = {"state_dict": checkpoint}
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("PixelDiT checkpoint must contain a state_dict entry")
    state = dict(checkpoint["state_dict"])
    state.pop("pos_embed", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"PixelDiT missing checkpoint keys: {missing}")
    if unexpected:
        print(f"PixelDiT unexpected checkpoint keys: {unexpected}")
    model.eval()
    return model, str(Path(checkpoint_path).resolve())


class PixelDiTViewDenoiser(ViewDenoiser):
    """Advance a three-channel image state with one official PixelDiT flow evaluation."""

    def __init__(
        self,
        model: Any,
        official_config: Any,
        official_modules: Any,
        *,
        cfg_scale: float,
        negative_prompt: str,
        flow_shift: Optional[float],
        interval_guidance: Sequence[float],
        release_text_encoder: bool,
        record_state_statistics: bool,
        checkpoint_path: str = "",
        official_commit: str = "",
        measure_performance: bool = False,
    ):
        self.model = model
        self.official_config = official_config
        self.official_modules = official_modules
        self.cfg_scale = float(cfg_scale)
        self.negative_prompt = negative_prompt
        self.interval_guidance = tuple(float(value) for value in interval_guidance)
        self.release_text_encoder = release_text_encoder
        self.state_diagnostics_enabled = record_state_statistics
        self.checkpoint_path = checkpoint_path
        self.official_commit = official_commit
        self.measure_performance = measure_performance
        selected_shift = official_config.scheduler.flow_shift if flow_shift is None else flow_shift
        self.solver = PixelDiTFirstOrderSolver(float(selected_shift))
        self.last_timings = {}
        self.last_model_prediction = None
        self._view_size = (0, 0)

    @classmethod
    def from_pretrained(
        cls,
        backend_config: Any,
        *,
        measure_performance: bool = False,
    ) -> "PixelDiTViewDenoiser":
        _, modules = _activate_official_repository(
            backend_config.repo_path, backend_config.expected_commit
        )
        official_config = _load_official_config(modules, backend_config.config_path)
        model, checkpoint_path = _load_official_model(modules, official_config, backend_config)
        return cls(
            model,
            official_config,
            modules,
            cfg_scale=backend_config.cfg_scale,
            negative_prompt=backend_config.negative_prompt,
            flow_shift=backend_config.flow_shift,
            interval_guidance=backend_config.interval_guidance,
            release_text_encoder=backend_config.release_text_encoder,
            record_state_statistics=backend_config.record_state_statistics,
            checkpoint_path=checkpoint_path,
            official_commit=backend_config.expected_commit,
            measure_performance=measure_performance,
        )

    def _timed(self, name: str, operation):
        if self.measure_performance and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        output = operation()
        if self.measure_performance and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_timings[name] = self.last_timings.get(name, 0.0) + time.perf_counter() - started
        return output

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def timesteps(self) -> torch.Tensor:
        return self.solver.timesteps

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def enable_model_cpu_offload(self) -> None:
        raise NotImplementedError("PixelDiT model CPU offload is not implemented")

    def prepare(self, *, num_steps: int, view_height: int, view_width: int) -> None:
        patch_size = int(self.official_config.model.extra.get("patch_size", 16))
        if view_height % patch_size or view_width % patch_size:
            raise ValueError(f"PixelDiT view dimensions must be divisible by {patch_size}")
        self._view_size = (view_height, view_width)
        self.solver.prepare(num_steps, device=self.device)

    @torch.no_grad()
    def prepare_prompt_conditioning(
        self, prompts: Sequence[str], negative_prompt: str = ""
    ) -> PixelDiTPromptBank:
        directional = expand_directional_prompts(prompts)
        tokenizer, text_encoder = self.official_modules.get_tokenizer_and_text_encoder(
            name=self.official_config.text_encoder.text_encoder_name,
            device=self.device,
        )
        maximum = int(self.official_config.text_encoder.model_max_length)
        chi_lines = self.official_config.text_encoder.chi_prompt
        if chi_lines:
            chi_prompt = "\n".join(chi_lines)
            prompt_values = [chi_prompt + prompt for prompt in directional.prompts]
            maximum_all = len(tokenizer.encode(chi_prompt)) + maximum - 2
        else:
            prompt_values = list(directional.prompts)
            maximum_all = maximum
        positive_tokens = tokenizer(
            prompt_values,
            max_length=maximum_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        select_index = [0] + list(range(-maximum + 1, 0))
        positive = text_encoder(
            positive_tokens.input_ids, positive_tokens.attention_mask
        )[0][:, None][:, :, select_index]
        positive_mask = positive_tokens.attention_mask[:, select_index]

        effective_negative = negative_prompt or self.negative_prompt
        negative_tokens = tokenizer(
            effective_negative,
            max_length=maximum,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        negative = text_encoder(
            negative_tokens.input_ids, negative_tokens.attention_mask
        )[0][:, None]
        bank = PixelDiTPromptBank(
            directional.directions,
            positive,
            positive_mask,
            negative,
            negative_tokens.attention_mask,
        )
        if self.release_text_encoder:
            del text_encoder
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return bank

    def conditioning_for_cameras(
        self,
        prepared_conditioning: PixelDiTPromptBank,
        cameras: Sequence[PerspectiveCamera],
        *,
        batch_size: int,
    ) -> PixelDiTConditioning:
        indices = camera_prompt_indices(cameras, prepared_conditioning.prompt_directions)
        indices = indices.repeat_interleave(batch_size).to(device=self.device)
        positive = prepared_conditioning.positive[indices]
        positive_mask = prepared_conditioning.positive_mask[indices]
        negative = prepared_conditioning.negative.expand(positive.shape[0], -1, -1, -1)
        negative_mask = prepared_conditioning.negative_mask.expand(positive.shape[0], -1)
        return PixelDiTConditioning(positive, positive_mask, negative, negative_mask)

    @staticmethod
    def image_metadata(
        state: torch.Tensor, *, double_batch: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = state.shape
        if double_batch:
            batch *= 2
        dimensions = torch.tensor(
            [float(height), float(width)], device=state.device, dtype=torch.float32
        ).view(1, 2).repeat(batch, 1)
        aspect_ratio = (dimensions[:, 0] / dimensions[:, 1]).unsqueeze(1)
        return dimensions, aspect_ratio


    def _official_solver(
        self, state: torch.Tensor, conditioning: PixelDiTConditioning
    ) -> Any:
        dimensions, aspect_ratio = self.image_metadata(state)
        return self.official_modules.DPMS(
            self.model.forward_with_dpmsolver,
            condition=conditioning.positive,
            uncondition=conditioning.negative,
            guidance_type="classifier-free",
            cfg_scale=self.cfg_scale,
            model_type="flow",
            model_kwargs={
                "data_info": {"img_hw": dimensions, "aspect_ratio": aspect_ratio},
                "mask": conditioning.positive_mask,
            },
            schedule="FLOW",
            interval_guidance=list(self.interval_guidance),
        )

    @torch.no_grad()
    def denoise_step(
        self, rgb_view: torch.Tensor, timestep: Any, conditioning: PixelDiTConditioning
    ) -> torch.Tensor:
        if rgb_view.ndim != 4 or rgb_view.shape[1] != 3:
            raise ValueError("PixelDiT state must have shape [B,3,H,W]")
        self.last_timings = {}
        current, following = self.solver.bounds_for(timestep)
        state = rgb_view.float()
        reference = self._official_solver(state, conditioning)
        clean_prediction = self._timed(
            "model_forward",
            lambda: reference.model_fn(state, current),
        )
        current_value = current.to(device=state.device, dtype=state.dtype)
        self.last_model_prediction = ((state - clean_prediction) / current_value).detach()
        return self._timed(
            "pixel_solver_step",
            lambda: reference.dpm_solver_first_update(
                state, current, following, model_s=clean_prediction
            ),
        )

    def sample_fixed_noise(
        self, *, batch_size: int, height: int, width: int, generator
    ) -> torch.Tensor:
        return torch.randn(
            batch_size,
            3,
            height,
            width,
            generator=generator,
            device=generator.device,
            dtype=torch.float32,
        )

    def make_initial_noisy_state(
        self, fixed_noise: torch.Tensor, timestep: Any
    ) -> torch.Tensor:
        self.solver.bounds_for(timestep)
        return fixed_noise.to(device=self.device, dtype=torch.float32)

    def encode_clean(self, rgb_clean: torch.Tensor) -> torch.Tensor:
        return rgb_clean.to(device=self.device, dtype=torch.float32)

    def add_fixed_noise(
        self, clean_state: torch.Tensor, fixed_noise: torch.Tensor, timestep: Any
    ) -> torch.Tensor:
        current, _ = self.solver.bounds_for(timestep)
        current = current.to(device=clean_state.device, dtype=clean_state.dtype)
        return (1.0 - current) * clean_state + current * fixed_noise.to(
            clean_state
        )

    def predict_clean_native(
        self,
        noisy_state: torch.Tensor,
        timestep: Any,
        conditioning: PixelDiTConditioning,
    ) -> torch.Tensor:
        if noisy_state.ndim != 4 or noisy_state.shape[1] != 3:
            raise ValueError("PixelDiT state must have shape [B,3,H,W]")
        self.last_timings = {}
        current, _ = self.solver.bounds_for(timestep)
        reference = self._official_solver(noisy_state, conditioning)
        predicted_clean = self._timed(
            "model_forward",
            lambda: reference.model_fn(noisy_state.float(), current),
        )
        self.last_model_prediction = predicted_clean.detach()
        return predicted_clean.float()

    def decode_clean(self, clean_state: torch.Tensor) -> torch.Tensor:
        return clean_state.to(device=self.device, dtype=torch.float32)

    def sample_native_rgb(self, *, batch_size: int, height: int, width: int, generator):
        del batch_size, height, width, generator
        raise RuntimeError("PixelDiT requires initialization.mode=pixel_gaussian")

    @torch.no_grad()
    def official_sample(
        self,
        initial_state: torch.Tensor,
        conditioning: PixelDiTConditioning,
        *,
        order: int = 2,
    ) -> torch.Tensor:
        """Run the unmodified official sampler for standalone image comparison only."""

        if order not in {1, 2}:
            raise ValueError("Official PixelDiT reference order must be 1 or 2")
        reference = self._official_solver(initial_state, conditioning)
        return reference.sample(
            initial_state,
            steps=len(self.timesteps),
            order=order,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=self.solver.flow_shift,
        )

    def official_order_one_sample(
        self, initial_state: torch.Tensor, conditioning: PixelDiTConditioning
    ) -> torch.Tensor:
        return self.official_sample(initial_state, conditioning, order=1)
