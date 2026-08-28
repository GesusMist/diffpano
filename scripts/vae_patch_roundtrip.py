#!/usr/bin/env python3
"""Measure cumulative patch-wise VAE encode/decode degradation on a wide image."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import torch
from diffusers import AutoencoderDC, AutoencoderKL
from PIL import Image

from diffpano.vae import decode_view_latents, encode_view_images


@dataclass(frozen=True)
class PatchLayout:
    canvas_height: int
    canvas_width: int
    patch_height: int
    patch_width: int
    positions: Tuple[Tuple[int, int], ...]

    @property
    def num_patches(self) -> int:
        return len(self.positions)


def patch_starts(length: int, patch_size: int, stride: int) -> Tuple[int, ...]:
    if patch_size > length:
        raise ValueError(f"Patch size {patch_size} exceeds canvas length {length}")
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def build_patch_layout(
    canvas_height: int,
    canvas_width: int,
    patch_height: int,
    patch_width: int,
    stride_height: int,
    stride_width: int,
) -> PatchLayout:
    y_starts = patch_starts(canvas_height, patch_height, stride_height)
    x_starts = patch_starts(canvas_width, patch_width, stride_width)
    return PatchLayout(
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        patch_height=patch_height,
        patch_width=patch_width,
        positions=tuple((y, x) for y in y_starts for x in x_starts),
    )


def scale_patch_layout(layout: PatchLayout, scale: int) -> PatchLayout:
    return PatchLayout(
        canvas_height=layout.canvas_height * scale,
        canvas_width=layout.canvas_width * scale,
        patch_height=layout.patch_height * scale,
        patch_width=layout.patch_width * scale,
        positions=tuple((y * scale, x * scale) for y, x in layout.positions),
    )


def extract_patches(canvas: torch.Tensor, layout: PatchLayout) -> torch.Tensor:
    if canvas.ndim != 4 or canvas.shape[0] != 1:
        raise ValueError(f"Expected canvas [1,C,H,W], got {tuple(canvas.shape)}")
    return torch.cat(
        [
            canvas[..., y : y + layout.patch_height, x : x + layout.patch_width]
            for y, x in layout.positions
        ],
        dim=0,
    )


MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "sana": {
        "id": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        "vae_class": AutoencoderDC,
        "dtype": torch.bfloat16,
        "precision": "bf16",
        "vae_scale_factor": 32,
        "latent_channels": 32,
    },
    "flux": {
        "id": "black-forest-labs/FLUX.1-dev",
        "vae_class": AutoencoderKL,
        "dtype": torch.bfloat16,
        "precision": "bf16",
        "vae_scale_factor": 8,
        "latent_channels": 16,
    },
    "sd2": {
        "id": "sd2-community/stable-diffusion-2-base",
        "vae_class": AutoencoderKL,
        "dtype": torch.float16,
        "precision": "fp16",
        "vae_scale_factor": 8,
        "latent_channels": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeatedly encode and decode overlapping RGB patches with one model VAE."
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--roundtrips", type=int, choices=(1, 2, 5, 10, 20), required=True)
    parser.add_argument("--input", default="experiments/vae/image.png")
    parser.add_argument("--output-root", default="test_outputs/vae_roundtrip_stride10_0828")
    parser.add_argument("--patch-pixels", type=int, default=640)
    parser.add_argument("--stride-latents", type=int, default=10)
    parser.add_argument("--vae-chunk-size", type=int, default=4)
    parser.add_argument(
        "--no-clamp-between-roundtrips",
        action="store_true",
        help="Do not clamp decoded tensors to the valid VAE RGB range [-1, 1].",
    )
    return parser.parse_args()


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return (tensor.to(device=device) / 127.5 - 1.0).float()


def save_rgb_tensor(tensor: torch.Tensor, path: Path) -> None:
    pixels = (
        ((tensor.detach().float().cpu()[0].clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(pixels, mode="RGB").save(path)


def error_metrics(current: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    difference = current.float() - reference.float()
    mse = float(difference.square().mean().item())
    mae = float(difference.abs().mean().item())
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(4.0 / mse)
    return {"mse": mse, "mae": mae, "psnr_db": psnr}


def write_patches_directly(
    canvas: torch.Tensor,
    patches: torch.Tensor,
    layout: PatchLayout,
) -> torch.Tensor:
    """Put decoded patches back without averaging; later patches overwrite overlaps."""

    if patches.shape[0] != layout.num_patches:
        raise ValueError(f"Expected {layout.num_patches} decoded patches, got {patches.shape[0]}")
    output = canvas.clone()
    for patch, (y, x) in zip(patches, layout.positions):
        output[
            ...,
            y : y + layout.patch_height,
            x : x + layout.patch_width,
        ] = patch.unsqueeze(0).to(dtype=output.dtype)
    return output


def make_run_directory(output_root: Path, model: str, roundtrips: int) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"{model}--roundtrips-{roundtrips:02d}" / f"{timestamp}-{job_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run(args: argparse.Namespace) -> Path:
    if args.patch_pixels < 1 or args.stride_latents < 1 or args.vae_chunk_size < 1:
        raise ValueError("Patch size, latent stride, and VAE chunk size must all be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")

    source_path = Path(args.input).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    spec = MODEL_SPECS[args.model]
    scale = int(spec["vae_scale_factor"])
    if args.patch_pixels % scale:
        raise ValueError(f"Patch size {args.patch_pixels} is not divisible by VAE scale {scale}")

    device = torch.device("cuda")
    original = load_rgb_tensor(source_path, device)
    height, width = original.shape[-2:]
    if height % scale or width % scale:
        raise ValueError(f"Input size {(height, width)} is not divisible by VAE scale {scale}")
    if args.patch_pixels > height or args.patch_pixels > width:
        raise ValueError("Patch size exceeds the source image")

    patch_latents = args.patch_pixels // scale
    latent_layout = build_patch_layout(
        height // scale,
        width // scale,
        patch_latents,
        patch_latents,
        args.stride_latents,
        args.stride_latents,
    )
    rgb_layout = scale_patch_layout(latent_layout, scale)
    vae_call_config = SimpleNamespace(
        vae_chunk_size=args.vae_chunk_size,
        vae_sample_posterior=False,
    )

    run_dir = make_run_directory(Path(args.output_root), args.model, args.roundtrips)
    save_rgb_tensor(original, run_dir / "input.png")
    started_at = datetime.now().isoformat()
    (run_dir / "run.log").write_text(f"started_at={started_at}\n", encoding="utf-8")

    print(
        f"Loading {args.model} VAE from {spec['id']} at {spec['precision']} "
        f"(scale={scale}, latent_channels={spec['latent_channels']})",
        flush=True,
    )
    vae = spec["vae_class"].from_pretrained(
        spec["id"],
        subfolder="vae",
        torch_dtype=spec["dtype"],
        local_files_only=True,
    )
    vae.requires_grad_(False).eval().to(device)

    actual_latent_channels = int(getattr(vae.config, "latent_channels", -1))
    if actual_latent_channels != spec["latent_channels"]:
        raise ValueError(
            f"Expected {spec['latent_channels']} latent channels, got {actual_latent_channels}"
        )

    print(
        f"input={height}x{width} patch={args.patch_pixels}x{args.patch_pixels} "
        f"latent_patch={patch_latents}x{patch_latents} stride={args.stride_latents} "
        f"patches={latent_layout.num_patches} rounds={args.roundtrips}",
        flush=True,
    )

    current = original
    metrics = []
    total_start = time.perf_counter()
    for round_index in range(1, args.roundtrips + 1):
        round_start = time.perf_counter()
        previous = current
        rgb_patches = extract_patches(previous, rgb_layout)
        latent_patches = encode_view_images(vae, rgb_patches, vae_call_config)
        expected_shape = (patch_latents, patch_latents)
        if latent_patches.shape[-2:] != expected_shape:
            raise ValueError(
                f"Encoded latent patch size {tuple(latent_patches.shape[-2:])} "
                f"does not match expected {expected_shape}"
            )
        decoded_patches = decode_view_latents(vae, latent_patches, vae_call_config)
        current = write_patches_directly(previous, decoded_patches, rgb_layout)
        if not args.no_clamp_between_roundtrips:
            current = current.clamp(-1.0, 1.0)
        elapsed = time.perf_counter() - round_start
        item = {
            "roundtrip": round_index,
            "seconds": elapsed,
            "against_input": error_metrics(current, original),
            "against_previous": error_metrics(current, previous),
            "value_min": float(current.min().item()),
            "value_max": float(current.max().item()),
        }
        metrics.append(item)
        print(
            f"round={round_index}/{args.roundtrips} seconds={elapsed:.2f} "
            f"mse_input={item['against_input']['mse']:.8g} "
            f"psnr_input={item['against_input']['psnr_db']:.3f}",
            flush=True,
        )

    result_path = run_dir / "result.png"
    save_rgb_tensor(current, result_path)
    metadata = {
        "experiment": "vae_patch_roundtrip_stride10_0828",
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": args.model,
        "model_id": spec["id"],
        "vae_class": spec["vae_class"].__name__,
        "precision": spec["precision"],
        "vae_scale_factor": scale,
        "latent_channels": actual_latent_channels,
        "vae_scaling_factor": float(getattr(vae.config, "scaling_factor", 1.0)),
        "vae_shift_factor": getattr(vae.config, "shift_factor", None),
        "input_path": str(source_path),
        "input_height": height,
        "input_width": width,
        "roundtrips": args.roundtrips,
        "cumulative_tensor_roundtrips": True,
        "clamp_between_roundtrips": not args.no_clamp_between_roundtrips,
        "posterior_sampling": False,
        "patch_pixels": args.patch_pixels,
        "patch_latents": patch_latents,
        "stride_latents": args.stride_latents,
        "stride_pixels": args.stride_latents * scale,
        "num_patches": latent_layout.num_patches,
        "patch_positions_latent": [list(position) for position in latent_layout.positions],
        "overlap_writeback": "direct_row_major_last_patch_wins",
        "vae_chunk_size": args.vae_chunk_size,
        "total_seconds": time.perf_counter() - total_start,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "metrics_by_roundtrip": metrics,
        "result_path": str(result_path),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"completed_at={metadata['completed_at']}\nresult={result_path}\n")
    print(f"VAE roundtrip result saved to {run_dir}", flush=True)
    return run_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
