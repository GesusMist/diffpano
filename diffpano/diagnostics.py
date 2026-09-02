"""Opt-in diagnostics for the ERP-RGB loop; nothing large is saved by default."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict

import torch


def tensor_to_pil(rgb: torch.Tensor):
    """Convert one VAE-range ``[-1,1]`` RGB tensor to a PIL image."""

    from PIL import Image

    image = rgb.detach().float().cpu().clamp(-1, 1)
    image = ((image + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(image)


class DiagnosticsWriter:
    def __init__(self, directory: Path, debug_config: Any):
        self.directory = Path(directory)
        self.config = debug_config
        if debug_config.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def on_step(self, step_index: int, source: torch.Tensor, result: Any, stats: Any) -> None:
        if not self.config.enabled:
            return
        step_dir = self.directory / f"step_{step_index:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_erp_each_step:
            tensor_to_pil(source[0]).save(step_dir / "erp_before.png")
            tensor_to_pil(result.erp_rgb[0]).save(step_dir / "erp_after.png")
        tensors: Dict[str, torch.Tensor] = {}
        if self.config.save_masks:
            tensors["coverage_mask"] = result.coverage_mask.detach().cpu()
            tensors["contributor_count"] = result.contributor_count.detach().cpu()
        if self.config.save_weights:
            tensors["accumulated_weight"] = result.accumulated_weight.detach().cpu()
        if tensors:
            torch.save(tensors, step_dir / "fusion_tensors.pt")
        (step_dir / "stats.txt").write_text(
            "\n".join(f"{key}={value}" for key, value in asdict(stats).items()) + "\n",
            encoding="utf-8",
        )

    def on_cameras(self, step_index: int, cameras) -> None:
        if not self.config.enabled:
            return
        step_dir = self.directory / f"step_{step_index:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        values = [
            {
                "yaw": camera.yaw,
                "pitch": camera.pitch,
                "roll": camera.roll,
                "fov_x": camera.fov_x,
                "fov_y": camera.fov_y,
                "height": camera.height,
                "width": camera.width,
            }
            for camera in cameras
        ]
        (step_dir / "cameras.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")

    def on_view(
        self,
        step_index: int,
        view_index: int,
        before: torch.Tensor,
        after: torch.Tensor,
        contribution: Any,
    ) -> None:
        if not self.config.enabled:
            return
        if not any(
            (
                self.config.save_views_before_denoise,
                self.config.save_views_after_denoise,
                self.config.save_masks,
                self.config.save_weights,
                self.config.save_lod_maps,
            )
        ):
            return
        view_dir = self.directory / f"step_{step_index:04d}" / f"view_{view_index:03d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_views_before_denoise:
            tensor_to_pil(before[0]).save(view_dir / "before.png")
        if self.config.save_views_after_denoise:
            tensor_to_pil(after[0]).save(view_dir / "after.png")
        tensors = {}
        if self.config.save_masks:
            tensors["valid_mask"] = contribution.valid_mask.detach().cpu()
        if self.config.save_weights:
            tensors["weight"] = contribution.weight.detach().cpu()
        if self.config.save_lod_maps and contribution.lod_map is not None:
            tensors["lod_map"] = contribution.lod_map.detach().cpu()
        if tensors:
            torch.save(tensors, view_dir / "projection.pt")
