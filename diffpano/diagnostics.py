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
        if self.config.save_step_indices and step_index not in self.config.save_step_indices:
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
        if self.config.save_step_indices and step_index not in self.config.save_step_indices:
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
        if self.config.save_step_indices and step_index not in self.config.save_step_indices:
            return
        if self.config.save_view_indices and view_index not in self.config.save_view_indices:
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

    def on_clean_consensus_view(
        self,
        step_index: int,
        view_index: int,
        clean_view: torch.Tensor,
        predicted_clean: torch.Tensor,
        contribution: Any,
    ) -> None:
        """Save only clean RGB images from the x0-consensus path."""

        if not self.config.enabled:
            return
        if self.config.save_step_indices and step_index not in self.config.save_step_indices:
            return
        if self.config.save_view_indices and view_index not in self.config.save_view_indices:
            return
        view_dir = self.directory / f"step_{step_index:04d}" / f"view_{view_index:03d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        if clean_view is not None and self.config.save_views_before_denoise:
            tensor_to_pil(clean_view[0]).save(view_dir / "clean_view.png")
        if self.config.save_views_after_denoise:
            tensor_to_pil(predicted_clean[0]).save(
                view_dir / "predicted_clean_view.png"
            )
            tensor_to_pil(contribution.rgb[0]).save(
                view_dir / "predicted_clean_erp_contribution.png"
            )
        tensors = {}
        if self.config.save_masks:
            tensors["predicted_clean_valid_mask"] = contribution.valid_mask.detach().cpu()
        if self.config.save_weights:
            tensors["predicted_clean_weight"] = contribution.weight.detach().cpu()
        if self.config.save_lod_maps and contribution.lod_map is not None:
            tensors["predicted_clean_lod_map"] = contribution.lod_map.detach().cpu()
        if tensors:
            torch.save(tensors, view_dir / "predicted_clean_projection.pt")

    def on_clean_consensus_step(
        self, step_index: int, source: Any, result: Any, stats: Any
    ) -> None:
        """Save explicitly named clean ERP snapshots without changing old diagnostics."""

        if not self.config.enabled:
            return
        if self.config.save_step_indices and step_index not in self.config.save_step_indices:
            return
        step_dir = self.directory / f"step_{step_index:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_erp_each_step:
            if source is not None:
                tensor_to_pil(source[0]).save(step_dir / "clean_erp_before.png")
            tensor_to_pil(result.erp_rgb[0]).save(step_dir / "fused_clean_erp.png")
        (step_dir / "clean_consensus_stats.txt").write_text(
            "\n".join(f"{key}={value}" for key, value in asdict(stats).items())
            + "\n",
            encoding="utf-8",
        )

    def on_planar_patch(
        self,
        step_index: int,
        patch_index: int,
        before: Any,
        after: torch.Tensor,
    ) -> None:
        """Save direct planar crops without projection-shaped diagnostics."""

        if not self.config.enabled:
            return
        if self.config.save_step_indices and step_index not in self.config.save_step_indices:
            return
        if self.config.save_view_indices and patch_index not in self.config.save_view_indices:
            return
        if not (
            self.config.save_views_before_denoise
            or self.config.save_views_after_denoise
        ):
            return
        patch_dir = (
            self.directory
            / f"step_{step_index:04d}"
            / f"patch_{patch_index:03d}"
        )
        patch_dir.mkdir(parents=True, exist_ok=True)
        if before is not None and self.config.save_views_before_denoise:
            tensor_to_pil(before[0]).save(patch_dir / "source_patch.png")
        if self.config.save_views_after_denoise:
            tensor_to_pil(after[0]).save(patch_dir / "predicted_patch.png")

    def on_planar_step(
        self,
        step_index: int,
        source: Any,
        result: Any,
        stats: Any,
        *,
        clean: bool,
    ) -> None:
        if not self.config.enabled:
            return
        if self.config.save_step_indices and step_index not in self.config.save_step_indices:
            return
        step_dir = self.directory / f"step_{step_index:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_erp_each_step:
            if source is not None:
                tensor_to_pil(source[0]).save(step_dir / "planar_canvas_before.png")
            name = "fused_clean_planar_canvas.png" if clean else "planar_canvas_after.png"
            tensor_to_pil(result.canvas_rgb[0]).save(step_dir / name)
        tensors: Dict[str, torch.Tensor] = {}
        if self.config.save_masks:
            tensors["coverage_mask"] = result.coverage_mask.detach().cpu()
            tensors["contributor_count"] = result.contributor_count.detach().cpu()
        if self.config.save_weights:
            tensors["accumulated_weight"] = result.accumulated_weight.detach().cpu()
        if tensors:
            torch.save(tensors, step_dir / "planar_fusion_tensors.pt")
        filename = "planar_clean_consensus_stats.txt" if clean else "planar_stats.txt"
        (step_dir / filename).write_text(
            "\n".join(f"{key}={value}" for key, value in asdict(stats).items())
            + "\n",
            encoding="utf-8",
        )


class TensorStatisticsAccumulator:
    """Streaming scalar moments for pixel-state diagnostics."""

    def __init__(self):
        self.count = 0
        self.total = None
        self.square_total = None
        self.minimum = None
        self.maximum = None

    def add(self, tensor: torch.Tensor) -> None:
        value = tensor.detach().float()
        self.count += value.numel()
        total = value.sum()
        square_total = value.square().sum()
        minimum = value.min()
        maximum = value.max()
        if self.total is None:
            self.total = total
            self.square_total = square_total
            self.minimum = minimum
            self.maximum = maximum
        else:
            self.total = self.total + total
            self.square_total = self.square_total + square_total
            self.minimum = torch.minimum(self.minimum, minimum)
            self.maximum = torch.maximum(self.maximum, maximum)

    def values(self, prefix: str) -> Dict[str, float]:
        if not self.count:
            return {}
        mean = self.total / self.count
        variance = (self.square_total / self.count - mean * mean).clamp_min(0.0)
        return {
            f"{prefix}_mean": float(mean),
            f"{prefix}_std": float(variance.sqrt()),
            f"{prefix}_min": float(self.minimum),
            f"{prefix}_max": float(self.maximum),
            f"{prefix}_l2": float(self.square_total.sqrt()),
        }


def tensor_state_statistics(
    tensor: torch.Tensor,
    prefix: str,
    *,
    spatial_correlation: bool = False,
) -> Dict[str, float]:
    accumulator = TensorStatisticsAccumulator()
    accumulator.add(tensor)
    result = accumulator.values(prefix)
    if spatial_correlation:
        value = tensor.detach().float()
        centered = value - value.mean()
        variance = centered.square().mean().clamp_min(1.0e-12)
        horizontal = (centered * centered.roll(-1, dims=-1)).mean() / variance
        if value.shape[-2] > 1:
            vertical = (centered[..., :-1, :] * centered[..., 1:, :]).mean() / variance
        else:
            vertical = value.new_zeros(())
        result[f"{prefix}_horizontal_correlation"] = float(horizontal)
        result[f"{prefix}_vertical_correlation"] = float(vertical)
    return result
