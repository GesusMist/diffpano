import json
import unittest
from dataclasses import dataclass

from output_metadata import build_output_metadata


@dataclass
class FakeArgs:
    pretrained_model_name_or_path: str = "example/model"
    variant: str = "bf16"
    mixed_precision: str = "bf16"
    prompt_to_log: list = None


class FakePipeline:
    sphere_diff_run_metadata = {
        "pixel_fusion_config": {
            "pixel_fusion_enabled": True,
            "warp_mode": "lpw",
        },
        "n_spherical_points": 2600,
        "num_denoising_steps": 2,
        "denoise_timesteps": [1000.0, 500.0],
        "num_dynamic_view_patches_per_step": 3,
        "generator_initial_seeds": [1234],
        "denoise_patch_point_counts_by_step": [[400, 400, 441], [400, 400, 441]],
        "pixel_fusion_applied_by_step": [True, False],
        "view_directions": [[0.0, 0.0, 1.0]],
    }

    def __call__(self, num_inference_steps=20, guidance_scale=4.5, n_spherical_points=2600):
        return None


class FakePlanarPipeline:
    sphere_diff_run_metadata = {
        "planar_fusion_config": {
            "patch_latent_height": 20,
            "patch_latent_width": 20,
            "patch_stride_height": 10,
            "patch_stride_width": 10,
        },
        "num_denoising_steps": 1,
        "denoise_timesteps": [1000.0],
        "num_dynamic_view_patches_per_step": 72,
        "generator_initial_seeds": [1234],
        "denoise_patch_point_counts_by_step": [[400] * 72],
        "pixel_fusion_applied_by_step": [True],
        "planar_latent_shape": [1, 32, 64, 128],
    }

    def __call__(
        self,
        num_inference_steps=20,
        height=2048,
        width=4096,
        planar_fusion_config_path="configs/planar_patch_test.yaml",
    ):
        return None


class OutputMetadataTests(unittest.TestCase):
    def test_metadata_is_compact_and_summarizes_runtime_patch_counts(self):
        args = FakeArgs(prompt_to_log=["a panoramic scene"])
        metadata = build_output_metadata(
            args,
            FakePipeline(),
            {"num_inference_steps": 7, "custom_override": "kept"},
            "outputs/example.png",
        )

        self.assertEqual(metadata["generation_parameters"]["num_inference_steps"], 7)
        self.assertEqual(metadata["generation_parameters"]["guidance_scale"], 4.5)
        self.assertEqual(metadata["generation_parameters"]["custom_override"], "kept")
        self.assertEqual(metadata["runtime"]["n_spherical_points"], 2600)
        self.assertEqual(metadata["runtime"]["denoise_timesteps"], [1000.0, 500.0])
        self.assertEqual(
            metadata["runtime"]["denoise_patch_points"],
            {"same_for_all_steps": True, "histogram": {"400": 2, "441": 1}},
        )
        self.assertEqual(metadata["runtime"]["pixel_fusion_applied_steps"], [0])
        self.assertEqual(metadata["pixel_fusion_config"]["warp_mode"], "lpw")
        self.assertNotIn("view_directions", metadata["runtime"])
        self.assertNotIn("components", metadata)
        self.assertNotIn("environment", metadata)
        self.assertNotIn("launch_config", metadata)

        json.dumps(metadata, allow_nan=False)

    def test_planar_metadata_records_grid_patch_count_and_config(self):
        metadata = build_output_metadata(
            FakeArgs(prompt_to_log=["five directional prompts"]),
            FakePlanarPipeline(),
            {},
            "outputs/planar.png",
        )

        self.assertEqual(metadata["generation_parameters"]["height"], 2048)
        self.assertEqual(metadata["generation_parameters"]["width"], 4096)
        self.assertEqual(metadata["runtime"]["planar_latent_shape"], [1, 32, 64, 128])
        self.assertEqual(metadata["runtime"]["planar_patch_count"], 72)
        self.assertEqual(
            metadata["runtime"]["denoise_patch_points"],
            {"same_for_all_steps": True, "histogram": {"400": 72}},
        )
        self.assertEqual(metadata["planar_fusion_config"]["patch_latent_height"], 20)

        json.dumps(metadata, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
