import unittest

from diffpano.config import ExperimentConfig, PixelFusionConfig, load_experiment_config
from diffpano.pipelines.base import resolve_model_source
from scripts.legacy import translate_legacy_config


class ExperimentConfigTests(unittest.TestCase):
    def test_root_config_preserves_active_fusion_defaults(self):
        config = load_experiment_config("config.yaml")
        self.assertEqual(config.model.pipeline, "sana")
        self.assertEqual(config.fusion.warp.mode, "lpw")
        self.assertEqual(config.fusion.lpw.levels, 4)
        self.assertEqual(config.fusion.lpw.lod_interpolation, "nearest")
        self.assertEqual(config.fusion.aggregation.mode, "detail_preserving_average")
        self.assertEqual(config.reinjection.mode, "noise_consistent")
        self.assertEqual(config.writeback.mode, "exclusive")

    def test_planar_config_uses_dense_five_latent_stride(self):
        config = load_experiment_config("experiments/planar/config.yaml")
        self.assertEqual(config.model.pipeline, "planar_sana")
        self.assertEqual(config.planar["patch_stride_height"], 5)
        self.assertEqual(config.planar["patch_stride_width"], 5)

    def test_invalid_pipeline_and_fusion_window_fail_early(self):
        config = ExperimentConfig()
        config.model.pipeline = "unknown"
        with self.assertRaisesRegex(ValueError, "model.pipeline"):
            config.validate()
        fusion = PixelFusionConfig(pixel_fusion_start_ratio=0.8, pixel_fusion_end_ratio=0.2)
        with self.assertRaisesRegex(ValueError, "ratios"):
            fusion.validate()

    def test_local_model_path_has_precedence(self):
        self.assertEqual(resolve_model_source("/models/local", "remote/id"), "/models/local")

    def test_legacy_flux_arguments_route_to_canonical_defaults(self):
        config = translate_legacy_config(
            {
                "pipeline_cls": "SphericalFluxPipeline",
                "call_kwargs": {"prompt_txt_path": "data/prompts/ruins.txt"},
            },
            video=False,
        )
        self.assertEqual(config.model.pipeline, "flux")
        self.assertEqual(config.prompt.path, "prompts/ruins.txt")
        self.assertEqual(config.generation.num_inference_steps, 28)
        self.assertEqual(config.sphere.num_points, 26500)
        self.assertFalse(config.fusion.enabled)


if __name__ == "__main__":
    unittest.main()
