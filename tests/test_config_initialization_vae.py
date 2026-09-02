import tempfile
import unittest
from pathlib import Path

import torch

from diffpano.config import ExperimentConfig, load_experiment_config
from diffpano.camera import CameraSampler, camera_for_direction
from diffpano.config import FusionConfig, InitializationConfig, SamplingDirectionConfig, WarpConfig
from diffpano.pipelines.base import MockViewDenoiser
from diffpano.initialization import initialize_erp_canvas
from diffpano.vae import encode_view_images
from diffpano.warp import StandardWarpOperator


class ConfigTests(unittest.TestCase):
    def test_root_config_is_new_architecture(self):
        config = load_experiment_config("config.yaml")
        self.assertEqual(config.initialization.mode, "erp_rgb_noise")
        self.assertFalse(hasattr(config, "sphere"))
        config.validate()

    def test_stale_spherical_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stale.yaml"
            path.write_text("sphere:\n  num_points: 2600\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_config(str(path))

    def test_invalid_option_matrix_member_is_rejected(self):
        config = ExperimentConfig()
        config.warp.erp_to_perspective.interpolation = "bicubic"
        with self.assertRaises(ValueError):
            config.validate()


class Posterior:
    def __init__(self, value):
        self.mean = value + 100
        self.value = value
        self.samples = 0

    def mode(self):
        return self.value

    def sample(self, generator=None):
        del generator
        self.samples += 1
        return self.value + 999


class Encoded:
    def __init__(self, posterior):
        self.latent_dist = posterior


class FakeVAE:
    dtype = torch.float32

    class Config:
        scaling_factor = 2.0
        shift_factor = 0.5

    config = Config()

    def __init__(self):
        self.posterior = None

    def encode(self, value):
        self.posterior = Posterior(value.mean(dim=1, keepdim=True))
        return Encoded(self.posterior)


class VAETests(unittest.TestCase):
    def test_encode_is_deterministic_posterior_mode_with_scale_and_shift(self):
        vae = FakeVAE()
        image = torch.ones(2, 3, 4, 4)
        first = encode_view_images(vae, image, chunk_size=2)
        second = encode_view_images(vae, image, chunk_size=2)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, torch.ones(2, 1, 4, 4)))
        self.assertEqual(vae.posterior.samples, 0)


class OneCameraSampler(CameraSampler):
    def sample(self, step_index, num_steps):
        del step_index, num_steps
        return [camera_for_direction(0, 0, height=8, width=8, fov_x=100, fov_y=100)]


class InitializationTests(unittest.TestCase):
    def test_latent_native_bootstrap_finishes_as_erp_rgb(self):
        fusion = FusionConfig(mode="average", weight_mode="uniform")
        warp = StandardWarpOperator(
            WarpConfig(
                mode="standard",
                erp_to_perspective=SamplingDirectionConfig("nearest"),
                perspective_to_erp=SamplingDirectionConfig("bilinear"),
            ),
            fusion,
        )
        generator = torch.Generator().manual_seed(9)
        erp = initialize_erp_canvas(
            InitializationConfig(mode="latent_native_bootstrap"),
            batch_size=1,
            height=8,
            width=16,
            device=torch.device("cpu"),
            generator=generator,
            camera_sampler=OneCameraSampler(),
            warp_operator=warp,
            fusion_config=fusion,
            view_denoiser=MockViewDenoiser(),
        )
        self.assertEqual(tuple(erp.shape), (1, 3, 8, 16))
        self.assertTrue(bool(torch.isfinite(erp).all()))


if __name__ == "__main__":
    unittest.main()
