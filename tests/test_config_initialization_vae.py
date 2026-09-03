import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from diffpano.config import ExperimentConfig, load_experiment_config
from diffpano.camera import CameraSampler, camera_for_direction
from diffpano.config import FusionConfig, InitializationConfig, SamplingDirectionConfig, WarpConfig
from diffpano.pipelines import build_view_denoiser
from diffpano.pipelines.base import (
    MockViewDenoiser,
    make_first_order_scheduler,
    release_prompt_encoders,
    reset_scheduler_step_state,
)
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

    def test_sd2_backend_is_registered(self):
        config = ExperimentConfig()
        config.model.pipeline = "sd2"
        config.model.id = "test/sd2"
        sentinel = object()
        with patch("diffpano.pipelines.sd2.SD2ViewDenoiser.from_pretrained", return_value=sentinel):
            self.assertIs(build_view_denoiser(config), sentinel)


class FakeSecondOrderScheduler:
    order = 1

    def __init__(self, solver_order=2):
        self.config = SimpleNamespace(solver_order=solver_order)

    @classmethod
    def from_config(cls, config, **kwargs):
        del config
        return cls(**kwargs)


class SchedulerTests(unittest.TestCase):
    def test_configurable_multistep_scheduler_is_rebuilt_at_order_one(self):
        scheduler = make_first_order_scheduler(FakeSecondOrderScheduler())
        self.assertEqual(scheduler.config.solver_order, 1)

    def test_all_per_view_scheduler_history_is_reset(self):
        scheduler = SimpleNamespace(
            _step_index=3,
            model_outputs=[torch.ones(1), torch.ones(1)],
            lower_order_nums=2,
            last_sample=torch.ones(1),
            timestep_list=[1, 2],
        )
        reset_scheduler_step_state(scheduler)
        self.assertIsNone(scheduler._step_index)
        self.assertEqual(scheduler.model_outputs, [None, None])
        self.assertEqual(scheduler.lower_order_nums, 0)
        self.assertIsNone(scheduler.last_sample)
        self.assertEqual(scheduler.timestep_list, [None, None])

    def test_one_shot_prompt_encoders_are_released(self):
        pipeline = SimpleNamespace(text_encoder=object(), text_encoder_2=object())
        release_prompt_encoders(pipeline)
        self.assertIsNone(pipeline.text_encoder)
        self.assertIsNone(pipeline.text_encoder_2)


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


class DirectLatentVAE(FakeVAE):
    class EncodedDirectLatent:
        def __init__(self, value):
            self.latent = value

    def encode(self, value):
        return self.EncodedDirectLatent(value.mean(dim=1, keepdim=True))


class VAETests(unittest.TestCase):
    def test_encode_is_deterministic_posterior_mode_with_scale_and_shift(self):
        vae = FakeVAE()
        image = torch.ones(2, 3, 4, 4)
        first = encode_view_images(vae, image, chunk_size=2)
        second = encode_view_images(vae, image, chunk_size=2)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, torch.ones(2, 1, 4, 4)))
        self.assertEqual(vae.posterior.samples, 0)

    def test_encode_supports_direct_latent_outputs_used_by_sana_dc_ae(self):
        encoded = encode_view_images(DirectLatentVAE(), torch.ones(2, 3, 4, 4), chunk_size=2)
        self.assertTrue(torch.equal(encoded, torch.ones(2, 1, 4, 4)))


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
