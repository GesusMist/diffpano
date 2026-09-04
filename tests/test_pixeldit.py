import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from diffpano.camera import CameraSampler, camera_for_direction
from diffpano.conditioning import camera_prompt_indices, expand_directional_prompts
from diffpano.config import (
    ExperimentConfig,
    FusionConfig,
    InitializationConfig,
    SamplingDirectionConfig,
    WarpConfig,
    load_experiment_config,
)
from diffpano.erp_pipeline import ERPRGBPipeline
from diffpano.initialization import initialize_erp_canvas
from diffpano.pipelines import build_view_denoiser
from diffpano.pipelines.base import MockViewDenoiser
from diffpano.pipelines.pixeldit import (
    PixelDiTConditioning,
    PixelDiTPromptBank,
    PixelDiTViewDenoiser,
)
from diffpano.pipelines.pixeldit_solver import (
    PixelDiTFirstOrderSolver,
    pixeldit_time_schedule,
)
from diffpano.warp import StandardWarpOperator


class FakePixelModel:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.zeros(()))
        self.last_data_info = None
        self.last_timestep = None

    def parameters(self):
        yield self.parameter

    def to(self, *args, **kwargs):
        self.parameter.data = self.parameter.data.to(*args, **kwargs)
        return self

    def forward_with_dpmsolver(self, state, timestep, text, mask=None, data_info=None):
        del text, mask
        self.last_data_info = data_info
        self.last_timestep = timestep
        return torch.zeros_like(state)



class FakeOfficialDPM:
    """Minimal test double for the official stateless order-one interfaces."""

    def __init__(
        self,
        model,
        condition,
        uncondition,
        cfg_scale,
        model_kwargs=None,
        interval_guidance=None,
        **kwargs,
    ):
        del kwargs
        self.model = model
        self.condition = condition
        self.uncondition = uncondition
        self.cfg_scale = cfg_scale
        self.model_kwargs = model_kwargs or {}
        self.interval_guidance = interval_guidance or [0.0, 1.0]

    def model_fn(self, state, timestep):
        continuous = timestep.expand(state.shape[0])
        guidance_active = (
            self.cfg_scale != 1.0
            and self.uncondition is not None
            and self.interval_guidance[0] < float(continuous[0]) < self.interval_guidance[1]
        )
        if guidance_active:
            state_in = torch.cat([state, state])
            time_in = torch.cat([continuous, continuous])
            text_in = torch.cat([self.uncondition, self.condition])
            flow_pair = self.model(
                state_in, time_in * 1000.0, text_in, **self.model_kwargs
            )
            coefficient = (1.0 - time_in).view(-1, 1, 1, 1).to(state_in)
            noise_pair = coefficient * flow_pair + state_in
            noise_unconditional, noise_conditional = noise_pair.chunk(2)
            noise = noise_unconditional + self.cfg_scale * (
                noise_conditional - noise_unconditional
            )
        else:
            flow = self.model(
                state, continuous * 1000.0, self.condition, **self.model_kwargs
            )
            coefficient = (1.0 - continuous).view(-1, 1, 1, 1).to(state)
            noise = coefficient * flow + state
        return (state - timestep * noise) / (1.0 - timestep)

    @staticmethod
    def dpm_solver_first_update(state, current, following, model_s=None, **kwargs):
        del kwargs
        lambda_current = torch.log(1.0 - current) - torch.log(current)
        lambda_following = torch.log(1.0 - following) - torch.log(following)
        phi_one = torch.expm1(-(lambda_following - lambda_current))
        return (
            following / current * state
            - (1.0 - following) * phi_one * model_s
        )

class TokenBatch:
    def __init__(self, batch, length):
        self.input_ids = torch.arange(length).view(1, length).repeat(batch, 1)
        self.attention_mask = torch.ones(batch, length, dtype=torch.long)

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class FakeTokenizer:
    def encode(self, text):
        del text
        return [1, 2, 3]

    def __call__(self, text, *, max_length, **kwargs):
        del kwargs
        batch = len(text) if isinstance(text, list) else 1
        return TokenBatch(batch, max_length)


class FakeTextEncoder:
    def __call__(self, input_ids, attention_mask):
        del attention_mask
        return (input_ids.float().unsqueeze(-1).repeat(1, 1, 4),)


def official_config():
    return SimpleNamespace(
        model=SimpleNamespace(extra={"patch_size": 2}),
        scheduler=SimpleNamespace(flow_shift=4.0),
        text_encoder=SimpleNamespace(
            text_encoder_name="fake",
            model_max_length=3,
            chi_prompt=[],
        ),
    )


def fake_modules():
    return SimpleNamespace(
        DPMS=FakeOfficialDPM,
        get_tokenizer_and_text_encoder=lambda **kwargs: (FakeTokenizer(), FakeTextEncoder())
    )


def make_adapter(model=None, *, diagnostics=False):
    return PixelDiTViewDenoiser(
        model or FakePixelModel(),
        official_config(),
        fake_modules(),
        cfg_scale=2.75,
        negative_prompt="bad",
        flow_shift=None,
        interval_guidance=[0.0, 1.0],
        release_text_encoder=True,
        record_state_statistics=diagnostics,
    )


def conditioning(batch=1, length=3, width=4):
    return PixelDiTConditioning(
        positive=torch.ones(batch, 1, length, width),
        positive_mask=torch.ones(batch, length, dtype=torch.long),
        negative=torch.zeros(batch, 1, length, width),
        negative_mask=torch.ones(batch, length, dtype=torch.long),
    )


class OrderedSampler(CameraSampler):
    def __init__(self, cameras):
        self.cameras = cameras

    def sample(self, step_index, num_steps):
        del step_index, num_steps
        return list(self.cameras)


class IdentityPixelDenoiser(MockViewDenoiser):
    def __init__(self, num_steps=3):
        super().__init__(num_steps=num_steps)
        self.state_diagnostics_enabled = True
        self.last_model_prediction = None


class PixelDiTConfigAndInvariantTests(unittest.TestCase):
    def test_checked_in_pixeldit_configs_match_typed_schema(self):
        for path, expected_steps in (
            ("configs/pixeldit_smoke.yaml", 1),
            ("configs/pixeldit_standard_average.yaml", 50),
        ):
            config = load_experiment_config(path)
            self.assertEqual(config.model.pipeline, "pixeldit")
            self.assertEqual(config.generation.num_inference_steps, expected_steps)

    def test_pixeldit_backend_registered_lazily(self):
        config = ExperimentConfig()
        config.model.pipeline = "pixeldit"
        config.model.id = None
        config.initialization.mode = "pixel_gaussian"
        sentinel = object()
        with patch(
            "diffpano.pipelines.pixeldit.PixelDiTViewDenoiser.from_pretrained",
            return_value=sentinel,
        ):
            self.assertIs(build_view_denoiser(config), sentinel)

    def test_pixeldit_backend_has_no_forbidden_dependency(self):
        source = inspect.getsource(PixelDiTViewDenoiser).lower()
        for forbidden in (
            "vae",
            "autoencoder",
            "latent",
            "encode_view_images",
            "decode_view_latents",
            "reinject",
        ):
            self.assertNotIn(forbidden, source)

    def test_pixeldit_initialization_is_direct_unclamped_rgb_gaussian(self):
        config = InitializationConfig(mode="pixel_gaussian", mean=0.25, std=1.5, clamp=False)
        generator = torch.Generator().manual_seed(12)
        state = initialize_erp_canvas(
            config,
            batch_size=1,
            height=16,
            width=32,
            device=torch.device("cpu"),
            generator=generator,
            camera_sampler=None,
            warp_operator=None,
            fusion_config=None,
        )
        expected_generator = torch.Generator().manual_seed(12)
        expected = torch.randn(1, 3, 16, 32, generator=expected_generator) * 1.5 + 0.25
        self.assertTrue(torch.equal(state, expected))
        self.assertGreater(float(state.max()), 1.0)

    def test_pixeldit_does_not_sample_noise_per_view(self):
        source = inspect.getsource(PixelDiTViewDenoiser.denoise_step)
        self.assertNotIn("randn", source)
        self.assertNotIn("rand", source)


class PixelDiTSolverTests(unittest.TestCase):
    def test_timestep_schedule_matches_official_time_uniform_flow(self):
        actual = pixeldit_time_schedule(2, 4.0)
        base = 1.0 - torch.linspace(1.0, 0.001, 3)
        expected = (4.0 * base / (1.0 + 3.0 * base)).flip(0)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(bool((actual[:-1] > actual[1:]).all()))
        self.assertEqual(float(actual[-1]), 0.0)

    def test_first_order_single_step(self):
        solver = PixelDiTFirstOrderSolver(4.0)
        timesteps = solver.prepare(4, device=torch.device("cpu"))
        state = torch.ones(1, 3, 2, 2)
        flow = torch.full_like(state, 2.0)
        output = solver.step(state, flow, timesteps[0])
        expected = state + (solver.schedule[1] - solver.schedule[0]) * flow
        self.assertTrue(torch.allclose(output, expected, atol=2e-4, rtol=1e-5))

    def test_first_order_step_matches_official_pixeldit_primitive(self):
        repository = Path("third_party/PixelDiT")
        if not repository.is_dir():
            self.skipTest("Pinned official PixelDiT checkout is not installed")
        for directory in (repository.resolve(), (repository / "t2i").resolve()):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        from diffusion.model.flow_dpm import DPMS

        def raw_flow(state, timestep, **kwargs):
            del timestep, kwargs
            return torch.full_like(state, 0.125)

        reference = DPMS(
            raw_flow,
            condition=None,
            uncondition=None,
            cfg_scale=1.0,
            guidance_type="uncond",
            model_type="flow",
            schedule="FLOW",
        )
        solver = PixelDiTFirstOrderSolver(4.0)
        current = solver.prepare(5, device=torch.device("cpu"))[0]
        following = solver.schedule[1]
        state = torch.randn(1, 3, 4, 4)
        expected = reference.dpm_solver_first_update(state, current, following)
        actual = solver.step(state, torch.full_like(state, 0.125), current)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=1e-6))


    def test_adapter_cfg_arithmetic_matches_official_order_one_trajectory(self):
        repository = Path("third_party/PixelDiT")
        if not repository.is_dir():
            self.skipTest("Pinned official PixelDiT checkout is not installed")
        for directory in (repository.resolve(), (repository / "t2i").resolve()):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        from diffusion.model.flow_dpm import DPMS

        class ConditioningAwareModel(FakePixelModel):
            def forward_with_dpmsolver(self, state, timestep, text, mask=None, data_info=None):
                del timestep, mask, data_info
                value = text.float().mean(dim=tuple(range(1, text.ndim)))
                return value.view(-1, 1, 1, 1).expand_as(state)

        adapter = make_adapter(ConditioningAwareModel())
        adapter.official_modules.DPMS = DPMS
        adapter.prepare(num_steps=50, view_height=8, view_width=8)
        selected = conditioning()
        initial = torch.randn(1, 3, 8, 8)
        actual = initial.clone()
        for timestep in adapter.timesteps:
            actual = adapter.denoise_step(actual, timestep, selected)
        expected = adapter.official_order_one_sample(initial.clone(), selected)
        self.assertTrue(torch.allclose(actual, expected, atol=2.0e-6, rtol=1.0e-6))

class PixelDiTConditioningTests(unittest.TestCase):
    def test_prompt_conditioning_shapes(self):
        adapter = make_adapter()
        bank = adapter.prepare_prompt_conditioning(["scene"] * 5)
        self.assertEqual(tuple(bank.positive.shape), (20, 1, 3, 4))
        self.assertEqual(tuple(bank.positive_mask.shape), (20, 3))
        self.assertEqual(tuple(bank.negative.shape), (1, 1, 3, 4))
        cameras = [camera_for_direction(0, 90, height=4, width=4)]
        selected = adapter.conditioning_for_cameras(bank, cameras, batch_size=2)
        self.assertEqual(tuple(selected.positive.shape), (2, 1, 3, 4))
        self.assertEqual(tuple(selected.negative.shape), (2, 1, 3, 4))

    def test_north_south_prompt_convention_matches_camera_pitch(self):
        directional = expand_directional_prompts(
            ["north", "upper", "equator", "lower", "south"]
        )
        cameras = [
            camera_for_direction(0, 90, height=2, width=2),
            camera_for_direction(0, -90, height=2, width=2),
        ]
        indices = camera_prompt_indices(cameras, directional.directions)
        self.assertEqual(directional.prompts[int(indices[0])], "north")
        self.assertEqual(directional.prompts[int(indices[1])], "south")

    def test_camera_metadata_uses_actual_rectangular_shape(self):
        state = torch.zeros(2, 3, 8, 4)
        dimensions, aspect_ratio = PixelDiTViewDenoiser.image_metadata(state)
        self.assertTrue(torch.equal(dimensions, torch.tensor([[8.0, 4.0], [8.0, 4.0]])))
        self.assertTrue(torch.equal(aspect_ratio, torch.tensor([[2.0], [2.0]])))

    def test_model_receives_shifted_time_and_actual_metadata(self):
        model = FakePixelModel()
        adapter = make_adapter(model)
        timestep = adapter.solver.prepare(2, device=torch.device("cpu"))[0]
        state = torch.randn(1, 3, 8, 4)
        adapter.denoise_step(state, timestep, conditioning())
        self.assertTrue(torch.equal(
            model.last_data_info["img_hw"], torch.tensor([[8.0, 4.0]])
        ))
        self.assertTrue(torch.equal(
            model.last_data_info["aspect_ratio"], torch.tensor([[2.0]])
        ))
        self.assertAlmostEqual(float(model.last_timestep[0]), float(timestep * 1000), places=4)


class PixelDiTPipelineTests(unittest.TestCase):
    @staticmethod
    def warp():
        return StandardWarpOperator(
            WarpConfig(
                mode="standard",
                erp_to_perspective=SamplingDirectionConfig("bilinear"),
                perspective_to_erp=SamplingDirectionConfig("bilinear"),
            ),
            FusionConfig(mode="average", weight_mode="uniform"),
        )

    def test_pixeldit_synchronous_camera_order_independence(self):
        cameras = [
            camera_for_direction(-45, 0, height=8, width=8, fov_x=100, fov_y=100),
            camera_for_direction(45, 0, height=8, width=8, fov_x=100, fov_y=100),
        ]
        initial = torch.randn(1, 3, 8, 16)
        outputs = []
        for order in (cameras, list(reversed(cameras))):
            adapter = make_adapter()
            adapter.prepare(num_steps=1, view_height=8, view_width=8)
            directions = expand_directional_prompts(["scene"] * 5).directions
            bank = PixelDiTPromptBank(
                prompt_directions=directions,
                positive=torch.ones(20, 1, 3, 4),
                positive_mask=torch.ones(20, 3),
                negative=torch.zeros(1, 1, 3, 4),
                negative_mask=torch.ones(1, 3),
            )
            pipeline = ERPRGBPipeline(
                camera_sampler=OrderedSampler(order),
                warp_operator=self.warp(),
                fusion_config=FusionConfig(mode="average", weight_mode="uniform"),
                view_denoiser=adapter,
            )
            outputs.append(pipeline.run(initial.clone(), bank).erp_rgb)
        self.assertTrue(torch.allclose(outputs[0], outputs[1], atol=2e-6, rtol=0))

    def test_geometry_only_variance_diagnostics(self):
        cameras = [camera_for_direction(0, 0, height=8, width=8, fov_x=120, fov_y=120)]
        denoiser = IdentityPixelDenoiser(num_steps=3)
        pipeline = ERPRGBPipeline(
            camera_sampler=OrderedSampler(cameras),
            warp_operator=self.warp(),
            fusion_config=FusionConfig(mode="average", weight_mode="uniform"),
            view_denoiser=denoiser,
        )
        result = pipeline.run(torch.randn(1, 3, 8, 16), None)
        self.assertEqual(len(result.steps), 3)
        for step in result.steps:
            self.assertIn("fusion_variance_ratio", step.state_statistics)
            self.assertIn("erp_next_horizontal_correlation", step.state_statistics)


if __name__ == "__main__":
    unittest.main()
