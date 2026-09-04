import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from diffpano.camera import (
    CameraSampler,
    camera_for_direction,
    spherediff_camera_cover,
)
from diffpano.config import (
    CleanConsensusConfig,
    ExperimentConfig,
    FusionConfig,
    GlobalPipelineConfig,
    WarpConfig,
    load_experiment_config,
)
from diffpano.erp_x0_pipeline import ERPX0ConsensusPipeline
from diffpano.fusion import RGBFusionAccumulator
from diffpano.noise import FixedPatchNoiseBank
from diffpano.pipelines.clean_prediction import (
    ddim_predicted_clean,
    flow_add_noise,
    flow_predicted_clean,
)
from diffpano.pipelines.flux import FluxViewDenoiser
from diffpano.pipelines.pixeldit import (
    PixelDiTConditioning,
    PixelDiTViewDenoiser,
)
from diffpano.projection import ERPContribution
from diffpano.warp import StandardWarpOperator
from scripts.generate import _generate_with_selected_global_pipeline


class OrderedSampler(CameraSampler):
    def __init__(self, cameras):
        self.cameras = list(cameras)

    def sample(self, step_index, num_steps):
        del step_index, num_steps
        return list(self.cameras)


class FullFrameWarp:
    def __init__(self):
        self.forward_inputs = []
        self.inverse_inputs = []

    def erp_to_perspective(self, erp_rgb, camera):
        del camera
        self.forward_inputs.append(erp_rgb.clone())
        return erp_rgb.clone()

    def perspective_to_erp(self, rgb_view, camera, erp_size):
        del camera
        assert tuple(rgb_view.shape[-2:]) == tuple(erp_size)
        self.inverse_inputs.append(rgb_view.clone())
        mask = torch.ones(
            rgb_view.shape[0], 1, *erp_size, dtype=torch.float32
        )
        return ERPContribution(
            rgb=rgb_view.float(),
            valid_mask=mask,
            weight=mask,
        )


class FullFrameJointAccumulator:
    def __init__(self, owner, previous):
        self.owner = owner
        self.inner = RGBFusionAccumulator(
            previous, FusionConfig(mode="average", weight_mode="uniform")
        )

    def accumulate(self, rgb_view, camera):
        del camera
        self.owner.joint_inputs.append(rgb_view.clone())
        mask = torch.ones(
            rgb_view.shape[0],
            1,
            *rgb_view.shape[-2:],
            device=rgb_view.device,
            dtype=torch.float32,
        )
        projected = ERPContribution(rgb_view.float(), mask, mask)
        self.inner.accumulate(projected)
        return projected

    def finalize(self):
        return self.inner.finalize()


class FullFrameJointWarp(FullFrameWarp):
    def __init__(self):
        super().__init__()
        self.joint_inputs = []
        self.direct_inverse_calls = 0

    def perspective_to_erp(self, rgb_view, camera, erp_size):
        del rgb_view, camera, erp_size
        self.direct_inverse_calls += 1
        raise AssertionError("joint LPW path must not call per-view inverse warp")

    def create_fusion_accumulator(self, previous):
        return FullFrameJointAccumulator(self, previous)


class MockCleanBackend:
    def __init__(self, *, constant=7.0):
        self._timesteps = torch.tensor([3.0, 2.0, 1.0])
        self.constant = constant
        self.noise_sample_calls = 0
        self.noise_history = []
        self.model_timesteps = []
        self.last_model_prediction = None

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def timesteps(self):
        return self._timesteps

    def sample_fixed_noise(
        self, *, batch_size, height, width, generator
    ):
        self.noise_sample_calls += 1
        return torch.randn(
            batch_size, 3, height, width, generator=generator
        )

    def make_initial_noisy_state(self, fixed_noise, timestep):
        del timestep
        self.noise_history.append(fixed_noise.clone())
        return fixed_noise

    def encode_clean(self, rgb_clean):
        return rgb_clean

    def add_fixed_noise(self, clean_state, fixed_noise, timestep):
        del timestep
        self.noise_history.append(fixed_noise.clone())
        return clean_state + fixed_noise

    def conditioning_for_cameras(
        self, prepared_conditioning, cameras, *, batch_size
    ):
        del prepared_conditioning
        return torch.tensor(
            [camera.yaw for camera in cameras for _ in range(batch_size)]
        )

    def predict_clean_native(self, noisy_state, timestep, conditioning):
        del conditioning
        self.model_timesteps.extend(
            [float(timestep)] * noisy_state.shape[0]
        )
        self.last_model_prediction = torch.zeros_like(noisy_state)
        return torch.full_like(noisy_state, self.constant)

    def decode_clean(self, clean_state):
        return clean_state


def run_mock(cameras, backend=None):
    backend = backend or MockCleanBackend()
    warp = FullFrameWarp()
    pipeline = ERPX0ConsensusPipeline(
        camera_sampler=OrderedSampler(cameras),
        warp_operator=warp,
        fusion_config=FusionConfig(mode="average", weight_mode="uniform"),
        consensus_config=CleanConsensusConfig(),
        backend=backend,
        seed=19,
    )
    result = pipeline.run(
        None, batch_size=1, erp_height=4, erp_width=4
    )
    return result, backend, warp


class GlobalPipelineConfigTests(unittest.TestCase):
    def test_old_configs_default_to_existing_pipeline(self):
        config = load_experiment_config("configs/sana_smoke.yaml")
        self.assertEqual(config.global_pipeline.mode, "erp_rgb_state")

    def test_explicit_old_and_new_routing(self):
        config = ExperimentConfig()
        old_result = object()
        new_result = object()
        with patch(
            "scripts.generate.generate_erp_rgb", return_value=old_result
        ) as old, patch(
            "scripts.generate.generate_erp_x0_consensus",
            return_value=new_result,
        ) as new:
            self.assertIs(
                _generate_with_selected_global_pipeline(config, object(), None),
                old_result,
            )
            old.assert_called_once()
            new.assert_not_called()
            config.global_pipeline.mode = "erp_x0_consensus"
            self.assertIs(
                _generate_with_selected_global_pipeline(config, object(), None),
                new_result,
            )
            new.assert_called_once()

    def test_pixeldit_initialization_requirement_is_conditional(self):
        config = ExperimentConfig()
        config.model.pipeline = "pixeldit"
        config.model.id = None
        with self.assertRaises(ValueError):
            config.validate()
        config.global_pipeline.mode = "erp_x0_consensus"
        config.validate()

    def test_loader_accepts_typed_clean_consensus_section(self):
        text = """
global_pipeline:
  mode: erp_x0_consensus
  clean_consensus:
    bootstrap: native_noise
    noise_storage: cpu
    noise_binding: camera_index
    noise_dtype: fp32
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            config = load_experiment_config(str(path))
        self.assertEqual(config.global_pipeline.mode, "erp_x0_consensus")

    def test_checked_in_x0_configs_are_fixed_89_view_plain_baselines(self):
        for backend in ("sana", "flux", "sd2", "pixeldit"):
            with self.subTest(backend=backend):
                config = load_experiment_config(
                    f"configs/x0_consensus_{backend}.yaml"
                )
                self.assertEqual(config.global_pipeline.mode, "erp_x0_consensus")
                self.assertEqual(config.sampling.strategy, "spherediff_fixed")
                self.assertEqual(config.warp.mode, "standard")
                self.assertEqual(
                    config.warp.erp_to_perspective.interpolation, "bilinear"
                )
                self.assertEqual(
                    config.warp.perspective_to_erp.interpolation, "bilinear"
                )
                self.assertEqual(config.fusion.mode, "average")
                self.assertEqual(config.fusion.weight_mode, "uniform")
                self.assertEqual(len(spherediff_camera_cover(config.view)), 89)


class FixedNoiseTests(unittest.TestCase):
    def test_same_camera_is_exactly_reused_and_cameras_differ(self):
        backend = MockCleanBackend()
        bank = FixedPatchNoiseBank(
            CleanConsensusConfig(),
            backend=backend,
            num_cameras=2,
            batch_size=1,
            height=4,
            width=4,
            seed=31,
        )
        first = bank.get([0], device=torch.device("cpu"))
        later = bank.get([0], device=torch.device("cpu"))
        other = bank.get([1], device=torch.device("cpu"))
        self.assertTrue(torch.equal(first, later))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(backend.noise_sample_calls, 2)
        self.assertEqual(bank.identity(0), bank.identity(0))
        self.assertNotEqual(bank.identity(0), bank.identity(1))

    def test_pipeline_samples_only_once_per_camera_not_per_step(self):
        cameras = [
            camera_for_direction(-20, 0, height=4, width=4),
            camera_for_direction(20, 0, height=4, width=4),
        ]
        result, backend, _ = run_mock(cameras)
        self.assertEqual(backend.noise_sample_calls, 2)
        self.assertEqual(len(backend.noise_history), 6)
        self.assertTrue(
            torch.equal(
                backend.noise_history[0], backend.noise_history[2]
            )
        )
        self.assertTrue(
            torch.equal(
                backend.noise_history[0], backend.noise_history[4]
            )
        )
        self.assertTrue(
            torch.equal(
                backend.noise_history[1], backend.noise_history[3]
            )
        )
        self.assertEqual(len(result.fixed_noise_identities), 2)


class X0PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cameras = [
            camera_for_direction(-20, 0, height=4, width=4),
            camera_for_direction(20, 0, height=4, width=4),
        ]

    def test_bootstrap_consumes_first_timestep_once_and_total_is_n_views(self):
        _, backend, _ = run_mock(self.cameras)
        self.assertEqual(
            backend.model_timesteps,
            [3.0, 3.0, 2.0, 2.0, 1.0, 1.0],
        )

    def test_persistent_and_fused_state_is_predicted_clean_rgb(self):
        result, _, warp = run_mock(self.cameras)
        self.assertTrue(
            torch.equal(result.erp_rgb, torch.full_like(result.erp_rgb, 7.0))
        )
        self.assertTrue(
            all(
                torch.equal(value, torch.full_like(value, 7.0))
                for value in warp.inverse_inputs
            )
        )
        self.assertEqual(len(warp.forward_inputs), 4)
        self.assertTrue(
            all(
                torch.equal(value, torch.full_like(value, 7.0))
                for value in warp.forward_inputs
            )
        )

    def test_lpw_style_accumulator_sees_predicted_clean_rgb_only(self):
        backend = MockCleanBackend(constant=7.0)
        warp = FullFrameJointWarp()
        pipeline = ERPX0ConsensusPipeline(
            camera_sampler=OrderedSampler(self.cameras),
            warp_operator=warp,
            fusion_config=FusionConfig(mode="average", weight_mode="uniform"),
            consensus_config=CleanConsensusConfig(),
            backend=backend,
            seed=19,
        )
        result = pipeline.run(
            None, batch_size=1, erp_height=4, erp_width=4
        )
        self.assertEqual(warp.direct_inverse_calls, 0)
        self.assertEqual(len(warp.joint_inputs), 6)
        self.assertTrue(
            all(
                torch.equal(value, torch.full_like(value, 7.0))
                for value in warp.joint_inputs
            )
        )
        self.assertTrue(
            torch.equal(
                result.erp_rgb, torch.full_like(result.erp_rgb, 7.0)
            )
        )

    def test_camera_processing_order_does_not_change_result(self):
        forward, _, _ = run_mock(self.cameras)
        reverse, _, _ = run_mock(list(reversed(self.cameras)))
        self.assertTrue(torch.equal(forward.erp_rgb, reverse.erp_rgb))

    def test_global_loop_never_calls_old_denoise_step_or_scheduler_step(self):
        source = inspect.getsource(ERPX0ConsensusPipeline)
        self.assertNotIn("denoise_step", source)
        self.assertNotIn("scheduler.step", source)
        self.assertNotIn("prev_sample", source)

    def test_small_erp_smoke_uses_real_standard_geometry(self):
        camera = camera_for_direction(0, 0, height=8, width=8)
        fusion = FusionConfig(mode="average", weight_mode="uniform")
        pipeline = ERPX0ConsensusPipeline(
            camera_sampler=OrderedSampler([camera]),
            warp_operator=StandardWarpOperator(WarpConfig(), fusion),
            fusion_config=fusion,
            consensus_config=CleanConsensusConfig(),
            backend=MockCleanBackend(),
            seed=41,
        )
        result = pipeline.run(
            None, batch_size=1, erp_height=8, erp_width=16
        )
        self.assertTrue(torch.isfinite(result.erp_rgb).all())
        self.assertGreater(result.steps[-1].coverage_percent, 0.0)
        self.assertLess(result.steps[-1].coverage_percent, 100.0)


class FakeFlowScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([1000.0, 500.0])
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.begin_index = None
        self.step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        schedule = self.timesteps if schedule_timesteps is None else schedule_timesteps
        return int((schedule == timestep).nonzero()[0].item())

    def scale_noise(self, sample, timestep, noise):
        indices = [
            self.index_for_timestep(value, self.timesteps.to(value.device))
            for value in timestep
        ]
        sigma = self.sigmas[indices].to(sample).reshape(-1, 1, 1, 1)
        return sigma * noise + (1.0 - sigma) * sample


class FakeDPMFlowScheduler(FakeFlowScheduler):
    scale_noise = None

    def add_noise(self, sample, noise, timestep):
        indices = [
            self.index_for_timestep(value, self.timesteps.to(value.device))
            for value in timestep
        ]
        sigma = self.sigmas[indices].to(sample).reshape(-1, 1, 1, 1)
        return (1.0 - sigma) * sample + sigma * noise


class SchedulerMathTests(unittest.TestCase):
    def test_flux_bootstrap_is_direct_native_gaussian(self):
        backend = object.__new__(FluxViewDenoiser)
        backend.pipeline = SimpleNamespace(_execution_device="cpu")
        noise = torch.randn(1, 16, 4, 4)
        initial = backend.make_initial_noisy_state(noise, torch.tensor(1000.0))
        self.assertTrue(torch.equal(initial, noise))

    def test_sana_flow_oracle_recovers_clean_state(self):
        scheduler = FakeFlowScheduler()
        clean = torch.randn(2, 4, 3, 3)
        noise = torch.randn_like(clean)
        timestep = scheduler.timesteps[1]
        noisy = flow_add_noise(scheduler, clean, noise, timestep)
        recovered = flow_predicted_clean(
            scheduler, noisy, noise - clean, timestep
        )
        self.assertTrue(torch.allclose(recovered, clean, atol=1e-6))

    def test_sana_dpmsolver_add_noise_oracle_recovers_clean_state(self):
        scheduler = FakeDPMFlowScheduler()
        clean = torch.randn(2, 4, 3, 3)
        noise = torch.randn_like(clean)
        timestep = scheduler.timesteps[1]
        noisy = flow_add_noise(scheduler, clean, noise, timestep)
        recovered = flow_predicted_clean(
            scheduler, noisy, noise - clean, timestep
        )
        self.assertTrue(torch.allclose(recovered, clean, atol=1e-6))

    def test_sana_actual_flow_dpmsolver_recovers_clean_state(self):
        from diffusers import DPMSolverMultistepScheduler

        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            prediction_type="flow_prediction",
            solver_order=1,
            use_flow_sigmas=True,
            flow_shift=3.0,
        )
        scheduler.set_timesteps(10)
        clean = torch.randn(1, 4, 3, 3)
        noise = torch.randn_like(clean)
        timestep = scheduler.timesteps[4]
        noisy = flow_add_noise(scheduler, clean, noise, timestep)
        recovered = flow_predicted_clean(
            scheduler, noisy, noise - clean, timestep
        )
        self.assertTrue(torch.allclose(recovered, clean, atol=1e-6))

    def test_flux_pack_flow_recovery_unpack_recovers_raw_clean(self):
        scheduler = FakeFlowScheduler()
        clean = torch.randn(1, 4, 4, 4)
        noise = torch.randn_like(clean)
        timestep = scheduler.timesteps[1]
        noisy_raw = flow_add_noise(scheduler, clean, noise, timestep)
        pack = lambda value: value.flatten(2).transpose(1, 2)
        unpack = lambda value: value.transpose(1, 2).reshape_as(clean)
        predicted_clean_packed = flow_predicted_clean(
            scheduler,
            pack(noisy_raw),
            pack(noise - clean),
            timestep,
        )
        self.assertTrue(
            torch.allclose(unpack(predicted_clean_packed), clean, atol=1e-6)
        )

    def test_sd2_add_noise_and_epsilon_recovery(self):
        from diffusers import DDIMScheduler

        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            prediction_type="epsilon",
            clip_sample=False,
        )
        clean = torch.randn(1, 4, 3, 3)
        noise = torch.randn_like(clean)
        timestep = torch.tensor([700], dtype=torch.long)
        noisy = scheduler.add_noise(clean, noise, timestep)
        recovered = ddim_predicted_clean(
            scheduler, noisy, noise, timestep[0]
        )
        self.assertTrue(torch.allclose(recovered, clean, atol=2e-5))


class FakePixelModel:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.zeros(()))

    def parameters(self):
        yield self.parameter

    def to(self, *args, **kwargs):
        self.parameter.data = self.parameter.data.to(*args, **kwargs)
        return self


def pixel_adapter():
    config = SimpleNamespace(
        model=SimpleNamespace(extra={"patch_size": 2}),
        scheduler=SimpleNamespace(flow_shift=4.0),
    )
    return PixelDiTViewDenoiser(
        FakePixelModel(),
        config,
        SimpleNamespace(DPMS=None),
        cfg_scale=2.75,
        negative_prompt="bad",
        flow_shift=None,
        interval_guidance=[0.0, 1.0],
        release_text_encoder=True,
        record_state_statistics=False,
    )


class PixelCleanConsensusTests(unittest.TestCase):
    def test_shifted_time_noising_and_direct_clean_prediction(self):
        adapter = pixel_adapter()
        adapter.prepare(num_steps=4, view_height=4, view_width=4)
        timestep = adapter.timesteps[1]
        clean = torch.randn(1, 3, 4, 4)
        noise = torch.randn_like(clean)
        noisy = adapter.add_fixed_noise(clean, noise, timestep)
        current, _ = adapter.solver.bounds_for(timestep)
        expected = (1.0 - current) * clean + current * noise
        self.assertTrue(torch.equal(noisy, expected))
        adapter._official_solver = lambda state, conditioning: SimpleNamespace(
            model_fn=lambda value, time: clean
        )
        condition = PixelDiTConditioning(
            positive=torch.ones(1, 1, 1, 1),
            positive_mask=torch.ones(1, 1),
            negative=torch.zeros(1, 1, 1, 1),
            negative_mask=torch.ones(1, 1),
        )
        recovered = adapter.predict_clean_native(
            noisy, timestep, condition
        )
        self.assertTrue(torch.equal(recovered, clean))

    def test_clean_consensus_methods_have_no_autoencoder_or_solver_update(self):
        methods = "\n".join(
            inspect.getsource(getattr(PixelDiTViewDenoiser, name))
            for name in (
                "sample_fixed_noise",
                "make_initial_noisy_state",
                "encode_clean",
                "add_fixed_noise",
                "predict_clean_native",
                "decode_clean",
            )
        ).lower()
        for forbidden in (
            "vae",
            "autoencoder",
            "latent",
            "dpm_solver_first_update",
        ):
            self.assertNotIn(forbidden, methods)


if __name__ == "__main__":
    unittest.main()
