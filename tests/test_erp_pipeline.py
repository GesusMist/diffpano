import inspect
import unittest

import torch

from diffpano.camera import CameraSampler, camera_for_direction
from diffpano.config import FusionConfig, SamplingDirectionConfig, WarpConfig
from diffpano.erp_pipeline import ERPRGBPipeline
from diffpano.fusion import FusionResult, RGBFusionAccumulator
from diffpano.pipelines.base import MockViewDenoiser
from diffpano.projection import ERPContribution
from diffpano.warp import StandardWarpOperator


class OrderedSampler(CameraSampler):
    def __init__(self, cameras):
        self.cameras = cameras

    def sample(self, step_index, num_steps):
        del step_index, num_steps
        return list(self.cameras)


class RecordingDenoiser(MockViewDenoiser):
    def __init__(self):
        super().__init__(num_steps=1, delta=0.1)
        self.inputs = []

    def denoise_step(self, rgb_view, timestep, conditioning):
        self.inputs.append(rgb_view.clone())
        return super().denoise_step(rgb_view, timestep, conditioning)


class RecordingJointAccumulator:
    def __init__(self, owner, previous):
        self.owner = owner
        self.inner = RGBFusionAccumulator(
            previous, FusionConfig(mode="average", weight_mode="uniform")
        )

    def accumulate(self, rgb_view, camera):
        del camera
        self.owner.joint_inputs.append(rgb_view.clone())
        mask = torch.ones(
            rgb_view.shape[0], 1, *rgb_view.shape[-2:]
        )
        projected = ERPContribution(rgb_view.float(), mask, mask)
        self.inner.accumulate(projected)
        return projected

    def finalize(self) -> FusionResult:
        return self.inner.finalize()


class RecordingJointWarp:
    def __init__(self):
        self.joint_inputs = []
        self.direct_inverse_calls = 0

    def erp_to_perspective(self, erp_rgb, camera):
        if tuple(erp_rgb.shape[-2:]) != (camera.height, camera.width):
            raise AssertionError("test warp requires a full-frame camera")
        return erp_rgb.clone()

    def perspective_to_erp(self, rgb_view, camera, erp_size):
        del rgb_view, camera, erp_size
        self.direct_inverse_calls += 1
        raise AssertionError("joint LPW path must not call per-view inverse warp")

    def create_fusion_accumulator(self, previous):
        return RecordingJointAccumulator(self, previous)


def standard_warp():
    return StandardWarpOperator(
        WarpConfig(
            mode="standard",
            erp_to_perspective=SamplingDirectionConfig("bilinear"),
            perspective_to_erp=SamplingDirectionConfig("bilinear"),
        ),
        FusionConfig(mode="weighted_average", weight_mode="uniform"),
    )


class ERPPipelineTests(unittest.TestCase):
    def setUp(self):
        self.cameras = [
            camera_for_direction(-60, 0, height=10, width=10, fov_x=110, fov_y=100),
            camera_for_direction(0, 0, height=10, width=10, fov_x=110, fov_y=100),
            camera_for_direction(60, 0, height=10, width=10, fov_x=110, fov_y=100),
        ]
        self.initial = torch.randn(1, 3, 10, 20)

    def run_order(self, cameras):
        denoiser = MockViewDenoiser(num_steps=1, delta=0.05)
        pipeline = ERPRGBPipeline(
            camera_sampler=OrderedSampler(cameras),
            warp_operator=standard_warp(),
            fusion_config=FusionConfig(mode="weighted_average", weight_mode="uniform"),
            view_denoiser=denoiser,
        )
        return pipeline.run(self.initial.clone(), prepared_conditioning=None).erp_rgb

    def test_camera_order_does_not_change_synchronous_result(self):
        forward = self.run_order(self.cameras)
        reverse = self.run_order(list(reversed(self.cameras)))
        self.assertTrue(torch.allclose(forward, reverse, atol=2e-6, rtol=0))

    def test_every_view_reads_frozen_source(self):
        camera = camera_for_direction(0, 0, height=8, width=8, fov_x=100, fov_y=100)
        denoiser = RecordingDenoiser()
        pipeline = ERPRGBPipeline(
            camera_sampler=OrderedSampler([camera, camera]),
            warp_operator=standard_warp(),
            fusion_config=FusionConfig(mode="average", weight_mode="uniform"),
            view_denoiser=denoiser,
        )
        pipeline.run(self.initial.clone(), prepared_conditioning=None)
        self.assertEqual(len(denoiser.inputs), 2)
        self.assertTrue(torch.equal(denoiser.inputs[0], denoiser.inputs[1]))

    def test_one_step_mock_smoke(self):
        output = self.run_order(self.cameras)
        self.assertEqual(tuple(output.shape), tuple(self.initial.shape))
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_lpw_style_accumulator_receives_rgb_proposals(self):
        camera = camera_for_direction(
            0, 0, height=4, width=4, fov_x=100, fov_y=100
        )
        warp = RecordingJointWarp()
        pipeline = ERPRGBPipeline(
            camera_sampler=OrderedSampler([camera]),
            warp_operator=warp,
            fusion_config=FusionConfig(mode="average", weight_mode="uniform"),
            view_denoiser=MockViewDenoiser(num_steps=1, delta=0.1),
        )
        initial = torch.zeros(1, 3, 4, 4)
        result = pipeline.run(initial, prepared_conditioning=None)
        self.assertEqual(warp.direct_inverse_calls, 0)
        self.assertEqual(len(warp.joint_inputs), 1)
        self.assertTrue(
            torch.allclose(
                warp.joint_inputs[0],
                torch.full_like(warp.joint_inputs[0], 0.1),
            )
        )
        self.assertTrue(
            torch.allclose(
                result.erp_rgb, torch.full_like(result.erp_rgb, 0.1)
            )
        )

    def test_main_loop_has_no_spherical_persistent_state_api(self):
        source = inspect.getsource(ERPRGBPipeline)
        for forbidden in (
            "dynamic_latent_sampling",
            "spherical_indices",
            "ExclusiveOwnerMap",
            "reinject_fused_latents",
            "write_back_views",
            "n_sphere",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("erp_source = erp_rgb", source)


if __name__ == "__main__":
    unittest.main()
