import unittest

import torch

from diffpano.config import FusionConfig, LPWConfig, SamplingDirectionConfig, WarpConfig
from diffpano.fusion import RGBFusionAccumulator, create_view_weight_map, detail_preserving_average
from diffpano.lpw import build_laplacian_pyramid, lod_level_confidence, reconstruct_laplacian_pyramid
from diffpano.projection import ERPContribution
from diffpano.warp import LaplacianPyramidWarpOperator
from diffpano.camera import camera_for_direction


def contribution(rgb, weight=1.0, mask=1.0):
    return ERPContribution(
        rgb=rgb,
        valid_mask=torch.full((rgb.shape[0], 1, *rgb.shape[-2:]), mask),
        weight=torch.full((rgb.shape[0], 1, *rgb.shape[-2:]), weight),
    )


class FusionTests(unittest.TestCase):
    def test_average_red_blue_overlap(self):
        previous = torch.zeros(1, 3, 2, 2)
        red = torch.zeros_like(previous); red[:, 0] = 1
        blue = torch.zeros_like(previous); blue[:, 2] = 1
        accumulator = RGBFusionAccumulator(previous, FusionConfig(mode="average"))
        accumulator.accumulate(contribution(red))
        accumulator.accumulate(contribution(blue))
        output = accumulator.finalize().erp_rgb
        self.assertTrue(torch.allclose(output[:, 0], torch.full((1, 2, 2), 0.5)))
        self.assertTrue(torch.allclose(output[:, 2], torch.full((1, 2, 2), 0.5)))

    def test_weighted_average_arithmetic(self):
        previous = torch.zeros(1, 3, 1, 1)
        first = torch.ones_like(previous)
        second = torch.zeros_like(previous)
        accumulator = RGBFusionAccumulator(previous, FusionConfig(mode="weighted_average"))
        accumulator.accumulate(contribution(first, 3))
        accumulator.accumulate(contribution(second, 1))
        self.assertTrue(torch.allclose(accumulator.finalize().erp_rgb, torch.full_like(previous, 0.75)))

    def test_uncovered_pixels_keep_previous(self):
        previous = torch.rand(1, 3, 3, 4)
        accumulator = RGBFusionAccumulator(previous, FusionConfig(mode="average"))
        self.assertTrue(torch.equal(accumulator.finalize().erp_rgb, previous))

    def test_spherediff_center_confidence_decreases_to_edge(self):
        weight = create_view_weight_map(31, 31, "spherediff_center", temperature=0.1)
        self.assertGreater(float(weight[0, 0, 15, 15]), float(weight[0, 0, 0, 0]))

    def test_dpa_alpha_zero_is_weighted_average(self):
        values = torch.tensor([[[[[1.0]]]], [[[[3.0]]]]])
        masks = torch.ones(2, 1, 1, 1, 1)
        weights = torch.tensor([1.0, 3.0]).view(2, 1, 1, 1, 1)
        result = detail_preserving_average(
            values, masks, weights, alpha=0, power=2, epsilon=1e-6
        )
        self.assertAlmostEqual(float(result), 2.5, places=5)


class LPWTests(unittest.TestCase):
    def test_pyramid_reconstruction(self):
        image = torch.randn(2, 3, 32, 48)
        pyramid = build_laplacian_pyramid(image, 4, spherical_erp=True)
        self.assertTrue(torch.allclose(reconstruct_laplacian_pyramid(pyramid), image, atol=1e-5))

    def test_lod_nearest_and_linear_are_distinct(self):
        lod = torch.tensor([[[[0.4]]]])
        nearest = lod_level_confidence(lod, 0, 4, "nearest")
        linear = lod_level_confidence(lod, 0, 4, "linear")
        self.assertEqual(float(nearest), 1.0)
        self.assertAlmostEqual(float(linear), 0.6, places=5)

    def test_lpw_runs_as_first_class_warp(self):
        warp_config = WarpConfig(
            mode="lpw",
            erp_to_perspective=SamplingDirectionConfig("nearest"),
            perspective_to_erp=SamplingDirectionConfig("bilinear"),
            lpw=LPWConfig(3, "jacobian", "linear", "reflect"),
        )
        operator = LaplacianPyramidWarpOperator(warp_config, FusionConfig(mode="average"))
        erp = torch.randn(1, 3, 16, 32)
        camera = camera_for_direction(170, 45, height=12, width=12)
        view = operator.erp_to_perspective(erp, camera)
        back = operator.perspective_to_erp(view, camera, (16, 32))
        self.assertEqual(tuple(view.shape), (1, 3, 12, 12))
        self.assertEqual(tuple(back.rgb.shape), (1, 3, 16, 32))
        self.assertIsNotNone(back.lod_map)


if __name__ == "__main__":
    unittest.main()
