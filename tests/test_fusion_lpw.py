import unittest
from unittest.mock import patch

import torch

from diffpano.config import FusionConfig, LPWConfig, SamplingDirectionConfig, WarpConfig
from diffpano.fusion import RGBFusionAccumulator, create_view_weight_map, detail_preserving_average
from diffpano.lpw import (
    build_laplacian_pyramid,
    lod_level_confidence,
    reconstruct_laplacian_pyramid,
    reconstruct_masked_laplacian_pyramid,
)
from diffpano.projection import ERPContribution
from diffpano.warp import LaplacianPyramidWarpOperator, StandardWarpOperator
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

    def test_confidence_is_in_numerator_and_denominator(self):
        previous = torch.zeros(1, 3, 1, 1)
        value = torch.ones_like(previous)
        accumulator = RGBFusionAccumulator(
            previous, FusionConfig(mode="average")
        )
        accumulator.accumulate(
            contribution(value), confidence=torch.ones(1, 1, 1, 1)
        )
        accumulator.accumulate(
            contribution(value), confidence=torch.zeros(1, 1, 1, 1)
        )
        result = accumulator.finalize()
        self.assertTrue(torch.equal(result.erp_rgb, value))
        self.assertTrue(
            torch.equal(result.accumulated_weight, torch.ones(1, 1, 1, 1))
        )

    def test_standard_operator_does_not_create_lpw_accumulator(self):
        operator = StandardWarpOperator(
            WarpConfig(mode="standard"),
            FusionConfig(mode="average", weight_mode="uniform"),
        )
        self.assertIsNone(
            operator.create_fusion_accumulator(torch.zeros(1, 3, 8, 16))
        )


class LPWTests(unittest.TestCase):
    @staticmethod
    def operator(
        mode="average",
        *,
        weight_mode="uniform",
        levels=3,
        lod_mode="none",
        lod_interpolation="nearest",
    ):
        warp_config = WarpConfig(
            mode="lpw",
            erp_to_perspective=SamplingDirectionConfig("bilinear"),
            perspective_to_erp=SamplingDirectionConfig("bilinear"),
            lpw=LPWConfig(
                levels, lod_mode, lod_interpolation, "reflect"
            ),
        )
        return LaplacianPyramidWarpOperator(
            warp_config,
            FusionConfig(
                mode=mode,
                weight_mode=weight_mode,
                alpha=1.0,
                power=1.0,
                epsilon=1.0e-6,
            ),
        )

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

    def test_lod_bands_and_coarsest_base_residual(self):
        lod = torch.tensor([[[[1.4]]]])
        nearest = [
            float(lod_level_confidence(lod, level, 4, "nearest"))
            for level in range(4)
        ]
        linear = [
            float(lod_level_confidence(lod, level, 4, "linear"))
            for level in range(4)
        ]
        self.assertEqual(nearest, [0.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(linear[0], 0.0, places=6)
        self.assertAlmostEqual(linear[1], 0.6, places=6)
        self.assertAlmostEqual(linear[2], 0.4, places=6)
        self.assertEqual(linear[3], 1.0)

    def test_masked_reconstruction_does_not_darken_boundary(self):
        fine = torch.zeros(1, 3, 8, 8)
        coarse_mask = torch.zeros(1, 1, 4, 4)
        coarse_mask[:, :, 1:3, 1:3] = 1
        coarse = coarse_mask.expand(1, 3, 4, 4).clone()
        fine_mask = torch.nn.functional.interpolate(
            coarse_mask, size=(8, 8), mode="nearest"
        )
        reconstructed, valid = reconstruct_masked_laplacian_pyramid(
            [fine, coarse], [fine_mask, coarse_mask], 1.0e-6
        )
        covered = valid > 1.0e-6
        self.assertTrue(
            torch.allclose(
                reconstructed[covered.expand_as(reconstructed)],
                torch.ones_like(
                    reconstructed[covered.expand_as(reconstructed)]
                ),
                atol=1.0e-6,
                rtol=0,
            )
        )
        unmasked = reconstruct_laplacian_pyramid([fine, coarse])
        self.assertLess(
            float(unmasked[covered.expand_as(unmasked)].min()), 0.9
        )

    def test_constant_overlaps_stay_constant_near_boundaries(self):
        cameras = [
            camera_for_direction(
                -20, 0, height=24, width=24, fov_x=100, fov_y=90
            ),
            camera_for_direction(
                20, 0, height=24, width=24, fov_x=100, fov_y=90
            ),
        ]
        expected = 0.625
        for mode, weight_mode in (
            ("average", "distance_to_boundary"),
            ("weighted_average", "gaussian"),
        ):
            with self.subTest(mode=mode):
                operator = self.operator(mode, weight_mode=weight_mode)
                accumulator = operator.create_fusion_accumulator(
                    torch.zeros(1, 3, 24, 48)
                )
                for camera in cameras:
                    accumulator.accumulate(
                        torch.full((
                            1, 3, camera.height, camera.width
                        ), expected),
                        camera,
                    )
                result = accumulator.finalize()
                covered = result.coverage_mask.expand_as(result.erp_rgb)
                self.assertGreater(int(covered.sum()), 0)
                self.assertLess(
                    float((result.erp_rgb[covered] - expected).abs().max()),
                    2.0e-5,
                )

    def test_lod_none_keeps_every_pyramid_level_active(self):
        operator = self.operator(levels=4, lod_mode="none")
        accumulator = operator.create_fusion_accumulator(
            torch.zeros(1, 3, 24, 48)
        )
        camera = camera_for_direction(
            0, 0, height=24, width=24, fov_x=100, fov_y=90
        )
        accumulator.accumulate(torch.ones(1, 3, 24, 24), camera)
        self.assertEqual(len(accumulator.level_accumulators), 4)
        self.assertTrue(
            all(
                float(level.ordinary_den.max()) > 0
                for level in accumulator.level_accumulators
            )
        )

    def test_jacobian_nearest_and_linear_inverse_paths_are_finite(self):
        camera = camera_for_direction(
            15, 55, height=16, width=16, fov_x=90, fov_y=80
        )
        view = torch.randn(1, 3, 16, 16)
        for interpolation in ("nearest", "linear"):
            with self.subTest(interpolation=interpolation):
                operator = self.operator(
                    levels=4,
                    lod_mode="jacobian",
                    lod_interpolation=interpolation,
                )
                accumulator = operator.create_fusion_accumulator(
                    torch.zeros(1, 3, 20, 40)
                )
                accumulator.accumulate(view, camera)
                result = accumulator.finalize()
                self.assertTrue(bool(torch.isfinite(result.erp_rgb).all()))
                self.assertTrue(
                    bool(accumulator.level_accumulators[-1].ordinary_den.gt(0).any())
                )

    def test_forward_lpw_preserves_constant_at_seam_and_poles(self):
        operator = self.operator(levels=4, lod_mode="none")
        erp = torch.full((1, 3, 24, 48), 0.375)
        cameras = [
            camera_for_direction(
                180, 0, height=16, width=16, fov_x=80, fov_y=80
            ),
            camera_for_direction(
                0, 85, height=16, width=16, fov_x=80, fov_y=80
            ),
            camera_for_direction(
                0, -85, height=16, width=16, fov_x=80, fov_y=80
            ),
        ]
        for camera in cameras:
            view = operator.erp_to_perspective(erp, camera)
            self.assertTrue(
                torch.allclose(
                    view, torch.full_like(view, 0.375), atol=2.0e-5, rtol=0
                )
            )

    def test_dpa_fuses_coefficients_before_reconstruction(self):
        operator = self.operator(
            "detail_preserving_average", levels=2, lod_mode="none"
        )
        accumulator = operator.create_fusion_accumulator(
            torch.zeros(1, 3, 2, 2)
        )
        fine_a = torch.full((1, 3, 2, 2), 2.0)
        coarse_a = torch.zeros(1, 3, 1, 1)
        fine_b = torch.zeros(1, 3, 2, 2)
        coarse_b = torch.full((1, 3, 1, 1), 2.0)
        pyramids = [[fine_a, coarse_a], [fine_b, coarse_b]]

        def fake_project(values, camera, height, width, **kwargs):
            del camera, kwargs
            self.assertEqual(tuple(values.shape[-2:]), (height, width))
            mask = torch.ones(values.shape[0], 1, height, width)
            return ERPContribution(values.clone(), mask, mask)

        camera = camera_for_direction(0, 0, height=2, width=2)
        with patch(
            "diffpano.warp.build_laplacian_pyramid",
            side_effect=pyramids,
        ), patch("diffpano.warp.perspective_to_erp", side_effect=fake_project):
            accumulator.accumulate(torch.ones(1, 3, 2, 2), camera)
            accumulator.accumulate(torch.full((1, 3, 2, 2), 2.0), camera)
        result = accumulator.finalize().erp_rgb

        fused_levels = []
        for first, second in zip(pyramids[0], pyramids[1]):
            values = torch.stack([first, second])
            masks = torch.ones(
                2, 1, 1, *first.shape[-2:]
            )
            fused_levels.append(
                detail_preserving_average(
                    values,
                    masks,
                    masks,
                    alpha=1.0,
                    power=1.0,
                    epsilon=1.0e-6,
                )
            )
        expected = reconstruct_laplacian_pyramid(fused_levels)
        reconstructed_views = torch.stack(
            [
                reconstruct_laplacian_pyramid(pyramids[0]),
                reconstruct_laplacian_pyramid(pyramids[1]),
            ]
        )
        full_masks = torch.ones(2, 1, 1, 2, 2)
        reconstruct_then_dpa = detail_preserving_average(
            reconstructed_views,
            full_masks,
            full_masks,
            alpha=1.0,
            power=1.0,
            epsilon=1.0e-6,
        )
        self.assertTrue(torch.allclose(result, expected, atol=1.0e-6))
        self.assertFalse(
            torch.allclose(result, reconstruct_then_dpa, atol=1.0e-4)
        )

    def test_streaming_camera_order_invariance_all_fusion_modes(self):
        cameras = [
            camera_for_direction(
                yaw, 0, height=12, width=12, fov_x=100, fov_y=90
            )
            for yaw in (-30, 0, 30)
        ]
        grid = torch.linspace(-1, 1, 12)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        views = [
            torch.stack(
                [xx + offset, yy - offset, xx * yy + offset], dim=0
            ).unsqueeze(0)
            for offset in (-0.2, 0.0, 0.25)
        ]
        for mode in (
            "average",
            "weighted_average",
            "detail_preserving_average",
        ):
            with self.subTest(mode=mode):
                forward = self.operator(
                    mode, weight_mode="gaussian"
                ).create_fusion_accumulator(torch.zeros(1, 3, 12, 24))
                reverse = self.operator(
                    mode, weight_mode="gaussian"
                ).create_fusion_accumulator(torch.zeros(1, 3, 12, 24))
                for view, camera in zip(views, cameras):
                    forward.accumulate(view, camera)
                for view, camera in reversed(list(zip(views, cameras))):
                    reverse.accumulate(view, camera)
                first = forward.finalize()
                second = reverse.finalize()
                self.assertTrue(
                    torch.equal(first.coverage_mask, second.coverage_mask)
                )
                self.assertTrue(
                    torch.allclose(
                        first.erp_rgb, second.erp_rgb, atol=2.0e-6, rtol=0
                    )
                )

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
        accumulator = operator.create_fusion_accumulator(
            torch.zeros_like(erp)
        )
        diagnostic = accumulator.accumulate(view, camera)
        back = accumulator.finalize()
        self.assertEqual(tuple(view.shape), (1, 3, 12, 12))
        self.assertEqual(tuple(back.erp_rgb.shape), (1, 3, 16, 32))
        self.assertIsNotNone(diagnostic.lod_map)
        with self.assertRaises(RuntimeError):
            operator.perspective_to_erp(view, camera, (16, 32))


if __name__ == "__main__":
    unittest.main()
