import math
import unittest

import torch

from diffpano.camera import (
    SphereDiffFixedCameraSampler,
    SphereDiffRotatedCameraSampler,
    camera_for_direction,
)
from diffpano.config import RotationConfig, ViewConfig
from diffpano.geometry import pairwise_angular_distances
from diffpano.projection import (
    ProjectionCache,
    erp_to_perspective,
    perspective_to_erp,
    perspective_to_erp_grid,
    projection_lod_map,
)


class CameraSamplerTests(unittest.TestCase):
    def test_spherediff_fixed_is_original_89_view_cover(self):
        cameras = SphereDiffFixedCameraSampler(ViewConfig(height=8, width=8)).sample(0, 20)
        self.assertEqual(len(cameras), 89)
        self.assertTrue(all(camera.fov_x == 80 and camera.fov_y == 80 for camera in cameras))

    def test_rotated_cover_is_one_rigid_deterministic_rotation(self):
        view = ViewConfig(height=8, width=8)
        rotation = RotationConfig(True, 8, 4, 3)
        sampler = SphereDiffRotatedCameraSampler(view, rotation, seed=17)
        first = sampler.sample(2, 5)
        repeated = sampler.sample(2, 5)
        base = SphereDiffFixedCameraSampler(view).sample(0, 5)
        first_dirs = torch.stack([camera.forward() for camera in first])
        repeated_dirs = torch.stack([camera.forward() for camera in repeated])
        base_dirs = torch.stack([camera.forward() for camera in base])
        self.assertTrue(torch.equal(first_dirs, repeated_dirs))
        self.assertTrue(
            torch.allclose(
                pairwise_angular_distances(first_dirs),
                pairwise_angular_distances(base_dirs),
                atol=2e-3,
                rtol=0,
            )
        )

    def test_fixed_cover_has_full_erp_coverage_and_overlap(self):
        view = ViewConfig(height=8, width=8)
        cameras = SphereDiffFixedCameraSampler(view).sample(0, 1)
        coverage = torch.zeros(1, 1, 24, 48)
        cache = ProjectionCache()
        for camera in cameras:
            _, mask = perspective_to_erp_grid(camera, 24, 48, device=torch.device("cpu"), cache=cache)
            coverage += mask
        self.assertTrue(bool((coverage > 0).all()))
        self.assertGreater(float((coverage > 1).float().mean()), 0.95)

    def test_rotated_cover_preserves_full_expected_coverage(self):
        view = ViewConfig(height=8, width=8)
        cameras = SphereDiffRotatedCameraSampler(
            view, RotationConfig(True, 8, 4, 3), seed=3
        ).sample(1, 3)
        coverage = torch.zeros(1, 1, 20, 40)
        cache = ProjectionCache()
        for camera in cameras:
            _, mask = perspective_to_erp_grid(camera, 20, 40, device=torch.device("cpu"), cache=cache)
            coverage += mask
        self.assertTrue(bool((coverage > 0).all()))
        self.assertGreater(float((coverage > 1).float().mean()), 0.95)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.cache = ProjectionCache()

    def test_camera_center_samples_expected_erp_pixel(self):
        erp = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).expand(1, 3, 4, 8)
        camera = camera_for_direction(22.5, 0, height=1, width=1)
        view = erp_to_perspective(erp, camera, interpolation="nearest", cache=self.cache)
        self.assertEqual(float(view[0, 0, 0, 0]), 4.0)

    def test_bilinear_is_independently_selectable_and_numeric(self):
        erp = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).expand(1, 3, 4, 8)
        camera = camera_for_direction(0, 0, height=1, width=1)
        nearest = erp_to_perspective(erp, camera, interpolation="nearest")
        bilinear = erp_to_perspective(erp, camera, interpolation="bilinear")
        self.assertIn(float(nearest[0, 0, 0, 0]), {3.0, 4.0})
        self.assertAlmostEqual(float(bilinear[0, 0, 0, 0]), 3.5, places=5)

    def test_seam_crossing_camera_samples_both_erp_edges(self):
        erp = torch.zeros(1, 3, 8, 32)
        erp[:, 0, :, :3] = 1
        erp[:, 2, :, -3:] = 1
        camera = camera_for_direction(180, 0, height=5, width=13, fov_x=60, fov_y=40)
        view = erp_to_perspective(erp, camera, interpolation="nearest", cache=self.cache)
        self.assertGreater(float(view[:, 0].max()), 0.9)
        self.assertGreater(float(view[:, 2].max()), 0.9)

    def test_north_and_south_pole_projection(self):
        erp = torch.zeros(1, 3, 12, 24)
        erp[:, :, 0] = 1
        erp[:, :, -1] = -1
        north = camera_for_direction(0, 90, height=1, width=1)
        south = camera_for_direction(0, -90, height=1, width=1)
        self.assertGreater(float(erp_to_perspective(erp, north, interpolation="nearest").mean()), 0.9)
        self.assertLess(float(erp_to_perspective(erp, south, interpolation="nearest").mean()), -0.9)

    def test_perspective_footprint_is_irregular_not_a_resized_rectangle(self):
        camera = camera_for_direction(0, 0, height=16, width=16, fov_x=90, fov_y=90)
        _, mask = perspective_to_erp_grid(camera, 32, 64, device=torch.device("cpu"))
        widths = mask[0, 0].sum(dim=1)
        nonzero = widths[widths > 0]
        self.assertGreater(nonzero.numel(), 2)
        self.assertGreater(torch.unique(nonzero).numel(), 1)

    def test_projection_round_trip_is_accurate_inside_covered_region(self):
        y = torch.linspace(-1, 1, 32).view(1, 1, 32, 1)
        x = torch.sin(torch.linspace(-math.pi, math.pi, 64)).view(1, 1, 1, 64)
        erp = torch.cat([x.expand(1, 1, 32, 64), y.expand(1, 1, 32, 64), (x * y)], dim=1)
        camera = camera_for_direction(30, 10, height=32, width=32, fov_x=80, fov_y=80)
        view = erp_to_perspective(erp, camera, interpolation="bilinear", cache=self.cache)
        back = perspective_to_erp(
            view,
            camera,
            32,
            64,
            interpolation="bilinear",
            weight_map=torch.ones(1, 1, 32, 32),
            cache=self.cache,
        )
        interior = (back.valid_mask > 0) & (back.weight > 0.999)
        error = (back.rgb - erp).abs()[interior.expand_as(erp)]
        self.assertLess(float(error.mean()), 0.08)

    def test_jacobian_lod_is_finite_and_nonnegative(self):
        camera = camera_for_direction(0, 55, height=16, width=16)
        lod = projection_lod_map(camera, 24, 48, device=torch.device("cpu"))
        self.assertTrue(bool(torch.isfinite(lod).all()))
        self.assertGreaterEqual(float(lod.min()), 0.0)


if __name__ == "__main__":
    unittest.main()
