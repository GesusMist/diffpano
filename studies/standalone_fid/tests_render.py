"""CPU geometry checks and a singular-covariance check for the FID protocol."""
import math
import unittest

import numpy as np

from studies.standalone_fid.render import render, _rays
from studies.panorama_metrics.render import render as legacy_render
from studies.panorama_metrics.numerics import frechet


def world_texture(height=512, width=1024):
    longitude = ((np.arange(width) + 0.5) / width - 0.5) * 2 * np.pi
    latitude = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    xyz = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat),
                    np.cos(lon) * np.cos(lat)], axis=-1)
    return np.rint((xyz + 1) * 127.5).astype(np.uint8)


class Rendering(unittest.TestCase):
    def test_world_axes_and_poles(self):
        texture = world_texture()
        for yaw, pitch, direction in [
            (0, 0, (0, 0, 1)), (90, 0, (1, 0, 0)),
            (-90, 0, (-1, 0, 0)), (180, 0, (0, 0, -1)),
            (0, 90, (0, 1, 0)), (0, -90, (0, -1, 0)),
        ]:
            with self.subTest(yaw=yaw, pitch=pitch):
                actual = render(texture, yaw, pitch, size=65)
                target = (np.asarray(direction) + 1) * 127.5
                np.testing.assert_allclose(actual[32, 32], target, atol=1)
                self.assertEqual(actual.dtype, np.uint8)
                self.assertEqual(actual.shape, (65, 65, 3))

    def test_analytic_frustum_independent_of_raster(self):
        texture = world_texture()
        extent = math.tan(math.radians(40))
        for size in (65, 129):
            xx, yy = np.meshgrid((2 * (np.arange(size) + 0.5) / size - 1) * extent,
                                 (1 - 2 * (np.arange(size) + 0.5) / size) * extent)
            expected = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
            expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
            actual = render(texture, 0, 0, size=size)
            np.testing.assert_allclose(actual, (expected + 1) * 127.5, atol=1.5)
            ray = _rays(0, 0, size, 80.0)[size // 2, -1]
            self.assertAlmostEqual(ray[0] / ray[2], (1 - 1 / size) * extent)
            # Half a pixel beyond the last centre is the exact 40-degree edge.
            self.assertAlmostEqual(math.degrees(math.atan(ray[0] / ray[2] /
                                                        (1 - 1 / size))), 40.0)
            self.assertGreater(int(actual[size // 2, -1, 0]), int(actual[size // 2, 0, 0]))
            self.assertGreater(int(actual[0, size // 2, 1]), int(actual[-1, size // 2, 1]))

    def test_periodic_seam_blends_last_and_first_columns(self):
        image = np.zeros((32, 64, 3), dtype=np.uint8)
        image[:, 0] = (200, 40, 100)
        image[:, -1] = (0, 160, 240)
        positive = render(image, 180, 0, size=65)
        negative = render(image, -180, 0, size=65)
        np.testing.assert_array_equal(positive, negative)
        np.testing.assert_array_equal(positive[32, 32], (100, 100, 170))

    def test_poles_clamp_latitude_without_crossing(self):
        image = np.zeros((32, 64, 3), dtype=np.uint8)
        image[0] = (10, 80, 200)
        image[-1] = (240, 100, 5)
        np.testing.assert_array_equal(render(image, 0, 90, size=65)[32, 32], image[0, 0])
        np.testing.assert_array_equal(render(image, 0, -90, size=65)[32, 32], image[-1, 0])

    def test_90_degree_legacy_equivalence(self):
        rng = np.random.default_rng(19)
        image = rng.integers(0, 256, (128, 256, 3), dtype=np.uint8)
        for yaw, pitch in ((0, 0), (45, 45), (-70, -45), (180, 0), (0, 90), (225, -90)):
            expected = legacy_render(image, ((yaw, pitch),), size=63)[0]
            np.testing.assert_array_equal(render(image, yaw, pitch, size=63, fov=90), expected)

    def test_invalid_inputs_rejected(self):
        image = np.zeros((8, 16, 3), dtype=np.uint8)
        for args in ({"size": 0}, {"size": 2.5}, {"size": True},
                     {"fov": 180}, {"fov": 0}, {"fov": float("nan")}):
            with self.assertRaises(ValueError):
                render(image, 0, 0, **args)
        with self.assertRaises(ValueError):
            render(image.astype(np.float32), 0, 0)
        with self.assertRaises(ValueError):
            render(image, 0, 91)


class FrechetNumerics(unittest.TestCase):
    def test_full_dimension_singular_covariance(self):
        # Independent covariance-eigenvalue reference with fewer samples than dimensions.
        rng = np.random.default_rng(11)
        x, y = rng.normal(size=(5, 20)), rng.normal(size=(9, 20))
        a, b = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(a)
        root_a = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
        product = root_a @ b @ root_a
        eigenvalues = np.linalg.eigvalsh((product + product.T) / 2)
        covariance_term = np.sqrt(np.maximum(eigenvalues, 0)).sum()
        expected = np.sum((x.mean(0) - y.mean(0)) ** 2) + np.trace(a) + np.trace(b) - 2 * covariance_term
        self.assertAlmostEqual(frechet(x, y), expected, places=5)
        self.assertAlmostEqual(frechet(x, y), frechet(y, x), places=10)
        self.assertLess(frechet(x, x), 1e-10)

    def test_85_reference_2048_dimensions_known_translation(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(85, 2048))
        self.assertAlmostEqual(frechet(x, x + 0.25), 2048 * 0.25 ** 2, places=8)


if __name__ == "__main__":
    unittest.main()
