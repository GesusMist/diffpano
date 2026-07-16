import math
import unittest

import torch

from pipelines_ours.pixel_fusion import (
    PixelFusionConfig,
    aggregate_overlap_contributions,
    create_patch_weight_map,
)
from pipelines_ours.planar_patch_fusion import (
    PlanarPatchFusionConfig,
    blend_planar_patches,
    build_planar_owner_map,
    build_planar_patch_layout,
    extract_planar_patches,
    planar_patch_prompt_indices,
    scale_planar_patch_layout,
    write_back_planar_latents,
)
from pipelines_ours.spherical_functions import SphericalFunctions


class PlanarPatchLayoutTests(unittest.TestCase):
    def test_layout_covers_edges_when_stride_does_not_divide_extent(self):
        layout = build_planar_patch_layout(8, 11, 3, 4, 2, 3)

        self.assertIn((5, 7), layout.positions)
        coverage = torch.zeros(8, 11, dtype=torch.long)
        for y, x in layout.positions:
            coverage[y:y + layout.patch_height, x:x + layout.patch_width] += 1
        self.assertTrue(coverage.ge(1).all())

    def test_extraction_is_exact_tensor_slicing(self):
        layout = build_planar_patch_layout(5, 7, 3, 3, 2, 2)
        canvas = torch.arange(5 * 7, dtype=torch.float32).reshape(1, 1, 5, 7)

        patches = extract_planar_patches(canvas, layout)

        for patch, (y, x) in zip(patches, layout.positions):
            self.assertTrue(torch.equal(patch, canvas[0, :, y:y + 3, x:x + 3]))

    def test_scaled_layout_preserves_patch_placement(self):
        latent_layout = build_planar_patch_layout(4, 6, 2, 3, 2, 3)
        rgb_layout = scale_planar_patch_layout(latent_layout, 32, 32)

        self.assertEqual((rgb_layout.canvas_height, rgb_layout.canvas_width), (128, 192))
        self.assertEqual((rgb_layout.patch_height, rgb_layout.patch_width), (64, 96))
        self.assertEqual(rgb_layout.positions[-1], (64, 96))


class PlanarPatchBlendTests(unittest.TestCase):
    def _reference_full_canvas_blend(self, patches, layout, config):
        values = patches.new_zeros(
            layout.num_patches,
            patches.shape[1],
            layout.canvas_height,
            layout.canvas_width,
        )
        masks = patches.new_zeros(
            layout.num_patches,
            1,
            layout.canvas_height,
            layout.canvas_width,
        )
        weights = torch.zeros_like(masks)
        patch_weight = create_patch_weight_map(
            layout.patch_height,
            layout.patch_width,
            config.weight_mode,
            device=patches.device,
            dtype=torch.float32,
            eps=config.dpa_eps,
        )
        for patch_index, (patch, (y, x)) in enumerate(zip(patches, layout.positions)):
            values[patch_index, :, y:y + layout.patch_height, x:x + layout.patch_width] = patch
            masks[patch_index, :, y:y + layout.patch_height, x:x + layout.patch_width] = 1
            weights[patch_index, :, y:y + layout.patch_height, x:x + layout.patch_width] = patch_weight
        return aggregate_overlap_contributions(
            values,
            masks,
            weights,
            config.aggregation_mode,
            dpa_alpha=config.dpa_alpha,
            dpa_power=config.dpa_power,
            dpa_eps=config.dpa_eps,
        )

    def test_constant_patches_reconstruct_full_canvas(self):
        layout = build_planar_patch_layout(6, 8, 4, 4, 2, 2)
        patches = torch.full((layout.num_patches, 3, 4, 4), 0.375)
        config = PixelFusionConfig(
            aggregation_mode="detail_preserving_average",
            weight_mode="distance_to_boundary",
        )

        result = blend_planar_patches(patches, layout, config)

        self.assertTrue(torch.allclose(result.fused_values, torch.full((3, 6, 8), 0.375)))
        self.assertTrue(result.valid_output_mask.bool().all())
        self.assertTrue(result.contributor_count.ge(1).all())

    def test_planar_dpa_matches_existing_overlap_formula(self):
        torch.manual_seed(17)
        layout = build_planar_patch_layout(6, 7, 4, 4, 2, 3)
        patches = torch.randn(layout.num_patches, 3, 4, 4)
        config = PixelFusionConfig(
            aggregation_mode="detail_preserving_average",
            weight_mode="gaussian",
            dpa_alpha=0.6,
            dpa_power=1.3,
        )

        planar = blend_planar_patches(patches, layout, config)
        reference = self._reference_full_canvas_blend(patches, layout, config)

        self.assertTrue(torch.allclose(planar.fused_values, reference.fused_values, atol=1e-6))
        self.assertTrue(torch.allclose(planar.accumulated_weight, reference.accumulated_weight, atol=1e-6))
        self.assertTrue(torch.equal(planar.contributor_count, reference.contributor_count))


class PlanarPatchWriteBackTests(unittest.TestCase):
    def test_exclusive_owner_map_covers_and_writes_every_cell_once(self):
        layout = build_planar_patch_layout(5, 7, 3, 3, 2, 2)
        config = PixelFusionConfig(weight_mode="distance_to_boundary")
        owner_map = build_planar_owner_map(layout, config, device=torch.device("cpu"))
        corrected = torch.stack(
            [torch.full((1, 3, 3), float(index)) for index in range(layout.num_patches)]
        )

        output = write_back_planar_latents(
            torch.zeros(1, 1, 5, 7),
            corrected,
            layout,
            config,
            mode="exclusive",
            owner_map=owner_map,
        )

        self.assertTrue(owner_map.covered_mask.all())
        expected = owner_map.owner_patch_id.reshape(1, 1, 5, 7).to(dtype=output.dtype)
        self.assertTrue(torch.equal(output, expected))

    def test_weighted_writeback_preserves_identical_overlaps(self):
        layout = build_planar_patch_layout(5, 7, 3, 3, 2, 2)
        config = PixelFusionConfig(weight_mode="cosine")
        corrected = torch.full((layout.num_patches, 2, 3, 3), -0.25)

        output = write_back_planar_latents(
            torch.zeros(1, 2, 5, 7),
            corrected,
            layout,
            config,
            mode="weighted_average",
        )

        self.assertTrue(torch.allclose(output, torch.full_like(output, -0.25)))


class PlanarPromptTests(unittest.TestCase):
    def test_patch_rows_select_same_five_prompt_groups_as_spherical_pipeline(self):
        layout = build_planar_patch_layout(20, 4, 4, 4, 4, 4)
        thetas = []
        phis = []
        for phi_degrees in (-90, -10, 0, 10, 90):
            for theta_degrees in (0, 90, 180, 270):
                thetas.append(math.radians(theta_degrees))
                phis.append(math.radians(phi_degrees))
        prompt_directions = SphericalFunctions.spherical_to_cartesian(
            torch.tensor(thetas),
            torch.tensor(phis),
        )

        prompt_indices = planar_patch_prompt_indices(layout, prompt_directions)
        prompt_groups = torch.div(prompt_indices, 4, rounding_mode="floor")

        self.assertEqual(prompt_groups.tolist(), [0, 1, 2, 3, 4])

    def test_config_rejects_gapped_patch_stride(self):
        with self.assertRaisesRegex(ValueError, "leave gaps"):
            PlanarPatchFusionConfig(
                patch_latent_height=4,
                patch_latent_width=4,
                patch_stride_height=5,
            ).validate()


if __name__ == "__main__":
    unittest.main()
