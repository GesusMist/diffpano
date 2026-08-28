import math
import unittest

import torch

from diffpano.pixel_fusion import (
    PixelFusionConfig,
    aggregate_overlap_contributions,
    build_identity_preserving_vae_target,
    create_patch_weight_map,
    decode_view_latents,
    predict_clean_latents,
    step_with_fused_clean_prediction,
)
from experiments.planar.fusion import (
    PlanarPatchFusionConfig,
    blend_planar_patches,
    build_planar_owner_map,
    build_planar_patch_layout,
    build_planar_patch_layout_for_step,
    extract_planar_patches,
    planar_patch_prompt_indices,
    scale_planar_patch_layout,
    write_back_planar_latents,
)
from diffpano.geometry import SphericalFunctions


class _PlanarEncodeOutput:
    def __init__(self, latent):
        self.latent = latent


class _PlanarVAEConfig:
    scaling_factor = 1.0
    shift_factor = None


class _NonPerfectPlanarVAE:
    dtype = torch.float32
    config = _PlanarVAEConfig()

    def decode(self, latents, return_dict=False):
        return (torch.tanh(latents[:, :3]),)

    def encode(self, images):
        extra = torch.full(
            (images.shape[0], 1, images.shape[-2], images.shape[-1]),
            0.25,
            device=images.device,
            dtype=images.dtype,
        )
        return _PlanarEncodeOutput(torch.cat([images * 0.75, extra], dim=1))


class _PlanarFlowConfig:
    prediction_type = "flow_prediction"


class _PlanarFlowScheduler:
    config = _PlanarFlowConfig()

    def __init__(self):
        self.timesteps = torch.tensor([1000.0])
        self.sigmas = torch.tensor([1.0, 0.5])
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps):
        return 0

    def step(self, model_output, timestep, sample, return_dict=False):
        if self._step_index is None:
            self._step_index = 0
        previous = sample.float() + (self.sigmas[1] - self.sigmas[0]) * model_output.float()
        self._step_index += 1
        return (previous.to(dtype=model_output.dtype),)


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

    def test_dynamic_layout_shifts_interior_and_anchors_horizontal_edges(self):
        config = PlanarPatchFusionConfig(
            patch_latent_height=5,
            patch_latent_width=5,
            patch_stride_height=5,
            patch_stride_width=4,
            patch_strategy="dynamic",
            dynamic_patch_step_size=1,
        )

        step_zero = build_planar_patch_layout_for_step(5, 13, config, 0)
        step_one = build_planar_patch_layout_for_step(5, 13, config, 1)

        self.assertEqual(step_zero.positions, ((0, 0), (0, 4), (0, 8)))
        self.assertEqual(step_one.positions, ((0, 0), (0, 1), (0, 5), (0, 8)))
        self.assertEqual(step_one.positions[0], step_zero.positions[0])
        self.assertEqual(step_one.positions[-1], step_zero.positions[-1])

    def test_dynamic_layout_is_periodic_and_preserves_full_coverage(self):
        config = PlanarPatchFusionConfig(
            patch_latent_height=4,
            patch_latent_width=5,
            patch_stride_height=2,
            patch_stride_width=4,
            patch_strategy="dynamic",
            dynamic_patch_step_size=3,
        )

        for step_index in range(8):
            layout = build_planar_patch_layout_for_step(9, 14, config, step_index)
            coverage = torch.zeros(layout.canvas_height, layout.canvas_width, dtype=torch.bool)
            for y, x in layout.positions:
                coverage[y:y + layout.patch_height, x:x + layout.patch_width] = True
            self.assertTrue(coverage.all())
        self.assertEqual(
            build_planar_patch_layout_for_step(9, 14, config, 0),
            build_planar_patch_layout_for_step(9, 14, config, 4),
        )


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

    def test_noop_planar_rgb_fusion_is_identity_at_zero_and_full_reinjection_strength(self):
        torch.manual_seed(23)
        layout = build_planar_patch_layout(4, 6, 2, 3, 1, 2)
        clean_grid = torch.randn(1, 4, 4, 6)
        clean_patches = extract_planar_patches(clean_grid, layout)
        config = PixelFusionConfig(aggregation_mode="average", weight_mode="uniform")
        decoded_patches = decode_view_latents(_NonPerfectPlanarVAE(), clean_patches, config)
        fused_canvas = blend_planar_patches(decoded_patches, layout, config).fused_values.unsqueeze(0)
        fused_patches = extract_planar_patches(fused_canvas, layout)
        bridge = build_identity_preserving_vae_target(
            _NonPerfectPlanarVAE(),
            clean_patches,
            decoded_patches,
            fused_patches,
            config,
        )

        self.assertGreater(bridge.vae_roundtrip_error_norm.item(), 0)
        self.assertTrue(torch.allclose(bridge.fusion_delta_vae_latents, torch.zeros_like(clean_patches), atol=1e-7))
        scheduler = _PlanarFlowScheduler()
        timestep = torch.tensor(1000.0)
        current = torch.randn_like(clean_patches)
        flow = current - clean_patches
        predicted_clean, _, _ = predict_clean_latents(scheduler, flow, timestep, current)
        original_prev = scheduler.step(flow, timestep, current, return_dict=False)[0]

        zero_strength = step_with_fused_clean_prediction(
            scheduler,
            timestep,
            current,
            flow,
            predicted_clean,
            bridge.target_clean_latents,
            0.0,
            original_prev_latents=original_prev,
        )
        full_strength = step_with_fused_clean_prediction(
            scheduler,
            timestep,
            current,
            flow,
            predicted_clean,
            bridge.target_clean_latents,
            1.0,
            original_prev_latents=original_prev,
        )
        self.assertTrue(torch.equal(zero_strength, original_prev))
        self.assertTrue(torch.allclose(full_strength, original_prev, atol=1e-6, rtol=0))


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

    def test_latent_fusion_mode_disables_pixel_fusion(self):
        config = PlanarPatchFusionConfig(fusion_space="latent")

        config.validate()

        self.assertFalse(config.to_pixel_fusion_config().pixel_fusion_enabled)

    def test_config_rejects_unknown_fusion_space(self):
        with self.assertRaisesRegex(ValueError, "fusion_space"):
            PlanarPatchFusionConfig(fusion_space="frequency").validate()

    def test_config_rejects_unknown_patch_strategy(self):
        with self.assertRaisesRegex(ValueError, "patch_strategy"):
            PlanarPatchFusionConfig(patch_strategy="random").validate()


if __name__ == "__main__":
    unittest.main()
