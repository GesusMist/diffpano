import math
import unittest
from pathlib import Path
from unittest import mock

import torch

import pipelines_ours.pixel_fusion as pixel_fusion_module
from pipelines_ours.pixel_fusion import (
    PixelFusionConfig,
    aggregate_overlap_contributions,
    apply_pixel_space_fusion,
    build_exclusive_owner_map,
    circular_pad_horizontal,
    decode_view_latents,
    detail_preserving_average,
    encode_view_images,
    exclusive_owner_diagnostics,
    extract_views_from_erp_standard,
    forward_lpw_to_views,
    get_or_build_exclusive_owner_map,
    inverse_lpw_to_erp,
    project_views_to_erp_standard,
    predict_clean_latents,
    reconstruct_laplacian_pyramid,
    reinject_fused_latents,
    render_views_to_erp_standard_weighted,
    spherical_pad_erp,
    build_laplacian_pyramid,
    should_apply_pixel_fusion,
    temporary_save_fused_clean_erp_debug,
    temporary_save_original_clean_erp_debug,
    write_back_views_exclusive,
    write_back_views_weighted_average,
)
from pipelines_ours.spherical_functions import SphericalFunctions


class FakeLatentDist:
    def __init__(self, mean):
        self.mean = mean

    def sample(self, generator=None):
        return self.mean


class FakeEncodeOutput:
    def __init__(self, mean):
        self.latent_dist = FakeLatentDist(mean)


class FakeDirectEncodeOutput:
    def __init__(self, latent):
        self.latent = latent


class FakeVAEConfig:
    scaling_factor = 0.5
    shift_factor = 0.1


class FakeVAE:
    dtype = torch.float32
    config = FakeVAEConfig()

    def decode(self, latents, return_dict=False):
        image = torch.tanh(latents[:, :3])
        return (image,)

    def encode(self, images):
        extra = torch.zeros(images.shape[0], 1, images.shape[-2], images.shape[-1], device=images.device, dtype=images.dtype)
        return FakeEncodeOutput(torch.cat([images, extra], dim=1))


class FakeDirectEncodeVAE(FakeVAE):
    def encode(self, images):
        extra = torch.zeros(images.shape[0], 1, images.shape[-2], images.shape[-1])
        return FakeDirectEncodeOutput(torch.cat([images, extra], dim=1))


class FakeFlowMatchScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([1000.0, 500.0])
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])

    def index_for_timestep(self, timestep, schedule_timesteps):
        return int((schedule_timesteps == timestep).nonzero().flatten()[0])


class FakeSchedulerConfig:
    prediction_type = "flow_prediction"


class FakeDPMSolverScheduler(FakeFlowMatchScheduler):
    config = FakeSchedulerConfig()


class PixelFusionTests(unittest.TestCase):
    @staticmethod
    def _exclusive_fixture(num_points=5):
        indices = [torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.tensor([2, 4])]
        scores = [torch.tensor([0.5, 0.8, 0.4]), torch.tensor([0.9, 0.4, 0.7]), torch.tensor([0.4, 0.5])]
        patch_ids = [20, 10, 30]
        owner_map = build_exclusive_owner_map(
            num_points,
            indices,
            scores,
            patch_ids,
            device=torch.device("cpu"),
        )
        return indices, scores, patch_ids, owner_map

    def test_exclusive_owner_map_uses_max_score_stable_ties_and_coverage(self):
        _, _, _, owner_map = self._exclusive_fixture()
        self.assertTrue(torch.equal(owner_map.owner_patch_id, torch.tensor([20, 10, 10, 10, 30])))
        self.assertTrue(torch.equal(owner_map.coverage_count, torch.tensor([1, 2, 3, 1, 1])))
        self.assertTrue(owner_map.covered_mask.all())
        self.assertEqual(owner_map.owner_score.dtype, torch.float32)

    def test_exclusive_owner_map_is_independent_of_record_order(self):
        indices, scores, patch_ids, expected = self._exclusive_fixture()
        order = [2, 0, 1]
        reordered = build_exclusive_owner_map(
            5,
            [indices[index] for index in order],
            [scores[index] for index in order],
            [patch_ids[index] for index in order],
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.equal(reordered.owner_patch_id, expected.owner_patch_id))
        self.assertTrue(torch.equal(reordered.owner_score, expected.owner_score))
        self.assertTrue(torch.equal(reordered.coverage_count, expected.coverage_count))

    def test_exclusive_owner_map_detects_uncovered_points(self):
        _, _, _, owner_map = self._exclusive_fixture(num_points=6)
        self.assertFalse(owner_map.covered_mask[-1])
        self.assertEqual(owner_map.owner_patch_id[-1].item(), -1)
        self.assertEqual(owner_map.coverage_count[-1].item(), 0)

    def test_exclusive_writeback_selects_exact_owner_values_once(self):
        indices, _, patch_ids, owner_map = self._exclusive_fixture()
        patches = [
            torch.tensor([[[[200.0, 201.0, 202.0]]]]),
            torch.tensor([[[[101.0, 102.0, 103.0]]]]),
            torch.tensor([[[[302.0, 304.0]]]]),
        ]
        result = write_back_views_exclusive(torch.zeros(1, 1, 1, 5), patches, indices, patch_ids, owner_map)
        self.assertTrue(torch.equal(result.latents.flatten(), torch.tensor([200.0, 101.0, 102.0, 103.0, 304.0])))
        self.assertTrue(result.exclusive_write_count.eq(1).all())

    def test_non_owner_values_do_not_affect_exclusive_writeback(self):
        indices, _, patch_ids, owner_map = self._exclusive_fixture()
        patches = [torch.full((1, 1, 1, len(item)), float(patch_id)) for item, patch_id in zip(indices, patch_ids)]
        first = write_back_views_exclusive(torch.zeros(1, 1, 1, 5), patches, indices, patch_ids, owner_map).latents
        patches[0][..., 1:] = -999.0
        patches[2][..., :1] = -999.0
        second = write_back_views_exclusive(torch.zeros(1, 1, 1, 5), patches, indices, patch_ids, owner_map).latents
        self.assertTrue(torch.equal(first, second))

    def test_exclusive_writeback_is_independent_of_patch_order(self):
        indices, _, patch_ids, owner_map = self._exclusive_fixture()
        patches = [torch.full((1, 1, 1, len(item)), float(patch_id)) for item, patch_id in zip(indices, patch_ids)]
        expected = write_back_views_exclusive(torch.zeros(1, 1, 1, 5), patches, indices, patch_ids, owner_map).latents
        order = [1, 2, 0]
        actual = write_back_views_exclusive(
            torch.zeros(1, 1, 1, 5),
            [patches[index] for index in order],
            [indices[index] for index in order],
            [patch_ids[index] for index in order],
            owner_map,
        ).latents
        self.assertTrue(torch.equal(actual, expected))

    def test_exclusive_writeback_supports_one_patch_and_sana_layout(self):
        indices = [torch.tensor([0, 1, 2])]
        owner_map = build_exclusive_owner_map(3, indices, [torch.ones(3)], [7], device=torch.device("cpu"))
        patch = torch.arange(12.0).reshape(2, 2, 1, 3)
        result = write_back_views_exclusive(torch.zeros_like(patch), [patch], indices, [7], owner_map)
        self.assertTrue(torch.equal(result.latents, patch))
        self.assertEqual(result.latents.shape, (2, 2, 1, 3))

    def test_exclusive_writeback_supports_flux_restored_spherical_layout(self):
        indices = [torch.tensor([0, 1, 2, 3])]
        owner_map = build_exclusive_owner_map(4, indices, [torch.ones(4)], [3], device=torch.device("cpu"))
        restored_flux_patch = torch.arange(64.0).reshape(1, 16, 1, 4).to(torch.bfloat16)
        result = write_back_views_exclusive(
            torch.zeros_like(restored_flux_patch), [restored_flux_patch], indices, [3], owner_map
        )
        self.assertTrue(torch.equal(result.latents, restored_flux_patch))
        self.assertEqual(result.latents.dtype, torch.bfloat16)

    def test_weighted_writeback_matches_original_accumulation(self):
        indices = [torch.tensor([0, 1]), torch.tensor([1, 2])]
        scores = [torch.tensor([[[1.0, 2.0]]]), torch.tensor([[[3.0, 1.0]]])]
        patches = [torch.tensor([[[[2.0, 4.0]]]]), torch.tensor([[[[10.0, 20.0]]]])]
        result = write_back_views_weighted_average(torch.zeros(1, 1, 1, 3), patches, indices, scores)
        expected = torch.tensor([[[[2.0, 7.6, 20.0]]]])
        self.assertTrue(torch.allclose(result, expected))

    def test_exclusive_uncovered_error_includes_required_context(self):
        indices, _, patch_ids, owner_map = self._exclusive_fixture(num_points=6)
        patches = [torch.zeros(1, 1, 1, len(item)) for item in indices]
        with self.assertRaisesRegex(RuntimeError, r"1/6 uncovered points .*3 patches; view geometry: test geometry"):
            write_back_views_exclusive(
                torch.zeros(1, 1, 1, 6),
                patches,
                indices,
                patch_ids,
                owner_map,
                geometry_summary="test geometry",
            )

    def test_weighted_fallback_changes_only_uncovered_points(self):
        indices, _, patch_ids, owner_map = self._exclusive_fixture(num_points=6)
        patches = [torch.full((1, 1, 1, len(item)), float(patch_id)) for item, patch_id in zip(indices, patch_ids)]
        fallback = torch.full((1, 1, 1, 6), 77.0)
        result = write_back_views_exclusive(
            torch.zeros_like(fallback),
            patches,
            indices,
            patch_ids,
            owner_map,
            uncovered_mode="weighted_average_fallback",
            weighted_average_fallback=fallback,
        )
        self.assertEqual(result.latents[..., -1].item(), 77.0)
        self.assertTrue(torch.equal(result.latents.flatten()[:-1], torch.tensor([20.0, 10.0, 10.0, 10.0, 30.0])))

    def test_static_owner_cache_matches_rebuild_and_invalidates_on_geometry_change(self):
        indices, scores, patch_ids, _ = self._exclusive_fixture()
        directions = [torch.tensor([[float(index), 0.0, 1.0]]) for index in range(3)]
        fovs = [(80.0, 80.0)] * 3
        config = PixelFusionConfig(exclusive_owner_map_static=True)
        first, _, first_reused = get_or_build_exclusive_owner_map(
            5, indices, scores, patch_ids, directions, fovs, config, device=torch.device("cpu")
        )
        second, _, second_reused = get_or_build_exclusive_owner_map(
            5, indices, scores, patch_ids, directions, fovs, config, device=torch.device("cpu")
        )
        changed_scores = [item.clone() for item in scores]
        changed_scores[0][0] += 0.1
        rebuilt, _, rebuilt_reused = get_or_build_exclusive_owner_map(
            5, indices, changed_scores, patch_ids, directions, fovs, config, device=torch.device("cpu")
        )
        self.assertFalse(first_reused)
        self.assertTrue(second_reused)
        self.assertIs(first, second)
        self.assertFalse(rebuilt_reused)
        self.assertIsNot(first, rebuilt)

    def test_exclusive_diagnostics_report_owner_and_write_counts(self):
        indices, _, patch_ids, owner_map = self._exclusive_fixture()
        patches = [torch.zeros(1, 1, 1, len(item)) for item in indices]
        result = write_back_views_exclusive(torch.zeros(1, 1, 1, 5), patches, indices, patch_ids, owner_map)
        diagnostics = exclusive_owner_diagnostics(owner_map, result.exclusive_write_count, patch_ids)
        for key in (
            "owner_patch_id",
            "owner_score",
            "coverage_count",
            "covered_mask",
            "owner_patch_histogram",
            "uncovered_count",
            "multiply_covered_count",
            "exclusive_write_count",
        ):
            self.assertIn(key, diagnostics)
        self.assertTrue(diagnostics["exclusive_write_count"].eq(1).all())

    @staticmethod
    def asymmetric_view(size=16):
        y = torch.linspace(-1, 1, size)
        x = torch.linspace(-1, 1, size)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        checker = (((torch.arange(size)[:, None] + 2 * torch.arange(size)[None, :]) % 5) < 2).float() * 2 - 1
        return torch.stack([xx, yy, checker * (0.25 + 0.75 * (xx > yy))], dim=0).unsqueeze(0)

    def test_erp_world_grid_round_trips_exact_pixel_centers(self):
        height, width = 16, 32
        world = pixel_fusion_module._erp_world_grid(height, width, device=torch.device("cpu"), dtype=torch.float32)
        grid = pixel_fusion_module._world_to_erp_grid(
            world.reshape(height, width, 3),
            erp_height=height,
            erp_width=width,
        )
        expected_x = 2 * (torch.arange(width, dtype=torch.float32) + 0.5) / width - 1
        expected_y = 2 * (torch.arange(height, dtype=torch.float32) + 0.5) / height - 1
        expected_x, expected_y = torch.meshgrid(expected_x, expected_y, indexing="xy")
        longitude_error = torch.remainder(grid[..., 0] - expected_x + 1, 2) - 1
        self.assertLess(longitude_error.abs().max().item(), 2e-6)
        self.assertLess((grid[..., 1] - expected_y).abs().max().item(), 2e-6)

    def test_erp_impulse_sampling_targets_pixel_centers(self):
        height, width = 8, 16
        locations = [(1, 1), (height // 2, width // 2), (2, width - 2), (1, 6), (height - 2, 10)]
        for y, x in locations:
            erp = torch.zeros(1, 1, height, width)
            erp[0, 0, y, x] = 1
            grid = torch.tensor([[[[2 * (x + 0.5) / width - 1, 2 * (y + 0.5) / height - 1]]]])
            sampled = pixel_fusion_module._sample_erp_image(erp, grid)
            self.assertEqual(sampled.item(), 1.0)

    def test_one_patch_returns_itself_for_constant_patch(self):
        config = PixelFusionConfig(weight_mode="uniform")
        view = torch.full((1, 3, 8, 8), 0.25)
        view_dir = torch.tensor([[0.0, 0.0, 1.0]])
        projected, mask, weight = project_views_to_erp_standard(view, view_dir, [(80, 80)], 32, 64, config)
        aggregate = aggregate_overlap_contributions(projected, mask, weight, "weighted_average")
        extracted, valid = extract_views_from_erp_standard(aggregate.fused_values, aggregate.valid_output_mask, view, view_dir, [(80, 80)], config)
        self.assertFalse(torch.isnan(extracted).any())
        self.assertTrue(torch.allclose(extracted, view, atol=1e-4))
        self.assertGreater(valid.sum().item(), 0)

    def test_one_textured_patch_round_trip_preserves_spatial_content(self):
        config = PixelFusionConfig(weight_mode="uniform", projection_chunk_size=1)
        y = torch.linspace(-1, 1, 16)
        x = torch.linspace(-1, 1, 16)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        view = torch.stack([xx, yy, torch.sin(xx * math.pi * 3) * torch.cos(yy * math.pi * 2)], dim=0).unsqueeze(0)
        view_dir = torch.tensor([[0.35, -0.2, 0.9151503]])
        projected, mask, weight = project_views_to_erp_standard(view, view_dir, [(80, 80)], 128, 256, config)
        aggregate = aggregate_overlap_contributions(projected, mask, weight, "weighted_average")
        extracted, valid = extract_views_from_erp_standard(
            aggregate.fused_values,
            aggregate.valid_output_mask,
            view,
            view_dir,
            [(80, 80)],
            config,
        )
        valid_error = ((extracted - view).abs() * valid).sum() / valid.sum().clamp_min(1)
        self.assertLess(valid_error.item(), 0.08)

    def test_asymmetric_patch_round_trip_across_view_directions(self):
        config = PixelFusionConfig(weight_mode="uniform", projection_chunk_size=1)
        view = self.asymmetric_view()
        inv_sqrt = 1 / math.sqrt(2)
        view_dirs = [
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, -1.0]]),
            torch.tensor([[-1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, inv_sqrt, inv_sqrt]]),
            torch.tensor([[0.0, -inv_sqrt, inv_sqrt]]),
            SphericalFunctions.spherical_to_cartesian(torch.tensor([math.pi - 1e-4]), torch.tensor([0.0])),
        ]
        for view_dir in view_dirs:
            projected, mask, weight = project_views_to_erp_standard(
                view, view_dir, [(100, 100)], 128, 256, config
            )
            aggregate = aggregate_overlap_contributions(projected, mask, weight, "weighted_average")
            extracted, valid = extract_views_from_erp_standard(
                aggregate.fused_values,
                aggregate.valid_output_mask,
                view,
                view_dir,
                [(100, 100)],
                config,
            )
            valid_error = ((extracted - view).abs() * valid).sum() / (valid.sum().clamp_min(1) * view.shape[1])
            self.assertLess(valid_error.item(), 0.08, f"view_dir={view_dir.tolist()}")

    def test_bfloat16_inputs_use_float32_projection_and_fusion(self):
        config = PixelFusionConfig(weight_mode="uniform")
        view = torch.linspace(-1, 1, 3 * 8 * 8, dtype=torch.bfloat16).reshape(1, 3, 8, 8)
        view_dir = torch.tensor([[0.35, -0.2, 0.9151503]], dtype=torch.bfloat16)
        projected, mask, weight = project_views_to_erp_standard(
            view,
            view_dir,
            [(80, 80)],
            32,
            64,
            config,
        )
        aggregate = aggregate_overlap_contributions(projected, mask, weight, "weighted_average")
        extracted, valid = extract_views_from_erp_standard(
            aggregate.fused_values,
            aggregate.valid_output_mask,
            view,
            view_dir,
            [(80, 80)],
            config,
        )
        for tensor in (projected, mask, weight, aggregate.fused_values, extracted, valid):
            self.assertEqual(tensor.dtype, torch.float32)

    def test_temporary_fused_erp_debug_export_writes_png(self):
        output_dir = Path("/home/shig/diffpano/test_outputs/temporary_fused_erp_export_test")
        config = PixelFusionConfig(
            temporary_save_fused_erp_per_step=True,
            temporary_fused_erp_dir=str(output_dir),
        )
        filename = temporary_save_fused_clean_erp_debug(
            torch.zeros(3, 8, 16),
            step_index=2,
            timestep=torch.tensor(345),
            config=config,
            pipeline_name="test",
        )
        try:
            self.assertIsNotNone(filename)
            self.assertTrue(Path(filename).is_file())
            self.assertGreater(Path(filename).stat().st_size, 0)
        finally:
            if filename is not None:
                Path(filename).unlink(missing_ok=True)
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()

    def test_temporary_original_clean_erp_debug_export_writes_png(self):
        output_dir = Path("/home/shig/diffpano/test_outputs/temporary_original_clean_erp_export_test")
        config = PixelFusionConfig(
            pixel_fusion_enabled=False,
            temporary_save_original_clean_erp_per_step=True,
            temporary_original_clean_erp_dir=str(output_dir),
        )
        filename = temporary_save_original_clean_erp_debug(
            [(torch.zeros(1, 3, 8, 8), torch.tensor([[0.0, 0.0, 1.0]]), (80, 80))],
            step_index=3,
            timestep=torch.tensor(250),
            erp_height=8,
            erp_width=16,
            weighted_average_temperature=0.1,
            config=config,
            pipeline_name="test",
        )
        try:
            self.assertIsNotNone(filename)
            self.assertTrue(Path(filename).is_file())
            self.assertGreater(Path(filename).stat().st_size, 0)
        finally:
            if filename is not None:
                Path(filename).unlink(missing_ok=True)
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()

    def test_identical_overlapping_patches_remain_unchanged(self):
        values = torch.ones(4, 3, 2, 2) * 0.7
        masks = torch.ones(4, 1, 2, 2)
        weights = torch.ones(4, 1, 2, 2)
        result = aggregate_overlap_contributions(values, masks, weights, "weighted_average")
        self.assertTrue(torch.allclose(result.fused_values, torch.ones(3, 2, 2) * 0.7))

    def test_three_overlapping_patches_weighted_average(self):
        values = torch.tensor([1.0, 3.0, 5.0]).view(3, 1, 1, 1)
        masks = torch.ones(3, 1, 1, 1)
        weights = torch.tensor([1.0, 2.0, 1.0]).view(3, 1, 1, 1)
        result = aggregate_overlap_contributions(values, masks, weights, "weighted_average")
        self.assertTrue(torch.allclose(result.fused_values, torch.tensor([[[3.0]]])))

    def test_patch_order_does_not_change_result(self):
        torch.manual_seed(7)
        values = torch.randn(5, 3, 4, 4)
        masks = (torch.rand(5, 1, 4, 4) > 0.25).float()
        weights = torch.rand(5, 1, 4, 4)
        first = aggregate_overlap_contributions(values, masks, weights, "detail_preserving_average").fused_values
        order = torch.tensor([3, 0, 4, 1, 2])
        second = aggregate_overlap_contributions(values[order], masks[order], weights[order], "detail_preserving_average").fused_values
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_invalid_patches_contribute_zero(self):
        values = torch.ones(2, 3, 2, 2)
        masks = torch.zeros(2, 1, 2, 2)
        weights = torch.ones(2, 1, 2, 2)
        result = aggregate_overlap_contributions(values, masks, weights, "weighted_average")
        self.assertEqual(result.accumulated_weight.sum().item(), 0)
        self.assertTrue(torch.all(result.fused_values == 0))

    def test_invalid_erp_pixels_fallback_without_nan(self):
        config = PixelFusionConfig()
        erp = torch.zeros(3, 8, 16)
        mask = torch.zeros(1, 8, 16)
        original = torch.randn(1, 3, 4, 4)
        view_dir = torch.tensor([[0.0, 0.0, 1.0]])
        extracted, valid = extract_views_from_erp_standard(erp, mask, original, view_dir, [(80, 80)], config)
        self.assertFalse(torch.isnan(extracted).any())
        self.assertTrue(torch.allclose(extracted, original))
        self.assertEqual(valid.sum().item(), 0)

    def test_erp_to_view_sampling_honors_projection_chunk_size(self):
        config = PixelFusionConfig(projection_chunk_size=2)
        erp = torch.randn(3, 16, 32)
        mask = torch.ones(1, 16, 32)
        original = torch.randn(5, 3, 4, 4)
        view_dirs = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        original_sample = pixel_fusion_module._sample_erp_image
        sampled_batch_sizes = []

        def checked_sample(values, grid, **kwargs):
            sampled_batch_sizes.append(values.shape[0])
            self.assertLessEqual(values.shape[0], config.projection_chunk_size)
            return original_sample(values, grid, **kwargs)

        with mock.patch.object(pixel_fusion_module, "_sample_erp_image", side_effect=checked_sample):
            extracted, valid = extract_views_from_erp_standard(
                erp,
                mask,
                original,
                view_dirs,
                [(80, 80)] * len(view_dirs),
                config,
            )

        self.assertEqual(extracted.shape, original.shape)
        self.assertEqual(valid.shape, (len(view_dirs), 1, 4, 4))
        self.assertEqual(max(sampled_batch_sizes), config.projection_chunk_size)

    def test_dpa_alpha_zero_equals_weighted_average(self):
        torch.manual_seed(11)
        values = torch.randn(4, 3, 3, 3)
        masks = torch.ones(4, 1, 3, 3)
        weights = torch.rand(4, 1, 3, 3)
        weighted = aggregate_overlap_contributions(values, masks, weights, "weighted_average").fused_values
        dpa = detail_preserving_average(values, masks, weights, alpha=0.0, power=1.0, eps=1e-6)
        self.assertTrue(torch.allclose(weighted, dpa, atol=1e-6))

    def test_single_valid_dpa_contributor_unchanged(self):
        values = torch.tensor([2.0, 9.0]).view(2, 1, 1, 1)
        masks = torch.tensor([1.0, 0.0]).view(2, 1, 1, 1)
        weights = torch.ones_like(masks)
        result = aggregate_overlap_contributions(values, masks, weights, "detail_preserving_average")
        self.assertTrue(torch.allclose(result.fused_values, torch.tensor([[[2.0]]])))

    def test_laplacian_reconstruction_reproduces_input(self):
        torch.manual_seed(13)
        image = torch.randn(2, 3, 16, 16)
        pyramid = build_laplacian_pyramid(image, 4)
        reconstructed = reconstruct_laplacian_pyramid(pyramid)
        self.assertTrue(torch.allclose(image, reconstructed, atol=1e-5))

    def test_lpw_smoke_uses_cached_lod_map(self):
        config = PixelFusionConfig(warp_mode="lpw", weight_mode="uniform", lpw_num_levels=3)
        views = torch.full((2, 3, 8, 8), 0.2)
        view_dirs = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        aggregate = inverse_lpw_to_erp(views, view_dirs, [(80, 80), (80, 80)], 32, 64, config)
        extracted, valid = forward_lpw_to_views(
            aggregate.fused_values,
            aggregate.valid_output_mask,
            views,
            view_dirs,
            [(80, 80), (80, 80)],
            config,
        )
        self.assertEqual(extracted.shape, views.shape)
        self.assertFalse(torch.isnan(extracted).any())
        max_error = (extracted - views).abs().max().item()
        self.assertTrue(
            torch.allclose(extracted, views, atol=1e-4),
            f"LPW constant round-trip max_error={max_error}, range=({extracted.min().item()}, {extracted.max().item()})",
        )
        self.assertGreater(valid.sum().item(), 0)
        self.assertEqual(len(config.projection_cache.lod_maps), 1)

    def test_erp_horizontal_seam_padding_is_circular(self):
        image = torch.arange(8, dtype=torch.float32).view(1, 1, 2, 4)
        padded = circular_pad_horizontal(image, 1, vertical_padding_mode="replicate")
        self.assertTrue(torch.equal(padded[0, 0, 1], torch.tensor([3.0, 0.0, 1.0, 2.0, 3.0, 0.0])))

    def test_north_and_south_pole_padding_rolls_half_turn(self):
        height, width = 3, 8
        erp = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width)
        padded = spherical_pad_erp(erp, pad_y=1, pad_x=1)
        north = padded[0, 0, 0, 1:-1]
        south = padded[0, 0, -1, 1:-1]
        self.assertTrue(torch.equal(north, torch.roll(erp[0, 0, 0], shifts=width // 2)))
        self.assertTrue(torch.equal(south, torch.roll(erp[0, 0, -1], shifts=width // 2)))

    def test_pole_padding_requires_even_erp_width(self):
        with self.assertRaisesRegex(ValueError, "even ERP width"):
            spherical_pad_erp(torch.zeros(1, 1, 3, 7), pad_y=1, pad_x=1)

    def test_spherical_gaussian_blur_crosses_poles_at_opposite_longitude(self):
        height, width = 8, 16
        north = torch.zeros(1, 1, height, width)
        north[0, 0, 0, 1] = 1
        north_blurred = pixel_fusion_module._pyramid_blur(north, "reflect", circular_horizontal=True)
        self.assertGreater(north_blurred[0, 0, 0, 1 + width // 2].item(), 0)

        south = torch.zeros_like(north)
        south[0, 0, -1, 3] = 1
        south_blurred = pixel_fusion_module._pyramid_blur(south, "reflect", circular_horizontal=True)
        self.assertGreater(south_blurred[0, 0, -1, 3 + width // 2].item(), 0)

    def test_original_debug_renderer_matches_standard_weighted_projection(self):
        config = PixelFusionConfig(
            warp_mode="standard",
            aggregation_mode="weighted_average",
            weight_mode="uniform",
            projection_chunk_size=1,
        )
        views = torch.cat([self.asymmetric_view(8), self.asymmetric_view(8).flip(-1)], dim=0)
        view_dirs = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        fovs = [(80, 80), (80, 80)]
        direct = pixel_fusion_module._fuse_views_to_erp_standard(views, view_dirs, fovs, 32, 64, config)
        streamed = render_views_to_erp_standard_weighted(
            [
                (views[0:1], view_dirs[0:1], fovs[0]),
                (views[1:2], view_dirs[1:2], fovs[1]),
            ],
            32,
            64,
            config,
        )
        self.assertIsNotNone(streamed)
        self.assertTrue(torch.allclose(streamed.fused_values, direct.fused_values, atol=1e-6))
        self.assertTrue(torch.equal(streamed.valid_output_mask, direct.valid_output_mask))

    def test_seam_crossing_round_trip_constant_patch(self):
        config = PixelFusionConfig(weight_mode="uniform")
        theta = torch.tensor([math.pi - 1e-4])
        phi = torch.tensor([0.0])
        view_dir = SphericalFunctions.spherical_to_cartesian(theta, phi)
        view = torch.full((1, 3, 8, 8), -0.4)
        projected, mask, weight = project_views_to_erp_standard(view, view_dir, [(100, 100)], 32, 64, config)
        aggregate = aggregate_overlap_contributions(projected, mask, weight, "weighted_average")
        extracted, _ = extract_views_from_erp_standard(aggregate.fused_values, aggregate.valid_output_mask, view, view_dir, [(100, 100)], config)
        self.assertFalse(torch.isnan(extracted).any())
        self.assertTrue(torch.allclose(extracted, view, atol=1e-4))

    def test_reinjection_strength_zero_and_one(self):
        clean = torch.ones(1, 2, 2, 2)
        fused = clean + 2
        model_output = torch.ones_like(clean) * 0.5
        sigma_next = torch.tensor(0.25)
        prev = clean + sigma_next * model_output
        zero_config = PixelFusionConfig(reinjection_strength=0.0)
        one_config = PixelFusionConfig(reinjection_strength=1.0)
        self.assertTrue(torch.allclose(reinject_fused_latents(clean, fused, prev, model_output, sigma_next, zero_config), prev))
        expected = prev + (1 - sigma_next) * (fused - clean)
        self.assertTrue(torch.allclose(reinject_fused_latents(clean, fused, prev, model_output, sigma_next, one_config), expected))

    def test_dpm_flow_prediction_converts_to_clean_sample(self):
        scheduler = FakeDPMSolverScheduler()
        sample = torch.full((1, 2, 2, 2), 3.0)
        flow = torch.full_like(sample, 2.0)
        clean, sigma, sigma_next = predict_clean_latents(scheduler, flow, torch.tensor(1000.0), sample)
        self.assertTrue(torch.allclose(clean, torch.ones_like(sample)))
        self.assertEqual(sigma.item(), 1.0)
        self.assertEqual(sigma_next.item(), 0.5)

    def test_disabling_pixel_fusion_leaves_schedule_off(self):
        config = PixelFusionConfig(pixel_fusion_enabled=False)
        self.assertFalse(should_apply_pixel_fusion(0, 10, config))

    def test_fixed_seed_deterministic_aggregation(self):
        torch.manual_seed(21)
        values = torch.randn(3, 3, 4, 4)
        masks = torch.ones(3, 1, 4, 4)
        weights = torch.rand(3, 1, 4, 4)
        first = aggregate_overlap_contributions(values, masks, weights, "weighted_average").fused_values
        torch.manual_seed(21)
        values = torch.randn(3, 3, 4, 4)
        masks = torch.ones(3, 1, 4, 4)
        weights = torch.rand(3, 1, 4, 4)
        second = aggregate_overlap_contributions(values, masks, weights, "weighted_average").fused_values
        self.assertTrue(torch.equal(first, second))

    def test_vae_encode_decode_shapes_and_ranges(self):
        config = PixelFusionConfig(vae_chunk_size=2)
        latents = torch.randn(3, 4, 5, 5)
        decoded = decode_view_latents(FakeVAE(), latents, config)
        encoded = encode_view_images(FakeVAE(), decoded, config)
        self.assertEqual(decoded.shape, (3, 3, 5, 5))
        self.assertEqual(encoded.shape, (3, 4, 5, 5))
        self.assertLessEqual(decoded.max().item(), 1.0)
        self.assertGreaterEqual(decoded.min().item(), -1.0)

    def test_deterministic_vae_encoder_output_is_supported(self):
        images = torch.zeros(2, 3, 4, 4)
        encoded = encode_view_images(FakeDirectEncodeVAE(), images, PixelFusionConfig())
        self.assertEqual(encoded.shape, (2, 4, 4, 4))

    def test_apply_pixel_space_fusion_smoke_with_fake_vae(self):
        config = PixelFusionConfig(pixel_fusion_enabled=True, weight_mode="uniform", save_diagnostics=True)
        config.projection_chunk_size = 1
        clean = torch.zeros(2, 4, 4, 4, dtype=torch.bfloat16)
        current = torch.ones_like(clean)
        model_output = torch.ones_like(clean)
        prev = clean + 0.5 * model_output
        view_dirs = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        result = apply_pixel_space_fusion(
            vae=FakeVAE(),
            scheduler=FakeFlowMatchScheduler(),
            timestep=torch.tensor(1000.0),
            clean_latents=clean,
            current_latents=current,
            model_output=model_output,
            prev_latents=prev,
            view_dirs=view_dirs,
            fovs=[(80, 80), (80, 80)],
            erp_height=16,
            erp_width=32,
            config=config,
        )
        self.assertEqual(result.fused_prev_latents.shape, clean.shape)
        self.assertEqual(result.fused_prev_latents.dtype, torch.bfloat16)
        self.assertEqual(result.fused_clean_latents.dtype, torch.float32)
        self.assertEqual(result.fused_views_rgb.dtype, torch.float32)
        self.assertEqual(result.fused_erp.dtype, torch.float32)
        self.assertEqual(result.accumulated_weight.dtype, torch.float32)
        self.assertFalse(torch.isnan(result.fused_prev_latents).any())
        for key in (
            "erp_grid_min",
            "erp_grid_max",
            "erp_pixel_center_max_error",
            "perspective_round_trip_mean_error",
            "perspective_round_trip_max_error",
        ):
            self.assertIn(key, result.diagnostics)
        self.assertLess(result.diagnostics["erp_pixel_center_max_error"].item(), 2e-6)


if __name__ == "__main__":
    unittest.main()
