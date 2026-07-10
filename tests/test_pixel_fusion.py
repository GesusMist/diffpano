import math
import unittest
from unittest import mock

import torch

import pipelines_ours.pixel_fusion as pixel_fusion_module
from pipelines_ours.pixel_fusion import (
    PixelFusionConfig,
    aggregate_overlap_contributions,
    apply_pixel_space_fusion,
    circular_pad_horizontal,
    decode_view_latents,
    detail_preserving_average,
    encode_view_images,
    extract_views_from_erp_standard,
    forward_lpw_to_views,
    inverse_lpw_to_erp,
    project_views_to_erp_standard,
    predict_clean_latents,
    reconstruct_laplacian_pyramid,
    reinject_fused_latents,
    build_laplacian_pyramid,
    should_apply_pixel_fusion,
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
        original_sample = pixel_fusion_module._sample_with_grid
        sampled_batch_sizes = []

        def checked_sample(values, grid, **kwargs):
            sampled_batch_sizes.append(values.shape[0])
            self.assertLessEqual(values.shape[0], config.projection_chunk_size)
            return original_sample(values, grid, **kwargs)

        with mock.patch.object(pixel_fusion_module, "_sample_with_grid", side_effect=checked_sample):
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
        self.assertTrue(torch.allclose(reinject_fused_latents(clean, fused, prev, model_output, sigma_next, one_config), fused + sigma_next * model_output))

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
        config = PixelFusionConfig(pixel_fusion_enabled=True, weight_mode="uniform")
        config.projection_chunk_size = 1
        clean = torch.zeros(2, 4, 4, 4)
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
        self.assertFalse(torch.isnan(result.fused_prev_latents).any())


if __name__ == "__main__":
    unittest.main()
