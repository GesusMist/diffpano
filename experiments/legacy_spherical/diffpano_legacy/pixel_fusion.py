"""Compatibility facade for the pre-package ``pixel_fusion`` API.

New code should import from the responsibility-based modules directly.
"""

from experiments.legacy_spherical.diffpano_legacy.config import PixelFusionConfig, ProjectionCache, build_pixel_fusion_config, load_pixel_fusion_config
from experiments.legacy_spherical.diffpano_legacy.diagnostics import (
    save_exclusive_owner_diagnostics,
    temporary_save_fused_clean_erp_debug,
    temporary_save_original_clean_erp_debug,
    write_pixel_fusion_diagnostics,
)
from experiments.legacy_spherical.diffpano_legacy.diffusion import (
    PixelFusionResult,
    apply_pixel_space_fusion,
    run_time_travel,
    should_apply_pixel_fusion,
    should_apply_time_travel,
)
from experiments.legacy_spherical.diffpano_legacy.fusion import (
    OverlapAggregationResult,
    _fuse_views_to_erp_standard,
    aggregate_overlap_contributions,
    create_patch_weight_map,
    detail_preserving_average,
    project_views_to_erp_standard,
    render_views_to_erp_standard_weighted,
)
from experiments.legacy_spherical.diffpano_legacy.initialization import apply_configured_random_seed
from experiments.legacy_spherical.diffpano_legacy.lpw import (
    _pyramid_blur,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    circular_pad_horizontal,
    forward_lpw_to_views,
    inverse_lpw_to_erp,
    reconstruct_laplacian_pyramid,
)
from experiments.legacy_spherical.diffpano_legacy.projection import (
    _erp_world_grid,
    _sample_erp_image,
    _world_to_erp_grid,
    extract_views_from_erp_standard,
    spherical_pad_erp,
)
from experiments.legacy_spherical.diffpano_legacy.reinjection import predict_clean_latents, reinject_fused_latents, step_with_fused_clean_prediction
from experiments.legacy_spherical.diffpano_legacy.vae import (
    VaeResidualBridgeResult,
    build_identity_preserving_vae_target,
    decode_view_latents,
    encode_view_images,
)
from experiments.legacy_spherical.diffpano_legacy.writeback import (
    ExclusiveOwnerMap,
    ExclusiveWriteBackResult,
    build_exclusive_owner_map,
    exclusive_owner_diagnostics,
    get_or_build_exclusive_owner_map,
    summarize_patch_geometry,
    write_back_views_exclusive,
    write_back_views_weighted_average,
)
