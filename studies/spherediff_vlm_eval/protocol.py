"""Versioned, verified SphereDiff VLM prompt and independent evaluation cameras.

The published prompt is licensed CC BY 4.0; its attribution and verification
record live beside the text asset. Generation camera configurations are never
used here. This module performs no network requests and loads no models.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent
PROMPT_PATH = STUDY / 'assets' / 'spherediff_arxiv_v2_table5_prompt_v1.txt'
PROMPT_PROVENANCE_PATH = STUDY / 'assets' / 'spherediff_arxiv_v2_table5_prompt_v1.provenance.json'
PROMPT_SHA256 = 'c7899680c0aafdc76c05389e869b11d3f8e3435c73382ce8bea1b86c21e83540'
PAPER_URL = 'https://arxiv.org/pdf/2504.14396v2'
CRITERIA = ('Image Quality', 'Aesthetic Appeal', 'Distortion Level', 'Connectivity')
RENDERER_PATH = STUDY.parent / 'panorama_metrics' / 'render.py'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _cameras():
    rows = []
    for pitch, group, suffix in ((-45, 'lower', 'm45'), (0, 'equator', 'p00'), (45, 'upper', 'p45')):
        for yaw in (0, 90, 180, 270):
            rows.append(dict(view_id=f'yaw{yaw:03d}_pitch{suffix}', yaw=float(yaw),
                             pitch=float(pitch), elevation_group=group, is_seam=yaw == 180,
                             is_polar=False, roll=0.0))
    for pitch, group in ((90, 'north_pole'), (-90, 'south_pole')):
        rows.append(dict(view_id=group, yaw=0.0, pitch=float(pitch), elevation_group=group,
                         is_seam=False, is_polar=True, roll=0.0))
    return tuple(rows)


CAMERAS = _cameras()
VIEWS = tuple((row['yaw'], row['pitch']) for row in CAMERAS)


def prompt_text():
    """Return the exact wording asset, refusing unversioned prompt edits."""
    raw = PROMPT_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROMPT_SHA256:
        raise ValueError('Evaluation prompt integrity check failed; create a new version for changes')
    return raw.decode('utf-8')


def _package_version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return 'unavailable'


def rendering_protocol_record():
    """Portable, content-addressed geometry record; paths do not enter its hash."""
    return dict(
        version='spherediff-v2-14views-pixel-center-v1',
        source=dict(url=PAPER_URL, appendix='F.2', pdf_page=19),
        renderer='studies.panorama_metrics.render.render',
        renderer_source_sha256=hashlib.sha256(RENDERER_PATH.read_bytes()).hexdigest(),
        libraries=dict(numpy=_package_version('numpy'), pillow=_package_version('Pillow'),
                       opencv=_package_version('opencv-python-headless', 'opencv-python')),
        width=512, height=512, fov_x=90.0, fov_y=90.0,
        cameras=[dict(c) for c in CAMERAS],
        axes='x right, y north, z yaw0; yaw atan2(x,z); positive pitch is north',
        erp_coordinates='u=(longitude/(2*pi)+0.5)*width-0.5; v=(0.5-latitude/pi)*height-0.5',
        pixel_centers=True, interpolation='OpenCV INTER_LINEAR',
        longitude_boundary='periodic by one-pixel wrap padding', latitude_boundary='clamp',
        seam_yaw_degrees=180.0, pole_yaw_degrees=0.0,
        pole_orientation='yaw0 right=+x; north image-up=-z; south image-up=+z',
        original_erp_operations='read only; no rotation, shift, crop, normalization, or enhancement',
        clarification='512px resolution and yaw0 pole orientation are fixed reproduction choices; Appendix F.2 specifies the 14 directions and 90-degree FOV.',
    )


def rendering_protocol_sha256():
    return digest(rendering_protocol_record())


def protocol_record():
    """Complete protocol metadata, suitable for output/protocol.json."""
    prompt_text()
    prompt_source = json.loads(PROMPT_PROVENANCE_PATH.read_text())
    rendering = rendering_protocol_record()
    return dict(
        version='spherediff-vlm-eval-v1',
        prompt=dict(asset=str(PROMPT_PATH.relative_to(STUDY)), sha256=PROMPT_SHA256,
                    reproduction='exact published wording; PDF line wrapping normalized',
                    provenance=prompt_source),
        criteria=list(CRITERIA), rating_scale=dict(minimum=1, maximum=5, type='integer', larger_is_better=True),
        judge=dict(requested_model='gpt-4o', images_per_request=1, generation_prompt_context=False,
                   method_names=False, competing_images=False, previous_scores=False,
                   template_image_marker='{image}',
                   prompt_injection='Replace {image} with one image content part; preserve text before and after it'),
        rendering=rendering, rendering_protocol_sha256=digest(rendering),
        scoring=dict(panorama='arithmetic mean of all 14 successful view scores',
                     method='equal-weight mean of complete panorama means on the matched cohort',
                     confidence_interval_unit='paired prompts, never individual views',
                     connectivity_display_name='End Continuity'),
    )
