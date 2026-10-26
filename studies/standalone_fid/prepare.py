"""Authenticate input collections and freeze a method-independent protocol."""
import argparse
from datetime import datetime, timezone
import fcntl

from studies.layout24 import common as panorama
from studies.followup24.common import PROMPTS24
from studies.faster_uniform_resume.completion import load_verifier
from studies.standalone_fid.common import *
from studies.standalone_fid.reference import build_reference_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaws', type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if PLAN.exists():
            print('VALID_PLAN', load_plan()['expected'], flush=True)
            return
        references = build_reference_manifest()
        assert references['unique_count'] == 85
        assert references['logical_count'] == 120
        manifest = panorama.freeze_check()
        verifier = load_verifier(strict_complete=panorama.complete)
        rows = []
        for variant in VARIANTS:
            for prompt in PROMPTS24:
                row = next(r for r in manifest['rows']
                           if r['variant'] == variant and r['prompt'] == prompt)
                assert verifier.checked_complete(row, verify=True), row['case_id']
                directory = panorama.folder(row)
                rows.append(dict(index=row['index'], case_id=row['case_id'],
                                 variant=variant, prompt=prompt,
                                 image_path=str(directory / 'final.png'),
                                 image_sha256=sha(directory / 'final.png'),
                                 metadata_path=str(directory / 'metadata.json'),
                                 metadata_sha256=sha(directory / 'metadata.json')))
        assert len(rows) == 168
        checkpoint = read(ASSETS / 'asset_status.json')['inception']
        assert checkpoint['status'] == 'available'
        assert sha(checkpoint['path']) == checkpoint['sha256']
        pitches = (90., 10., 0., -10., -90.)
        yaws = (0., 90., 180., 270.) if args.yaws == 4 else (0.,)
        views = [dict(yaw=yaw, pitch=pitch, direction_index=index)
                 for index, pitch in enumerate(pitches) for yaw in yaws]
        protocol = dict(
            name='FID against 85 unique standalone FLUX images', direction='lower',
            reference=dict(kind='synthetic standalone FLUX, not real photographs',
                           logical_count=120, unique_count=85,
                           deduplication='Exact final PNG SHA256; each unique image has equal weight',
                           resolution=[1024, 1024], enforced_camera_fov=None),
            generated=dict(variants=list(VARIANTS), prompts=list(PROMPTS24),
                           panoramas_per_variant=24, views=views,
                           views_per_panorama=len(views), views_per_variant=24 * len(views),
                           fov_x=80., fov_y=80., width=1024, height=1024, roll=0.,
                           weighting='Equal weight per rendered view; same scene/direction/yaw mixture for every method',
                           poles='Different yaws at poles rotate the image plane; these are correlated views'),
            features=dict(extractor='torch-fidelity inception-v3-compat', layer='2048',
                          dimensions=2048, checkpoint=checkpoint,
                          preprocessing='Both collections: RGB uint8 1024x1024, unchanged torch-fidelity internal TensorFlow-compatible bilinear resize to299 and normalization (x-128)/128',
                          precision='float32 inference; float64 FID statistics'),
            metric=dict(implementation='studies.panorama_metrics.numerics.frechet',
                        covariance='sample covariance, denominator n-1; full2048 dimensions',
                        singular_covariance='Exact low-rank cross-matrix SVD; no PCA or covariance regularization',
                        aggregation='One collection-level FID per method across all24 scenes'),
            limitations=[
                '85 unique reference images and24 scenes are small; finite-sample FID is exploratory.',
                'All standalone samples reset seed0; uniqueness is not a claim of independence.',
                'Deduplication changes reference scene/direction weights relative to the120 logical slots.',
                'Rendered views from one panorama are correlated; pole yaws are image-plane rotations.',
                'Standalone generated images do not have a guaranteed physical80-degree optical FOV.',
                'This measures similarity to standalone FLUX, not real-image realism or global panorama consistency.',
            ])
        write(OUT / 'reference-manifest.json', references)
        write(OUT / 'panorama-manifest.json', dict(rows=rows, variants=list(VARIANTS), prompts=list(PROMPTS24)))
        write(OUT / 'protocol.json', protocol)
        for name in ('logs', 'cache'):
            (OUT / name).mkdir(parents=True, exist_ok=True)
        input_paths = [OUT / name for name in ('reference-manifest.json', 'panorama-manifest.json', 'protocol.json')]
        input_paths += [panorama.OUT / 'manifest.json', ROOT / 'outputs/10.15perspective24-flux/seed0-v1/manifest.json']
        plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    source_hashes=sources(), input_hashes={str(p): sha(p) for p in input_paths},
                    protocol_sha256=digest(protocol), expected=dict(reference=85, variants=7,
                    panoramas=168, views_per_variant=24 * len(views), generated_views=168 * len(views)),
                    generation_required=False)
        write(PLAN, plan)
        (OUT / 'plan.sha256').write_text(sha(PLAN) + '\n')
        load_plan()
        print('PREPARED_FID', plan['expected'], flush=True)


if __name__ == '__main__':
    main()
