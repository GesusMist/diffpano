"""CPU-only fixtures for identity, provenance, alpha separation and no mutation."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from .inventory import (COUNTS, DIFFPANO_CHECKPOINT, INITIAL_LOCAL_SHA, SCHEDULE_SHA, OFFICIAL_COMMIT, OFFICIAL_PIPELINE_SHA, NEW, OLD89, ORIGINAL, InventoryBuilder, Rejected,
                        Resolver, assemble_cohorts, build_inventory, canonical_prompt_path,
                        select_duplicates, sha)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ORIGINAL + NEW:
            path = self.root / 'prompts' / ('ournew' if name in NEW else '') / (name + '.txt')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('one\ntwo\nthree\nfour\nfive\n')
        self.output = self.root / 'evaluation'

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, alpha=.5, name='air_balloons', mode='poisson_mean_max', folder=None):
        folder = folder or self.root / 'outputs' / 'fixture' / name / str(alpha)
        folder.mkdir(parents=True, exist_ok=True)
        image = folder / 'final.png'
        Image.new('RGB', (4096, 2048), (20, 40, 80)).save(image)
        prompt = self.root / 'prompts' / (name + '.txt')
        fusion = dict(mode=mode, reference_mode='screened', lambda_color=.1)
        if mode == 'poisson_mean_max':
            fusion['gradient_mix_alpha'] = alpha
        config = dict(model=dict(pipeline='flux', id=DIFFPANO_CHECKPOINT[0], revision=DIFFPANO_CHECKPOINT[1], precision='bf16'),
                      generation=dict(num_inference_steps=20, guidance_scale=3.5, true_cfg_scale=1.0, batch_size=1),
                      experiment=dict(seed=0), canvas=dict(mode='erp'), erp=dict(height=2048, width=4096),
                      view=dict(width=1024, height=1024, fov_x=80, fov_y=80),
                      sampling=dict(rotation=dict(enabled=False)),
                      consensus_transition=dict(mode='preserve_current_state', vae_residual_correction=True),
                      warp=dict(mode='standard', erp_to_perspective=dict(interpolation='nearest'), perspective_to_erp=dict(interpolation='bilinear')),
                      fusion=dict(mode='weighted_average', weight_mode='spherediff_center', spherediff_temperature=.1),
                      global_pipeline=dict(mode='erp_bridge_factorial', refinement=dict(last_fraction=0)))
        put(folder / 'config.json', config)
        meta = dict(configuration=config, camera_geometry_sha256=OLD89, counts=COUNTS,
                    initialization=dict(initial_local_sha256=INITIAL_LOCAL_SHA), schedule_sha256=SCHEDULE_SHA,
                    prompt=dict(path=str(prompt), original_sha256=sha(prompt), effective_sha256=sha(prompt),
                                adapted=False, effective_lines=prompt.read_text().splitlines()),
                    fusion=fusion, refinement=dict(last_fraction=0),
                    artifacts={'final.png': sha(image), 'config.json': sha(folder / 'config.json')},
                    fusions=[dict(fusion, converged=True, terminal=i == 20) for i in range(21)])
        put(folder / 'metadata.json', meta)
        put(folder / 'status.json', dict(state='complete', metadata_sha256=sha(folder / 'metadata.json'), image_sha256=sha(image)))
        return folder, meta

    def test_completed_fixture_and_no_mutation(self):
        folder, _ = self.fixture()
        before = {str(p): sha(p) for p in self.root.rglob('*') if p.is_file()}
        result = build_inventory(self.root, self.output)
        self.assertEqual(len(result['records']), 1)
        self.assertEqual(result['records'][0]['method_id'], 'meanmax050')
        self.assertEqual(result['cohorts']['primary']['matched_prompt_count'], 0)
        for path, value in before.items():
            self.assertEqual(sha(path), value)
        self.assertTrue((self.output / 'missing_cases.csv').exists())

    def test_deleted_status_preserved_audit_and_tampered_metadata(self):
        folder, meta = self.fixture()
        (folder / 'status.json').unlink()
        proof = dict(path=str(folder / 'final.png'), image_sha256=sha(folder / 'final.png'),
                     metadata_path=str(folder / 'metadata.json'), metadata_sha256=sha(folder / 'metadata.json'))
        put(self.root / 'outputs' / 'report' / 'provenance' / 'figures.json', dict(figures=[dict(panels=[proof])]))
        result = build_inventory(self.root, self.output)
        self.assertEqual(len(result['records']), 1)
        meta['counts']['denoiser'] = 1
        put(folder / 'metadata.json', meta)
        result = build_inventory(self.root, self.output)
        self.assertEqual(result['records'], [])

    def test_missing_completion_not_accepted(self):
        folder, _ = self.fixture()
        (folder / 'status.json').unlink()
        result = build_inventory(self.root, self.output)
        self.assertEqual(result['records'], [])

    def test_same_basename_different_full_path_is_not_substituted(self):
        folder, meta = self.fixture()
        meta['prompt']['path'] = str(self.root / 'prompts/other/air_balloons.txt')
        builder = InventoryBuilder(self.root, self.output)
        with self.assertRaisesRegex(Rejected, 'Outside'):
            builder.standard(folder / 'metadata.json', meta)
        self.assertEqual(canonical_prompt_path('/old/repo/prompts/ournew/bamboo_grove.txt'),
                         'prompts/ournew/bamboo_grove.txt')

    def test_effective_hash_and_rotation_are_rejected(self):
        folder, meta = self.fixture()
        builder = InventoryBuilder(self.root, self.output)
        meta['prompt']['effective_sha256'] = '0' * 64
        with self.assertRaisesRegex(Rejected, 'Effective prompt hash'):
            builder.standard(folder / 'metadata.json', meta)
        meta['prompt']['effective_sha256'] = meta['prompt']['original_sha256']
        meta['configuration']['sampling']['rotation']['enabled'] = True
        with self.assertRaisesRegex(Rejected, 'rotation'):
            builder.standard(folder / 'metadata.json', meta)

    def test_nonconverged_or_wrong_alpha_not_accepted(self):
        folder, meta = self.fixture()
        builder = InventoryBuilder(self.root, self.output)
        meta['fusions'][2]['gradient_mix_alpha'] = .75
        with self.assertRaisesRegex(Rejected, 'alpha mismatch'):
            builder.standard(folder / 'metadata.json', meta)
        meta['fusions'][2]['gradient_mix_alpha'] = .5
        meta['fusions'][2]['converged'] = False
        with self.assertRaisesRegex(Rejected, 'not converged'):
            builder.standard(folder / 'metadata.json', meta)

    def test_different_checkpoint_guidance_or_warp_rejected(self):
        folder, meta = self.fixture()
        builder = InventoryBuilder(self.root, self.output)
        for key, value, message in [('model', {'id': 'another/checkpoint'}, 'checkpoint'),
                                     ('generation', {'guidance_scale': 7.0}, 'guidance'),
                                     ('warp', {'mode': 'lpw'}, 'warp')]:
            altered = json.loads(json.dumps(meta))
            altered['configuration'][key].update(value)
            with self.assertRaisesRegex(Rejected, message):
                builder.standard(folder / 'metadata.json', altered)

    def test_alpha_intersections_and_sphere28_separation(self):
        records = []
        methods = ('sphere20', 'rgb', 'select', 'meanmax050', 'meanmax025', 'meanmax075', 'sphere28')
        for prompt in ('p1', 'p2'):
            for method in methods:
                if (prompt, method) in (('p2', 'meanmax025'), ('p1', 'sphere28')):
                    continue
                records.append(dict(prompt_id=prompt, method_id=method, panorama_id=prompt + method))
        cohorts = assemble_cohorts(records)
        self.assertEqual(cohorts['primary']['matched_prompt_count'], 2)
        self.assertEqual(cohorts['alpha025']['prompt_ids'], ['p1'])
        self.assertEqual(cohorts['alpha075']['matched_prompt_count'], 2)
        self.assertEqual(cohorts['sphere28']['prompt_ids'], ['p2'])
        self.assertNotIn('sphere28', cohorts['primary']['method_ids'])

    def test_duplicate_identical_preference_and_different_image_exclusion(self):
        base = dict(prompt_id='p', method_id='rgb', image_sha256='a', metadata_path='m1', image_path='im1', evidence=[])
        second = dict(base, metadata_path='m2', image_path='im2', evidence=[{}])
        selected, dup, ambiguous = select_duplicates([base, second])
        self.assertEqual(selected[0]['metadata_path'], 'm2')
        self.assertEqual(len(dup), 1)
        self.assertEqual(ambiguous, [])
        selected, dup, ambiguous = select_duplicates([base, dict(second, image_sha256='b')])
        self.assertEqual(selected, [])
        self.assertEqual(len(ambiguous), 1)

    def test_explicit_relocation_and_extra_historical_root(self):
        moved = self.root / 'faster'
        (moved / 'experiment').mkdir(parents=True)
        file = moved / 'experiment' / 'metadata.json'
        file.write_text('{}')
        resolver = Resolver(self.root, [moved], {'/old/output-data': str(moved)})
        self.assertEqual(resolver.resolve('/old/output-data/experiment/metadata.json'), file.resolve())
        self.assertEqual(resolver.resolve('/old/repo/outputs/experiment/metadata.json'), file.resolve())

    def test_native_flux28_completion_audit_and_manifest(self):
        suite = self.root / 'outputs' / 'original-reference' / 'seed0-v1'
        folder = suite / 'flux' / 'air_balloons' / 'seed0'
        folder, old = self.fixture(folder=folder)
        (folder / 'final.png').rename(folder / 'final_result.png')
        call = dict(num_inference_steps=28, guidance_scale=3.5, true_cfg_scale=1.,
                    n_spherical_points=26500, weighted_average_temperature=.1,
                    erp_height=2048, erp_width=4096)
        config = dict(pipeline='SphericalFluxPipeline', model_source='black-forest-labs/FLUX.1-dev',
                      revision='official-revision', variant=None, mixed_precision='bf16',
                      enable_model_cpu_offload=False, enable_vae_tiling=False, call=call)
        meta = dict(backend='flux', source_commit=OFFICIAL_COMMIT,
                    official_source_hashes={'pipelines_ours/pipeline_spherical_flux.py': OFFICIAL_PIPELINE_SHA},
                    prompt=old['prompt'], spec=config, seed=0, transformer_forwards=2492,
                    prepared_schedule=dict(timesteps=list(range(28)), sigmas=list(range(29))))
        put(folder / 'metadata.json', meta)
        put(suite / 'manifest.json', dict(rows=[dict(backend='flux', prompt='air_balloons', seed=0, spec=config)]))
        put(suite / 'completion.json', dict(complete=True, rows=[dict(backend='flux', prompt='air_balloons',
            seed=0, steps=28, forwards=2492, status='complete', output=str(folder / 'final_result.png'),
            image_sha256=sha(folder / 'final_result.png'))]))
        put(suite / 'final_audit.json', dict(passed=True, all_four_complete=True,
            artifact_hashes={name: sha(suite / name) for name in ('manifest.json', 'completion.json')}))
        builder = InventoryBuilder(self.root, self.output)
        builder.collect_audits([suite / 'completion.json', suite / 'manifest.json'])
        record = builder.native_sphere(folder / 'metadata.json', meta)
        self.assertEqual(record['method_id'], 'sphere28')
        meta['spec']['call']['guidance_scale'] = 4.
        with self.assertRaisesRegex(Rejected, 'invocation differs'):
            builder.native_sphere(folder / 'metadata.json', meta)

    def test_unrelated_remapping_does_not_hide_conflicting_copies(self):
        current = self.root / 'outputs' / 'experiment' / 'metadata.json'
        current.parent.mkdir(parents=True)
        current.write_text('current')
        extra = self.root / 'extra'
        (extra / 'experiment').mkdir(parents=True)
        (extra / 'experiment' / 'metadata.json').write_text('different')
        resolver = Resolver(self.root, [extra], {'/unrelated': '/irrelevant'})
        with self.assertRaisesRegex(Rejected, 'Ambiguous relocated path'):
            resolver.resolve('/old/repo/outputs/experiment/metadata.json')


if __name__ == '__main__':
    unittest.main()
