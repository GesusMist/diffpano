"""Small fixture tests for identity, exclusion, alias and missing-cell behavior."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from . import cohorts
from .inventory import Auditor, Incompatible, method_id, runtime_from


def prompt_fixture(root):
    for name in cohorts.ORIGINAL20 + cohorts.NEW4:
        path = root / 'prompts' / ('ournew' if name in cohorts.NEW4 else '') / (name + '.txt')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(name + ' direction ' + str(i) for i in range(5)) + '\n')
    return cohorts.prompt_records(root)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.prompts = prompt_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_scopes_and_full_path_identity(self):
        self.assertEqual(len(self.prompts), 24)
        self.assertEqual(sum('development15' in p['memberships'] for p in self.prompts), 15)
        p = next(p for p in self.prompts if p['prompt_name'] == 'bamboo_grove')
        self.assertTrue(p['prompt_id'].startswith('prompts/ournew/bamboo_grove.txt@'))
        with self.assertRaises(ValueError):
            cohorts.canonical_path('bamboo_grove.txt')
        bad = dict(p, path='prompts/bamboo_grove.txt')
        bad.pop('full_path')
        with self.assertRaises(ValueError):
            cohorts.validate_prompt(bad, p)

    def test_byte_and_line_mutations_rejected(self):
        p = self.prompts[0]
        record = dict(p, path=p['prompt_path'])
        record['effective_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            cohorts.validate_prompt(record, p)
        record = copy.deepcopy(p)
        record['effective_lines'][0] += ' changed'
        with self.assertRaises(ValueError):
            cohorts.validate_prompt(record, p)

    def test_matched_cohort_deduplicates_aliases_and_excludes_conflicts(self):
        p, q = self.prompts[:2]
        rows = []
        for method in cohorts.PRIMARY_METHODS:
            for prompt in (p, q):
                rows.append(dict(case_id=prompt['prompt_name']+'/'+method, prompt_id=prompt['prompt_id'],
                    method_id=method, panorama_id=prompt['prompt_name']+method, state='complete', eligible=True,
                    image_sha256=prompt['prompt_name']+method, selection_priority=0))
        alias = dict(rows[0], case_id='historical-alias', selection_priority=10)
        rows.append(alias)
        result = cohorts.assemble(rows, self.prompts)
        self.assertEqual(result['cohorts']['development15']['matched']['primary']['prompt_count'], 2)
        self.assertEqual(len(result['selected_cells']), 8)
        rows.append(dict(rows[1], case_id='conflicting-image', image_sha256='different'))
        result = cohorts.assemble(rows, self.prompts)
        self.assertEqual(result['cohorts']['development15']['matched']['primary']['prompt_count'], 1)
        self.assertEqual(len(result['ambiguities']), 1)

    def test_missing_primary_not_filled_with_other_methods(self):
        audit = Auditor(self.root)
        records = []
        audit.missing_primary(records)
        self.assertEqual(len(records), 24 * 4)
        self.assertTrue(all(r['state'] == 'missing' and not r['eligible'] for r in records))


class ScienceTests(unittest.TestCase):
    def test_actual_guidance_and_checkpoint_are_required(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            prompt_fixture(root)
            auditor = Auditor(root)
            p = auditor.prompts[0]
            cfg = dict(model=dict(pipeline='flux', id='ModelsLab/flux.1-dev', revision='fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4'),
                generation=dict(num_inference_steps=20, guidance_scale=3.5, true_cfg_scale=1.),
                experiment=dict(seed=0), erp=dict(width=4096,height=2048),
                view=dict(width=1024,height=1024,fov_x=80,fov_y=80),
                consensus_transition=dict(mode='preserve_current_state',vae_residual_correction=True),
                global_pipeline=dict(mode='erp_bridge_factorial'),
                warp=dict(mode='standard',erp_to_perspective=dict(interpolation='nearest'),perspective_to_erp=dict(interpolation='bilinear')))
            row = dict(prompt=p['prompt_name'], gradient_mode='rgb', steps=20, seed=0)
            meta = dict(configuration=cfg, prompt=dict(p,path=p['prompt_path']), reused=True,
                fusion=dict(mode='rgb'), counts=dict(denoiser=1780,encode=3560,decode=1869,initialize=0))
            auditor.scientific(row,meta,p)
            meta['configuration']['generation']['guidance_scale'] = 7.0
            with self.assertRaises(Incompatible):
                auditor.scientific(row,meta,p)
            meta['configuration']['generation']['guidance_scale'] = 3.5
            meta['configuration']['model']['revision'] = 'other-checkpoint'
            with self.assertRaises(Incompatible):
                auditor.scientific(row,meta,p)

    def test_ablation_ids_do_not_replace_primary(self):
        self.assertEqual(method_id(dict(gradient_mode='poisson_mean_max',gradient_mix_alpha=.5,reference_mode='screened')), 'meanmax050')
        self.assertEqual(method_id(dict(gradient_mode='poisson_mean_max',gradient_mix_alpha=.5,reference_mode='coarse_color')), 'meanmax050_coarse_color')
        self.assertEqual(method_id(dict(gradient_mode='poisson_select',steps=28,refinement_steps=5)), 'select_refine05of28')
        self.assertEqual(method_id(dict(gradient_mode='poisson_select',steps=28,last_fraction=0)), 'select_steps28')

    def test_missing_runtime_is_null(self):
        runtime = runtime_from({'generation_seconds': 12.5})
        self.assertEqual(runtime['generation_seconds'], 12.5)
        self.assertIsNone(runtime['solver_seconds'])
        self.assertIn('solver_seconds', runtime['missing_fields'])


if __name__ == '__main__':
    unittest.main()
