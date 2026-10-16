"""Focused legacy adapters: reject scientific changes, tolerate source evolution."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from studies.layout24 import reuse
from studies.layout24.common import definitions


class ReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row=next(r for r in definitions() if r['prompt']=='air_balloons' and r['variant']=='diffpano_old89_screened_select')
        cls.source=reuse.ROOT/'outputs/10.7gradient-blending-flux/seed0-v1/cases/air_balloons/poisson_select/metadata.json'
        cls.n=reuse.normalize_metadata(cls.source)

    def rejected(self, change):
        n=copy.deepcopy(self.n);change(n)
        with self.assertRaises((ValueError,KeyError)):reuse.compatibility(n,self.row)

    def test_accept_own_archived_source_despite_unrelated_current_source(self):
        before=reuse.sha(self.source)
        with patch('studies.layout24.common.sources',side_effect=AssertionError('Never compare current whole source')):
            n=reuse.normalize_metadata(self.source);reuse.compatibility(n,self.row)
        self.assertEqual(before,reuse.sha(self.source))
        self.assertNotEqual(n['original_provenance']['source_revision'],'be64a2fdefb23e7e722d5b97a599ec4b2fb78c71')

    def test_reject_CEA_even_if_base_config_says_ERP(self):
        self.rejected(lambda n:n.update(consensus_projection='cea'))
        self.rejected(lambda n:n['details'].update(active_consensus_projection='cea'))

    def test_reject_CEA_wrapped_as_ERP(self):
        self.rejected(lambda n:n.update(verified_source_projection='cea'))

    def test_inventory_projection_must_match_original_log(self):
        r=next(r for r in reuse.read(reuse.INVENTORY)['rows'] if r['family']=='benchmark' and r['method']=='diffpano' and r['backend']=='flux' and r['prompt_id']=='air_balloons' and r['consensus_projection']=='erp')
        r=copy.deepcopy(r);r['consensus_projection']='cea'
        with self.assertRaisesRegex(ValueError,'Locator projection'):
            reuse.normalize_inventory(r)

    def test_reject_28_steps(self):self.rejected(lambda n:n.update(steps=28))
    def test_reject_wrong_prompt_bytes(self):self.rejected(lambda n:n['prompt_record'].update(original_sha256='0'*64))
    def test_reject_wrong_prompt_folder(self):self.rejected(lambda n:n['prompt_record'].update(path=str(reuse.ROOT/'prompts/ournew/air_balloons.txt')))
    def test_reject_mixed_guidance(self):self.rejected(lambda n:n.update(mode='poisson_mean_max'))
    def test_reject_magnitude_max(self):self.rejected(lambda n:n.update(mode='poisson_max'))
    def test_reject_coarse_reference(self):self.rejected(lambda n:n['fusion'].update(reference_mode='coarse_color'))
    def test_reject_tail(self):self.rejected(lambda n:n.update(refinement=.1))
    def test_reject_geometry(self):self.rejected(lambda n:n.update(geometry=reuse.UNIFORM))
    def test_reject_shared_noise(self):self.rejected(lambda n:n['initialization'].update(method='shared_nearest'))
    def test_reject_changed_local_noise(self):self.rejected(lambda n:n['initialization'].update(initial_local_sha256='0'*64))
    def test_reject_changed_noise_slots(self):self.rejected(lambda n:n['initialization'].update(camera_slots=list(reversed(range(89)))))
    def test_reject_weakened_solver(self):self.rejected(lambda n:n['fusion'].update(relative_tolerance=2e-5))
    def test_reject_nonconverged_terminal(self):self.rejected(lambda n:n['fusions'][-1].update(converged=False))
    def test_reject_changed_guidance(self):self.rejected(lambda n:n['config']['generation'].update(guidance_scale=4.))
    def test_reject_bad_artifact_hash(self):
        r=next(r for r in reuse.read(reuse.INVENTORY)['rows'] if r['family']=='step_control' and r['method']=='spherediff' and r['prompt_id']=='air_balloons')
        r=copy.deepcopy(r);r['image_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Image bytes'):reuse.normalize_inventory(r)

    def test_legacy_reference_default_is_screened(self):
        n=copy.deepcopy(self.n);self.assertNotIn('reference_mode',n['fusion'])
        reuse.compatibility(n,self.row)
        n['fusion']['reference_mode']='screened';reuse.compatibility(n,self.row)

    def test_exact_six_noise_banks_distinct(self):
        refs=reuse.references();banks=[refs[k]for k in ('old89','old70','old50','uniform89','uniform70','uniform50')]
        self.assertEqual(len(set(b['geometry']for b in banks)),6)
        self.assertEqual(len(set(b['initial']for b in banks)),6)
        self.assertEqual(len(banks[0]['slots']),89)
        self.assertTrue(refs['weight_evidence']['all_shared_files_identical'])


if __name__=='__main__':unittest.main()
