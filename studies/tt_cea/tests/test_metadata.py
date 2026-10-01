import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from studies.tt_cea.metadata import (terminal_pole_rgb, validate_pole_record,
    assert_capture_only_run_change, compatible_source_hashes, certify_metadata_revision, assert_same_recovery_sample)
from studies.tt_cea.common import study_hashes


class PoleMetadataTests(unittest.TestCase):
    def test_exact_values_order_no_mutation_or_rng(self):
        poles=torch.arange(6,dtype=torch.float32).reshape(1,3,2,1)-2
        before=poles.clone();rng=torch.random.get_rng_state().clone()
        value=terminal_pole_rgb(SimpleNamespace(pole_rgb=poles))
        self.assertEqual(value,dict(north=[-2.,0.,2.],south=[-1.,1.,3.]))
        self.assertTrue(torch.equal(poles,before));self.assertTrue(torch.equal(torch.random.get_rng_state(),rng))
        value['north'][0]=999
        self.assertTrue(torch.equal(poles,before))

    def test_missing_wrong_shape_nonfinite_rejected(self):
        for poles in (None,torch.zeros(1,3,1,2),torch.full((1,3,2,1),float('nan')),
                      torch.zeros(1,3,2,1,dtype=torch.float64)):
            with self.assertRaises(AssertionError):terminal_pole_rgb(SimpleNamespace(pole_rgb=poles))

    def test_record_validation_raw_values_not_clipped(self):
        value=dict(north=[-2.,0.,2.],south=[3.,4.,5.])
        self.assertEqual(validate_pole_record(value),value)
        for bad in (None,{},dict(north=[0,0],south=[0,0,0]),dict(north=[0,0,float('inf')],south=[0,0,0])):
            with self.assertRaises(AssertionError):validate_pole_record(bad)

    def test_generation_ast_guard(self):
        before="def run():\n    metadata=dict(answer=42)\n"
        after="from studies.tt_cea.metadata import terminal_pole_rgb\ndef run():\n    metadata=dict(answer=42,terminal_pole_rgb=terminal_pole_rgb(native) if row['projection']=='cea' else None)\n"
        assert_capture_only_run_change(before,after)
        with self.assertRaises(AssertionError):assert_capture_only_run_change(before,after.replace('answer=42','answer=43'))
        with self.assertRaises(AssertionError):assert_capture_only_run_change(before,after.replace("row['projection']=='cea'","True"))

    def test_prior_sources_require_exact_passed_certificate(self):
        current={'new':'hash'};prior={'old':'hash'};certificate={'prior_source_hashes':prior}
        with patch('studies.tt_cea.metadata.study_hashes',return_value=current),patch('studies.tt_cea.metadata.certify_metadata_revision',return_value=certificate):
            self.assertEqual(compatible_source_hashes(dict(passed=True,source_hashes=current)),[current])
            self.assertEqual(compatible_source_hashes(dict(passed=True,source_hashes=current,metadata_only_compatibility=certificate)),[current,prior])
            with self.assertRaises(AssertionError):compatible_source_hashes(dict(passed=False,source_hashes=current))
            with self.assertRaises(AssertionError):compatible_source_hashes(dict(passed=True,source_hashes=prior))
            with self.assertRaises(AssertionError):compatible_source_hashes(dict(passed=True,source_hashes=current,metadata_only_compatibility={}))

    def test_real_revision_certificate_and_protected_source_rejection(self):
        certificate=certify_metadata_revision()
        self.assertTrue(certificate['generation_AST_unchanged_except_metadata'])
        changed=copy.deepcopy(study_hashes());changed['studies/tt_cea/pipeline.py']='0'*64
        with patch('studies.tt_cea.metadata.study_hashes',return_value=changed):
            with self.assertRaises(AssertionError):certify_metadata_revision()

    def test_recovery_requires_state_configuration_and_image_match(self):
        keys=('backend','prompt','case','generation_configuration','model_checkpoint','model_revision',
              'camera_geometry_sha256','local_native_resolution','local_RGB_resolution','native_channels',
              'native_scale','prepared_intervals','conditioning_sha256','counts','manifest_sha256')
        original={key:'same' for key in keys}
        original.update(initialization={'initial_local_sha256':'initial'},diagnostics={'terminal_state_sha256':'terminal'},
            artifacts={f:'identical' for f in ('final.png','final_cea.png','viewports.png','schedule.csv')})
        candidate=copy.deepcopy(original);candidate['terminal_pole_rgb']=dict(north=[0.,0.,0.],south=[1.,1.,1.])
        assert_same_recovery_sample(original,candidate)
        for kind in ('state','configuration','image'):
            bad=copy.deepcopy(candidate)
            if kind=='state':bad['diagnostics']['terminal_state_sha256']='different'
            elif kind=='configuration':bad['generation_configuration']='different'
            else:bad['artifacts']['final.png']='different'
            with self.assertRaises(AssertionError):assert_same_recovery_sample(original,bad)
