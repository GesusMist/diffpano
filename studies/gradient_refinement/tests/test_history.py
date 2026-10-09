import sys,unittest,tarfile,types,hashlib
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path('tests').resolve()))
from test_gwtf_noise_initialization import GWTFlowTests,NativeStub
from diffpano.erp_noise_initialization import initialize_erp_noise,ERPNoiseConfig,states_digest
from diffpano.gwtf_noise_initialization import initialize_gwtf_shared_noise,GWTFlowNoiseConfig
from diffpano.gradient_fusion import GradientSettings,GuidanceAccumulator,edges,reconstruct
from diffpano.projection import ERPContribution
from studies.gradient_refinement.common import OUT,read,sha

def archived(name,path):
    with tarfile.open(OUT/'history/source-616d8ec.tar') as archive:source=archive.extractfile(path).read().decode()
    module=types.ModuleType(name);sys.modules[name]=module;exec(compile(source,path,'exec'),module.__dict__);return module

class HistoryTests(unittest.TestCase):
    def test_initializer_preserved_source_and_original_behavior(self):
        historical=read('outputs/bridge-factorial-ruins/20260918/validation.json')
        before=read(OUT/'history/source-hashes.json')
        for path in ('diffpano/erp_noise_initialization.py','diffpano/noise.py'):
            self.assertEqual(sha(path),historical['source_hashes'][path]);self.assertEqual(sha(path),before[path])
        helper=GWTFlowTests();b=NativeStub();cams=helper.cams();rng=torch.random.get_rng_state().clone()
        for mode in ('S-direct-local','V-independent-erp','V-shared-erp'):
            a,_=initialize_erp_noise(b,cams,ERPNoiseConfig(mode,17,34,3))
            initialize_gwtf_shared_noise(b,cams,GWTFlowNoiseConfig(17,34,3))
            z,_=initialize_erp_noise(b,cams,ERPNoiseConfig(mode,17,34,3))
            self.assertEqual(states_digest(a),states_digest(z))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
    def test_exact_archived_screened_guidance_and_solver(self):
        old=archived('_legacy_gradient','diffpano/gradient_fusion.py');torch.manual_seed(9)
        ref=torch.randn(1,3,8,16);ones=torch.ones(1,1,8,16)
        for mode in ('poisson_mean','poisson_select'):
            a=old.GuidanceAccumulator(ref,mode);b=GuidanceAccumulator(ref,mode)
            for i in (7,3,0):
                contribution=ERPContribution(torch.randn_like(ref),ones,torch.rand_like(ones))
                a.accumulate(contribution,i);b.accumulate(contribution,i)
            ga,sa=a.guidance();gb,sb=b.guidance()
            for x,y in zip(ga,gb):self.assertTrue(torch.equal(x,y))
            x,dx=old.reconstruct(ref,ga,sa,old.GradientSettings(mode));y,dy=reconstruct(ref,gb,sb,GradientSettings(mode))
            self.assertTrue(torch.equal(x,y));self.assertEqual(dx['iterations'],dy['iterations'])
    def test_archived_study_off_path_exact(self):
        from studies.tt_cea.tests.test_pipeline import IndependentMock,PipelineTests,config
        from studies.tt_cea.pipeline import ExperimentalPipeline
        from studies.tt_cea.schedule import prepare_interval_table
        from studies.all_prompts.runtime import ordinary_run
        old=archived('_legacy_pipeline','studies/tt_cea/pipeline.py');h=PipelineTests()
        for pixel in (False,True):
            outputs=[];inputs=[]
            for cls in (old.ExperimentalPipeline,ExperimentalPipeline):
                b=IndependentMock(pixel);p=cls(b,h.cams(),config(pixel),size=(8,16))
                _,erp,_=ordinary_run(p,h.initial(),[{'offset':torch.tensor(.1)}]*89,prepare_interval_table(b))
                outputs.append(erp);inputs.append(b.inputs)
            self.assertTrue(torch.equal(*outputs))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(*inputs)))

    def test_old_metadata_certificate_still_rejects_changed_generation_source(self):
        from studies.tt_cea.metadata import certify_metadata_revision
        with self.assertRaisesRegex(AssertionError,'Protected source changed'):
            certify_metadata_revision()
        before=read(OUT/'history/source-hashes.json')
        self.assertEqual(sha('studies/tt_cea/metadata.py'),before['studies/tt_cea/metadata.py'])
