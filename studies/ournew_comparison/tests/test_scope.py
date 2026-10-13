import unittest
from studies.ournew_comparison.common import *
from studies.ournew_comparison.configuration import resolved

class ScopeTests(unittest.TestCase):
    def test_exact_matrix_and_paths(self):
        rows=definitions();self.assertEqual(len(rows),28);self.assertEqual(len({r['case_id'] for r in rows}),28)
        for name in PROMPTS:
            info,path=prompt(name);self.assertEqual(path.parent,ROOT/'prompts/ournew');self.assertEqual(info['line_count'],5)
            group=[r for r in rows if r['prompt']==name];self.assertEqual(tuple(r['variant'] for r in group),VARIANTS)
            self.assertEqual(sum(r['method']=='spherediff' for r in group),1)
            for r in group:self.assertEqual((r['seed'],r['steps'],r['last_fraction']),(0,20,0.))
            for r in group[3:]:self.assertEqual((r['layout'],r['gradient_mode']),('uniform89','poisson_select'))
    def test_real_layouts_and_frozen_uniform_group(self):
        rows=definitions();old=resolved(rows[1]);uniform=resolved(rows[2])
        self.assertNotEqual(old[5]['angular_sha256'],uniform[5]['angular_sha256'])
        self.assertEqual(len(old[1]),89);self.assertEqual(len(uniform[1]),89)
        self.assertTrue(all(isinstance(s,str) for s in uniform[3]));self.assertEqual(uniform[2],list(range(89)))
        self.assertEqual(old[3],list(range(89)))
        for row in rows[2:7]:
            c,cams,owners,slots,s,g,prior=resolved(row)
            self.assertEqual(g,uniform[5]);self.assertEqual(c.global_pipeline.refinement.last_fraction,0.)
            self.assertEqual(s.relative_tolerance,1e-4 if row['reference_mode']=='coarse_color' else 1e-5)
            self.assertEqual((s.absolute_tolerance,s.max_iterations,s.constrained_max_iterations),(1e-7,200,400))
            self.assertEqual(s.single_pixel_source,'fused_rgb_erp')
    def test_official_twenty_step_control(self):
        from studies.flux_step_controls.runtime import original_spec
        from studies.original_spherediff.common import specs
        spec,delta=original_spec();self.assertEqual(spec['call']['num_inference_steps'],20)
        self.assertEqual(specs()['flux']['call']['num_inference_steps'],28)
        self.assertEqual({r['path'] for r in delta},{'call.num_inference_steps'})

    def test_noise_slot_strings_preserved_with_independent_owners(self):
        import sys,torch
        from dataclasses import replace
        sys.path.insert(0,str(ROOT/'tests'))
        from test_gwtf_noise_initialization import NativeStub
        from diffpano.gwtf_noise_initialization import initialize_gwtf_shared_noise,GWTFlowNoiseConfig
        from diffpano.erp_noise_initialization import states_digest
        c,cams,owners,slots,*_=resolved(definitions()[2])
        cams=tuple(replace(cam,height=16,width=16) for cam in cams[:3]);slots=slots[:3]
        a,_=initialize_gwtf_shared_noise(NativeStub(),cams,GWTFlowNoiseConfig(17,34,0),1,camera_slots=slots)
        b,_=initialize_gwtf_shared_noise(NativeStub(),cams,GWTFlowNoiseConfig(17,34,0),1,camera_slots=slots,execution_order=[2,1,0])
        wrong,_=initialize_gwtf_shared_noise(NativeStub(),cams,GWTFlowNoiseConfig(17,34,0),1,camera_slots=owners[:3])
        self.assertEqual(states_digest(a),states_digest(b));self.assertNotEqual(states_digest(a),states_digest(wrong))

