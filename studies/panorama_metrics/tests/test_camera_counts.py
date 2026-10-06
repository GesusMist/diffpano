import copy
import unittest
from unittest.mock import patch
from studies.panorama_metrics.common import *
from studies.panorama_metrics import camera_counts as counts
from studies.panorama_metrics.report import key
from studies.all_prompts.common import inventory as prompts

class CameraCounts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts={p['name']:p for p in prompts()}
        cls.layouts={(s,n):read(counts.cc.layout_path(s,n)) for s in counts.cc.STRATEGIES for n in counts.cc.COUNTS}

    def descriptor(self,row):
        return counts.descriptor(row,self.prompts[row['prompt']],self.layouts[row['strategy'],row['num_cameras']])

    def record(self,row):
        r=self.descriptor(row);m=read(counts.cc.folder(row)/'config.json')
        if row['family']=='diffpano':model=m['runtime_audit']['model_checkpoint'];revision=m['runtime_audit']['model_revision']
        else:model=m['configuration']['model_source'];revision=m['configuration']['revision']
        r.update(configuration=m,model_id=model,model_revision=revision,schedule=m['schedule'],schedule_sha256=m['schedule_sha256'],initialization=m['initialization'],completion_status='complete')
        return r

    def test_no_count_or_original_baseline_collisions(self):
        rr=[self.descriptor(r) for r in counts.cc.rows()]
        self.assertEqual(len({r['id'] for r in rr}),126)
        self.assertEqual(len({key(r) for r in rr}),42)
        self.assertEqual(sum(r['method']=='diffpano' for r in rr),108)
        sf=[r for r in rr if r['method']=='spherediff_camera_override']
        self.assertEqual(len(sf),18)
        self.assertTrue(all(r['camera_layout']=='old' and r['steps']==20 for r in sf))
        for n,seed in ((70,0),(50,1),(30,7)):
            random=[r for r in rr if r['camera_layout']=='random' and r['camera_count']==n]
            self.assertEqual(len(random),12)
            self.assertTrue(all(r['layout_seed']==seed for r in random))

    def test_failed_sphere_geometry_never_becomes_scorable(self):
        row=next(r for r in counts.cc.rows() if r['family']=='spherediff_camera_override' and r['num_cameras']==70)
        with patch.object(counts.cc,'complete',side_effect=AssertionError('must not inspect missing output')):
            r=counts.inspect(row,self.prompts[row['prompt']],self.layouts['old',70],{})
        self.assertEqual(r['completion_status'],'blocked')
        self.assertNotIn('image_sha256',r)

    def test_count_pair_rejects_scientific_mismatches(self):
        row=next(r for r in counts.cc.rows() if r['family']=='diffpano' and r['backend']=='flux' and r['strategy']=='old' and r['num_cameras']==70 and r['prompt']=='firework')
        b=self.record(row)
        old=read(ROOT/'history/20261002-before-camera-count-refresh/inventory.json')['rows']
        a=next(r for r in old if r['method']=='diffpano' and r['backend']=='flux' and r['steps']==20 and r['consensus_projection']=='erp' and r['camera_strategy']=='old89' and r['prompt_id']=='firework')
        self.assertTrue(counts.paired_compatible(a,b,'count_vs_89'))
        for field in ('effective_prompt_sha256','model_revision','steps'):
            changed=copy.deepcopy(b);changed[field]='different'
            with self.assertRaises(AssertionError):counts.paired_compatible(a,changed,'count_vs_89')
        changed=copy.deepcopy(b);changed['initialization']['source_sha256']='different'
        with self.assertRaises(AssertionError):counts.paired_compatible(a,changed,'count_vs_89')

    def test_method_pair_requires_same_physical_geometry(self):
        rr=[r for r in counts.cc.rows() if r['backend']=='flux' and r['num_cameras']==50 and r['strategy']=='old' and r['prompt']=='firework']
        a=self.record(next(r for r in rr if r['family']=='spherediff_camera_override'))
        b=self.record(next(r for r in rr if r['family']=='diffpano'))
        self.assertTrue(counts.paired_compatible(a,b,'method_at_fixed_count'))
        changed=copy.deepcopy(b);changed['camera_geometry_sha256']='different'
        with self.assertRaises(AssertionError):counts.paired_compatible(a,changed,'method_at_fixed_count')

if __name__=='__main__':unittest.main()
