"""Collection checks: matched prompts, factor separation, and correct pooled IS."""
import unittest
import numpy as np
from studies.gradient_score_refresh.evaluate import group_rows,FACTOR_PROMPTS,MODES

class CollectionTests(unittest.TestCase):
    def records(self,suite,prompts,mode='poisson_select',reference='screened'):
        rows=[];features={}
        for i,prompt in enumerate(prompts):
            key=prompt+'/'+mode
            rows.append(dict(suite=suite,key=key,prompt=prompt,mode=mode,reference_mode=reference,last_fraction=0.,DS=float(i),CS=float(20+i),effective_prompt_sha256=prompt))
            logits=np.zeros((8,4),dtype=np.float32);logits[:,i%4]=20.
            features[(suite,key)]={'logits':logits}
        return rows,features
    def test_is_uses_whole_matched_collection(self):
        rows,features=self.records('refinement64',FACTOR_PROMPTS)
        groups=group_rows('refinement64',rows,features,FACTOR_PROMPTS)
        complete=[g for g in groups if g['complete']]
        self.assertEqual(len(complete),1)
        result=complete[0]
        self.assertEqual(result['panorama_count'],4);self.assertEqual(result['view_count'],32)
        self.assertEqual(result['DS'],1.5);self.assertEqual(result['CS'],21.5)
        # Every individual panorama's eight identical class distributions gives
        # IS=1; pooling the four different classes must instead give nearly 4.
        self.assertGreater(result['IS'],3.99)
    def test_missing_prompt_never_produces_shorter_collection_score(self):
        rows,features=self.records('refinement64',FACTOR_PROMPTS[:-1])
        groups=group_rows('refinement64',rows,features,FACTOR_PROMPTS)
        result=next(g for g in groups if g['mode']=='poisson_select' and g['reference_mode']=='screened' and g['last_fraction']==0.)
        self.assertFalse(result['complete']);self.assertEqual(result['missing_prompts'],['firework'])
        for name in ('DS','CS','IS'):self.assertIsNone(result[name])
    def test_twenty_prompt_modes_and_matched_four_remain_separate(self):
        prompts=list(FACTOR_PROMPTS)+['p'+str(i) for i in range(16)]
        all_rows=[];all_features={}
        for mode in MODES:
            rows,features=self.records('blending20',prompts,mode,'rgb' if mode=='rgb' else 'screened')
            all_rows+=rows;all_features.update(features)
        groups=group_rows('blending20',all_rows,all_features,prompts)
        self.assertEqual(len(groups),6);self.assertTrue(all(g['complete'] for g in groups))
        self.assertEqual([g['view_count'] for g in groups if g['subset']=='all20'],[160]*3)
        self.assertEqual([g['view_count'] for g in groups if g['subset']=='matched_factorial4'],[32]*3)
    def test_duplicate_prompt_is_rejected(self):
        rows,features=self.records('refinement64',FACTOR_PROMPTS)
        with self.assertRaises(AssertionError):group_rows('refinement64',rows+[rows[0]],features,FACTOR_PROMPTS)

if __name__=='__main__':unittest.main()
