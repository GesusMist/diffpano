"""Validate sweep scope and collection accounting without new model inference."""
import unittest
import numpy as np
from studies.gradient_blending.common import *
from studies.gradient_blending.evaluate import collection_summary

class SceneChecks(unittest.TestCase):
    def test_explicit_scope_and_no_duplicate_pilot_generation(self):
        rows=scene_rows()
        self.assertEqual(len(rows),34)
        self.assertEqual(len({(r['prompt'],r['mode']) for r in rows}),34)
        self.assertEqual(len(NEW_FLUX_PROMPTS),17)
        self.assertEqual(len(FLUX_SCENE_PROMPTS),20)
        self.assertFalse(set(NEW_FLUX_PROMPTS)&set(PILOT_PROMPTS))
        self.assertNotIn('native_control',FLUX_SCENE_PROMPTS)
        self.assertEqual({r['mode'] for r in rows},{'poisson_mean','poisson_select'})
        for backend in BACKENDS:
            root,prompts=suite_context(backend,'pilot')
            self.assertEqual(prompts,('ruins','underwater','firework'))
            self.assertEqual(root,BASE_OUT if backend=='sana' else BASE_OUT/backend)
        with self.assertRaises(ValueError):suite_context('sana','flux-scenes20')
        with self.assertRaises(ValueError):suite_context('flux','unknown')
        root,prompts=suite_context('flux','flux-scenes20')
        self.assertNotEqual(root,BASE_OUT/'flux');self.assertEqual(len(prompts),20)
    def test_collection_counts_and_matched_subsets(self):
        prompts=('a','b');rows=[];features={}
        for index,prompt in enumerate(prompts):
            for mode in MODES:
                rows.append(dict(prompt=prompt,mode=mode,metrics={k:1.+2*index for k in ['DS','CS','Seam-SSIM','Seam-Sobel']}))
                features[(prompt,mode)]={'logits':np.zeros((2+index,4))}
        both=collection_summary(prompts,rows,features)
        single=collection_summary(('b',),rows,features)
        for mode in MODES:
            self.assertEqual(both[mode]['panorama_count'],2);self.assertEqual(both[mode]['view_count'],5)
            self.assertEqual(both[mode]['DS'],2.);self.assertAlmostEqual(both[mode]['IS'],1.)
            self.assertEqual(single[mode]['panorama_count'],1);self.assertEqual(single[mode]['view_count'],3)
            self.assertEqual(single[mode]['DS'],3.)
        with self.assertRaises(AssertionError):collection_summary(prompts,rows[:-1],features)
    def test_original_three_prompt_scores_unchanged(self):
        base=BASE_OUT/'flux';old=read(base/'evaluation/summary.json');features={}
        for prompt in PILOT_PROMPTS:
            for mode in MODES:
                with np.load(base/'evaluation'/(prompt+'-'+mode+'-features.npz'),allow_pickle=False) as values:
                    features[(prompt,mode)]={'logits':values['logits']}
        now=collection_summary(PILOT_PROMPTS,old['rows'],features)
        for mode in MODES:
            for metric in ('DS','CS','IS','Seam-SSIM','Seam-Sobel'):
                self.assertAlmostEqual(now[mode][metric],old['collections'][mode][metric],places=12)
            self.assertEqual(now[mode]['view_count'],24)

if __name__=='__main__':unittest.main()
