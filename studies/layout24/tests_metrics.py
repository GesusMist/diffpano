"""Matched-collection and content/protocol reuse tests without model downloads."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from studies.layout24.evaluate import summarize,compatible_features
from studies.layout24.common import VARIANTS,sha
from studies.followup24.common import PROMPTS24,PROMPTS20,NEW4

class Collections(unittest.TestCase):
    def fixture(self):
        rows=[];features={}
        for variant in VARIANTS:
            for i,name in enumerate(PROMPTS24):
                key=variant+'/'+name
                rows.append(dict(case_id=key,variant=variant,prompt=name,camera_count=89,
                    completion_status='complete',scoring_status='complete',DS=.5,CS=30.,generation_seconds=None))
                logits=np.zeros((8,10),dtype=np.float32);logits[:,i%10]=4.
                features[key]={'logits':logits}
        return rows,features
    def test_exact_collections_and_subset_is(self):
        from studies.panorama_metrics.numerics import inception_score
        rows,features=self.fixture()
        main=summarize(rows,features);old=summarize(rows,features,PROMPTS20,'original20');new=summarize(rows,features,NEW4,'new4')
        self.assertEqual(len(main),9)
        for group in main:self.assertEqual(group['view_count'],192);self.assertTrue(group['complete']);self.assertIsNone(group['mean_generation_seconds'])
        self.assertEqual(old[0]['view_count'],160);self.assertEqual(new[0]['view_count'],32)
        expected=inception_score(np.concatenate([features[VARIANTS[0]+'/'+n]['logits'] for n in NEW4]))
        self.assertAlmostEqual(new[0]['IS'],expected)
        self.assertGreater(main[0]['IS'],1.1) # Average single-image IS would incorrectly equal one.
    def test_partial_stays_null(self):
        rows,features=self.fixture();rows[0].update(scoring_status='pending_generation',completion_status='missing',DS=None,CS=None)
        group=summarize(rows,features)[0]
        self.assertFalse(group['complete']);self.assertEqual(group['view_count'],184)
        for key in ('mean_DS','mean_CS','IS'):self.assertIsNone(group[key])
    def test_cache_requires_content_and_prompt_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'features.npz';record=Path(temporary)/'record.json';record.write_text('{}')
            np.savez(path,logits=np.ones((8,10)),clip_views=np.ones((10,3)))
            candidate=dict(feature_path=str(path),feature_sha256=sha(path),effective_prompt_sha256='prompt',record_source=str(record))
            self.assertIsNotNone(compatible_features({'image':[candidate]},'image','prompt')[0])
            self.assertIsNone(compatible_features({'image':[candidate]},'wrong','prompt')[0])
            self.assertIsNone(compatible_features({'image':[candidate]},'image','wrong')[0])
            candidate['feature_sha256']='corrupt'
            self.assertIsNone(compatible_features({'image':[candidate]},'image','prompt')[0])

if __name__=='__main__':unittest.main()
