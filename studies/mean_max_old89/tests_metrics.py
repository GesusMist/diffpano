import unittest
import numpy as np
from studies.mean_max_old89.common import *
from studies.mean_max_old89.evaluate import summarize


class MetricTests(unittest.TestCase):
    def test_matched_six_groups_and_subsets(self):
        rows=[dict(r,completion_status='complete',scoring_status='complete',DS=.1,CS=20.,generation_seconds=1800.) for r in definitions()]
        features={r['case_id']:dict(logits=np.zeros((8,10),dtype=np.float32)) for r in rows}
        for names,subset in ((PROMPTS,'all15'),(ORIGINAL,'original11'),(NEW,'new4')):
            groups=summarize(rows,features,names,subset);self.assertEqual(len(groups),6)
            for group in groups:
                self.assertEqual(group['matched_prompt_ids'],list(names));self.assertEqual(group['view_count'],8*len(names))
                self.assertAlmostEqual(group['IS'],1.,places=12);self.assertEqual(group['scored_panorama_count'],len(names))
                self.assertEqual(group['subset'],subset)
        rows[0].update(scoring_status='pending_generation',completion_status='missing',DS=None,CS=None)
        group=summarize(rows,features)[0];self.assertIsNone(group['IS']);self.assertIsNone(group['mean_DS']);self.assertIsNone(group['mean_CS'])
        self.assertEqual(group['scored_panorama_count'],14);self.assertTrue(group['provisional'])

    def test_subset_is_from_subset_features_not_image_scores(self):
        from studies.panorama_metrics.numerics import inception_score
        rows=[dict(r,completion_status='complete',scoring_status='complete',DS=.1,CS=20.,generation_seconds=1800.) for r in definitions()]
        features={}
        for r in rows:
            logits=np.zeros((8,20),dtype=np.float32);logits[:,PROMPTS.index(r['prompt'])]=10
            features[r['case_id']]=dict(logits=logits)
        for names in (PROMPTS,ORIGINAL,NEW):
            group=summarize(rows,features,names)[0]
            expected=inception_score(np.concatenate([features[r['case_id']]['logits'] for r in rows if r['prompt'] in names and r['reference_mode']=='screened' and r['gradient_mix_alpha']==.25]))
            self.assertEqual(group['IS'],expected);self.assertGreater(group['IS'],1.)
