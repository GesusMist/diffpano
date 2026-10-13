import unittest
from studies.ournew_comparison.common import *

class MetricCollectionTests(unittest.TestCase):
    def test_collection_membership_and_missing_scores(self):
        import numpy as np
        from studies.ournew_comparison.evaluate import summarize
        rows=[];features={}
        for r in definitions():
            x=dict(r,completion_status='complete',scoring_status='complete',DS=.1,CS=20.)
            rows.append(x);features[r['case_id']]=dict(logits=np.zeros((8,10),dtype=np.float32))
        groups=summarize(rows,features)
        self.assertEqual(len(groups),7)
        for g in groups:
            self.assertEqual(g['matched_prompt_ids'],list(PROMPTS));self.assertEqual(g['view_count'],32)
            self.assertEqual(g['IS'],1.);self.assertEqual(g['scored_panorama_count'],4)
        rows[6].update(scoring_status='pending_generation',completion_status='missing',DS=None,CS=None)
        groups=summarize(rows,features);self.assertIsNone(groups[-1]['IS']);self.assertIsNone(groups[-1]['mean_DS']);self.assertIsNone(groups[-1]['mean_CS'])
        self.assertEqual(groups[-1]['scored_panorama_count'],3);self.assertTrue(groups[-1]['provisional'])
