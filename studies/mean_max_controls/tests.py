"""Small matrix/provenance tests; no alternate image-generation implementation."""
import unittest
from studies.mean_max_controls.common import definitions, PROMPTS, REFERENCES


class MissingMatrixTests(unittest.TestCase):
    def test_unique_full_selection(self):
        rows=definitions([dict(prompt=p,reference_mode=r) for p in PROMPTS for r in REFERENCES])
        self.assertEqual(len(rows),30)
        self.assertEqual(len(set(r['case_id'] for r in rows)),30)
        self.assertEqual([r['index'] for r in rows],list(range(30)))
        self.assertTrue(all(r['layout']=='old89' and r['steps']==20 and r['last_fraction']==0 for r in rows))
    def test_only_explicit_missing(self):
        rows=definitions({'missing':[dict(prompt='tiled_courtyard',reference_mode='screened')]})
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['case_id'],'prompts/ournew/tiled_courtyard/screened/poisson_select/refine000')
    def test_duplicate_and_unknown_rejected(self):
        item=dict(prompt='ruins',reference_mode='coarse_color')
        for entries in ([item,item],[dict(prompt='elsewhere',reference_mode='screened')],[dict(prompt='ruins',reference_mode='rgb')]):
            with self.assertRaises(ValueError):definitions(entries)
    def test_fixed_order(self):
        a=dict(prompt='ruins',reference_mode='screened');b=dict(prompt='aurora',reference_mode='coarse_color')
        self.assertEqual(definitions([a,b]),definitions([b,a]))


if __name__=='__main__':unittest.main()
