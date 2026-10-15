"""Focused checks that the retry cannot widen scope or alter the trajectory loop."""
import copy
import unittest
from dataclasses import asdict
from studies.mean_max_firefly_retry.common import *


class RetryScope(unittest.TestCase):
    def test_exactly_one_semantic_change(self):
        rows = definitions()
        self.assertEqual(len(rows), 1)
        check_scope(rows[0])
        self.assertNotEqual(rows[0]['semantic_sha256'], base_row()['semantic_sha256'])
        self.assertEqual(rows[0]['case_id'], base_row()['case_id'])
        self.assertNotEqual(folder(rows[0]), base.folder(base_row()))

    def test_rejects_unapproved_changes(self):
        row = definitions()[0]
        for key, value in [('gradient_mix_alpha', .5), ('relative_tolerance', 1e-3), ('constrained_max_iterations', 500)]:
            changed = copy.deepcopy(row)
            changed['semantic']['solver'][key] = value
            changed['semantic_sha256'] = digest(changed['semantic'])
            with self.assertRaises(AssertionError):
                check_scope(changed)

    def test_resolved_config_matches_original(self):
        from studies.mean_max_firefly_retry.configuration import resolved
        row = definitions()[0]
        c, cameras, owners, slots, settings, geometry, old = resolved(row)
        self.assertEqual(safe(asdict(c)), base_row()['semantic']['configuration'])
        self.assertEqual(safe(geometry), base_row()['semantic']['geometry'])
        self.assertEqual(settings.to_record(), row['semantic']['solver'])
        self.assertEqual(settings.absolute_tolerance, 1e-7)
        self.assertEqual(settings.constrained_max_iterations, 400)
        self.assertEqual(settings.relative_tolerance, NEW_RTOL)

    def test_runtime_is_existing_wrapper(self):
        source = (ROOT / 'studies/mean_max_old89/runtime.py').read_text()
        expected = source.replace('from studies.mean_max_old89.common import *',
                                  'from studies.mean_max_firefly_retry.common import *').replace(
                                  'from studies.mean_max_old89.configuration import resolved',
                                  'from studies.mean_max_firefly_retry.configuration import resolved')
        self.assertEqual((ROOT / 'studies/mean_max_firefly_retry/runtime.py').read_text(), expected)


if __name__ == '__main__':
    unittest.main()
