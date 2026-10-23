"""Regression checks for destructive stale-state and out-of-scope dispatch."""
from pathlib import Path
import tempfile
import unittest

from studies.faster_uniform_resume.dispatch import validate_selection, safe_to_generate


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(pending_indices=[17, 22], protected_indices=[4],
                         profile_index=17, bulk_indices=[22])

    def test_scope_and_stage(self):
        validate_selection(self.plan, 'profile', 17)
        validate_selection(self.plan, 'bulk', 22)
        for stage, index in [('bulk', 4), ('bulk', 17), ('profile', 22), ('bulk', 11),
                             ('unknown', 22)]:
            with self.assertRaises(ValueError):
                validate_selection(self.plan, stage, index)

    def test_existing_image_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'final.png').write_bytes(b'protected')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                safe_to_generate({}, complete=lambda *a, **k: False,
                                 folder=lambda _: directory)

    def test_complete_is_skipped_and_partial_requires_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertFalse(safe_to_generate({}, complete=lambda *a, **k: True,
                                             folder=lambda _: directory))
            self.assertTrue(safe_to_generate({}, complete=lambda *a, **k: False,
                                            folder=lambda _: directory))
            (directory / 'status.json').write_text('{"state":"failed"}')
            with self.assertRaisesRegex(ValueError, 'explicit review'):
                safe_to_generate({}, complete=lambda *a, **k: False,
                                 folder=lambda _: directory)


if __name__ == '__main__':
    unittest.main()
