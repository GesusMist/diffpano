"""Entry-point safeguards against paid execution and historical writes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .cli import environment_check, main, prepare_output, select_views


class CLITests(unittest.TestCase):
    def test_foreign_output_directory_is_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'inventory.json').write_text('{"old_study":true}')
            (root / 'provenance_audit.json').write_text('{}')
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with self.assertRaises(ValueError):
                prepare_output(root)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_execute_only_available_for_evaluate(self):
        with tempfile.TemporaryDirectory() as folder:
            for command in ('prepare', 'dry-run', 'report', 'inventory'):
                with self.assertRaises(ValueError):
                    main([command, '--execute', '--output', str(Path(folder) / command)])
            self.assertFalse(list(Path(folder).iterdir()))

    def test_environment_default_is_offline_and_does_not_log_secret(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'private-test-value'}), patch('urllib.request.urlopen', side_effect=AssertionError('network')):
                result = environment_check(Path(folder), network=False)
            self.assertTrue(result['credential_available'])
            self.assertEqual(result['paid_requests'], 0)
            self.assertNotIn('private-test-value', (Path(folder) / 'environment_preflight.json').read_text())

    def test_scope_union_reuses_primary_without_sphere28(self):
        inv = {'cohorts': {'primary': {'panorama_ids': ['a', 'b']},
                           'alpha025': {'panorama_ids': ['a', 'b', 'c']},
                           'alpha075': {'panorama_ids': ['a', 'b', 'd']},
                           'sphere28': {'panorama_ids': ['a', 'b', 'e']}}}
        views = [{'panorama_id': p} for p in 'abcde']
        self.assertEqual([r['panorama_id'] for r in select_views(inv, views, 'ablations')], list('abcd'))
        self.assertEqual([r['panorama_id'] for r in select_views(inv, views, 'all')], list('abcde'))


if __name__ == '__main__':
    unittest.main()
