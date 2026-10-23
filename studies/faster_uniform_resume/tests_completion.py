"""CPU-only regression tests for historical pruning and resume boundaries."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from studies.faster_uniform_resume.completion import CompletionVerifier, sha256


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for index, layout in enumerate(('uniform89', 'old89', 'uniform70', 'old70')):
            self.rows.append(dict(
                index=index, case_id='prompts/scene/diffpano_' + layout + '_rgb',
                semantic_sha256='semantic-' + str(index), source_sha256='frozen-source',
                variant='diffpano_' + layout + '_rgb', prompt='scene',
                layout=layout, camera_count=int(layout.replace('uniform', '').replace('old', '')),
                subset='original20', semantic={'seed': 0, 'fov': 80},
            ))
        self.records = []
        for row in self.rows[:2]:
            directory = self.root / 'cases' / row['case_id']
            directory.mkdir(parents=True)
            (directory / 'final.png').write_bytes(b'original-image-' + str(row['index']).encode())
            image_hash = sha256(directory / 'final.png')
            metadata = dict(
                reused=bool(row['index']), artifacts={'final.png': image_hash,
                                                     'deleted-tensor.pt': 'never-required'},
            )
            if metadata['reused']:
                metadata.update({key: row[key] for key in ('case_id', 'prompt', 'variant')})
            else:
                metadata.update(
                    case={k: v for k, v in row.items() if k != 'semantic'},
                    semantic_sha256=row['semantic_sha256'], source_sha256=row['source_sha256'],
                )
            (directory / 'metadata.json').write_text(json.dumps(metadata))
            record = {k: v for k, v in row.items() if k != 'semantic'}
            record.update(
                image_path=str(directory / 'final.png'),
                image_sha256_recorded=image_hash,
                metadata_sha256_recorded=sha256(directory / 'metadata.json'),
                reused=metadata['reused'],
            )
            self.records.append(record)
        pending = []
        for row in self.rows[2:]:
            record = {k: v for k, v in row.items() if k != 'semantic'}
            record['expected_image_path'] = str(self.root / 'cases' / row['case_id'] / 'final.png')
            pending.append(record)
        self.manifest = {'rows': self.rows}
        self.inventory = {'completed_rows': self.records, 'pending_rows': pending}
        self.strict = Mock(return_value=True)
        self.verifier = CompletionVerifier(self.manifest, self.inventory, self.root, self.strict)

    def test_known_generated_and_reused_results_survive_intentional_pruning(self):
        for row in self.rows[:2]:
            self.assertTrue(self.verifier.checked_complete(row))
            directory = self.root / 'cases' / row['case_id']
            self.assertFalse((directory / 'status.json').exists())
            self.assertFalse((directory / 'deleted-tensor.pt').exists())
        self.strict.assert_not_called()

    def test_changed_png_is_rejected_even_when_verify_false(self):
        path = self.root / 'cases' / self.rows[0]['case_id'] / 'final.png'
        path.write_bytes(b'changed-image')
        self.assertFalse(self.verifier.checked_complete(self.rows[0], verify=False))

    def test_changed_metadata_is_rejected(self):
        path = self.root / 'cases' / self.rows[1]['case_id'] / 'metadata.json'
        metadata = json.loads(path.read_text())
        metadata['prompt'] = 'different-scene'
        path.write_text(json.dumps(metadata))
        self.assertFalse(self.verifier.checked_complete(self.rows[1]))

    def test_missing_retained_artifact_is_rejected(self):
        (self.root / 'cases' / self.rows[0]['case_id'] / 'metadata.json').unlink()
        self.assertFalse(self.verifier.checked_complete(self.rows[0]))

    def test_altered_manifest_semantics_are_rejected(self):
        row = copy.deepcopy(self.rows[0])
        row['semantic']['fov'] = 60
        self.assertFalse(self.verifier.checked_complete(row))

    def test_unknown_identity_cannot_borrow_protected_index(self):
        row = dict(self.rows[0], case_id='prompts/other/diffpano_uniform89_rgb')
        self.assertFalse(self.verifier.checked_complete(row))
        self.assertFalse(self.verifier.checked_complete(dict(self.rows[0], index=999)))
        self.strict.assert_not_called()

    def test_snapshot_fields_do_not_change_fixed_identity(self):
        row = dict(self.rows[0], snapshot_complete=True, snapshot_image_sha256='ignored')
        self.assertTrue(self.verifier.checked_complete(row))

    def test_new_uniform_result_requires_unchanged_strict_validator(self):
        self.strict.return_value = False
        self.assertFalse(self.verifier.checked_complete(self.rows[2]))
        self.strict.assert_called_once_with(self.rows[2], verify=True)
        self.assertFalse(self.verifier.protected_complete(self.rows[2]))

    def test_other_pending_layout_cannot_be_marked_complete(self):
        self.assertFalse(self.verifier.checked_complete(self.rows[3]))
        self.strict.assert_not_called()

    def test_corrupt_inventory_partition_and_semantics_fail_closed(self):
        invalid = copy.deepcopy(self.inventory)
        invalid['completed_rows'][0]['semantic_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            CompletionVerifier(self.manifest, invalid, self.root, self.strict)
        invalid = copy.deepcopy(self.inventory)
        invalid['pending_rows'].append(copy.deepcopy(invalid['completed_rows'][0]))
        with self.assertRaises(ValueError):
            CompletionVerifier(self.manifest, invalid, self.root, self.strict)


if __name__ == '__main__':
    unittest.main()
