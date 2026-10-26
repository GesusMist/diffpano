"""Content-based preservation, immutable preparation, and scoped recovery gates."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from studies.refinement28_resume import common, prepare


class RecoveryCommonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        target = self.root / 'stored-outputs'
        target.mkdir()
        (self.root / 'outputs').symlink_to(target, target_is_directory=True)
        self.old = self.root / 'outputs/refinement28-flux48/seed0-v1'
        self.out = self.old / 'recovery-v1'
        for name, value in dict(ROOT=self.root, ORIGINAL_OUT=self.old, OUT=self.out,
                                RESUME_OUT=self.out, PLAN=self.out / 'manifest.json',
                                AUDIT=self.out / 'audit.json').items():
            patch = mock.patch.object(common, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.source = self.root / 'studies/refinement28/common.py'
        self.source.parent.mkdir(parents=True)
        self.source.write_text('original frozen code')
        self.helper = self.root / 'studies/refinement28_resume/common.py'
        self.helper.parent.mkdir(parents=True)
        self.helper.write_text('new frozen helper')
        self.evidence = self.root / 'evidence.json'
        self.evidence.write_text('historical evidence')
        hashes = {'studies/refinement28/common.py': common.sha(self.source)}
        self.manifest = dict(source_hashes=hashes, source_sha256=common.digest(hashes),
                             prompts={str(index): {} for index in range(24)},
                             rows=[dict(index=index, case_id='prompt/case-%02d' % index,
                                        semantic_sha256='semantic-%02d' % index,
                                        source_sha256=common.digest(hashes)) for index in range(48)])
        common.write(self.old / 'manifest.json', self.manifest)
        common.write(self.old / 'startup-preservation.json', dict(
            hashes=dict(hashes, **{str(self.evidence): common.sha(self.evidence)}),
            git_head='original-head', branch='no_sphere', outputs_symlink=str(target)))
        common.write(self.old / 'validation.json', dict(
            passed=True, manifest_sha256=common.sha(self.old / 'manifest.json'),
            source_sha256=self.manifest['source_sha256']))
        self.head = 'prepared-head'
        patch = mock.patch.object(common, 'git_evidence', side_effect=self.git_evidence)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(common.original, 'freeze_check', return_value=self.manifest)
        patch.start()
        self.addCleanup(patch.stop)

    def git_evidence(self, original_head):
        return dict(original_git_head=original_head, observed_git_head=self.head,
                    branch='no_sphere', original_is_ancestor=True,
                    name_status=['A\tstudies/refinement28_resume/common.py'])

    def freeze(self):
        inputs = common.audit_original_inputs()
        cases, protected = [], []
        for row in self.manifest['rows']:
            directory = common.case_directory(row)
            value = dict(row, directory=str(directory), snapshot=common.snapshot(directory))
            (cases if row['index'] in common.INDICES else protected).append(value)
        audit = dict(original_inputs=inputs, cases=cases, protected_completed=protected)
        prepare.create_json(common.AUDIT, audit)
        source_hashes = common.sources()
        plan = dict(indices=list(common.INDICES), allowed_indices=list(common.INDICES),
                    cases=cases, protected_completed=protected,
                    audit_sha256=common.sha(common.AUDIT),
                    source_hashes=source_hashes, source_sha256=common.digest(source_hashes),
                    original_manifest_sha256=inputs['original_manifest_sha256'])
        prepare.create_json(common.PLAN, plan)
        prepare.create_json(common.OUT / 'validation.json', dict(
            passed=True, plan_sha256=common.sha(common.PLAN), audit_sha256=plan['audit_sha256'],
            source_sha256=plan['source_sha256']))
        return plan

    def test_exact_whitelist_preserves_original_row_semantics(self):
        self.assertEqual(common.INDICES, (24, 25, 26, 27, 42, 43, 44, 45, 46, 47))
        self.assertEqual(common.definitions(), [self.manifest['rows'][index] for index in common.INDICES])

    def test_benign_later_commit_is_recorded_without_invalidating_bytes(self):
        plan = self.freeze()
        self.head = 'later-benign-commit'
        evidence = common.preserve_original()
        self.assertEqual(evidence['preparation_git_head'], 'prepared-head')
        self.assertEqual(evidence['observed_git_head'], 'later-benign-commit')
        self.assertEqual(common.gate(), plan)

    def test_source_evidence_and_helper_mutations_are_rejected(self):
        self.freeze()
        for path in (self.source, self.evidence, self.helper):
            before = path.read_bytes()
            path.write_bytes(b'changed')
            with self.subTest(path=path), self.assertRaises(AssertionError):
                common.gate()
            path.write_bytes(before)
        common.gate()

    def test_original_manifest_and_preservation_records_are_immutable(self):
        self.freeze()
        for path in (self.old / 'manifest.json', self.old / 'startup-preservation.json'):
            before = path.read_bytes()
            value = common.read(path)
            value['unapproved_change'] = True
            common.write(path, value)
            with self.subTest(path=path), self.assertRaises(AssertionError):
                common.gate()
            path.write_bytes(before)

    def test_branch_and_outputs_link_remain_protected(self):
        self.freeze()
        changed_branch = self.git_evidence('original-head')
        changed_branch['branch'] = 'another-branch'
        with mock.patch.object(common, 'git_evidence', return_value=changed_branch), self.assertRaises(AssertionError):
            common.gate()
        old_link = self.root / 'outputs'
        old_link.unlink()
        old_link.symlink_to(self.root / 'elsewhere', target_is_directory=True)
        with self.assertRaises((AssertionError, OSError)):
            common.gate()

    def test_gate_allows_authorized_result_and_status_changes(self):
        plan = self.freeze()
        for row in common.definitions():
            directory = common.case_directory(row)
            common.write(directory / 'status.json', dict(state='complete'))
            (directory / 'final.png').write_bytes(b'new generated artifact')
        self.assertEqual(common.gate(), plan)

    def test_case_snapshot_detects_byte_name_size_or_symlink_changes(self):
        directory = self.root / 'case'
        missing = common.snapshot(directory)
        self.assertFalse(missing['exists'])
        directory.mkdir()
        image = directory / 'final.png'
        image.write_bytes(b'keep me')
        frozen = common.snapshot(directory)
        common.verify_snapshot(directory, frozen)
        image.write_bytes(b'changed')
        with self.assertRaises(AssertionError):
            common.verify_snapshot(directory, frozen)
        image.write_bytes(b'keep me')
        image.rename(directory / 'other.png')
        with self.assertRaises(AssertionError):
            common.verify_snapshot(directory, frozen)
        (directory / 'link').symlink_to(self.evidence)
        with self.assertRaises(AssertionError):
            common.snapshot(directory)

    def test_immutable_json_publication_never_overwrites(self):
        path = self.root / 'immutable.json'
        prepare.create_json(path, {'first': True})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare.create_json(path, {'replaced': True})
        self.assertEqual(path.read_bytes(), before)

    def test_plan_whitelist_and_audit_hash_are_bound(self):
        plan = self.freeze()
        bad = copy.deepcopy(plan)
        bad['indices'].append(0)
        common.write(common.PLAN, bad)
        with self.assertRaises(AssertionError):
            common.gate()
        common.write(common.PLAN, plan)
        audit = common.read(common.AUDIT)
        audit['altered'] = True
        common.write(common.AUDIT, audit)
        with self.assertRaises(AssertionError):
            common.gate()


if __name__ == '__main__':
    unittest.main()
