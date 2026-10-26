"""Real temporary-directory archival and mocked model/scheduler orchestration."""
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studies.refinement28_resume import common, run


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original_out = self.root/'original'
        self.out = self.original_out/'recovery-v1'
        self.plan_path = self.out/'manifest.json'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module, field, value in (
            (common, 'ROOT', self.root), (common, 'ORIGINAL_OUT', self.original_out),
            (common, 'OUT', self.out), (common, 'PLAN', self.plan_path),
            (run.original, 'ROOT', self.root), (run.original, 'OUT', self.original_out),
        ):
            self.stack.enter_context(patch.object(module, field, value))
        self.stack.enter_context(patch.dict(os.environ, SLURM_JOB_ID='resume-test-job'))
        self.rows = run.original.definitions()
        for row in self.rows:
            row.update(source_sha256='frozen-source', semantic_sha256='semantic-'+str(row['index']))
        self.row = self.rows[24]
        self.directory = run.original.folder(self.row)
        self.directory.mkdir(parents=True)
        (self.directory/'case.lock').touch()
        (self.directory/'final.png').write_bytes(b'original image, preserve all bytes')
        common.write(self.directory/'status.json', dict(state='running', job='old-timed-out-job'))
        self.manifest = dict(rows=self.rows)
        common.write(self.original_out/'manifest.json', self.manifest)
        common.write(self.original_out/'launch.json', dict(jobs={'bulk': {'job_id': '111'}}))
        self.plan = dict(original_manifest_sha256=common.sha(self.original_out/'manifest.json'),
                         source_sha256='resume-source', cases=[])
        for index in common.INDICES:
            row = self.rows[index]
            self.plan['cases'].append(dict(index=index, case_id=row['case_id'],
                source_sha256=row['source_sha256'], semantic_sha256=row['semantic_sha256'],
                directory=str(run.original.folder(row)), snapshot=common.snapshot(run.original.folder(row))))
        common.write(self.plan_path, self.plan)
        self.gate = self.stack.enter_context(patch.object(common, 'gate', return_value=self.plan))
        self.preservation = dict(content_verified=True, original_git_head='old',
                                 preparation_git_head='prepared', observed_git_head='current')
        self.preserve = self.stack.enter_context(patch.object(common, 'preserve_original', return_value=self.preservation))
        self.original_preserved = run.original.preserved
        self.stack.enter_context(patch.object(run.original_run, 'require_allocation'))
        self.stack.enter_context(patch.object(run.original, 'freeze_check', return_value=self.manifest))
        self.stack.enter_context(patch.object(run.original, 'validation_gate'))
        self.stack.enter_context(patch.object(run, 'require_no_original_jobs', return_value={'active_original_jobs': []}))
        self.stack.enter_context(patch.object(run.original, 'complete', side_effect=self.complete))
        self.generator = self.stack.enter_context(patch.object(run.original_run, 'run', side_effect=self.generate))

    def complete(self, row, verify=True):
        self.assertTrue(verify)
        directory = run.original.folder(row)
        return (directory/'metadata.json').exists() and (directory/'status.json').exists() and common.read(directory/'status.json').get('state') == 'complete'

    def generate(self, index, stage):
        self.assertEqual(stage, 'bulk')
        self.assertIs(run.original.preserved, common.preserve_original)
        self.assertEqual(run.original.preserved(), self.preservation)
        row = self.rows[index]
        directory = run.original.folder(row)
        self.assertFalse(directory.exists())
        directory.mkdir(parents=True)
        (directory/'final.png').write_bytes(b'new independently generated output')
        metadata = dict(index=index, source_sha256=row['source_sha256'], execution={'job': 'resume-test-job'},
                        artifacts={'final.png': common.sha(directory/'final.png')})
        common.write(directory/'metadata.json', metadata)
        common.write(directory/'status.json', dict(state='complete', job='resume-test-job'))
        return metadata

    def archive_path(self, row=None):
        return self.out/'attempts'/(row or self.row)['case_id']

    def journal_path(self, row=None):
        return self.out/'journals'/(row or self.row)['case_id']/'attempt.json'

    def test_archive_real_directory_and_restore_original_callable(self):
        expected = common.snapshot(self.directory)
        old_inode = (self.directory/'final.png').stat().st_ino
        result = run.run(24)
        self.assertEqual(common.snapshot(self.archive_path()), expected)
        self.assertEqual((self.archive_path()/'final.png').stat().st_ino, old_inode)
        self.assertNotEqual((self.directory/'final.png').read_bytes(), (self.archive_path()/'final.png').read_bytes())
        self.assertIs(run.original.preserved, self.original_preserved)
        journal = common.read(self.journal_path())
        self.assertEqual([event['state'] for event in journal['events']],
                         ['archive_intent', 'archived', 'generation_started', 'complete'])
        self.assertEqual(journal['preservation_before']['observed_git_head'], 'current')
        audit = common.read(self.journal_path().parent/'result.json')
        self.assertEqual(audit['original_execution']['job'], 'resume-test-job')
        self.assertEqual(result, common.read(self.directory/'metadata.json'))
        self.assertEqual(audit['original_metadata_sha256'], common.sha(self.directory/'metadata.json'))
        self.assertEqual(self.generator.call_args.kwargs, {'stage': 'bulk'})

    def test_actual_libc_rename_never_replaces_existing_destination(self):
        source, target = self.root/'source', self.root/'target'
        source.mkdir(); target.mkdir()
        (source/'image.png').write_bytes(b'source pixels')
        (target/'image.png').write_bytes(b'archived pixels')
        with self.assertRaises(FileExistsError):
            run._rename_no_replace(source, target)
        self.assertEqual((source/'image.png').read_bytes(), b'source pixels')
        self.assertEqual((target/'image.png').read_bytes(), b'archived pixels')

    def test_existing_archive_without_journal_is_preserved(self):
        archive = self.archive_path()
        archive.mkdir(parents=True)
        (archive/'final.png').write_bytes(b'earlier archived output')
        before = common.snapshot(self.directory)
        with self.assertRaises(FileExistsError):
            run.run(24)
        self.assertEqual(common.snapshot(self.directory), before)
        self.assertEqual((archive/'final.png').read_bytes(), b'earlier archived output')
        self.generator.assert_not_called()

    def test_completed_retry_is_idempotent_without_rewriting_evidence(self):
        first = run.run(24)
        journal_bytes = self.journal_path().read_bytes()
        result_bytes = (self.journal_path().parent/'result.json').read_bytes()
        metadata_bytes = (self.directory/'metadata.json').read_bytes()
        self.assertEqual(run.run(24), first)
        self.generator.assert_called_once()
        self.assertEqual(self.journal_path().read_bytes(), journal_bytes)
        self.assertEqual((self.journal_path().parent/'result.json').read_bytes(), result_bytes)
        self.assertEqual((self.directory/'metadata.json').read_bytes(), metadata_bytes)

    def test_changed_prepared_snapshot_blocks_archive(self):
        (self.directory/'progress.json').write_text('new unexpected progress')
        before = common.snapshot(self.directory)
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            run.run(24)
        self.assertEqual(common.snapshot(self.directory), before)
        self.assertFalse(self.archive_path().exists())
        self.generator.assert_not_called()

    def test_stale_gate_fails_before_archival(self):
        self.gate.side_effect = AssertionError('Protected scientific bytes changed')
        before = common.snapshot(self.directory)
        with self.assertRaises(AssertionError):
            run.run(24)
        self.assertEqual(common.snapshot(self.directory), before)
        self.assertFalse(self.archive_path().exists())
        self.generator.assert_not_called()

    def test_absent_case_starts_fresh_without_invented_archive(self):
        row = self.rows[44]
        self.assertFalse(run.original.folder(row).exists())
        run.run(44)
        self.assertFalse(self.archive_path(row).exists())
        self.assertTrue(self.complete(row))
        self.assertFalse(common.read(self.journal_path(row))['archive_exists'])

    def test_crash_after_atomic_move_can_resume_before_generation(self):
        real_rename = run._rename_no_replace
        def crash_after_move(source, target):
            real_rename(source, target)
            raise RuntimeError('simulated interruption after atomic rename')
        with patch.object(run, '_rename_no_replace', side_effect=crash_after_move):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                run.run(24)
        self.assertFalse(self.directory.exists())
        self.assertTrue(self.archive_path().exists())
        self.assertEqual(common.read(self.journal_path())['state'], 'archive_intent')
        self.generator.assert_not_called()
        run.run(24)
        self.generator.assert_called_once()
        self.assertEqual(common.read(self.journal_path())['state'], 'complete')

    def test_failed_generation_restores_callable_and_cannot_repeat(self):
        self.generator.side_effect = RuntimeError('simulated generation failure')
        with self.assertRaisesRegex(RuntimeError, 'generation failure'):
            run.run(24)
        self.assertIs(run.original.preserved, self.original_preserved)
        self.assertEqual(common.read(self.journal_path())['state'], 'failed')
        self.assertEqual((self.archive_path()/'final.png').read_bytes(), b'original image, preserve all bytes')
        with self.assertRaisesRegex(RuntimeError, 'duplicate generation'):
            run.run(24)
        self.generator.assert_called_once()

    def test_external_lock_blocks_duplicate_process(self):
        path = self.journal_path().parent/'lock'
        path.parent.mkdir(parents=True)
        with path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                run.run(24)
        self.assertFalse(self.archive_path().exists())
        self.generator.assert_not_called()

    def test_old_case_lock_blocks_archiving_an_active_writer(self):
        with (self.directory/'case.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                run.run(24)
        self.assertFalse(self.archive_path().exists())
        self.assertTrue(self.directory.exists())
        self.generator.assert_not_called()

    def test_unapproved_indices_never_reach_a_gate(self):
        for index in (0, 23, 28, 40, 48, True, '24'):
            with self.subTest(index=index), self.assertRaises(ValueError):
                run.run(index)
        self.gate.assert_not_called()
        self.generator.assert_not_called()


class SchedulerGuardTests(unittest.TestCase):
    def test_original_jobs_block_but_other_resume_cases_are_allowed(self):
        with tempfile.TemporaryDirectory() as name, patch.object(common, 'ORIGINAL_OUT', Path(name)):
            common.write(Path(name)/'launch.json', {'jobs': {'bulk': {'job_id': '111'}}})
            for text in ('111_24|renamed-job\n', '999|refinement28-bulk\n', '999|refinement28-report\n'):
                with self.subTest(text=text), self.assertRaises(RuntimeError):
                    run.require_no_original_jobs(query=lambda *args, **kwargs: text)
            result = run.require_no_original_jobs(query=lambda *args, **kwargs:
                '222_25|refinement28-resume\n333|unrelated-job\n')
            self.assertEqual(result['active_original_jobs'], [])
            with self.assertRaises(ValueError):
                run.require_no_original_jobs(query=lambda *args, **kwargs: 'unparseable')


if __name__ == '__main__':
    unittest.main()
