"""Small fixtures exercise scientific completion and destructive retry guards."""
import copy
import fcntl
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image

from studies.refinement28 import common, run


class StudyStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / 'outputs/refinement28-flux48/seed0-v1'
        self.patches = [mock.patch.object(common, 'ROOT', self.root),
                        mock.patch.object(common, 'OUT', self.output)]
        for patch in self.patches:
            patch.start()
        self.addCleanup(self.temporary.cleanup)
        for patch in self.patches:
            self.addCleanup(patch.stop)
        self.rows = common.definitions()
        for row in self.rows:
            self.add_semantic(row)
        self.row = self.rows[0]

    def add_semantic(self, row):
        info = dict(name=row['prompt'], path=row['prompt_identity'] + '.txt',
                    full_path=row['prompt_path'], effective_sha256='prompt-sha',
                    original_sha256='prompt-sha')
        row['semantic'] = dict(
            configuration=dict(generation=dict(num_inference_steps=28),
                               global_pipeline=dict(refinement=dict(last_fraction=row['last_fraction']))),
            solver=dict(mode='poisson_select', reference_mode='screened', lambda_color=.1,
                        relative_tolerance=1e-5, absolute_tolerance=1e-7, max_iterations=200),
            prompt=info, geometry=dict(angular_sha256='geometry-sha'),
            expected_initialization=dict(initial_local_sha256='native-sha',
                source_sha256='noise-sha', source_shape=[1, 16, 240, 480],
                scaling=1.0, scaling_applications=1),
            schedule=dict(timesteps=list(range(28)), sigmas=list(range(29))),
            refinement=dict(last_fraction=row['last_fraction'], total_intervals=28,
                            coupled_intervals=row['coupled_steps'], independent_intervals=row['refinement_steps'],
                            transition_step=row['coupled_steps'] + 1,
                            transition_timestep=float(row['coupled_steps'])))
        row['semantic_sha256'] = common.digest(row['semantic'])
        row['source_sha256'] = common.digest({'new-study.py': 'code-sha'})

    def metadata(self, row=None):
        row = self.row if row is None else row
        semantic = row['semantic']
        intervals = [dict(k=index, pass_kind='initial', phase='coupled' if index < row['coupled_steps']
                          else 'independent_refinement', fusion=None, model_timestep=float(index),
                          guided_predictions=89, stage_seconds={'independent_native': 0.01})
                     for index in range(28)]
        fusions = [dict(mode='poisson_select', reference_mode='screened', converged=True,
                        terminal=index == row['coupled_steps'], ownership_consistent=True,
                        stage='terminal' if index == row['coupled_steps'] else index + 1)
                   for index in range(row['coupled_steps'] + 1)]
        return dict(
            semantic_sha256=row['semantic_sha256'], source_sha256=row['source_sha256'],
            configuration=copy.deepcopy(semantic['configuration']), fusion=copy.deepcopy(semantic['solver']),
            prompt=copy.deepcopy(semantic['prompt']), camera_count=89, camera_geometry_sha256='geometry-sha',
            counts=common.expected_counts(row), initialization=copy.deepcopy(semantic['expected_initialization']),
            initialization_audit=dict(calls=dict(denoiser=0, encode=0, decode=0, initialize=89)),
            schedule_sha256=common.digest(semantic['schedule']),
            refinement=copy.deepcopy(semantic['refinement']),
            details=dict(refinement=copy.deepcopy(semantic['refinement']), guided_predictions=2492,
                         time_travel_enabled=False, original_noise_bank_retained=False,
                         active_consensus_projection='erp', initial_state_sha256='native-sha',
                         backward_calls=0, replay_count=0, original_noise_reinjections=0, additional_denoising_cycles=0,
                         refinement_transition=dict(original_index=row['coupled_steps'],
                            step=row['coupled_steps'] + 1, timestep=float(row['coupled_steps']),
                            native_state_sha256='a' * 64)),
            intervals=intervals, fusions=fusions, poisson_solves=len(fusions),
            terminal_fusion_mode='poisson_select', terminal_reference_mode='screened', reused=False,
            artifacts={name: 'placeholder' for name in common.ARTIFACT_NAMES})

    def artifacts(self, row=None, image_size=(4096, 2048)):
        row = self.row if row is None else row
        directory = common.folder(row)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = self.metadata(row)
        common.write(directory / 'config.json', dict(configuration=metadata['configuration'], fusion=metadata['fusion']))
        for name in ('final.png', 'terminal-rgb-reference.png'):
            Image.new('RGB', image_size, (12, 34, 56)).save(directory / name)
        for name in ('final-unclamped-fp32.pt', 'terminal-rgb-reference-fp32.pt'):
            (directory / name).write_bytes(b'fixture tensor: content hash verified, no torch import')
        metadata['artifacts'] = {name: common.sha(directory / name) for name in common.ARTIFACT_NAMES}
        return metadata

    def publish(self, metadata, row=None):
        row = self.row if row is None else row
        directory = common.folder(row)
        common.write(directory / 'metadata.json', metadata)
        common.write(directory / 'status.json', dict(
            state='complete', semantic_sha256=row['semantic_sha256'], source_sha256=row['source_sha256'],
            metadata_sha256=common.sha(directory / 'metadata.json'),
            image_sha256=metadata['artifacts']['final.png'], artifacts=metadata['artifacts']))

    def manifest(self):
        value = dict(rows=self.rows, source_hashes={'new-study.py': 'code-sha'},
                     source_sha256=self.row['source_sha256'])
        common.write(self.output / 'manifest.json', value)
        return value

    def test_exact_scope_and_operation_counts(self):
        self.assertEqual(len(self.rows), 48)
        self.assertEqual(len({row['case_id'] for row in self.rows}), 48)
        self.assertEqual({row['prompt'] for row in self.rows}, set(common.PROMPTS24))
        self.assertNotIn('native_control', {row['prompt'] for row in self.rows})
        self.assertEqual(self.rows[0]['case_id'], 'prompts/air_balloons/old89/screened/poisson_select/refine05of28')
        self.assertEqual(self.rows[40]['prompt_identity'], 'prompts/ournew/bamboo_grove')
        self.assertEqual(common.expected_counts(self.rows[0]), dict(denoiser=2492, encode=4094, decode=2136, initialize=0))
        self.assertEqual(common.expected_counts(self.rows[1]), dict(denoiser=2492, encode=3560, decode=1869, initialize=0))
        for row in self.rows:
            self.assertEqual(row['last_fraction'] * 28, row['refinement_steps'])
            self.assertEqual(row['coupled_steps'] + row['refinement_steps'], 28)

    def test_mutated_identity_or_science_is_outside_scope(self):
        for key, value in (('seed', 1), ('steps', 20), ('last_fraction', .18),
                           ('layout', 'uniform89'), ('prompt_path', '/tmp/another.txt'),
                           ('case_id', '../../outside'), ('index', True)):
            row = dict(self.row, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                common.folder(row)

    def test_complete_case_is_idempotent_and_preserves_image(self):
        metadata = self.artifacts()
        self.publish(metadata)
        before = (common.folder(self.row) / 'final.png').read_bytes()
        self.assertTrue(common.complete(self.row))
        self.assertFalse(run.safe_to_generate(self.row))
        self.assertFalse(run.safe_to_generate(self.row, retry=True))
        self.assertEqual((common.folder(self.row) / 'final.png').read_bytes(), before)

    def test_scientifically_invalid_metadata_cannot_claim_completion(self):
        valid = self.artifacts()
        mutations = {
            'wrong counts': lambda m: m['counts'].update(denoiser=1780),
            'wrong solver': lambda m: m['fusion'].update(lambda_color=.2),
            'wrong configuration': lambda m: m['configuration']['generation'].update(num_inference_steps=20),
            'wrong initial noise': lambda m: m['initialization'].update(initial_local_sha256='other'),
            'wrong schedule': lambda m: m.update(schedule_sha256='other'),
            'missing interval': lambda m: m['intervals'].pop(),
            'repeated interval': lambda m: m['intervals'][-1].update(k=26),
            'tail has fusion': lambda m: m['intervals'][-1].update(fusion={'copied': 'previous'}),
            'early tail': lambda m: m['intervals'][0].update(phase='independent_refinement'),
            'extra solve': lambda m: m['fusions'].append(dict(m['fusions'][-1])),
            'failed solve': lambda m: m['fusions'][0].update(converged=False),
            'wrong terminal': lambda m: m.update(terminal_reference_mode='coarse_color'),
            'not terminal': lambda m: m['fusions'][-1].update(terminal=False),
            'wrong tail count': lambda m: m['refinement'].update(independent_intervals=6),
            'replay': lambda m: m['details'].update(replay_count=1),
            'backward': lambda m: m['details'].update(backward_calls=1),
            'noise reinjection': lambda m: m['details'].update(original_noise_reinjections=1),
            'tail decode': lambda m: m['intervals'][-1]['stage_seconds'].update(decode=.1),
        }
        for reason, mutate in mutations.items():
            with self.subTest(reason=reason):
                metadata = copy.deepcopy(valid)
                mutate(metadata)
                self.publish(metadata)
                self.assertFalse(common.complete(self.row))

    def test_artifact_and_metadata_hashes_cannot_be_stale(self):
        metadata = self.artifacts()
        self.publish(metadata)
        directory = common.folder(self.row)
        (directory / 'final-unclamped-fp32.pt').write_bytes(b'changed')
        self.assertFalse(common.complete(self.row))
        metadata['artifacts']['final-unclamped-fp32.pt'] = common.sha(directory / 'final-unclamped-fp32.pt')
        self.publish(metadata)
        self.assertTrue(common.complete(self.row))
        metadata['extra'] = 'unrecorded modification'
        common.write(directory / 'metadata.json', metadata)
        self.assertFalse(common.complete(self.row, verify=False))

    def test_matching_hashes_do_not_accept_wrong_image_dimensions(self):
        metadata = self.artifacts(image_size=(128, 64))
        self.publish(metadata)
        self.assertFalse(common.complete(self.row))

    def test_existing_image_is_never_overwritten_even_on_retry(self):
        directory = common.folder(self.row)
        directory.mkdir(parents=True)
        for name in ('final.png', 'terminal-rgb-reference.png'):
            path = directory / name
            path.write_bytes(b'preserve')
            for retry in (False, True):
                with self.subTest(name=name, retry=retry), self.assertRaises(ValueError):
                    run.safe_to_generate(self.row, retry=retry)
                self.assertEqual(path.read_bytes(), b'preserve')
            path.unlink()
        (directory / 'final.png').symlink_to(directory / 'missing-target')
        with self.assertRaises(ValueError):
            run.safe_to_generate(self.row, retry=True)

    def test_partial_runs_need_explicit_retry_and_running_is_not_retried(self):
        directory = common.folder(self.row)
        directory.mkdir(parents=True)
        (directory / 'case.lock').touch()
        self.assertTrue(run.safe_to_generate(self.row))
        common.write(directory / 'status.json', dict(state='failed'))
        with self.assertRaises(ValueError):
            run.safe_to_generate(self.row)
        self.assertTrue(run.safe_to_generate(self.row, retry=True))
        common.write(directory / 'status.json', dict(state='running'))
        with self.assertRaises(ValueError):
            run.safe_to_generate(self.row, retry=True)

    def test_profiles_are_actual_first_two_cases_and_bulk_is_restricted(self):
        for index in (0, 1):
            run.validate_selection('profile', index)
        for index in range(2, 48):
            run.validate_selection('bulk', index)
        for stage, index in (('profile', 2), ('bulk', 0), ('bulk', 48), ('profile', -1), ('other', 0), ('profile', True)):
            with self.subTest(stage=stage, index=index), self.assertRaises(ValueError):
                run.validate_selection(stage, index)

    def test_freeze_verifies_scope_prompts_sources_and_semantics(self):
        manifest = self.manifest()
        prompts = {row['prompt']: row['semantic']['prompt'] for row in self.rows}
        with mock.patch.object(common, 'sources', return_value=manifest['source_hashes']), \
             mock.patch.object(common, 'prompt_record', side_effect=lambda name: prompts[name]):
            self.assertEqual(common.freeze_check(), manifest)
            bad = copy.deepcopy(manifest)
            bad['rows'][0]['semantic']['solver']['lambda_color'] = .2
            common.write(self.output / 'manifest.json', bad)
            with self.assertRaises(AssertionError):
                common.freeze_check()
            common.write(self.output / 'manifest.json', manifest)
            prompts['air_balloons'] = dict(prompts['air_balloons'], effective_sha256='changed')
            with self.assertRaises(AssertionError):
                common.freeze_check()
        with mock.patch.object(common, 'sources', return_value={'new-study.py': 'changed'}):
            with self.assertRaises(AssertionError):
                common.freeze_check()

    def test_validation_gate_binds_exact_source_and_manifest(self):
        manifest = self.manifest()
        gate = dict(passed=True, source_sha256=manifest['source_sha256'],
                    manifest_sha256=common.sha(self.output / 'manifest.json'))
        common.write(self.output / 'validation.json', gate)
        self.assertEqual(common.validation_gate(manifest), gate)
        for field, value in (('passed', False), ('source_sha256', 'stale'), ('manifest_sha256', 'stale')):
            common.write(self.output / 'validation.json', dict(gate, **{field: value}))
            with self.subTest(field=field), self.assertRaises(AssertionError):
                common.validation_gate(manifest)

    def test_preservation_is_read_only_and_detects_changed_evidence(self):
        historical = self.root / 'historical.json'
        historical.write_bytes(b'original evidence')
        common.write(self.output / 'startup-preservation.json', dict(hashes={'historical.json': common.sha(historical)}))
        common.preserved()
        self.assertEqual(historical.read_bytes(), b'original evidence')
        historical.write_bytes(b'changed evidence')
        with self.assertRaises(AssertionError):
            common.preserved()

    def runner_context(self):
        manifest = self.manifest()
        return [mock.patch.object(run, 'require_allocation'),
                mock.patch.object(common, 'freeze_check', return_value=manifest),
                mock.patch.object(common, 'validation_gate'), mock.patch.object(common, 'preserved'),
                mock.patch.dict('os.environ', SLURM_JOB_ID='fixture-job', SLURM_CLUSTER_NAME='faster',
                                SLURM_JOB_ACCOUNT=common.ACCOUNT)]

    def test_runner_idempotence_does_not_call_generation(self):
        self.publish(self.artifacts())
        patches = self.runner_context()
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        generator = mock.Mock(side_effect=AssertionError('must not generate again'))
        with mock.patch.dict(sys.modules, {'studies.refinement28.runtime': types.SimpleNamespace(generate=generator)}):
            run.run(0, stage='profile')
        generator.assert_not_called()

    def test_runner_failure_is_recorded_without_automatic_retry(self):
        patches = self.runner_context()
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        generator = mock.Mock(side_effect=RuntimeError('fixture runtime failure'))
        with mock.patch.dict(sys.modules, {'studies.refinement28.runtime': types.SimpleNamespace(generate=generator)}):
            with self.assertRaisesRegex(RuntimeError, 'fixture runtime failure'):
                run.run(0, stage='profile')
            self.assertEqual(common.read(common.folder(self.row) / 'status.json')['state'], 'failed')
            with self.assertRaisesRegex(ValueError, 'explicit --retry'):
                run.run(0, stage='profile')
        self.assertEqual(generator.call_count, 1)

    def test_bulk_waits_for_both_profiles(self):
        patches = self.runner_context()
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        with self.assertRaisesRegex(AssertionError, 'Both actual profile'):
            run.run(2, stage='bulk')
        self.assertFalse(common.folder(self.rows[2]).exists())

    def test_case_lock_excludes_concurrent_execution(self):
        patches = self.runner_context()
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        directory = common.folder(self.row)
        directory.mkdir(parents=True)
        with (directory / 'case.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                run.run(0, stage='profile')
        self.assertFalse((directory / 'status.json').exists())

    def test_successful_runner_writes_verifiable_completion(self):
        patches = self.runner_context()
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        generator = mock.Mock(side_effect=lambda row, directory: self.artifacts(row))
        with mock.patch.dict(sys.modules, {'studies.refinement28.runtime': types.SimpleNamespace(generate=generator)}):
            result = run.run(0, stage='profile')
        self.assertEqual(result['counts'], common.expected_counts(self.row))
        self.assertTrue(common.complete(self.row))
        self.assertEqual(generator.call_count, 1)


if __name__ == '__main__':
    unittest.main()
