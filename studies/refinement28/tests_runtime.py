"""CPU checks of actual StudyPipeline transitions over the full 28-step schedule."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from diffpano.config import GlobalPipelineConfig
from diffpano.gradient_fusion import GradientSettings
from diffpano.pipelines.base import reset_scheduler_step_state
from diffpano.refinement import RefinementConfig, independent_count, resolved_refinement
from studies.all_prompts.runtime import ordinary_run
from studies.gradient_blending.operator import StudyPipeline
from studies.refinement28 import common, runtime
from studies.tt_cea.schedule import prepare_interval_table
from studies.tt_cea.tests import test_pipeline as fixtures


class CountingBackend(fixtures.IndependentMock):
    def __init__(self):
        super().__init__(False)
        self.pipeline.scheduler.set_timesteps(28)
        self.timesteps = self.pipeline.scheduler.timesteps
        self.calls = dict(denoiser=0, encode=0, decode=0, initialize=0)

    def predict_clean_and_endpoint(self, x, timestep, conditioning):
        self.calls['denoiser'] += 1
        return super().predict_clean_and_endpoint(x, timestep, conditioning)

    def denoise_native_step(self, x, timestep, conditioning):
        self.calls['denoiser'] += 1
        self.guided_prediction_count += 1
        self.inputs.append(x.clone())
        prediction = .11*x+.07*torch.sin(x)+conditioning['offset']
        return self.pipeline.scheduler.step(prediction, timestep, x).prev_sample

    def encode_clean(self, image):
        self.calls['encode'] += 1
        return super().encode_clean(image)

    def decode_clean(self, state):
        self.calls['decode'] += 1
        return super().decode_clean(state)


class ObservedPipeline(StudyPipeline):
    """Record operations at boundaries without replacing any pipeline arithmetic."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations = []
        self.transition_inputs = None

    def advance_interval(self, states, conditions, interval, *args, **kwargs):
        before = dict(self.backend.calls)
        fusions = len(self.canvas.records)
        coupled = 28-independent_count(28, self.refinement_config.last_fraction)
        if interval.k == coupled:
            self.transition_inputs = [state.clone() for state in states]
        result = super().advance_interval(states, conditions, interval, *args, **kwargs)
        self.observations.append(dict(k=interval.k,
            calls={key: value-before[key] for key, value in self.backend.calls.items()},
            fusions=len(self.canvas.records)-fusions))
        return result

    def terminal(self, states):
        before = dict(self.backend.calls)
        fusions = len(self.canvas.records)
        result = super().terminal(states)
        self.terminal_observation = dict(
            calls={key: value-before[key] for key, value in self.backend.calls.items()},
            fusions=len(self.canvas.records)-fusions)
        return result


def run_mock(tail_steps):
    backend = CountingBackend()
    helper = fixtures.PipelineTests()
    cameras = helper.cams()
    config = fixtures.config()
    config.global_pipeline = GlobalPipelineConfig(refinement=RefinementConfig(tail_steps/28))
    pipeline = ObservedPipeline(backend, cameras, config,
        GradientSettings(mode='poisson_select', reference_mode='screened'), list(range(89)), size=(8, 16))
    conditions = [{'offset': torch.tensor(.1+i*.0001)} for i in range(89)]
    original_schedule = backend.timesteps.clone()
    initial = helper.initial()
    initial_copy = [state.clone() for state in initial]
    _, image, details = ordinary_run(pipeline, initial, conditions, prepare_interval_table(backend))
    row = dict(common.definitions()[0 if tail_steps == 5 else 1],
               semantic={'refinement': resolved_refinement(backend.timesteps, config.global_pipeline.refinement)})
    intervals, plan = runtime.validate_execution(row, pipeline, details, backend.calls)
    return dict(row=row, backend=backend, pipeline=pipeline, conditions=conditions, image=image,
                details=details, intervals=intervals, plan=plan, schedule=original_schedule,
                initial=initial, initial_copy=initial_copy)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {tail: run_mock(tail) for tail in (5, 8)}

    def test_exact_counts_original_schedule_and_no_tail_consensus(self):
        expected = {5: dict(denoiser=2492, encode=4094, decode=2136, initialize=0),
                    8: dict(denoiser=2492, encode=3560, decode=1869, initialize=0)}
        for tail, run in self.runs.items():
            with self.subTest(tail=tail):
                backend, pipeline, coupled = run['backend'], run['pipeline'], 28-tail
                self.assertEqual(backend.calls, expected[tail])
                self.assertTrue(torch.equal(backend.timesteps, run['schedule']))
                self.assertEqual([record['k'] for record in run['intervals']], list(range(28)))
                self.assertEqual([record['k'] for record in pipeline.refinement_tail.records], list(range(coupled, 28)))
                self.assertEqual(run['plan']['transition_step'], coupled+1)
                self.assertEqual(len(pipeline.canvas.records), coupled+1)
                self.assertTrue(all(record['converged'] for record in pipeline.canvas.records))
                for observation in pipeline.observations:
                    independent = observation['k'] >= coupled
                    self.assertEqual(observation['calls'], dict(denoiser=89, encode=0 if independent else 178,
                                                               decode=0 if independent else 89, initialize=0))
                    self.assertEqual(observation['fusions'], 0 if independent else 1)
                self.assertEqual(pipeline.terminal_observation,
                                 dict(calls=dict(denoiser=0, encode=0, decode=89, initialize=0), fusions=1))
                self.assertTrue(torch.isfinite(run['image']).all())
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(run['initial'], run['initial_copy'])))

    def test_paired_first_twenty_coupled_inputs_are_identical(self):
        a, b = self.runs[5], self.runs[8]
        self.assertEqual(a['details']['initial_state_sha256'], b['details']['initial_state_sha256'])
        self.assertEqual(len(a['backend'].inputs), 28*89)
        self.assertEqual(len(b['backend'].inputs), 28*89)
        for x, y in zip(a['backend'].inputs[:20*89], b['backend'].inputs[:20*89]):
            self.assertTrue(torch.equal(x, y))
        # The independent variant begins from the ordinary post-step-20 state.
        for x, y in zip(a['backend'].inputs[20*89:21*89], b['pipeline'].transition_inputs):
            self.assertTrue(torch.equal(x, y))

    def test_tail_equals_direct_original_scheduler_continuation(self):
        for tail, run in self.runs.items():
            direct = CountingBackend()
            coupled = 28-tail
            expected = []
            for state, condition in zip(run['pipeline'].transition_inputs, run['conditions']):
                reset_scheduler_step_state(direct.pipeline.scheduler)
                state = state.clone()
                for k in range(coupled, 28):
                    state = direct.denoise_native_step(state, direct.timesteps[k], condition)
                expected.append(state)
            for actual, reference in zip(run['backend'].terminal_inputs, expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            self.assertEqual(direct.calls, dict(denoiser=tail*89, encode=0, decode=0, initialize=0))

    def test_fraction_json_roundtrip_and_invalid_execution_rejected(self):
        for tail in (5, 8):
            fraction = json.loads(json.dumps(tail/28))
            self.assertEqual(independent_count(28, fraction), tail)
        self.assertEqual(independent_count(28, .18), 6)
        self.assertEqual(independent_count(28, .29), 9)
        run = self.runs[5]
        wrong = dict(run['backend'].calls, encode=run['backend'].calls['encode']+1)
        with self.assertRaises(AssertionError):
            runtime.validate_execution(run['row'], run['pipeline'], run['details'], wrong)
        changed = copy.deepcopy(run['details'])
        changed['original_noise_reinjections'] = 1
        with self.assertRaises(AssertionError):
            runtime.validate_execution(run['row'], run['pipeline'], changed, run['backend'].calls)
        record = run['pipeline'].interval_records[23]
        with patch.dict(record, fusion='unexpected solve'):
            with self.assertRaises(AssertionError):
                runtime.validate_execution(run['row'], run['pipeline'], run['details'], run['backend'].calls)

    def test_same_prompt_group_rejects_changed_conditioning(self):
        row = dict(prompt='example', prompt_identity='prompts/example-hash')
        config = dict(provenance={'conditioning_sha256': 'c'*64},
                      initialization={'initial_local_sha256': 'i'*64, 'source_sha256': 'n'*64},
                      geometry={'angular_sha256': 'g'*64}, schedule_sha256='s'*64)
        with tempfile.TemporaryDirectory() as name, patch.object(runtime, 'OUT', Path(name)):
            first = runtime.register_group(dict(row, refinement_steps=5), config)
            self.assertEqual(first, runtime.register_group(dict(row, refinement_steps=8), config))
            changed = copy.deepcopy(config)
            changed['provenance']['conditioning_sha256'] = 'z'*64
            with self.assertRaises(AssertionError):
                runtime.register_group(row, changed)
            self.assertEqual(first, runtime.read(Path(name)/'runtime-groups/prompts/example-hash/identity.json'))


if __name__ == '__main__':
    unittest.main()
