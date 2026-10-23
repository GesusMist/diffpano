"""Validate real saved configuration and the exact requested scientific matrix."""
import copy
import json
import unittest

from studies.followup24.common import PROMPTS24, prompt_path
from studies.refinement28.common import definitions, expected_counts
from studies.refinement28.configuration import resolved, schedule_evidence, semantic


class ConfigurationTests(unittest.TestCase):
    def test_exact_24_prompt_pairs_and_integer_refinement(self):
        from diffpano.refinement import independent_count
        rows = json.loads(json.dumps(definitions()))
        self.assertEqual(len(rows), 48)
        self.assertEqual({r['prompt'] for r in rows}, set(PROMPTS24))
        self.assertNotIn('native_control', {r['prompt'] for r in rows})
        for name in PROMPTS24:
            pair = [r for r in rows if r['prompt'] == name]
            self.assertEqual({r['refinement_steps'] for r in pair}, {5, 8})
            for row in pair:
                self.assertEqual(row['prompt_path'], str(prompt_path(name)))
                self.assertEqual(independent_count(row['steps'], row['last_fraction']), row['refinement_steps'])

    def test_verified_28_schedule_and_shared_pair_inputs(self):
        schedule, evidence = schedule_evidence()
        self.assertTrue(evidence['verified'])
        self.assertEqual((len(schedule['timesteps']), len(schedule['sigmas'])), (28, 29))
        self.assertEqual(schedule['sigmas'][-1], 0)
        for index in (0, 40):  # One original prompt and one new full-path prompt.
            first, second = definitions()[index:index + 2]
            a, b = semantic(first, schedule), semantic(second, schedule)
            for key in ('geometry', 'solver', 'prompt', 'schedule', 'expected_initialization', 'model'):
                self.assertEqual(a[key], b[key], key)
            self.assertEqual(a['refinement']['coupled_intervals'], 23)
            self.assertEqual(b['refinement']['coupled_intervals'], 20)
            self.assertEqual(a['refinement']['transition_step'], 24)
            self.assertEqual(b['refinement']['transition_step'], 21)
            ca, cb = copy.deepcopy(a['configuration']), copy.deepcopy(b['configuration'])
            ca['global_pipeline']['refinement']['last_fraction'] = 0
            cb['global_pipeline']['refinement']['last_fraction'] = 0
            self.assertEqual(ca, cb)

    def test_exact_counters_and_no_rounding_substitution(self):
        rows = definitions()
        self.assertEqual(expected_counts(rows[0]), dict(denoiser=2492, encode=4094, decode=2136, initialize=0))
        self.assertEqual(expected_counts(rows[1]), dict(denoiser=2492, encode=3560, decode=1869, initialize=0))
        wrong = dict(rows[0], last_fraction=.18)
        with self.assertRaises(AssertionError):
            resolved(wrong)


if __name__ == '__main__':
    unittest.main()
