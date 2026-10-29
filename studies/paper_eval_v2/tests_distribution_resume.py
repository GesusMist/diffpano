"""Estimator equivalence and append-only recovery checks for Group2 continuation."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from .distribution import paired_bootstrap
from .distribution_resume import longest_prefix, resume_linked, target, validate_saved


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(53)
        # Unequal multiplicities exercise matching by prompt, not view count.
        self.first = {str(p):rng.normal(size=(p+1, 8)) for p in range(3)}
        self.second = {str(p):rng.normal(size=(4-p, 8)) for p in range(3)}
        self.reference = {str(p):rng.normal(size=(2, 8)) for p in range(3)}

    def original(self, metric='KID', count=9):
        result = paired_bootstrap(self.first, self.second, self.reference, metric,
            linked_reference=True, replicates=count, seed=721, max_seconds=None)
        result.update(metric=metric, namespace='native_five_v2', reference_kind='native',
            bank='A', cohort_id='synthetic3', baseline='first', comparison='second')
        return result

    def partial(self, original, count=3):
        result = copy.deepcopy(original)
        result.update(replicates=result['replicates'][:count], achieved_replicates=count,
            state='pending_uncertainty_compute', ci95=None, seconds=0.)
        return result

    def test_linked_kid_exact_draws_match_frozen_function(self):
        expected = self.original('KID')
        partial = self.partial(expected)
        before = copy.deepcopy(partial)
        actual = resume_linked(partial, self.first, self.second, self.reference)
        self.assertEqual(actual['replicates'], expected['replicates'])
        self.assertEqual(actual['ci95'], expected['ci95'])
        self.assertEqual(actual['mean_difference'], expected['mean_difference'])
        self.assertEqual(partial, before)
        self.assertEqual(actual['state'], 'complete')

    def test_linked_fid_exact_draws_match_frozen_function(self):
        expected = self.original('FID')
        actual = resume_linked(self.partial(expected), self.first, self.second, self.reference)
        self.assertEqual(actual['replicates'], expected['replicates'])
        self.assertEqual(actual['ci95'], expected['ci95'])

    def test_zero_saved_draws_match_frozen_function(self):
        expected = self.original('KID', 4)
        actual = resume_linked(self.partial(expected, 0), self.first, self.second, self.reference)
        self.assertEqual(actual['replicates'], expected['replicates'])

    def test_interrupted_checkpoint_resumes_remaining_draws(self):
        expected = self.original('FID', 9)
        partial = self.partial(expected, 2)
        checkpoints = []
        tick = [0]
        def clock():
            tick[0] += 1
            return tick[0]
        first = resume_linked(partial, self.first, self.second, self.reference,
            clock=clock, deadline=5, checkpoint=checkpoints.append, checkpoint_every=2)
        self.assertLess(first['achieved_replicates'], 9)
        self.assertGreater(first['achieved_replicates'], 2)
        self.assertIsNone(first['ci95'])
        saved = longest_prefix(partial, [first, checkpoints[-1]])
        actual = resume_linked(saved, self.first, self.second, self.reference)
        self.assertEqual(actual['replicates'], expected['replicates'])
        self.assertEqual(actual['ci95'], expected['ci95'])

    def test_completed_contrast_does_no_numerical_work(self):
        expected = self.original('KID', 3)
        with patch('studies.paper_eval_v2.distribution_resume.kid', side_effect=AssertionError('recomputed')):
            actual = resume_linked(expected, self.first, self.second, self.reference)
        self.assertEqual(actual, expected)
        self.assertFalse(target(expected))

    def test_changed_prefix_is_rejected(self):
        original = self.partial(self.original('KID', 5), 2)
        changed = copy.deepcopy(original)
        changed['replicates'][0] += .1
        with self.assertRaisesRegex(ValueError, 'prefix changed'):
            longest_prefix(original, [changed])
        with self.assertRaisesRegex(ValueError, 'saved bootstrap prefix'):
            resume_linked(changed, self.first, self.second, self.reference)

    def test_altered_observed_estimate_is_rejected(self):
        original = self.partial(self.original('FID', 4), 2)
        original['mean_difference'] += 1.
        with self.assertRaisesRegex(ValueError, 'observed contrast'):
            resume_linked(original, self.first, self.second, self.reference)

    def test_checkpoint_identity_and_count_are_guarded(self):
        original = self.partial(self.original('FID', 4), 2)
        changed = copy.deepcopy(original)
        changed['bootstrap_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            longest_prefix(original, [changed])
        changed = copy.deepcopy(original)
        changed['achieved_replicates'] += 1
        with self.assertRaisesRegex(ValueError, 'replicate count'):
            validate_saved(changed)


if __name__ == '__main__':
    unittest.main()
