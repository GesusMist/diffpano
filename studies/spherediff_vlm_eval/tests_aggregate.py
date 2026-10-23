"""Statistical fixtures with unequal completion, fixed cohorts and missing data."""
import copy
import unittest

from .aggregate import CRITERIA, aggregate, paired_bootstrap


CAMERAS = [dict(view_id='v%02d' % i, pitch=90 if i == 12 else -90 if i == 13 else (-45, 0, 45)[i // 4],
                yaw=(i % 4) * 90 if i < 12 else 0,
                elevation_group='north' if i == 12 else 'south' if i == 13 else 'e%d' % (i // 4),
                is_seam=i < 12 and i % 4 == 2) for i in range(14)]


def fixture(prompts=('a', 'b'), methods=('sphere20', 'rgb', 'select', 'meanmax050')):
    records, views = [], []
    for method in methods:
        for prompt in prompts:
            uid = method + '-' + prompt
            records.append(dict(panorama_id=uid, method_id=method, prompt_id=prompt, prompt_name=prompt, image_sha256=uid))
            for i, camera in enumerate(CAMERAS):
                value = (2 if prompt == 'a' else 4) + (method == 'meanmax050')
                views.append(dict(records[-1], **camera, status='success', scores={c: value for c in CRITERIA}, resolved_model='gpt-4o-snapshot'))
    inventory = dict(records=records, cohorts={'primary': dict(method_ids=list(methods), prompt_ids=list(prompts))})
    return inventory, views


class AggregationTests(unittest.TestCase):
    def test_equal_panorama_weight_and_paired_differences(self):
        inv, views = fixture()
        result = aggregate(inv, views, cameras=CAMERAS, bootstrap_draws=1000)
        rgb = next(r for r in result['summaries'] if r['method_id'] == 'rgb')
        self.assertEqual(rgb['image_quality'], 3.)
        self.assertEqual(rgb['n_prompts'], 2)
        comparison = next(r for r in result['comparisons'] if r['method_id'] == 'meanmax050' and r['baseline'] == 'sphere20')
        self.assertEqual((comparison['mean_difference'], comparison['ci95_low'], comparison['ci95_high']), (1., 1., 1.))
        self.assertTrue(all(r['n_prompts'] == 2 for r in result['summaries']))

    def test_single_missing_view_removes_prompt_from_every_method(self):
        inv, views = fixture()
        views[0].update(status='failed', scores=None)
        result = aggregate(inv, views, cameras=CAMERAS, bootstrap_draws=1000)
        self.assertTrue(all(r['n_prompts'] == 1 and r['status'] == 'preliminary' for r in result['summaries']))
        incomplete = next(r for r in result['panoramas'] if r['panorama_id'] == 'sphere20-a')
        self.assertIsNone(incomplete['image_quality'])
        self.assertEqual(incomplete['successful_views'], 13)
        self.assertEqual(result['failed_views'], 1)
        self.assertIsNone(result['comparisons'][0]['ci95_low'])

    def test_no_scores_are_blank_not_zero(self):
        inv, _ = fixture()
        result = aggregate(inv, [], cameras=CAMERAS)
        self.assertTrue(all(r['image_quality'] is None for r in result['summaries']))
        self.assertEqual(result['missing_views'], len(inv['records']) * 14)
        self.assertTrue(all(r['rank'] is None for r in result['rankings']))

    def test_out_of_range_and_bool_never_aggregate(self):
        inv, views = fixture(prompts=('a',))
        views[0]['scores']['distortion'] = True
        views[1]['scores']['connectivity'] = 6
        result = aggregate(inv, views, cameras=CAMERAS)
        self.assertEqual(result['successful_views'], len(views) - 2)
        self.assertEqual(result['cohorts']['primary']['evaluated_prompts'], 0)

    def test_duplicate_view_and_changed_source_rejected(self):
        inv, views = fixture()
        with self.assertRaises(ValueError):
            aggregate(inv, views + views[:1], cameras=CAMERAS)
        views[0]['image_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            aggregate(inv, views, cameras=CAMERAS)

    def test_mixed_snapshots_are_not_combined(self):
        inv, views = fixture()
        for row in views:
            if row['method_id'] == 'rgb':
                row['resolved_model'] = 'other-snapshot'
        result = aggregate(inv, views, cameras=CAMERAS)
        self.assertEqual(result['cohorts']['primary']['status'], 'mixed_models')
        self.assertTrue(all(r['image_quality'] is None for r in result['summaries']))

    def test_alpha_cohort_does_not_change_primary(self):
        inv, views = fixture(methods=('sphere20', 'rgb', 'select', 'meanmax050', 'meanmax025'))
        inv['cohorts']['primary']['method_ids'].remove('meanmax025')
        inv['cohorts']['alpha025'] = dict(method_ids=['meanmax050', 'meanmax025'], prompt_ids=['b'])
        result = aggregate(inv, views, cameras=CAMERAS)
        self.assertEqual(result['cohorts']['primary']['evaluated_prompts'], 2)
        self.assertEqual(result['cohorts']['alpha025']['evaluated_prompts'], 1)
        self.assertEqual({r['n_prompts'] for r in result['alpha_ablation'] if r['cohort'] == 'alpha025'}, {1})

    def test_bootstrap_uses_two_prompt_units_and_seed(self):
        a = paired_bootstrap([-2, 2], draws=1000, seed=17)
        self.assertEqual(a, paired_bootstrap([-2, 2], draws=1000, seed=17))
        self.assertEqual(a, (-2., 2.))
        self.assertEqual(paired_bootstrap([1]), (None, None))

    def test_diagnostics_reuse_exact_cameras_and_tied_ranks(self):
        inv, views = fixture()
        result = aggregate(inv, views, cameras=CAMERAS)
        seam = [r for r in result['diagnostics'] if r['group'] == 'seam_centered']
        self.assertTrue(all(r['expected_views'] == 3 for r in seam))
        polar = [r for r in result['diagnostics'] if r['group'] == 'polar_combined']
        self.assertTrue(all(r['expected_views'] == 2 for r in polar))
        ranks = [r['rank'] for r in result['rankings'] if r['prompt_id'] == 'a' and r['criterion'] == 'distortion']
        self.assertEqual(sorted(ranks), [1, 2, 2, 2])


if __name__ == '__main__':
    unittest.main()
