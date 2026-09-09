import unittest
from unittest.mock import patch

from diffpano.config import load_experiment_config
from scripts.run_ablation_matrix import BACKENDS, EXPERIMENTS, ROOT, commands, main, selected_configs


class AblationMatrixTests(unittest.TestCase):
    def test_every_selected_config_validates(self):
        for backend in BACKENDS:
            for experiment in EXPERIMENTS:
                for path in selected_configs(backend, experiment):
                    config = load_experiment_config(str(ROOT / path))
                    self.assertEqual(config.model.pipeline, backend)
        self.assertEqual(len(commands(BACKENDS, ['native','rgb','x0'])), 12)

    def test_pairs_keep_model_prompt_sampling_budget_fusion_and_resolution(self):
        for backend in BACKENDS:
            for method in ('rgb','x0'):
                paths = [ROOT/'configs/experiments/planar_erp_pairs'/f'{backend}-{method}-{canvas}.yaml'
                         for canvas in ('planar','erp')]
                a, b = [load_experiment_config(str(path)) for path in paths]
                for key in ('model','pixeldit','prompt','generation','fusion','view','global_pipeline'):
                    self.assertEqual(a.to_dict()[key], b.to_dict()[key])
                self.assertEqual(a.experiment.seed, b.experiment.seed)
                self.assertEqual((a.planar.height,a.planar.width), (b.erp.height,b.erp.width))
                self.assertEqual(a.planar.patch_size,b.view.height)

    def test_fusion_presets_change_only_name_and_fusion(self):
        for backend in BACKENDS:
            configs = [load_experiment_config(str(ROOT/'configs/experiments/planar_fusion'/f'{backend}-{label}.yaml')).to_dict()
                       for label in 'abcd']
            controls = []
            for config in configs:
                config.pop('fusion')
                config['experiment'].pop('name')
                controls.append(config)
            self.assertTrue(all(value == controls[0] for value in controls[1:]))

    def test_dry_run_has_no_subprocess_or_model_loading(self):
        with patch('sys.argv', ['matrix','--backend','sana,flux','--experiment','trajectory','--dry-run']), \
                patch('scripts.run_ablation_matrix.subprocess.run') as execute, patch('builtins.print') as output:
            main()
        execute.assert_not_called()
        self.assertEqual(output.call_count, 2)
        self.assertTrue(all('single_patch_trajectory' in call.args[0] for call in output.call_args_list))


if __name__ == '__main__':
    unittest.main()
