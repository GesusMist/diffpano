#!/usr/bin/env python3
"""Print or execute selected controls sequentially; no automatic Slurm submission."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('sd2', 'sana', 'flux', 'pixeldit')
EXPERIMENTS = ('native', 'trajectory', 'rgb', 'x0', 'noise', 'fusion', 'paired', 'erp_later')


def selected_configs(backend, experiment):
    if backend not in BACKENDS or experiment not in EXPERIMENTS:
        raise ValueError('Unknown backend or experiment')
    prefix = Path('configs/experiments')
    if experiment == 'native':
        return [prefix / 'native_multidiffusion' / f'{backend}.yaml']
    if experiment == 'trajectory':
        return [prefix / 'trajectory' / f'{backend}.yaml']
    if experiment == 'rgb':
        return [prefix / 'representation' / f'{backend}-rgb.yaml']
    if experiment == 'x0':
        return [prefix / 'planar_x0_noise' / f'{backend}-camera_index.yaml']
    if experiment == 'noise':
        return [prefix / 'planar_x0_noise' / f'{backend}-{binding}.yaml'
                for binding in ('camera_index', 'global_native_canvas')]
    if experiment == 'fusion':
        return [prefix / 'planar_fusion' / f'{backend}-{label}.yaml' for label in 'abcd']
    if experiment == 'paired':
        return [prefix / 'planar_erp_pairs' / f'{backend}-{method}-{canvas}.yaml'
                for method in ('rgb', 'x0') for canvas in ('planar', 'erp')]
    return [prefix / 'erp_later' / f'{backend}-{label}.yaml' for label in (
        'standard_average', 'standard_weighted', 'standard_dpa', 'lpw_dpa', 'lpw_dpa_dynamic')]


def commands(backends, experiments, python=sys.executable):
    result = []
    for experiment in experiments:
        for backend in backends:
            for config in selected_configs(backend, experiment):
                if not (ROOT / config).is_file():
                    raise FileNotFoundError(config)
                module = 'scripts.single_patch_trajectory' if experiment == 'trajectory' else 'scripts.generate'
                result.append([python, '-m', module, '--config', str(config)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='sd2', help='Comma-separated backends, or all')
    parser.add_argument('--experiment', default='native', help='Comma-separated: ' + ', '.join(EXPERIMENTS))
    parser.add_argument('--dry-run', action='store_true', help='Only print commands; never load models')
    args = parser.parse_args()
    backends = list(BACKENDS) if args.backend == 'all' else args.backend.split(',')
    experiments = args.experiment.split(',')
    if any(b not in BACKENDS for b in backends) or any(e not in EXPERIMENTS for e in experiments):
        parser.error('Unknown backend or experiment')
    for command in commands(backends, experiments):
        print(' '.join(shlex.quote(part) for part in command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
