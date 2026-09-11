#!/usr/bin/env python3
"""Render three-way endpoints and numerical one-step errors from saved JSON."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.metadata:
        data = json.loads(path.read_text())
        backend = data['config']['model']['pipeline']
        steps = data['steps']
        if not all(v for k,v in data['fairness'].items() if k != 'conditioning_sha256'):
            raise ValueError('Failed fairness controls: '+str(path))
        if any(n != len(steps) for n in data['model_evaluations'].values()):
            raise ValueError('Unequal predictions: '+str(path))
        names = ('native', 'fixed_initial_noise', 'model_implied_endpoint')
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
        for ax, name in zip(axes, names):
            ax.imshow(plt.imread(path.parent/data['images'][name]))
            ax.set_title(name.replace('_', ' ')); ax.axis('off')
        fig.suptitle(backend + ': identical initial Gaussian and conditioning')
        fig.tight_layout()
        for suffix in ('png', 'pdf'): fig.savefig(args.output/(backend+'-threeway.'+suffix), dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        times = [s['timestep'] for s in steps]
        for ax, measure in zip(axes, ('mae', 'max_abs')):
            for name in ('implied', 'fixed'):
                ax.plot(times, [s['native_vs_'+name+'_'+measure] for s in steps], label='native vs '+name)
            ax.set_yscale('symlog', linthresh=1e-9)
            ax.set(xlabel='Actual scheduler timestep (denoising runs left to right)', ylabel=measure+' (native units)')
            ax.invert_xaxis(); ax.legend(); ax.grid(True, alpha=.3)
        fig.suptitle(backend+': one shared prediction along the native trajectory')
        fig.tight_layout()
        for suffix in ('png', 'pdf'): fig.savefig(args.output/(backend+'-one-step.'+suffix), dpi=160)
        plt.close(fig)
        implied = sum(s['native_vs_implied_mae'] for s in steps)/len(steps)
        fixed = sum(s['native_vs_fixed_mae'] for s in steps)/len(steps)
        rows.append(dict(backend=backend, native_implied_mae=implied, native_fixed_mae=fixed,
                         ratio=fixed/max(implied,1e-12), ratio_epsilon=1e-12,
                         max_native_implied_abs=max(s['native_vs_implied_max_abs'] for s in steps),
                         final_rgb_native_implied_mae=steps[-1]['final_rgb_native_vs_implied_mae'],
                         runtime_seconds=data['runtime_seconds'], peak_gpu_allocated_gib=data['peak_gpu_allocated_gib'],
                         job=data['slurm_job_id'], node=data['node'], metadata=str(path)))
    with (args.output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
