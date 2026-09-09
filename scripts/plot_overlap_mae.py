#!/usr/bin/env python3
"""Plot recorded pre-fusion overlap MAE, with separate panels for each run."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(args.metadata), 1, figsize=(9, 3.2 * len(args.metadata)), squeeze=False)
    summary = []
    for ax, path in zip(axes[:, 0], args.metadata):
        data = json.loads(path.read_text())
        steps = data['steps']
        mean = [s['state_statistics']['overlap_mae_mean'] for s in steps]
        maximum = [s['state_statistics']['overlap_mae_max'] for s in steps]
        index = [s['step_index'] for s in steps]
        ax.plot(index, mean, label='Mean pair MAE')
        ax.plot(index, maximum, label='Maximum pair MAE', linestyle='--')
        units = 'native state' if data['global_pipeline_mode'] == 'native_multidiffusion' else 'decoded RGB'
        ax.set(title=data['experiment']['name'], xlabel='Denoising step index', ylabel=f'MAE ({units} units)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        summary.append({'metadata': str(path), 'units': units, 'first_mean': mean[0], 'last_mean': mean[-1], 'first_max': maximum[0], 'last_max': maximum[-1]})
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.with_suffix('.pdf'))
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
