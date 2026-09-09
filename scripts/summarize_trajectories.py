#!/usr/bin/env python3
"""Validate saved A/B controls and render endpoint comparisons and a CSV summary."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectories', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.trajectories:
        data = json.loads(path.read_text())
        steps = data['steps']
        count = len(steps)
        assert data['same_initial_epsilon'] and all(data['fairness'].values()), path
        assert data['model_evaluations'] == {'native': count, 'x0_renoise': count}, path
        assert data['scheduler_timesteps'] == [s['scheduler_timestep'] for s in steps], path
        assert count == data['config']['generation']['num_inference_steps'], path
        backend = data['config']['model']['pipeline']
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
        for ax, name, title in zip(axes, ('native_final.png', 'x0_renoise_final.png'), ('Ordinary native', 'Fixed-epsilon x0-renoise')):
            ax.imshow(plt.imread(path.parent / name))
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'{backend}: shared epsilon, conditioning, schedule; {count} guided predictions each')
        fig.tight_layout()
        fig.savefig(args.output / f'{backend}-trajectory.png', dpi=160)
        fig.savefig(args.output / f'{backend}-trajectory.pdf')
        plt.close(fig)
        rows.append(dict(backend=backend, job=data['slurm_job_id'], node=data['node'], guided_predictions_each=count,
                         final_rgb_L1=steps[-1]['final_rgb_L1'], final_rgb_RMSE=steps[-1]['final_rgb_RMSE'],
                         runtime_seconds=data['runtime_seconds'], peak_gpu_allocated_gib=data['peak_gpu_allocated_gib'],
                         epsilon_sha256=data['epsilon_sha256'], trajectory=str(path)))
    with (args.output / 'trajectory-summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
