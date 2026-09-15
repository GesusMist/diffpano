#!/usr/bin/env python3
"""Audit all four I artifacts and render paired G/I comparisons without models."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

module_spec = importlib.util.spec_from_file_location('seams', Path(__file__).resolve().parents[1]/'diffpano/seams.py')
seams = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(seams)


def read(path):
    return json.loads(Path(path).read_text())


def save(fig, path):
    for extension in ('png', 'pdf'):
        fig.savefig(str(path)+'.'+extension, dpi=160)
    plt.close(fig)


def main():
    specs = read('configs/experiments/vae_residual/i-all-models.json')
    out = Path('outputs/vae-residual-controls/report/I')
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    audit = {}
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), layout='constrained')
    plots, curves = plt.subplots(4, 3, figsize=(15, 13), layout='constrained')
    transitions, transition_axes = plt.subplots(4,4,figsize=(18,13),layout='constrained')
    for row, (backend, spec) in enumerate(specs.items()):
        folder = Path(spec['output'])
        h = read(folder/'comparison.json')
        g = read(spec['reference'])
        metadata = [read(Path(spec['reference_generation'])/'metadata.json'), read(folder/'generation/metadata.json')]
        restored = copy.deepcopy(h['config'])
        assert restored.pop('consensus_transition') == {'mode': 'preserve_current_state'}
        assert restored['fusion']['mode'] == 'average'
        assert restored == g['config']
        assert h['initial_native_sha256'] == g['initial_native_sha256']
        expected_condition = (g['fairness'] if backend == 'pixeldit' else g['shared'] if backend == 'sd35' else g)['conditioning_sha256']
        assert h['conditioning_sha256'] == expected_condition
        assert h['gpu'] == 'NVIDIA A40'
        for key in h['runtime_fields_checked']:
            assert metadata[0].get(key) == metadata[1].get(key), (backend, key)
        for key in ('python', 'torch', 'cuda'):
            assert metadata[0]['environment'][key] == metadata[1]['environment'][key]
        a = h['audit']
        assert a['synchronous'] and not a['bridge_enabled'] and not a['global_native_state_persisted']
        assert a['initial_local_states_equal_global_native_crops']
        assert a['training_free_residual_correction'] == (backend != 'pixeldit')
        assert a['residual_fusion'] == (None if backend == 'pixeldit' else 'temporary uniform native-coordinate canvas')
        assert a['rgb_fusion'] == h['config']['fusion']
        assert a['consensus_transition'] == 'preserve_current_state'
        patch_records = read(folder/'transition_patches.json')
        assert len(patch_records) == h['guided_predictions']
        diagnostic_keys = ['clean_consensus_delta_mae','g_current_state_mismatch',
                           'i_current_state_mismatch','next_state_G_vs_I_mae']
        for step in metadata[1]['steps']:
            values = [r for r in patch_records if r['step'] == step['step_index']]
            assert len(values) == 3
            for key in diagnostic_keys:
                assert np.isclose(np.mean([r[key] for r in values]),step['state_statistics'][key+'_mean'])
                assert max(r[key] for r in values) == step['state_statistics'][key+'_max']
        assert max(r['i_current_state_error_max_abs'] for r in patch_records) < 1e-4
        assert max(r['next_state_identity_error_max_abs'] for r in patch_records) < 1e-4
        expected = len(metadata[0]['steps'])*len(h['geometry']['rgb']['patches'])
        assert h['guided_predictions'] == a['guided_predictions'] == a['expected_guided_predictions'] == expected
        assert h['extra_diagnostic_denoiser_calls'] == a['diagnostic_extra_denoiser_evaluations'] == 0
        assert h['training_free']
        if backend != 'pixeldit':
            assert all(r['correction_recovery_max_abs'] <= 2e-6 for r in read(folder/'real_vae_identity_preflight.json'))
        with (folder/'generation/steps.csv').open() as stream:
            assert len(list(csv.DictReader(stream))) == len(metadata[0]['steps'])
        for gs, hs in zip(metadata[0]['steps'], metadata[1]['steps']):
            assert gs['scheduler_timestep'] == hs['scheduler_timestep']
            assert hs['num_patches'] == 3 and hs['coverage_percent'] == 100.
            assert hs['weight_min'] == 1. and hs['weight_max'] == 2.
            for key in ('alpha', 'sigma', 'next_alpha', 'next_sigma'):
                assert gs['state_statistics'][key] == hs['state_statistics'][key]
            assert all(np.isfinite(v) for v in hs['state_statistics'].values())
        paths = [Path(spec['reference_generation'])/'result.png', folder/'generation/result.png']
        images = [np.asarray(Image.open(p).convert('RGB'))/255. for p in paths]
        assert images[0].shape == images[1].shape
        labels = [spec['baseline_label']+' / average', 'I / preserve current state']
        pair, pair_axes = plt.subplots(1, 2, figsize=(14, 4.2), layout='constrained')
        crops, crop_axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
        for col, (image, label) in enumerate(zip(images, labels)):
            for ax in (axes[row,col], pair_axes[col]):
                ax.imshow(image); ax.set_title(backend.upper()+' — '+label); ax.axis('off')
            width = image.shape[1]
            crop_axes[col].imshow(image[:,width//4:3*width//4])
            crop_axes[col].set_title(backend.upper()+' — '+label+'\nCentral overlap regions'); crop_axes[col].axis('off')
        save(pair, out/(backend+'-G-I'))
        save(crops, out/(backend+'-overlap-crops'))
        record = dict(control=h, images=[str(p) for p in paths], G={}, I={})
        for label, image, m in zip(('G', 'I'), images, metadata):
            record[label]['seams'] = seams.boundary_gradient_metrics(image, h['geometry']['rgb']['patches'])
            steps = m['steps']
            keys = ['pre_fusion_rgb_overlap_mae_mean','consensus_correction_mae_mean']
            if backend != 'pixeldit':
                keys += ['local_vae_residual_mae_mean','fused_residual_mae_mean','residual_correction_magnitude']
            record[label]['diagnostic_means'] = {key:float(np.mean([s['state_statistics'][key] for s in steps])) for key in keys}
            record[label]['runtime_seconds'] = m['runtime_seconds']
            record[label]['peak_gpu_memory_gib'] = m['peak_gpu_memory_gib']
            for col, key in enumerate(keys[:2]+(['fused_residual_mae_mean'] if backend != 'pixeldit' else [])):
                curves[row,col].plot([s['step_index'] for s in steps], [s['state_statistics'][key] for s in steps], label=label)
                curves[row,col].set_title(backend.upper()+' / '+key.replace('_mae_mean','').replace('_',' '), fontsize=9)
                curves[row,col].set_xlabel('Step');curves[row,col].legend()
        if backend == 'pixeldit':
            curves[row,2].axis('off');curves[row,2].text(.1,.5,'Pixel native: no VAE residual')
        record['transition_diagnostics'] = {key:dict(mean=float(np.mean([r[key] for r in patch_records])),
            max=max(r[key] for r in patch_records)) for key in patch_records[0] if key not in ('step','timestep','patch')}
        for col,key in enumerate(diagnostic_keys):
            ax=transition_axes[row,col]
            steps=metadata[1]['steps']
            ax.plot([s['step_index'] for s in steps],[s['state_statistics'][key+'_mean'] for s in steps],label='patch mean')
            ax.plot([s['step_index'] for s in steps],[s['state_statistics'][key+'_max'] for s in steps],label='patch max')
            ax.set_title(backend.upper()+' / '+key.replace('_',' '),fontsize=8)
            ax.set_xlabel('Step');ax.legend(fontsize=8)
        summary[backend] = record
        audit[backend] = dict(passed=True, job=h['job'], guided_predictions=expected, extra_denoiser_calls=0,
            only_setting_changed=h['only_setting_changed'], baseline=spec['baseline_label'])
    save(fig, out/'G-I-all-models')
    save(plots, out/'G-I-diagnostics')
    save(transitions, out/'I-transition-diagnostics')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    lines = ['| Model | G seam ratio | I seam ratio | G boundary excess | I boundary excess | I overlap MAE | I seconds | I peak allocated GiB | Job |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name, record in summary.items():
        g,h = record['G'], record['I']
        lines.append('| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.3f} | {:.3f} | {} |'.format(name,
            g['seams']['boundary_to_nearby_ratio'],h['seams']['boundary_to_nearby_ratio'],
            g['seams']['boundary_excess'],h['seams']['boundary_excess'],
            h['diagnostic_means']['pre_fusion_rgb_overlap_mae_mean'],h['runtime_seconds'],
            h['peak_gpu_memory_gib']['allocated_gib'],record['control']['job']))
    (out/'metrics.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print('All four I audits passed; guided predictions:',sum(v['guided_predictions'] for v in audit.values()))


if __name__ == '__main__':
    main()
