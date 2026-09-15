"""Audit K and compare full uncropped planar J images with terminal K ERPs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.plot_controlled_ladder import plt, save


def read(path):
    return json.loads(Path(path).read_text())


def rgb(path):
    image = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.
    assert np.isfinite(image).all()
    return image


def wrap_metrics(image):
    boundary = float(np.abs(image[:, 0] - image[:, -1]).mean())
    differences = np.abs(np.diff(image, axis=1)).mean(axis=(0, 2))
    nearby = float(np.concatenate((differences[:8], differences[-8:])).mean())
    return dict(boundary_gradient=boundary, nearby_gradient=nearby,
                boundary_excess=boundary-nearby, boundary_to_nearby_ratio=boundary/max(nearby, 1e-12),
                nearby_radius=8, rgb_range='displayed PNG RGB [0,1]')


def main():
    specs = read('configs/experiments/erp_later/k-all-models.json')
    out = Path('outputs/vae-residual-controls/report/K')
    out.mkdir(parents=True, exist_ok=True)
    summary, audits = {}, {}
    overview, axes = plt.subplots(4, 2, figsize=(14, 15), layout='constrained')
    for row, (backend, spec) in enumerate(specs.items()):
        folder = Path(spec['output'])
        c, m = read(folder/'comparison.json'), read(folder/'metadata.json')
        j = read(Path(spec['j_output'])/'comparison.json')
        pre = read(out/'geometry'/(backend+'.json'))
        a = c['audit']
        assert c['experiment'] == 'K' and a['synchronous']
        assert not a['vae_residual_correction'] and not a['fixed_initial_noise_renoising']
        assert not a['global_native_state_persisted'] and a['extra_denoiser_calls'] == 0
        assert a['warp'] == 'standard' and a['fusion'] == 'average/uniform'
        assert a['consensus_transition'] == 'preserve_current_state'
        assert c['camera_sha256'] == pre['camera_sha256'] == a['camera_sha256']
        assert all(h == c['camera_sha256'] for h in a['camera_hashes_per_step'])
        assert c['cameras'] == pre['cameras'] and c['coverage'] == pre['coverage']
        assert c['coverage']['coverage_percent'] == 100
        expected = len(m['steps'])*len(c['cameras'])
        assert a['guided_predictions'] == a['expected_guided_predictions'] == expected
        assert len(a['camera_hashes_per_step']) == len(m['steps'])
        assert a['conditioning_sha256'] == j['conditioning_sha256']
        initial = read(folder/'initialization.json')
        assert initial['sha256_by_camera'] == c['initial_local_sha256']
        assert len(set(initial['sha256_by_camera'])) == len(c['cameras'])
        assert not initial['erp_latent_field'] and not initial['j_k_bit_identical_initialization']
        config, jc = dict(c['config']), dict(j['config'])
        for key in ('canvas', 'global_pipeline', 'view', 'erp', 'sampling'):
            config.pop(key); jc.pop(key)
        assert config == jc
        jm = read(Path(spec['j_output'])/'generation/metadata.json')
        for key in c['runtime_fields_checked']:
            assert m.get(key) == jm.get(key)
        records = read(folder/'transition_patches.json')
        assert len(records) == expected
        max_error = max(r['i_current_state_error_max_abs'] for r in records)
        assert max_error < 1e-4
        images = [rgb(Path(spec['j_output'])/'generation/result.png'), rgb(folder/'final_erp.png')]
        pair, pax = plt.subplots(1, 2, figsize=(14, 4), layout='constrained')
        for col, (label, image) in enumerate(zip(('J planar', 'K ERP standard'), images)):
            for ax in (axes[row, col], pax[col]):
                ax.imshow(image); ax.axis('off'); ax.set_title(backend.upper()+' — '+label)
        save(pair, out/(backend+'-J-K'))
        views, vax = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
        for i, ax in enumerate(vax.flat):
            ax.imshow(rgb(folder/('final_view_%02d.png'%i))); ax.axis('off')
            ax.set_title('slot %d: yaw %.0f°, pitch %.0f°'%(i, np.rad2deg(c['cameras'][i]['yaw']), np.rad2deg(c['cameras'][i]['pitch'])))
        save(views, out/(backend+'-final-views'))
        steps = m['steps']; x = np.arange(len(steps))
        fig, dax = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
        keys = ['pre_fusion_rgb_overlap_mae_mean', 'rgb_consensus_delta_mae_mean',
                'latent_consensus_delta_mae_mean', 'i_current_state_error_max_abs_max', 'next_state_update_mae_mean']
        for ax, key in zip(dax.flat, keys):
            ax.plot(x, [s['state_statistics'][key] for s in steps]); ax.set_title(key, fontsize=9); ax.set_xlabel('step')
        for key in ('model', 'vae_decode', 'vae_encode', 'view_to_erp', 'erp_to_view'):
            dax.flat[-1].plot(x, [s['timings_seconds'][key] for s in steps], label=key)
        dax.flat[-1].set_title('stage seconds'); dax.flat[-1].legend(fontsize=8)
        save(fig, out/(backend+'-diagnostics'))
        overlaps = c['final_pairwise_overlap']
        summary[backend] = dict(path=str(folder/'final_erp.png'), job=c['job'], predictions=expected,
            timing=c['timing'], coverage=c['coverage'], camera_sha256=c['camera_sha256'],
            current_state_max_abs_error=max_error, wrap=wrap_metrics(images[1]),
            final_pairwise_overlap=overlaps,
            final_pairwise_overlap_mae_mean=float(np.mean([p['mae'] for p in overlaps])),
            final_pairwise_overlap_mae_max=max(p['mae'] for p in overlaps),
            diagnostics={k:dict(mean=float(np.mean([r[k] for r in records])), max=max(r[k] for r in records))
                         for k in ('rgb_consensus_delta_mae','latent_consensus_delta_mae','next_state_update_mae')},
            stage_seconds={k:sum(s['timings_seconds'][k] for s in steps) for k in steps[0]['timings_seconds']},
            terminal_vs_final_clean_rgb_mae=float(np.abs(images[1]-rgb(folder/'final_consensus_erp.png')).mean()))
        audits[backend] = dict(passed=True, job=c['job'], predictions=expected)
    save(overview, out/'J-K-all-models')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'audit.json').write_text(json.dumps(audits, indent=2)+'\n')
    print('K audit passed', sum(a['predictions'] for a in audits.values()), 'predictions')
    for b, r in summary.items():
        print(b, 'wrap ratio', r['wrap']['boundary_to_nearby_ratio'], 'overlap', r['final_pairwise_overlap_mae_mean'])


if __name__ == '__main__':
    main()
