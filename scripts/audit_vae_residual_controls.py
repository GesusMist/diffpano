#!/usr/bin/env python3
"""Audit saved training-free controls without loading models or invoking CUDA."""
import argparse
import json
import math
import struct
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('specs', nargs='+')
    args = parser.parse_args()
    rows = []
    required = ('pre_fusion_rgb_overlap_mae_mean', 'pre_fusion_rgb_overlap_mae_max',
                'local_vae_residual_mae_mean', 'local_vae_residual_mae_max',
                'fused_residual_mae_mean', 'fused_residual_std',
                'residual_correction_magnitude', 'consensus_correction_mae_mean')
    for spec_path in args.specs:
        spec = read(spec_path)
        root, previous = Path(spec['output']), Path(spec['previous_pair'])
        cd, g, old = read(root/'CD/comparison.json'), read(root/'G/comparison.json'), read(previous/'comparison.json')
        assert cd['training_free'] and g['training_free']
        assert cd['same_initial_model_settings_schedule'] and g['same_initial_model_settings_schedule']
        assert cd['config'] == g['config'] == old['config']
        assert g['initial_native_sha256'] == old['initial_native_sha256']
        assert cd['epsilon_sha256'] == read(previous/'one_patch_roundtrip/control.json')['epsilon_sha256']
        assert cd['conditioning_sha256'] == g['conditioning_sha256'] == old['fairness']['conditioning_sha256']
        assert cd['extra_diagnostic_denoiser_calls'] == g['extra_diagnostic_denoiser_calls'] == 0
        gm, fm = read(root/'G/generation/metadata.json'), read(previous/'implied_endpoint_consensus/metadata.json')
        assert gm['scheduler_timesteps'] == fm['scheduler_timesteps']
        assert gm['scheduler_sigmas'] == fm['scheduler_sigmas']
        steps = len(gm['steps'])
        assert set(cd['counts'].values()) == {steps}
        assert g['guided_predictions'] == old['model_evaluations']['implied_endpoint_consensus'] == 3*steps
        assert g['audit']['training_free_residual_correction'] and not g['audit']['bridge_enabled']
        assert not g['audit']['global_native_state_persisted']
        assert g['audit']['synchronous']
        assert finite(gm['steps'])
        for a, b in zip(gm['steps'], fm['steps']):
            assert a['scheduler_timestep'] == b['scheduler_timestep']
            assert a['coverage_percent'] == 100 and a['num_patches'] == 3
            for key in ('alpha', 'sigma', 'next_alpha', 'next_sigma'):
                assert a['state_statistics'][key] == b['state_statistics'][key]
            assert a['state_statistics']['guided_predictions'] == 3
            assert all(key in a['state_statistics'] for key in required)
        for phase in ('CD', 'G'):
            preflight = read(root/phase/'real_vae_identity_preflight.json')
            assert len(preflight) == 2 and finite(preflight)
            assert max(row['correction_recovery_max_abs'] for row in preflight) <= 2e-6
        d = read(root/'CD/D_steps.json')
        assert finite(d) and finite(read(root/'CD/C_steps.json'))
        assert max(row['correction_recovery_max_abs'] for row in d) <= 2e-6
        dimensions = {}
        for label, path in [('B',root/'CD/B_final.png'), ('C',root/'CD/C_final.png'),
                            ('D',root/'CD/D_final.png'), ('G',root/'G/generation/result.png')]:
            data = path.read_bytes()
            assert data[:8] == b'\x89PNG\r\n\x1a\n'
            dimensions[label] = list(struct.unpack('>II', data[16:24]))
        assert dimensions['B'] == dimensions['C'] == dimensions['D']
        assert dimensions['G'] == [old['geometry']['rgb']['canvas_width'], old['geometry']['rgb']['canvas_height']]
        rows.append(dict(backend=spec['backend'], passed=True, steps=steps,
                         G_guided_predictions=g['guided_predictions'], dimensions=dimensions,
                         recovery_max_abs=max(row['correction_recovery_max_abs'] for row in d)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(passed=True, backends=rows), indent=2)+'\n')
    print(output.read_text())


if __name__ == '__main__':
    main()
