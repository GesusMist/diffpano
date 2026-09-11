#!/usr/bin/env python3
"""Audit saved A-G fairness, identities, schedules and finite metrics without CUDA."""
import argparse
import json
import math
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def finite(value):
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',default='configs/experiments/trajectory/sd35-ladder.json')
    args=parser.parse_args();protocol=read(args.protocol)
    controls={label:read(Path(folder)/(label+'_control.json')) for label,folder in protocol['folders'].items()}
    reference=controls['A']['shared'];count=len(reference['timesteps'])
    for label,control in controls.items():
        assert control['experiment']==label and control['backend']==protocol['backend']
        assert control['shared']==reference and control['training_free']
        assert control['extra_denoiser_calls']==0
        assert control['guided_predictions']==count*(3 if label in 'EFG' else 1)
        assert control['correction']==(label in 'DG')
        assert control['gpu']==protocol['gpu'] and finite(control)
        assert Path(control['output']).is_file()
        if label in 'ABCD':assert control['initial_epsilon_sha256']==controls['A']['initial_epsilon_sha256']
        else:
            assert control['initial_native_sha256']==controls['E']['initial_native_sha256']
            assert control['geometry']==controls['E']['geometry']
            metadata=read(Path(protocol['folders'][label])/'metadata.json')
            assert metadata['scheduler_timesteps']==reference['timesteps']
            assert metadata['scheduler_sigmas']==reference['sigmas']
            assert len(metadata['steps'])==count and finite(metadata['steps'])
            for step in metadata['steps']:
                assert step['coverage_percent']==100 and step['num_patches']==3
                assert 'pre_fusion_rgb_overlap_mae_mean' in step['state_statistics']
            if label in 'FG':
                assert control['local_initial_crops_bit_identical']
                audit=control['audit']
                assert audit['synchronous'] and not audit['global_native_state_persisted']
                assert not audit['bridge_enabled']
                assert audit['training_free_residual_correction']==(label=='G')
    d=read(Path(protocol['folders']['D'])/'D_steps.json')
    assert finite(d) and len(d)==count
    recovery=max(row['corrected_vs_original_max_abs'] for row in d)
    assert recovery<2e-6
    for label in 'CD':
        preflight=read(Path(protocol['folders'][label])/(label+'_real_vae_identity_preflight.json'))
        assert len(preflight)==2 and finite(preflight)
        assert max(row['correction_recovery_max_abs'] for row in preflight)<2e-6
    gpreflight=read(Path(protocol['folders']['G']).parent/'real_vae_identity_preflight.json')
    assert finite(gpreflight) and max(row['correction_recovery_max_abs'] for row in gpreflight)<2e-6
    a=read(Path(protocol['folders']['A'])/'A_steps.json')
    assert len(a)==count and finite(a)
    result=dict(passed=True,backend=protocol['backend'],phases=list(controls),
        counts={label:value['guided_predictions'] for label,value in controls.items()},
        total_guided_predictions=sum(c['guided_predictions'] for c in controls.values()),
        same_settings_conditioning_schedule=True,same_single_initialization=True,
        same_multi_initialization=True,D_recovery_max_abs=recovery,
        native_one_step_max_abs=max(row['native_vs_implied_max_abs'] for row in a))
    out=Path(protocol['report']);out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
