"""Audited original loader; only its suite gate is adapted, not the algorithm."""
from unittest.mock import patch
from studies.layout24.common import *


def generate(row,directory):
    from studies.ournew_comparison.official import original
    evidence=read(OUT/'manifest.json')['official_preflight']
    with patch('studies.ournew_comparison.official.preflight_gate',return_value=evidence):
        image,record=original(row,prompt_path(row['prompt']))
    assert record['configuration']==row['semantic']['configuration'] and record['schedule']==row['semantic']['schedule']
    temporary=directory/('final.tmp-'+os.environ['SLURM_JOB_ID']+'.png');image.save(temporary);os.replace(str(temporary),str(directory/'final.png'))
    write(directory/'config.json',safe(record))
    return dict(prompt=prompt(row['prompt'])[0],configuration=record['configuration'],official=record,counts=record['execution']['counts'],
        generation_seconds=record['execution']['generation_seconds'],peak_allocated_gib=record['execution']['peak_gpu_allocated_gib'],
        label='Original SphereDiff; established matched 20-step algorithm',
        unclamped_output='Official PIL output interface exposes no raw final tensor; no reconstructed raw artifact',
        artifacts={n:sha(directory/n) for n in ('final.png','config.json')})
