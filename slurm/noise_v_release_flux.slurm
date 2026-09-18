#!/bin/bash
#SBATCH --job-name=noise-v-pixel-validity
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
cd /home/shig/diffpano
python3 - <<'PY_GATE'
import hashlib,json,math,os
from pathlib import Path
r=Path('/home/shig/diffpano/outputs/vae-residual-controls/20260917-noise-v');sroot=r.parent/'20260916-diagnostics-pt/S/pixeldit'
read=lambda p:json.loads(p.read_text())
s=read(sroot/'metadata.json');pre=read(r/'initialization-preflight.json')['backends']['pixeldit'];primary=next(q for q in pre['records'] if q['scale']=='primary')
methods=['V-independent-erp','V-shared-erp'];m=[read(r/'pixeldit'/v/'metadata.json') for v in methods]
execution=read(r/'execution.json');validation=execution['validation']
assert validation['passed'] and all(hashlib.sha256(Path('/home/shig/diffpano',p).read_bytes()).hexdigest()==h for p,h in validation['source_hashes'].items())
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
for name,x,draws in zip(methods,m,[len(pre['camera_poses']),1]):
 assert x['experiment']=='V' and x['variant']==name
 assert x['config']['sampler']==s['config'] and x['sampler_config_diff_from_S']=={}
 for key in ['camera_sha256','conditioning_sha256','model_checkpoint','model_revision','weights','prepared_schedule']:
  assert x[key]==s[key],key
 assert x['direct_S_initializer_sha256']==s['audit']['initial_local_sha256']
 assert x['audit']['guided_predictions']==s['audit']['guided_predictions']==4450
 assert x['actual_transformer_forward_invocations']==s['actual_transformer_forward_invocations']==4450
 assert x['source_hashes']==validation['source_hashes']
 init=x['initialization'];assert init['source_draws']==draws and init['source_released'] and init['vae_calls']==0
 assert init['map_sha256']==primary['maps_sha256'] and init['source_shape'][-2:]==pre['primary_noise_size']
 assert init['scaling']==pre['native_scale'] and init['scaling_applications']==1
 assert not init['empirical_normalization'] and not init['clipping'] and not init['fixed_noise_renoising'] and not init['persistent_erp_native_state']
 assert x['audit']['stage_audit']['diagnostic_vae_decodes']==0 and x['audit']['extra_denoiser_calls']==0
 finite(x['final_metrics']);finite(x['milestone_agreement'])
for key in ['first_camera_sha256','map_sha256','native_channels','native_local_shapes','scaling','source_shape']:
 assert m[0]['initialization'][key]==m[1]['initialization'][key],key
assert m[0]['initialization']['initial_local_sha256']!=m[1]['initialization']['initial_local_sha256']
assert (r/'pixeldit-comparison.png').is_file() and 'pixeldit' in read(r/'results.json')['backends']
execution['pixeldit_reviewed']=True
execution['pixeldit_review_kind']='Automated numerical validity and pair audit; visual interpretation remains pending and is not a gate on FLUX.'
execution['pixeldit_numerical_review']=dict(passed=True,job=os.environ.get('SLURM_JOB_ID'),methods=methods,first_camera_identical=True,maps_identical=True,scaling_identical=True,source_draws=[89,1],finite=True,guided_predictions=[4450,4450],sampler_unchanged=True)
execution['status']='PixelDiT numerical validity passed; FLUX dependency released; visual reviews pending'
(r/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
print(json.dumps(execution['pixeldit_numerical_review'],indent=2))

PY_GATE
