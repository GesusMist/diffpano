"""Audit ten N/O runs; save only one FLUX N/O progression sheet."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('outputs/vae-residual-controls/20260916-dense-no')
BASE=Path('outputs/vae-residual-controls/20260915-dense-lm')
MODELS=('sd2','sana','flux','sd35','pixeldit')

def read(p):return json.loads(p.read_text())

def canonical(s):
    s=dict(s)
    if 'config' in s:
        s['config']=dict(s['config'])
        if '_use_default_values' in s['config']:s['config']['_use_default_values']=sorted(s['config']['_use_default_values'])
    return s

def measurements(path):
    im=Image.open(path).convert('RGB');rgb=np.asarray(im,dtype=np.float32)/255
    gray=im.convert('L');g=np.asarray(gray,dtype=np.float32)/255
    smooth=np.asarray(gray.filter(ImageFilter.GaussianBlur(1)),dtype=np.float32)/255
    return dict(luminance_std=float(g.std()),high_frequency_mae=float(np.abs(g-smooth).mean()),
        wrap_gradient=float(np.abs(rgb[:,0]-rgb[:,-1]).mean()))

def main():
    validation=read(ROOT/'validation.json')
    for path,digest in validation['baseline_artifact_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, 'L/M artifact changed: '+path
    summary={};calls=0
    for backend in MODELS:
        l=read(BASE/'L'/backend/'metadata.json');summary[backend]={}
        summary[backend]['L']=dict(runtime=l['runtime_seconds'],allocated_gib=l['peak_allocated_gib'],reserved_gib=l['peak_reserved_gib'],**measurements(BASE/'L'/backend/'final_result.png'))
        pair=[]
        for label in ('N','O'):
            folder=ROOT/label/backend;m=read(folder/'metadata.json');a=m['audit'];pair.append(m)
            expected=['consensus_%03d.png'%p for p in range(10,100,10)]+['final_result.png','metadata.json']
            assert sorted(p.name for p in folder.iterdir())==sorted(expected)
            assert m['experiment']==label and m['backend']==backend
            for key in ('camera_sha256','camera_count','prompt_sha256','prompt_assignment','prompt_slot_histogram','semantic_band_histogram','seed','steps','fov','erp_size','view_size','model_checkpoint','model_revision','guidance_scale','negative_prompt'):
                assert m[key]==l[key], (backend,label,key)
            assert canonical(m['scheduler'])==canonical(l['scheduler'])
            assert m['geometry_file']==l['geometry_file']
            for key in ('initial_local_sha256','conditioning_sha256'):assert a[key]==l['audit'][key]
            assert a['guided_predictions']==a['expected_guided_predictions']==m['camera_count']*m['steps']
            calls+=a['guided_predictions']
            assert a['extra_denoiser_calls']==0 and a['synchronous'] and a['cameras_fixed'] and a['conditioning_fixed']
            assert not any(a[k] for k in ('vae_residual','fixed_noise_renoising','spherical_latent','erp_latent'))
            assert a['transition']=='preserve_current_state'
            assert a['warp']==('standard' if label=='N' else 'lpw') and a['fusion']=='detail_preserving_average'
            assert m['periodic_pyramid_reconstruction']==(label=='O')
            assert m['coverage']['coverage_percent']==100 and m['coverage']['minimum']==l['coverage']['minimum']
            assert len(m['snapshots'])==9
            progression=[]
            for p,s in zip(range(10,100,10),m['snapshots']):
                assert s['requested_percentage']==p and s['requested_percentages']==[p]
                assert s['completed_step']==(p*m['steps']+99)//100
                assert s['scheduler_timestep']==m['scheduler']['timesteps'][s['completed_step']-1]
                assert s['filename']=='consensus_%03d.png'%p
                progression.append(dict(percentage=p,**measurements(folder/s['filename'])))
            for path in folder.glob('*.png'):
                with Image.open(path) as im:assert im.size==(m['erp_size'][1],m['erp_size'][0])
            if label=='O':assert {'pyramid_construction','pyramid_level_fusion','erp_reconstruction'}<=set(m['stage_seconds'])
            summary[backend][label]=dict(job=m['slurm_job_id'],runtime=m['runtime_seconds'],allocated_gib=m['peak_allocated_gib'],reserved_gib=m['peak_reserved_gib'],
                error=m['aggregate_metrics']['current_state_error_max']['max'],seam_ratio=m['seam']['ratio'],calls=a['guided_predictions'],
                rgb_delta=m['aggregate_metrics']['rgb_consensus_delta_mean']['mean'],
                native_delta=m['aggregate_metrics']['native_consensus_delta_mean']['mean'],
                progression=progression,**measurements(folder/'final_result.png'))
            assert summary[backend][label]['error']<1e-4
        assert pair[0]['prompt_indices']==pair[1]['prompt_indices']
    fig,axes=plt.subplots(2,6,figsize=(18,3.5),layout='constrained')
    for row,label in enumerate(('N','O')):
        for col,p in enumerate((10,30,50,70,90,100)):
            path=ROOT/label/'flux'/('final_result.png' if p==100 else 'consensus_%03d.png'%p)
            axes[row,col].imshow(Image.open(path));axes[row,col].axis('off')
            axes[row,col].set_title(label+' / '+('terminal final' if p==100 else str(p)+'% clean'))
    fig.suptitle('FLUX.1-dev: N standard + DPA; O five-level LPW + DPA. L has no snapshots.')
    fig.savefig(ROOT/'flux-progression.png',dpi=140);plt.close(fig)
    storage={e:dict(images=len(list((ROOT/e).rglob('*.png'))),metadata=len(list((ROOT/e).rglob('metadata.json'))),
        files=sum(p.is_file() for p in (ROOT/e).rglob('*')),bytes=sum(p.stat().st_size for p in (ROOT/e).rglob('*') if p.is_file())) for e in ('N','O')}
    assert all(v['images']==50 and v['metadata']==5 and v['files']==55 for v in storage.values())
    validation.update(final_artifact_audit_passed=True,total_guided_predictions=calls,storage=storage)
    (ROOT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    total=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    print(json.dumps(dict(audit='passed',results=summary,storage=storage,total_output_bytes=total,total_guided_predictions=calls),indent=2),flush=True)

if __name__=='__main__':main()
