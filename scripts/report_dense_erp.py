"""Audit ten compact L/M runs and save one K/L/M comparison PNG."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('outputs/vae-residual-controls/20260915-dense-lm')
MODELS=('sd2','sana','flux','sd35','pixeldit')


def read(path):return json.loads(Path(path).read_text())


def canonical_scheduler(scheduler):
    """Normalize only Diffusers' unordered default-field provenance list."""
    result = dict(scheduler)
    config = dict(result.get("config", {}))
    if "_use_default_values" in config:
        config["_use_default_values"] = sorted(config["_use_default_values"])
    result["config"] = config
    return result


def main():
    summary={};fig,axes=plt.subplots(5,3,figsize=(15,13),layout='constrained')
    names={'sd2':'SD2','sana':'SANA','flux':'FLUX.1-dev','sd35':'SD3.5','pixeldit':'PixelDiT'}
    fig.suptitle('K: 6 views, 100° | L/M: 89 views, 80°; identical prompt text in all bands',fontsize=12)
    geometry={label:read(ROOT/'geometry'/('experiment_'+label.lower()+'.json')) for label in ('L','M')}
    for row,b in enumerate(MODELS):
        baseline=Path('outputs/vae-residual-controls/20260910-lookingglass-v1')/b/'K'
        old=read(baseline/'comparison.json') if (baseline/'comparison.json').exists() else None
        old_image=baseline/'final_erp.png'
        if old_image.exists():axes[row,0].imshow(Image.open(old_image))
        else:axes[row,0].text(.5,.5,'not available',ha='center',va='center')
        axes[row,0].set_title(names[b]+' / K');axes[row,0].axis('off')
        summary[b]={}
        for col,label in enumerate(('L','M'),1):
            folder=ROOT/label/b
            assert sorted(p.name for p in folder.iterdir())==['final_result.png','metadata.json']
            meta=read(folder/'metadata.json');a=meta['audit'];g=geometry[label]
            assert meta['experiment']==label and meta['backend']==b
            assert meta['camera_count']==g['camera_count']==89
            assert a['guided_predictions']==a['expected_guided_predictions']==meta['camera_count']*meta['steps']
            assert a['extra_denoiser_calls']==0 and a['synchronous'] and a['cameras_fixed'] and a['conditioning_fixed']
            assert not any(a[k] for k in ('vae_residual','fixed_noise_renoising','spherical_latent','erp_latent'))
            assert a['transition']=='preserve_current_state' and a['warp']=='standard' and a['fusion']=='average'
            assert meta['prompt_assignment']==g['prompt_assignment']
            assert meta['prompt_slot_histogram']==g['prompt_slot_histogram']
            verified=next(r for r in g['full_resolution_verification'] if r['erp_size']==meta['erp_size'])
            assert meta['camera_sha256']==a['camera_sha256']==verified['camera_sha256']
            assert meta['coverage']['coverage_percent']==100
            assert meta['coverage']['minimum']>= (5 if label=='M' else 1)
            for key in ('minimum','p01','p05','median','maximum'):
                assert meta['coverage'][key]==verified[key]
            assert abs(meta['coverage']['mean']-verified['mean'])<1e-4
            assert meta['aggregate_metrics']['current_state_error_max']['max']<1e-4
            if old:
                bm=read(baseline/'metadata.json')
                assert meta['scheduler']['timesteps']==bm['scheduler_timesteps']
                if bm['scheduler_sigmas'] is not None:assert meta['scheduler']['sigmas']==bm['scheduler_sigmas']
                assert meta['model_checkpoint']==bm['model']['source']
            image=Image.open(folder/'final_result.png').convert('RGB')
            assert image.size==(meta['erp_size'][1],meta['erp_size'][0])
            axes[row,col].imshow(image);axes[row,col].set_title(names[b]+' / '+label);axes[row,col].axis('off')
            summary[b][label]=dict(runtime=meta['runtime_seconds'],allocated_gib=meta['peak_allocated_gib'],reserved_gib=meta['peak_reserved_gib'],
                calls=a['guided_predictions'],job=meta['slurm_job_id'],seam_ratio=meta['seam']['ratio'],
                runtime_vs_k=meta['runtime_seconds']/old['timing']['runtime_seconds'] if old else None,
                error=meta['aggregate_metrics']['current_state_error_max']['max'])
        l=read(ROOT/'L'/b/'metadata.json');m=read(ROOT/'M'/b/'metadata.json')
        assert l['audit']['initial_local_sha256']==m['audit']['initial_local_sha256']
        assert canonical_scheduler(l['scheduler'])==canonical_scheduler(m['scheduler'])
        assert l['prompt_sha256']==m['prompt_sha256']
        assert l['audit']['conditioning_sha256']==m['audit']['conditioning_sha256']
        lhs=np.asarray(Image.open(ROOT/'L'/b/'final_result.png'),dtype=np.float32)/255
        rhs=np.asarray(Image.open(ROOT/'M'/b/'final_result.png'),dtype=np.float32)/255
        summary[b]['L_M_png_mae']=float(np.abs(lhs-rhs).mean())
        summary[b]['L_M_pixel_identical']=bool(np.array_equal(lhs,rhs))
    fig.savefig(ROOT/'K-L-M.png',dpi=150);plt.close(fig)
    storage={label:dict(images=len(list((ROOT/label).rglob('*.png'))),metadata=len(list((ROOT/label).rglob('metadata.json'))),bytes=sum(p.stat().st_size for p in (ROOT/label).rglob('*') if p.is_file())) for label in ('L','M')}
    assert sum(v['images'] for v in storage.values())==10
    assert sum(v['metadata'] for v in storage.values())==10
    print(json.dumps(dict(audit='passed',results=summary,storage=storage),indent=2),flush=True)

if __name__=='__main__':main()
