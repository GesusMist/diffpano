"""Audit J and expose the F/J/G/I factorial comparison without model loading."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.plot_controlled_ladder import plt, save, seams


def read(path):return json.loads(Path(path).read_text())


def main():
    specs=read('configs/experiments/implied_endpoint_consensus/j-all-models.json')
    old=read('configs/experiments/vae_residual/i-all-models.json')
    out=Path('outputs/vae-residual-controls/report/J');out.mkdir(parents=True,exist_ok=True)
    summary={};audit={}
    overview,axes=plt.subplots(4,4,figsize=(18,11),layout='constrained')
    for row,(b,spec) in enumerate(specs.items()):
        folder=Path(spec['output']);c=read(folder/'comparison.json');f=read(spec['reference'])
        restored=dict(c['config']);assert restored.pop('consensus_transition')==dict(mode='preserve_current_state',vae_residual_correction=False)
        assert restored==f['config']
        assert c['initial_native_sha256']==f['initial_native_sha256']
        assert not c['audit']['training_free_residual_correction'] and not c['audit']['own_rgb_roundtrip_diagnostic']
        assert not c['audit']['bridge_enabled'] and c['audit']['synchronous']
        assert c['audit']['consensus_transition']=='preserve_current_state'
        paths={'F':Path(spec['reference_generation']),'J':folder/'generation',
               'G':Path(old[b]['reference_generation']),'I':Path(old[b]['output'])/'generation'}
        ms={label:read(p/'metadata.json') for label,p in paths.items()}
        for key in c['runtime_fields_checked']:assert ms['J'].get(key)==ms['F'].get(key)
        expected=len(ms['F']['steps'])*3
        assert c['guided_predictions']==expected and c['extra_diagnostic_denoiser_calls']==0
        rs=read(folder/'transition_patches.json');assert len(rs)==expected
        assert max(r['i_current_state_error_max_abs'] for r in rs)<1e-4
        assert all('clean_native_roundtrip_mae' not in s['state_statistics'] for s in ms['J']['steps'])
        pair,pax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        record={'control':c,'diagnostics':{k:dict(mean=float(np.mean([r[k] for r in rs])),max=max(r[k] for r in rs)) for k in rs[0] if k not in ('step','patch','timestep')}}
        for col,label in enumerate(('F','J','G','I')):
            image=np.asarray(Image.open(paths[label]/'result.png').convert('RGB'))/255.
            title=label+' / '+('no residual' if label in 'FJ' else 'residual')+' / '+('current state' if label in 'JI' else 'endpoint')
            for ax in (axes[row,col],pax.flat[col]):
                ax.imshow(image);ax.axis('off');ax.set_title(b.upper()+' — '+title,fontsize=9)
            record[label]=dict(path=str(paths[label]/'result.png'),seams=seams.boundary_gradient_metrics(image,c['geometry']['rgb']['patches']),runtime_seconds=ms[label]['runtime_seconds'])
        save(pair,out/(b+'-F-J-G-I'));summary[b]=record;audit[b]=dict(passed=True,job=c['job'],predictions=expected)
    save(overview,out/'F-J-G-I-all-models')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    for b,r in summary.items():print(b,{l:round(r[l]['seams']['boundary_to_nearby_ratio'],5) for l in ('F','J','G','I')})
    print('J audit passed',sum(v['predictions'] for v in audit.values()),'predictions')

if __name__=='__main__':main()
