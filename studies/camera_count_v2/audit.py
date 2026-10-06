"""Read-only baseline audit; writes only this study's new manifest/evidence."""
import collections
import os
from .common import *

def run():
    assert os.environ.get('SLURM_JOB_ID'), 'Use a CPU allocation'
    from studies.all_prompts.audit import configuration,verify_sources,audit_cache
    from studies.camera_patching.common import complete as camera_complete,rows as camera_rows
    from studies.original_spherediff.common import official_hashes,COMMIT,GIT_SOURCE,SPHERE,specs
    from .spherediff_adapter import reference_geometry
    import torch
    plan=verify_sources()
    for rel,h in official_hashes().items():
        content=subprocess.check_output(['git','-C',str(GIT_SOURCE),'show',COMMIT+':'+rel])
        assert hashlib.sha256(content).hexdigest()==h
    records={}
    for backend in BACKENDS:
        c,_,cams,_,old=configuration(backend)
        audit_cache(backend,c)
        # These are actual completed N89 experiment configs, not proposed settings.
        refs=[]
        for strategy in ('fibonacci','random'):
            for prompt in PROMPTS:
                row=next(r for r in camera_rows() if r['backend']==backend and r['strategy']==strategy
                         and r['projection']=='erp' and r['prompt']==prompt)
                assert camera_complete(row,check_exit=True,verbose=True),(backend,strategy,prompt)
                p=Path(row['output'])/'config.json'
                refs.append(dict(path=str(p),sha256=sha(p),configuration=read(p)))
        records[backend]=dict(configuration=c.to_dict(),baseline_metadata_sha256=sha(
            REPO/'outputs/gwtf-erp4k/20260923'/backend/'metadata.json'),
            source_shape=old['initialization']['source_shape'],source_sha256=old['initialization']['source_sha256'],
            local_native_resolution=old['local_native_resolution'],native_channels=old['native_channels'],
            local_RGB_resolution=old['local_RGB_resolution'],schedule=old['prepared_schedule'],
            camera_n89_references=refs)
    sf=reference_geometry();dirs=sf.horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator()
    rings=collections.Counter(round(float(x),1) for x in torch.rad2deg(torch.asin(dirs[:,1])))
    reference=[rings[x] for x in (90.,67.5,45.,22.5,0.,-22.5,-45.,-67.5,-90.)]
    assert reference==[4,8,11,14,15,14,11,8,4] and len(dirs)==89,reference
    _,_,saved,_,_=configuration('flux')
    saved_counts=collections.Counter(round(__import__('math').degrees(c.pitch),1) for c in saved)
    assert [saved_counts[x] for x in (90.,67.5,45.,22.5,0.,-22.5,-45.,-67.5,-90.)]==reference
    original=specs()
    original['flux']=dict(original['flux'],call=dict(original['flux']['call'],num_inference_steps=20))
    controlled=read(REPO/'outputs/flux-step-controls-seed0/spherediff/flux/steps20/ruins/config.json')
    assert original['flux']==controlled['configuration']
    source=dict(time=now(),job=os.environ['SLURM_JOB_ID'],protected_sources=protected_hashes(),
        scratch_destination=str((REPO/'outputs').resolve()),outputs_link=str(REPO/'outputs'),
        git_commit=shell_command(['git','rev-parse','HEAD']),git_status=shell_command(['git','status','--short']),
        official_commit=COMMIT,official_import_paths=plan['import_paths'],diffpano=records,
        spherediff=original,official_n89_ring_counts=reference,saved_n89_ring_counts=reference,
        new_source_scope='studies/camera_count_v2 only; historical gates and artifacts preserved',
        latest_user_override='SphereDiff FLUX 20 steps, superseding attachment preset 28',
        existing_n89_reference_root=str(REPO/'outputs/all-prompts-erp-cea-seed0'))
    ROOT.mkdir(parents=True,exist_ok=True)
    immutable(ROOT/'audit.json',source)
    prompts={p:prompt_record(REPO/'prompts'/(p+'.txt'))[0] for p in PROMPTS}
    immutable(ROOT/'study.json',dict(version=VERSION,rows=rows(),expected=126,diffpano=108,spherediff=18,
        sphere_strategies=['old'],sphere_flux_steps=20,prompts=prompts,seed=0,fov=[80.,80.],
        output_width=4096,output_height=2048,output_projection='erp',time_travel=False,lpw=False,dpa=False,
        max_gpu_jobs=5,layout_selection='geometry-only; random first passing seed; no image selection',
        forbidden=['cea','sd2','sphere fibonacci','sphere random','training','metric-model evaluation'],
        audit_sha256=sha(ROOT/'audit.json')))
    emit('AUDIT_PASSED',count=126,diffpano=108,spherediff=18,old89_rings=reference,
         scratch=source['scratch_destination'])
if __name__=='__main__':run()
