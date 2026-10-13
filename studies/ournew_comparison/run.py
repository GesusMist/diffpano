"""Locked, atomic status wrapper; no independent denoising implementation."""
import argparse,fcntl,traceback
from studies.ournew_comparison.common import *


def run(index,retry=False,single_pixel_source='fused_rgb_erp'):
    assert os.environ.get('SLURM_JOB_ID'),'Generation requires a Slurm GPU allocation'
    assert single_pixel_source=='fused_rgb_erp','This exact suite fixes ERP-source v2; legacy is an explicit GradientSettings option only'
    manifest=freeze_check();row=manifest['rows'][index]
    gate('cpu.json');gate('weights.json');gate('metrics-cpu.json');validation=gate('gpu.json');gate('official.json');preserved()
    directory=folder(row);directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if complete(row):print('VALID_CACHE',row['case_id'],flush=True);return
    if (directory/'status.json').exists() and not retry:raise RuntimeError('Incomplete case requires explicit --retry')
    job=os.environ['SLURM_JOB_ID'];write(directory/'status.json',dict(state='running',job=job,semantic_sha256=row['semantic_sha256']))
    try:
        if row['method']=='diffpano':
            from studies.ournew_comparison.runtime import generate
            metadata=generate(row,directory,validation)
        else:
            from studies.ournew_comparison.official import original
            from studies.all_prompts.common import safe
            info,path=prompt(row['prompt']);image,record=original(row,path)
            assert record['configuration']==row['semantic']['configuration'] and record['schedule']==row['semantic']['schedule']
            temporary=directory/('final.tmp-'+job+'.png');image.save(temporary);os.replace(str(temporary),str(directory/'final.png'))
            write(directory/'config.json',safe(record))
            metadata=dict(prompt=info,configuration=record['configuration'],official=record,counts=record['execution']['counts'],
                generation_seconds=record['execution']['generation_seconds'],peak_allocated_gib=record['execution']['peak_gpu_allocated_gib'],
                label='Official SphereDiff with matched 20 steps (official default 28)',
                unclamped_output='Not exposed by the official PIL output interface; no reconstructed float artifact',
                artifacts={n:sha(directory/n) for n in ('final.png','config.json')})
        freeze_check();preserved()
        metadata.update(case={k:v for k,v in row.items() if k!='semantic'},semantic_sha256=row['semantic_sha256'],source_sha256=row['source_sha256'],
            source_hashes=manifest['source_hashes'],implementation_revision=manifest['implementation_revision'],reused=False,job=job)
        write(directory/'metadata.json',metadata)
        write(directory/'status.json',dict(state='complete',job=job,semantic_sha256=row['semantic_sha256'],metadata_sha256=sha(directory/'metadata.json'),image_sha256=sha(directory/'final.png')))
        assert complete(row);print('CASE_COMPLETE',row['case_id'],flush=True)
    except BaseException as error:
        write(directory/'status.json',dict(state='failed',job=job,semantic_sha256=row['semantic_sha256'],error=traceback.format_exc(),solver=getattr(error,'diagnostics',None)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int);p.add_argument('--retry',action='store_true')
    p.add_argument('--single-pixel-source',choices=('fused_rgb_erp','legacy_source_patch'),default='fused_rgb_erp')
    a=p.parse_args();run(int(os.environ['SLURM_ARRAY_TASK_ID']) if a.index is None else a.index,a.retry,a.single_pixel_source)
