"""Audit four original-reference entries and assemble their full-ERP overview."""
from PIL import Image, ImageDraw
from studies.original_spherediff.common import *


def main():
    plan=require_gate();rows=[]
    for row in plan['rows']:
        n,p=row['backend'],row['prompt']
        record=dict(backend=n,prompt=p,seed=0,steps=row['spec']['call']['num_inference_steps'],
            resolution='4096x2048',kind='reused' if row['reuse'] else 'new',output=row['output'],status='pending')
        if row['reuse']:
            assert sha(row['metadata'])==row['reused_metadata_sha256']
            assert image_audit(row['output'])==row['reused_image']
            m=read(row['metadata'])
            record.update(status='complete',job=m['job'],generation_seconds=m['runtime_seconds'],total_seconds=m['total_seconds'],
                gpu_allocated_gib=m['peak_allocated_gib'],gpu_reserved_gib=m['peak_reserved_gib'],host_gib=m['host_max_rss_gib'],
                forwards=m['transformer_forwards'],shapes=m['transformer_input_shape_histogram'],
                scheduler=row['expected_scheduler'],image_sha256=row['reused_image']['sha256'])
        else:
            status=folder(n,p)/'status.json'
            if status.exists():
                s=read(status);record.update(status=s['state'],job=s['job'])
                if s['state']=='complete':
                    assert sha(row['metadata'])==s['metadata_sha256'];m=read(row['metadata'])
                    assert m['source_hashes']==source_hashes() and m['manifest_sha256']==sha(ROOT/'manifest.json')
                    assert image_audit(row['output'])==m['output'] and m['output']['sha256']==s['image_sha256']
                    assert m['actual_import_paths']==plan['import_paths'] and m['official_source_hashes']==plan['official_source_hashes']
                    assert m['prompt']==plan['prompts'][p] and m['seed']==0 and m['generator_device'].split(':')[0]=='cuda'
                    record.update(generation_seconds=m['generation_seconds'],total_seconds=m['process_seconds_since_module_start'],
                        model_loading_seconds=m['model_loading_seconds'],gpu_allocated_gib=m['peak_allocated_gib'],
                        gpu_reserved_gib=m['peak_reserved_gib'],host_gib=m['host_peak_rss_gib'],forwards=m['transformer_forwards'],
                        shapes=m['model_input_shape_histogram'],scheduler=m['scheduler'],image_sha256=m['output']['sha256'])
                elif s['state']=='failed':record['error']=s['error']
        rows.append(record)
    width,height=1024,512
    sheet=Image.new('RGB',(2*width,2*(height+32)),'white');draw=ImageDraw.Draw(sheet)
    for r in rows:
        x=PROMPTS.index(r['prompt'])*width;y=BACKENDS.index(r['backend'])*(height+32)
        draw.text((x+8,y+8),'Original SphereDiff '+r['backend'].upper()+' | '+r['prompt']+' | seed 0 | '+str(r['steps'])+' steps | '+r['kind']+' | '+r['status'],fill='black')
        if r['status']=='complete':
            with Image.open(r['output']) as im:
                sheet.paste(im.convert('RGB').resize((width,height),Image.Resampling.LANCZOS),(x,y+32))
    sheet.save(ROOT/'overview.png')
    require_gate()
    atomic(ROOT/'completion.json',dict(complete=all(r['status']=='complete' for r in rows),rows=rows,
        overview=str(ROOT/'overview.png'),overview_sha256=sha(ROOT/'overview.png'),historical_sources_and_artifacts_unchanged=True))
    lines=['# Original SphereDiff SANA / FLUX references','',
        'Pinned official spherical-latent implementation: '+COMMIT+'. Verified frozen source: '+str(SPHERE)+'. No DiffPano generation adapter or RGB-consensus code produces these images.',
        '', '| Backend | Prompt | Seed | Steps | Resolution | Reused/New | Status | Output | Job ID |',
        '|---|---|---:|---:|---|---|---|---|---|']
    for r in rows:
        lines.append('| '+' | '.join(str(r[k]) for k in ['backend','prompt','seed','steps','resolution','kind','status'])+
                     ' | ['+r['backend']+' '+r['prompt']+']('+r['output']+') | '+r.get('job','pending')+' |')
    lines+=['','[Overview sheet](overview.png). Exact matching evidence, imported paths, complete prompts and line ordering, checkpoint cache inventory and source hashes: [manifest.json](manifest.json), [audit.json](audit.json).',
        '', '## Exact prompts and settings']
    for p in PROMPTS:
        v=plan['prompts'][p]
        lines.append('- '+p+': SHA256 `'+v['sha256']+'`; seed 0; five unchanged lines. Snapshot prompt exists: '+str(v['snapshot_prompt_exists'])+
                     '; snapshot byte-identical: '+str(v['snapshot_byte_identical'])+'; pinned Git prompt byte-identical: '+str(v['pinned_git_byte_identical'])+'.')
    lines+=['',
        'SANA: Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers at e2b3c0cbffebcd09d83805e88b9f5f106afc74ac, bf16 variant, BF16, 20 steps, guidance 4.5, nominal 1024x1024, 2600 spherical points.',
        'FLUX: black-forest-labs/FLUX.1-dev at 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21, no variant, BF16, 28 steps, guidance 3.5, true CFG 1.0, 26500 spherical points; local defaults remain official. Both use temperature 0.1, CPU offload false, VAE tiling false, final ERP 4096x2048.',
        'These are complete-method references, not compute-matched DiffPano ablations. Original FLUX uses 28 steps; existing DiffPano FLUX runs may use 20. Equal seeds do not imply identical initial noise or local shapes across methods.',
        '', '## Reuse and scheduler evidence',
        'The two ruins originals were reused after matching source commit/hashes, preserved execution wrapper, model IDs/revisions, exact prompt hash/content, seed/device, call arguments, original defaults, precision/offload/tiling, schedules, forward counts, terminal ERP semantics and readable 4096x2048 images. No historic metadata was edited.',
        'The initial two CPU audits stopped on different ordering of Diffusers _use_default_values bookkeeping names. Installed source constructs this list from a set; the audit now canonicalizes only that order and still checks all scheduler parameter values, timesteps and sigmas. Failed CPU attempts are preserved in validation-attempts and execution.json.',
        'SANA uses DPMSolverMultistepScheduler with flow prediction. The serialized solver_order mapping remains 2 while the official static launcher sets config.solver_order to 1. The scheduler step reads that attribute, so effective numerical order is 1. The preserved correction record and installed scheduler-source hash verify this interpretation. FLUX uses first-order FlowMatchEulerDiscreteScheduler with its original dynamic shift. Actual timesteps and sigmas are recorded in the audit and new metadata.',
        '', '## Runtime, memory and actual model inputs',
        '| Backend/prompt | Generation sec | Total recorded sec | Peak allocated GiB | Peak reserved GiB | Host peak GiB | Transformer calls | Input shapes |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if r['status']=='complete':
            lines.append('| '+r['backend']+'/'+r['prompt']+' | '+' | '.join('%.3f'%r[k] for k in
                ['generation_seconds','total_seconds','gpu_allocated_gib','gpu_reserved_gib','host_gib'])+
                ' | '+str(r['forwards'])+' | '+str(r['shapes'])+' |')
    lines+=['',
        'Generation timing brackets the official pipeline call with CUDA synchronization. Historical total starts before model loading; new process timing starts at module startup and includes imports/audits/loading/artifact work. Neither includes shell activation or queue wait. GPU peaks describe PyTorch allocator memory; host peak is process RSS. Slurm elapsed is recorded separately.',
        'The historical global seeding order is preserved before model loading; each generation receives a fresh seed-0 CUDA generator. FLUX keeps its preparatory latent draw before the spherical Gaussian draw. No attention implementation or deterministic flag was changed.',
        '', '## Validation and execution',
        '[Validation commands, exits and logs](validation.json); [submission ledger](execution.json); [completion audit](completion.json). Missing/failed cases: '+str([r for r in rows if r['status']!='complete'])+'.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print('REPORT',[(r['backend'],r['prompt'],r['kind'],r['status']) for r in rows],flush=True)


if __name__=='__main__':main()
