"""Build an auditable partial report from snapshots of completed evaluation artifacts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import textwrap
from .common import ROOT, RUN, PRIMARY_METHODS, read, write, sha, digest, atomic_text, csv_write, update_component, now

LABELS={'sphere20':'SphereDiff','rgb':'DiffPano RGB','select':'DiffPano Select','meanmax050':'DiffPano mean/max 0.50','spherope':'SpheRoPE'}


_PLOT_ADDON=None

def _plotting(run):
    """Load pinned report-only wheels in process-local temporary storage."""
    global _PLOT_ADDON
    try:
        import matplotlib
        return
    except ImportError:
        pass
    import atexit
    import os
    import sys
    import tempfile
    import zipfile
    folder=Path(run)/'report/addon-wheels'
    manifest=read(folder/'manifest.json')
    temporary=tempfile.TemporaryDirectory(prefix='paper-eval-report-')
    destination=Path(temporary.name)
    for entry in manifest['wheels']:
        source=folder/entry['filename']
        if sha(source)!=entry['sha256']:raise ValueError('Report addon wheel hash changed')
        with zipfile.ZipFile(source) as archive:archive.extractall(destination)
    os.environ.setdefault('MPLCONFIGDIR',str(destination/'mpl-config'))
    sys.path.insert(0,str(destination));_PLOT_ADDON=temporary;atexit.register(temporary.cleanup)
    import matplotlib


def _rows(value):
    return value if isinstance(value,list) else value.get('records',value.get('rows',value.get('results',[])))


def _table(rows, fields):
    def cell(value):
        if value is None:return '—'
        if isinstance(value,float):return '%.4g'%value
        if isinstance(value,list) and all(item is None or isinstance(item,(int,float)) for item in value):return '—' if any(item is None for item in value) else '['+', '.join('%.4g'%item for item in value)+']'
        if isinstance(value,(dict,list)):return json.dumps(value,separators=(',',':'))
        return str(value).replace('|','/')
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']+
        ['| '+' | '.join(cell(row.get(field)) for field in fields)+' |' for row in rows])


def _tex(value):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(char,char) for char in str(value))


def _scheduler(run, journal):
    result=dict(observed_utc=now(),submissions=journal.get('attempts',[]),accounting=[],state='unavailable')
    ids=sorted({str(row['job_id']).split(';')[0] for row in result['submissions'] if row.get('job_id')})
    if not ids:return result
    try:
        command=['sacct','-M','faster','-n','-P','-X','-j',','.join(ids),'--format=JobIDRaw,State,ExitCode,Elapsed,AllocCPUS,NodeList']
        process=subprocess.run(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
        if process.returncode:raise RuntimeError(process.stderr.strip())
        fields=('job_id','state','exit_code','elapsed','allocated_cpus','nodes')
        result.update(state='observed',command=command,accounting=[dict(zip(fields,line.split('|'))) for line in process.stdout.splitlines() if line.strip()])
    except (OSError,RuntimeError,subprocess.TimeoutExpired) as exc:result['error']=str(exc)
    return result


def _runtime(records):
    output=[]
    for row in records:
        runtime=row.get('runtime',{});parameters=row.get('parameters',{})
        output.append(dict(case_id=row['case_id'],prompt_id=row['prompt_id'],method_id=row['method_id'],seed=row.get('seed',0),
            image_sha256=row['image_sha256'],gpu=runtime.get('gpu'),generation_seconds=runtime.get('generation_seconds'),
            setup_seconds=runtime.get('setup_seconds'),solver_seconds=runtime.get('solver_seconds'),peak_allocated_gib=runtime.get('peak_allocated_gib'),
            resolution=parameters.get('erp_resolution'),steps=parameters.get('steps',parameters.get('num_inference_steps')),
            camera_count=parameters.get('camera_count'),local_resolution=parameters.get('local_resolution'),counts=parameters.get('counts'),
            runtime_policy=runtime.get('policy'),missing_fields=runtime.get('missing_fields',[]),parameters=parameters))
    return output


def _figures(run, directory, records, per_panorama, cohorts):
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .rendering import render_common_views
    outputs=[];by_case={row['case_id']:row for row in records}
    prompts=cohorts.get('cohorts',{}).get('development15',{}).get('prompt_ids',[])
    selected=[row for row in per_panorama if row.get('namespace')=='common_v2' and row.get('method_id') in PRIMARY_METHODS and row.get('prompt_id') in prompts]
    for metric in ('DS','CS-H','MUSIQ'):
        points=[row for row in selected if row.get('metrics',{}).get(metric) is not None]
        if not points:continue
        figure,axis=plt.subplots(figsize=(10,4))
        for method in PRIMARY_METHODS:
            series=[row for row in points if row['method_id']==method]
            if series:axis.plot([prompts.index(row['prompt_id'])+1 for row in series],[row['metrics'][metric] for row in series],'o',label=LABELS[method],markersize=4)
        axis.set(xlabel='Frozen development15 prompt index',ylabel=metric,title='Per-prompt '+metric+' (available common_v2 observations)')
        axis.set_xticks(range(1,len(prompts)+1));axis.grid(alpha=.2);axis.legend(fontsize=8,ncol=2);figure.tight_layout()
        target=directory/('per-prompt-'+metric.replace('-','_')+'.png');figure.savefig(target,dpi=150);plt.close(figure)
        outputs.append(dict(kind='per_prompt',metric=metric,path=str(target),sha256=sha(target),case_ids=[row['case_id'] for row in points]))
    # Explicit hardware, resolution and denoising-count strata; missing hardware is excluded.
    strata=defaultdict(list)
    for scored in selected:
        if scored.get('metrics',{}).get('MUSIQ') is None:continue
        row=by_case.get(scored['case_id']);runtime=row.get('runtime',{}) if row else {};parameters=row.get('parameters',{}) if row else {}
        if not runtime.get('gpu') or runtime.get('generation_seconds') is None:continue
        key=(runtime['gpu'],tuple(parameters.get('erp_resolution',[])),parameters.get('steps',parameters.get('num_inference_steps')))
        strata[key].append((row,scored))
    if strata:
        figure,axes=plt.subplots(len(strata),1,figsize=(9,4*len(strata)),squeeze=False)
        for axis,(key,values) in zip(axes[:,0],sorted(strata.items(),key=lambda item:str(item[0]))):
            for method in PRIMARY_METHODS:
                subset=[pair for pair in values if pair[0]['method_id']==method]
                if subset:axis.scatter([a['runtime']['generation_seconds'] for a,b in subset],[b['metrics']['MUSIQ'] for a,b in subset],label=LABELS[method],s=18)
            axis.set(xlabel='Recorded generation seconds (setup excluded)',ylabel='MUSIQ',title='%s | ERP%s | %s denoising steps'%key)
            axis.grid(alpha=.2);axis.legend(fontsize=8)
        figure.tight_layout();target=directory/'quality-time.png';figure.savefig(target,dpi=140);plt.close(figure)
        outputs.append(dict(kind='quality_time',path=str(target),sha256=sha(target),stratification=['gpu','native_resolution','denoising_steps'],
            limitation='Descriptive systems comparison; camera count, solvers and measured timing boundaries differ. Not an isolated speedup or causal quality claim. Missing hardware excluded.'))
    # Score-independent prompt selection remains identical on each incremental rebuild.
    picked=sorted(prompts,key=lambda prompt:digest(['representative-v1',prompt]))[:3]
    selection=dict(rule='First three development15 prompt IDs ordered by SHA256([representative-v1,prompt_id]); independent of scores',prompt_ids=picked,
        cameras=[dict(name='Front',yaw=0,pitch=0),dict(name='Seam',yaw=180,pitch=0),dict(name='North',yaw=0,pitch=90),dict(name='South',yaw=0,pitch=-90)],
        display='common_v2 2048x1024 ERP; 512x512 FOV90 crops; no native-detail claim')
    write(directory/'representative-selection.json',selection)
    for index,prompt in enumerate(picked):
        cells={row['method_id']:row for row in records if row['prompt_id']==prompt and row['method_id'] in PRIMARY_METHODS and row.get('seed',0)==0}
        if set(cells)!=set(PRIMARY_METHODS):continue
        target=directory/('representative-%02d.png'%(index+1));provenance=directory/('representative-%02d.json'%(index+1))
        identity=dict(prompt_id=prompt,rule_sha256=digest(selection),sources={method:dict(case_id=row['case_id'],image_path=row['image_path'],image_sha256=row['image_sha256']) for method,row in cells.items()})
        if not target.exists() or not provenance.exists() or read(provenance).get('identity')!=identity:
            figure,axes=plt.subplots(4,4,figsize=(11,11))
            for ri,method in enumerate(PRIMARY_METHODS):
                row=cells[method]
                if sha(row['image_path'])!=row['image_sha256']:raise ValueError('Representative source changed: '+row['case_id'])
                with Image.open(row['image_path']) as image:views=render_common_views(np.asarray(image.convert('RGB')),selection['cameras'])
                for ci,view in enumerate(views):
                    axes[ri,ci].imshow(view);axes[ri,ci].set_xticks([]);axes[ri,ci].set_yticks([])
                    if ri==0:axes[ri,ci].set_title(selection['cameras'][ci]['name'])
                    if ci==0:axes[ri,ci].set_ylabel(LABELS[method])
            figure.suptitle(Path(prompt.split('@')[0]).stem.replace('_',' '));figure.tight_layout();figure.savefig(target,dpi=120);plt.close(figure)
            write(provenance,dict(identity=identity,image_sha256=sha(target)))
        outputs.append(dict(kind='representative',path=str(target),sha256=sha(target),provenance=str(provenance)))
    return outputs


def _pdf(path, markdown, figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from itertools import zip_longest
    lines=[];source_lines=markdown.splitlines();index=0
    while index<len(source_lines):
        line=source_lines[index]
        if line.startswith('|'):
            block=[]
            while index<len(source_lines) and source_lines[index].startswith('|'):
                cells=[value.strip() for value in source_lines[index].strip().strip('|').split('|')]
                if not all(set(value)<=set('-: ') for value in cells):block.append(cells)
                index+=1
            widths=[min(30,max(len(row[col]) for row in block)) for col in range(len(block[0]))]
            while sum(widths)+2*(len(widths)-1)>104:
                largest=widths.index(max(widths));widths[largest]-=1
            for row in block:
                wrapped=[textwrap.wrap(cell,width=width) or [''] for cell,width in zip(row,widths)]
                lines.extend('  '.join(cell.ljust(width) for cell,width in zip(parts,widths)) for parts in zip_longest(*wrapped,fillvalue=''))
            continue
        lines.extend(textwrap.wrap(line,width=104,replace_whitespace=False,break_long_words=True) or [''])
        index+=1
    temporary=path.with_suffix('.tmp.pdf')
    with PdfPages(temporary) as pdf:
        for start in range(0,len(lines),62):
            figure=plt.figure(figsize=(8.27,11.69))
            figure.text(.06,.96,'\n'.join(lines[start:start+62]),va='top',family='DejaVu Sans Mono',fontsize=8,linespacing=1.4)
            figure.text(.94,.025,str(start//62+1),ha='right',fontsize=9)
            if start==0:figure.savefig(path.with_name('report-preview.png'),dpi=120)
            pdf.savefig(figure);plt.close(figure)
        for item in figures:
            figure=plt.figure(figsize=(11.69,8.27) if item['kind']!='representative' else (8.27,11.69))
            axis=figure.add_axes([.04,.06,.92,.89]);axis.imshow(plt.imread(item['path']));axis.axis('off')
            figure.text(.04,.025,Path(item['path']).name+' — '+item['sha256'][:16],fontsize=8);pdf.savefig(figure);plt.close(figure)
    temporary.replace(path)


def report(run=RUN):
    run=Path(run);directory=run/'report';directory.mkdir(exist_ok=True);tables=directory/'tables';tables.mkdir(exist_ok=True)
    from .progress import refresh as refresh_component_status
    refresh_component_status(run)
    figures_dir=directory/'figures';figures_dir.mkdir(exist_ok=True)
    evidence={}
    def load(relative,default):
        path=run/relative
        if not path.exists():return default
        content=path.read_bytes();value=json.loads(content)
        import hashlib
        evidence[relative]=dict(path=str(path),sha256=hashlib.sha256(content).hexdigest(),size_bytes=len(content))
        return value
    registry=load('protocol_registry.json',{});inventory=_rows(load('inventory/records.json',[]));cohorts=load('inventory/cohorts.json',{})
    selected_ids={row['case_id'] for row in cohorts.get('selected_cells',[])}
    records=[row for row in inventory if row['case_id'] in selected_ids and row.get('state')=='complete']
    historical_count=len(records)
    confirmation_inventory=_rows(load('generation/inventory_records.json',[]))
    records.extend(row for row in confirmation_inventory if row.get('state')=='complete' and row['case_id'] in selected_ids and row['case_id'] not in {item['case_id'] for item in records})
    record_by_id={row['case_id']:row for row in records}
    capture_coverage=load('dataset/capture_coverage.json',{});dataset=load('real_dataset_manifest.json',{});native=load('native/manifest.json',{});captions=load('captions.json',{})
    groups={name:_rows(load(name+'/group_metrics.json',[])) for name in ('group1','group2','vlm')}
    groups['group2'].extend(_rows(load('group2/confirmation/group_metrics.json',[])))
    pairs={name:_rows(load(name+'/paired_comparisons.json',[])) for name in ('group1','group2','vlm')}
    pairs['group2'].extend(_rows(load('group2/confirmation/paired_comparisons.json',[])))
    per_panorama=_rows(load('group1/per_panorama_metrics.json',[]))
    # Scores must remain tied to an audited original image; never silently remap aliases.
    for row in per_panorama:
        source=record_by_id.get(row['case_id'])
        if source is not None and (row['image_sha256']!=source['image_sha256'] or row['prompt_id']!=source['prompt_id']):
            raise ValueError('Metric/inventory identity mismatch: '+row['case_id'])
    statuses={name:load(name+'/status.json',{}) for name in ('group1','group2','vlm','human')}
    browser_validation=load('human/browser-validation/status.json',{})
    browser_sources_match=bool(browser_validation.get('source_hashes')) and all((ROOT/path).exists() and sha(ROOT/path)==expected for path,expected in browser_validation.get('source_hashes',{}).items())
    browser_checked=browser_validation.get('state')=='passed' and browser_sources_match and browser_validation.get('real_ratings_unchanged')
    usage=load('api/usage_summary.json',{});pilot=load('vlm/pilot_result.json',{});calibration=load('vlm/calibration_review.json',{})
    final_method=load('generation/final_method_evidence.json',{});agreement=load('vlm/judge_agreement.json',{});generation=load('generation/manifest.json',{})
    load('targeted_checks.json',{});load('scoring_prompts.json',{});load('run_config.json',{});load('native/seed_validation.json',{});load('group1/assets/musiq-validation.json',{})
    for relative in ('group1/protocol.json','group2/protocol.json','group2/reference_features.json','group2/real_reference_inputs.json',
                     'native/reference-A.json','native/reference-B.json','vlm/pilot_protocol.json','vlm/production_views.json',
                     'vlm/per_panorama.json','report/addon-wheels/manifest.json','inventory/provenance_audit.json','generation/optional_baseline_availability.json'):
        load(relative,{})
    scheduler=_scheduler(run,load('scheduler.json',{}));write(directory/'scheduler_snapshot.json',scheduler)
    pending=[];native_done=[]
    from .native import complete as native_complete
    for row in native.get('rows',[]):
        if native_complete(run,row):native_done.append(row['case_id'])
        else:pending.append(dict(component='native',case_id=row['case_id'],index=row['index'],state='pending_or_incomplete'))
    actual_generation=_rows(load('generation/inventory_records.json',load('generation/verified_records.json',load('generation/records.json',[]))))
    generated={row['case_id'] for row in actual_generation if row.get('state')=='complete'}
    generation_rows=_rows(generation)
    if generation_rows:
        for row in generation_rows:
            if not row.get('selected',row.get('tier','core')=='core'):continue
            if row['case_id'] not in generated:pending.append(dict(component='confirmation',case_id=row['case_id'],state='pending_or_incomplete'))
    else:
        for scene in dataset.get('rows',[]):
            for method in ('select','sphere20','spherope'):
                pending.append(dict(component='confirmation',case_id='confirmation24/'+scene['scene_id']+'/'+method+'/seed0',state='awaiting_frozen_generation_manifest',identity_kind='planned_logical_slot'))
    for row in records:
        common=[result for result in per_panorama if result['case_id']==row['case_id'] and result.get('namespace')=='common_v2']
        for metric in ('DS','CS-H','MUSIQ'):
            if not any(item.get('metrics',{}).get(metric) is not None for item in common):pending.append(dict(component='group1',case_id=row['case_id'],metric=metric,state='pending_metric'))
    for metric in ('FAED','Distort'):
        pending.append(dict(component='group2',metric=metric,state='blocked_compatible_asset_or_implementation',case_id=None))
    for name in ('group2','vlm'):
        if not groups[name]:pending.append(dict(component=name,state='awaiting_completed_group_evaluation',case_id=None))
    if not captions.get('reviewed'):pending.append(dict(component='captions',state='awaiting_complete_visual_review',case_id=None))
    runtime=_runtime(records)
    coverage=[]
    for cohort_name,cohort in cohorts.get('cohorts',{}).items():
        ids=set(cohort.get('prompt_ids',[]))
        for method in (('sphere20','select','spherope') if cohort_name=='confirmation24' else PRIMARY_METHODS):
            members=[row for row in records if row['prompt_id'] in ids and row['method_id']==method and row.get('seed',0)==0]
            coverage.append(dict(cohort=cohort_name,method=method,expected_prompts=len(ids),available_prompts=len({row['prompt_id'] for row in members}),seed=0))
    confirmation_groups=_rows(load('group1/confirmation/groups.json',[]));confirmation_pairs=_rows(load('group1/confirmation/paired_comparisons.json',[]))
    exported=dict(coverage=coverage,runtime=runtime,pending=pending,api_usage=usage.get('by_stage_and_judge',[]),
        confirmation_metrics=confirmation_groups,confirmation_paired=confirmation_pairs,
        gpt4o=[row for row in groups['vlm'] if row.get('judge')=='openai/gpt-4o'],
        gpt6sol=[row for row in groups['vlm'] if row.get('judge')=='openai/gpt-6-sol'],
        judge_agreement=_rows(agreement),
        repeat_reliability=_rows(load('vlm/repeat_reliability.json',[])),repeat_observations=_rows(load('vlm/repeat_observations.json',[])),
        real_distribution=[row for row in groups['group2'] if row.get('reference_kind')=='real'],
        native_distribution=[row for row in groups['group2'] if row.get('reference_kind') in ('native','native-to-native')],
        ablations=[row for row in groups['group1'] if row.get('method_id') not in PRIMARY_METHODS or row.get('cohort_id')=='ablation4'],
        failures=_rows(load('group1/failures.json',[])))
    main_cells={}
    for row in groups['group1']+confirmation_groups:
        if row.get('namespace')!='common_v2' or row.get('cohort_id') not in ('development15','development24','confirmation24') or row.get('method_id') not in set(PRIMARY_METHODS)|{'spherope'}:continue
        identity=(row['cohort_id'],row['method_id'],row.get('text_policy_id','not_applicable'))
        cell=main_cells.setdefault(identity,dict(cohort_id=identity[0],method_id=identity[1],text_policy_id=identity[2],namespace='common_v2'))
        metric=row['metric'];cell[metric]=row.get('mean');cell[metric+'_ci95']=row.get('ci95');cell[metric+'_prompts']=row.get('prompt_count');cell[metric+'_panoramas']=row.get('panorama_count')
    exported['main_objective']=list(main_cells.values())
    write(tables/'table_protocols.json',dict(registry_sha256=evidence['protocol_registry.json']['sha256'],
        metric_definitions=registry.get('metrics',{}),statistics=registry.get('statistics',{}),
        default_units=dict(DS='unitless, lower',CS='100 times raw cosine, higher',MUSIQ='native SPAQ model scale, higher',IS='collection-level unitless, higher',FID='raw squared feature-space distance, lower',KID='raw unbiased polynomial MMD-squared; not multiplied by 1000; lower',runtime='seconds',memory='GiB',VLM='adapted1-to-5 ordinal score, higher'),
        references='Distinct reference_kind/bank/cohort namespaces; counts and hashes retained in source rows and traceability input files',
        missing='null/blank; never synthetic zero',bootstrap='2000 paired prompt clusters; achieved replicate counts retained; human uses crossed prompt/rater clusters'))
    for name in groups:exported[name+'_metrics']=groups[name];exported[name+'_paired']=pairs[name]
    for name,rows in exported.items():
        write(tables/(name+'.json'),rows);csv_write(tables/(name+'.csv'),rows,sorted({key for row in rows for key in row}) or ['state'])
    # Full trace includes every audited source, complete parameters and retained metric provenance.
    write(directory/'traceability.json',dict(registry_sha256=evidence.get('protocol_registry.json',{}).get('sha256'),input_artifacts=evidence,
        report_sources={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),Path(__file__).with_name('human.py')]+sorted(Path(__file__).with_name('human_assets').glob('*')) if path.is_file()},
        case_records=[dict(case_id=row['case_id'],prompt_id=row['prompt_id'],method_id=row['method_id'],seed=row.get('seed',0),
            image_path=row['image_path'],image_sha256=row['image_sha256'],metadata_path=row.get('metadata_path'),metadata_sha256=row.get('metadata_sha256'),
            parameters=row.get('parameters'),provenance=row.get('provenance')) for row in records],
        metric_identity_policy='Image hash and prompt ID checked against audited selected case; input result files separately content pinned. This is an incremental multi-file snapshot, not an all-workers completion barrier.'))
    from .human import prepare as prepare_human, analyze as analyze_human
    human=prepare_human(run);human_analysis=analyze_human(run)
    observed_path=run/'human/ratings/observations.json'
    observed_ids={row['assignment_id'] for row in read(observed_path)} if observed_path.exists() else set()
    public_assignments=read(run/'human/assignments.json')['assignments']
    pending.extend(dict(component='human',case_id=row['assignment_id'],state='awaiting_collection' if row['state']=='ready' else 'awaiting_assets_and_collection') for row in public_assignments if row['assignment_id'] not in observed_ids)
    write(tables/'pending.json',pending);csv_write(tables/'pending.csv',pending,sorted({key for row in pending for key in row}))
    _plotting(run)
    figure_records=_figures(run,figures_dir,records,per_panorama,cohorts)
    write(directory/'figures.json',figure_records)
    primary=[row for row in groups['group1'] if row.get('cohort_id')=='development15' and row.get('namespace')=='common_v2' and row.get('method_id') in PRIMARY_METHODS and row.get('metric') in ('DS','CS-H','CS-polar','CS-directional','MUSIQ','MUSIQ-polar','IS')]
    primary.sort(key=lambda row:(row.get('metric',''),row['method_id']))
    primary_pairs=[row for row in pairs['group1'] if row.get('cohort_id')=='development15' and row.get('namespace')=='common_v2' and row.get('comparison')=='select' and row.get('metric') in ('DS','CS-H','MUSIQ')]
    limitations=[
        'Development prompts were used during method development. Their results do not substitute for held-out confirmation.',
        'The 24 ODI-SR scenes (6 indoor, 18 outdoor) are a small explicitly reviewed confirmation sample. Small-sample FID/KID/OmniFID are exploratory distribution comparisons, not paired perceptual errors or evidence of universal realism.',
        'The frozen ODI-SR set includes source-capture missing data: scenes 031 and 036 have black missing-capture caps at both poles, with larger nadir caps. At the recorded black-mask threshold, north/south crop pixel fractions are 3.76%/31.06% for 031 and 2.34%/51.51% for 036. These estimates describe 80-degree 320-square review crops, not full-sphere area; threshold sensitivity and solid-angle fractions are retained in dataset/capture_coverage.json. Physical capture occlusions in 027/032/021/077 are documented separately. Scenes are not replaced after evaluation. Real OmniFID polar components may reflect capture masks, so horizontal and polar contributions must be read separately.',
        'Native standalone references are ordinary FLUX images, not ground truth of a unique panorama or physically guaranteed camera FOV. Bank A is primary and bank B sensitivity; five-view/native-resolution fidelity and common-resolution quality remain separate.',
        'common_v2 resamples every ERP periodically to 2048x1024 before 512x512 FOV90 evaluation. Upsampling does not restore missing detail. Native system-fidelity results retain source-resolution effects.',
        'DS is the author-linked first-order Scharr reimplementation with its documented singleton-output fix, not exact reproduction of the unavailable original second-order formulation. Low seam discontinuity alone does not establish geometry.',
        'CLIP CS-H uses eight horizontal views and a separately pinned short scoring-text policy. Directional and polar CS are reported separately. Legacy generation-text and resolution results are not pooled with common_v2.',
        'FAED remains unavailable because the author-linked compatible RGB checkpoint returned HTTP 403. Distort is unavailable without a verified released inference implementation/checkpoint. No substitute or zero score is reported.',
        'MUSIQ uses the published SPAQ weights through pinned upstream PyIQA architecture. All 248 converted state tensors were compared exactly with the official checkpoint; this checks conversion, not broader domain validity.',
        'VLM scores use a declared adapted rubric and response schema. They are not a numerical replication of the published SphereDiff judging protocol. GPT-4o and GPT-6 Sol are never pooled.',
        'The real-image pilot returned 30/30 valid schemas. Both judges showed zero severe-minus-clean geometric-distortion change for the single-scene warp. GPT-6 Sol also showed zero seam-continuity change while GPT-4o decreased by 3; GPT-6 Sol grayscale geometry increased by 1. This is weak control sensitivity and possible appearance confounding, not validated measurement accuracy.',
        'Human collection currently contains %d of 240 planned pair evaluations. Preparation and synthetic software-test fixtures are not human observations.'%human_analysis['observations'],
        'Runtime observations retain actual hardware, native resolution, denoising count, camera count and accounting boundaries. Missing setup or hardware fields stay missing; quality-time plots do not imply isolated algorithmic speedups.',
        'All aggregate uncertainty resamples prompt clusters rather than treating crops as independent panoramas. Historical runs are seed0; additional seeds are nested within prompts if completed. Current partial groups may have fewer than the prespecified prompts.',
    ]
    claims=[dict(claim='Verified historical method comparisons are available',state='supported_inventory',evidence='inventory/records.json; inventory/cohorts.json',scope='selected cells only; aliases excluded'),
        dict(claim='Select improves quality, geometry or runtime generally',state='not_established',evidence='report/tables/group1_paired.json; pending confirmation and human observations',scope='Do not infer from isolated metric or unpaired partial means'),
        dict(claim='VLM outputs follow the requested schema',state='supported_on_pilot' if pilot.get('valid_requests')==30 else 'pending',evidence='vlm/pilot_result.json',scope='30 pilot responses; schema validity only'),
        dict(claim='VLM judges reliably detect geometry defects',state='not_supported_by_pilot',evidence='vlm/calibration_review.json',scope='single-scene controls; weak warp sensitivity'),
        dict(claim='Human preference supports a method',state='awaiting_collection',evidence='human/analysis.json',scope='No human ratings invented'),
        dict(claim='Common-resolution evaluation controls display resolution',state='protocol_supported',evidence='protocol_registry.json; report/traceability.json',scope='Does not recover native information lost by lower-resolution generation')]
    write(directory/'claims_evidence.json',claims);write(directory/'limitations.json',limitations)
    summary=dict(state='partial',built_utc=now(),run=str(run),registry_sha256=evidence['protocol_registry.json']['sha256'],
        inventory_records=len(inventory),selected_complete_cases=len(records),historical_selected_complete_cases=historical_count,raw_inventory_states=dict(Counter(row.get('state') for row in inventory)),
        group1_status=statuses['group1'],native_complete=len(native_done),native_expected=native.get('expected',240),
        confirmation_complete=len(generated),confirmation_expected=72,pilot_valid=pilot.get('valid_requests',0),pilot_expected=pilot.get('planned_requests',30),
        human_observations=human_analysis['observations'],human_expected=240,ready_human_assets=human['ready_anonymous_assets'],
        human_viewer_browser_validation='passed' if browser_checked else browser_validation.get('state','pending'),
        api_usage=usage.get('totals',{}),api_budget_mode=usage.get('budget_mode'),pending_items=len(pending),
        output_files=['report.md','report.pdf','results.tex','methods_evaluation.tex','report/tables','report/traceability.json','human/participant/index.html','RESUME.md'])
    write(directory/'summary.json',summary)
    parts=['# Panorama evaluation — partial report',
        'Snapshot UTC: '+summary['built_utc']+'. Registry SHA256: '+summary['registry_sha256']+'.',
        'This report contains completed observations and explicit missing work. Worker completion is not inferred from a submitted job or an existing image alone.',
        '## What changed and what ran',
        'Historical images and scientific execution evidence were preserved. The new run audits aliases and actual configurations, adds a shared periodic common-resolution renderer, preserves a separate exact legacy metric namespace, verifies the MUSIQ weight conversion, and executes cached/recomputed numeric scoring. New ordinary native-reference images, held-out confirmation generation, and paid judging have independent manifests and completion checks.',
        'The metric registry and per-case retained feature identities identify which historical features were compatible for reuse. Completed observations are exported without treating job submission, PNG presence, or code readiness as completed scientific evidence. Evaluation tests covered projection orientation and wrap periodicity, analytic seam behavior, prompt-cluster statistics, cache identities and provenance. Human software tests use temporary synthetic fixtures only.',
        '## Coverage',
        '%d historical inventory entries resolve to %d selected complete scientific cases. Native references: %d/%d. Confirmation ERPs: %d/72. Human ratings: %d/240.'%(len(inventory),historical_count,len(native_done),native.get('expected',240),len(generated),human_analysis['observations']),
        _table([row for row in coverage if row['cohort'] in ('development15','development24')],['cohort','method','available_prompts','expected_prompts','seed']),
        'The full provenance and all ablation cohorts are in report/traceability.json and report/tables/coverage.csv. Missing meanmax050 development24 prompts are not replaced by other configurations.',
        '## Common-resolution development15 observations',
        'Means and 95% prompt-bootstrap intervals below retain their actual prompt counts. All common_v2 CS rows here use common_v2_short_scoring_text. Partial method means with differing prompts should not be ranked. Paired contrasts use their own common prompt intersection.',
        _table(primary,['metric','method_id','mean','ci95','prompt_count']) if primary else 'Pending completed group1 metric rows.',
        '### Paired select contrasts',_table(primary_pairs,['metric','baseline','comparison','mean','ci95','prompt_count']) if primary_pairs else 'Pending paired rows.',
        'Full metrics, per-prompt values, bootstrap settings and every ablation are in report/tables/group1_metrics.json and group1_paired.json. Legacy values remain in their own namespace. IS is a distribution-level quantity; it is not assigned to individual panoramas.',
        '## Frozen confirmation-method decision',
        'Select is the predeclared fallback used for confirmation, not a declared winner. Development evidence leads differ: mean/max 0.50 leads MUSIQ and CS, while RGB leads DS. This tradeoff does not establish one uniformly best method. The decision is frozen before confirmation outcomes; its exact rationale is retained in generation/final_method_evidence.json.',
        '## Confirmation observations',
        _table([row for row in confirmation_groups if row.get('namespace')=='common_v2' and row.get('metric') in ('DS','CS-H','MUSIQ')],['metric','method_id','mean','ci95','prompt_count']) if confirmation_groups else 'Pending verified confirmation outputs and completed evaluation. The expected 72 core ERPs are not treated as observations.',
        '## Distribution comparisons',
        'Group2 currently contains %d aggregate rows and %d paired-bootstrap rows. Tables are report/tables/group2_metrics.csv and group2_paired.csv. Reference identities, image counts, subset sizes and failed bootstrap replicates remain in the source records.'%(len(groups['group2']),len(pairs['group2'])),
        'Real ODI-SR24, historical SUN360, ordinary-native A/B and native-to-native A/B are distinct reference questions. No unavailable FAED/Distort value is filled with zero.',
        '## VLM evidence and actual API usage',
        'Pilot: %d/%d schema-valid responses. Production aggregate rows: %d. Judges remain separate. The adapted rubric is supplementary subjective evidence.'%(pilot.get('valid_requests',0),pilot.get('planned_requests',30),len(groups['vlm'])),
        '\n'.join('- '+text for text in calibration.get('limitations',[])),
        _table(usage.get('by_stage_and_judge',[]),['stage','judge','attempts','actual_cost_usd','uncertain_reserved_cost_usd','states']),
        'Separate paper tables: report/tables/gpt4o.csv, gpt6sol.csv, judge_agreement.csv and repeat_reliability.csv. Real and native distributions are separated into real_distribution.csv and native_distribution.csv; main objective metrics, runtime and ablations have their own files.',
        'Budget mode: '+str(usage.get('budget_mode','unavailable'))+'. Provider-account limits govern calls under the current user authorization. Uncertain reservations, if any, are reported separately from actual billed costs. API credentials are not included.',
        '## Efficiency and representative views',
        'Recorded runtime components, memory, denoiser/VAE counters and settings are in report/tables/runtime.csv. Quality-time plots use only identified hardware and separate native-resolution/denoising-step strata; setup is excluded from generation seconds unless explicitly captured by source records.',
        'Three representative prompts are chosen by a fixed hash ordering of development15 prompt IDs before looking at scores. Their front, wrap seam, north and south crops use identical common_v2 cameras. Selection and every source hash are recorded in report/figures/representative-selection.json and adjacent provenance files.',
        '## Human evaluation',
        'The local viewer is human/participant/index.html; serve only its participant folder on 127.0.0.1 as described in human/participant_instructions.txt. There are 240 anonymous balanced assignments: 24 scenes × 2 comparisons × 5 raters. Each has separate A/B/tie questions for quality, geometry/continuity and prompt adherence. Both panorama views share camera movement and front/seam/pole presets. Assets use the same 2048x1024 ERP display resolution.',
        ('The viewer was exercised in Chromium '+browser_validation.get('browser_version','')+' using synthetic localhost-only fixtures: '+str(len(browser_validation.get('checks',[])))+' checks covered synchronized WebGL pixels, camera presets and drag, separate A/B/tie questions, CSV export, pending-pair protection, rapid navigation, projection aspect ratio and basic labels/narrow layout. No page JavaScript errors or external page requests occurred, and real human observations remained unchanged. This is a software workflow check, not a human result or full accessibility certification.' if browser_checked else 'Browser workflow validation is '+browser_validation.get('state','pending')+'; inspect human/browser-validation/status.json for exact evidence or blockers.'),
        'The method key is human/private/method_key.json outside the participant folder. Incomplete images or unreviewed descriptions disable ratings. No recruitment, publication or fabricated ratings occurred. Raw input CSVs are retained by content hash; analysis retains ties and uses 2000 crossed prompt-and-rater bootstrap replicates when there are enough observed clusters.',
        '## Claims and limitations',_table(claims,['claim','state','evidence']), '\n'.join('- '+text for text in limitations),
        '## Pending work and reproducibility',
        '%d pending items, with exact case/assignment IDs and metric labels, are listed in report/tables/pending.json. Native and confirmation case IDs are distinguished from planned logical slots if the generation manifest is not yet frozen.'%len(pending),
        'The job-accounting snapshot is report/scheduler_snapshot.json. Actual job states are included below; historical failed environment jobs remain in the record. RESUME.md contains executable local inspection/rebuild commands and points to the bounded shared-resource launcher.',
        _table(scheduler['accounting'],['job_id','state','exit_code','elapsed','nodes']),
        'Reproduction sources, original image/metadata hashes, registry identity and every input-result hash are recorded in report/traceability.json. Rebuilding this report performs no model generation or API calls.'
    ]
    markdown='\n\n'.join(parts)+'\n';atomic_text(run/'report.md',markdown)
    tex_rows=[]
    for row in primary:
        interval=row.get('ci95');ci='--' if not interval or any(value is None for value in interval) else '[%.3f, %.3f]'%tuple(interval)
        tex_rows.append(' & '.join(_tex(v) for v in (row['metric'],LABELS.get(row['method_id'],row['method_id']),'--' if row.get('mean') is None else '%.3f'%row['mean'],ci,row['prompt_count']))+r' \\')
    atomic_text(run/'results.tex','% Partial snapshot '+summary['built_utc']+'\n% Registry '+summary['registry_sha256']+'\n\\begin{tabular}{llrrr}\nMetric & Method & Mean & 95\\% CI & Prompts \\\\\n\\hline\n'+'\n'.join(tex_rows)+'\n\\end{tabular}\n% Other namespaces, references, judges, and complete machine-readable tables: report/tables/.\n')
    atomic_text(run/'methods_evaluation.tex','% Generated from the frozen evaluation protocol; no claim of completed pending experiments.\n'+ '\n\n'.join(_tex(text) for text in limitations[:9]+[parts[parts.index('## Human evaluation')+1]])+'\n')
    _pdf(run/'report.pdf',markdown,figure_records)
    commands='''# Resume and inspect this partial evaluation

This file is a report snapshot, not authorization to duplicate active jobs. Read report/scheduler_snapshot.json and scheduler.json first. The existing shared launcher enforces GPU/API resource limits. Keep the frozen protocol and old outputs unchanged.

```bash
cd %s
source studies/paper_eval_v2/clean_environment.sh
bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2 --run %s status
bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.snapshot --launch --run %s
bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.human --run %s prepare
bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.human --run %s analyze
# After actual ratings have been collected locally:
# bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.human --run %s ingest /absolute/path/ratings.csv
```

Exact unfinished native/confirmation/metric/assignment IDs: report/tables/pending.json. The confirmed submission history and dependencies are preserved in scheduler.json; accounting at report time is report/scheduler_snapshot.json. Resume worker stages only through the existing resource-aware launcher after checking current queue state; do not resubmit already-running arrays. Missing compatible FAED and Distort assets remain blockers. Human collection requires real participants and has not been performed.

Current generation, caption, native and production-scoring progress can change after this report. Rebuild the report after completed workers have published their verified records. Rebuilding makes no model/API calls.
'''%(ROOT,run,run,run,run,run)
    controller_command='bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.controller resume --run '+str(run)+' --hours 24 --interval 60'
    component_commands=[dict(component=name,command=controller_command,condition='Reconcile controller/state.json and actual scheduler jobs first; finite shared-resource resume, not duplicate submission') for name in ('confirmation_generation','confirmation_metrics','confirmation_API','confirmation_report')]
    component_commands.extend([
        dict(component='native_generation',command='bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.resume_native --run '+str(run),condition='Verify incomplete native groups; refuses conflicting generation allocations; preserves prior attempts'),
        dict(component='partial_report',command='bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.snapshot --launch --run '+str(run),condition='Submit one finite CPU report snapshot; reuse any active report job; no paid API calls'),
        dict(component='VLM_manifest_only',command='bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.prepare_production --run '+str(run),condition='After shared views are complete; canonical-equality adapter only; does not bill API calls'),
        dict(component='FAED_and_Distort',command='bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2 --run '+str(run)+' status',condition='Inspect recorded blocker; compatible released asset/implementation must be supplied and separately verified before scoring'),
        dict(component='human_ratings',command='bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.human --run '+str(run)+' analyze',condition='Await real locally collected CSVs; ingest each actual CSV first as shown above')])
    write(directory/'resume_commands.json',component_commands)
    commands+='\n## Finite continuation and per-component commands\n\n'+_table(component_commands,['component','command','condition'])+'\n\nThe controller CLI can continue within its existing deadline inside a CPU allocation with `bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.controller run --run '+str(run)+' --hours24 --interval60` (use `--hours 24 --interval 60`). After a finite deadline or error, use the journaled `resume` command above; plain `run` preserves the old deadline. `resume` archives the prior epoch and refuses if it is still active. Persistent controller state is controller/state.json; batches and finalizations are retained under controller/. A completed finite controller cannot operate after its allocation ends. It retains scientific failures as blockers instead of silently retrying altered science.\n'
    commands=commands.replace('--hours24 --interval60','--hours 24 --interval 60').replace(' (use `--hours 24 --interval 60`)','')
    atomic_text(run/'RESUME.md',commands)
    update_component(run,'report','partial',report_md=str(run/'report.md'),report_pdf=str(run/'report.pdf'),pending_items=len(pending),built_utc=summary['built_utc'])
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUN);args=parser.parse_args()
    print(json.dumps(report(args.run),indent=2))

if __name__=='__main__':main()
