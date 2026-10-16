"""Freeze 120 exact standalone identities before any GPU generation."""
import argparse,tarfile
from studies.perspective24.common import *
from studies.perspective24.configuration import semantic,BASELINE,WEIGHTS

def historical_native_candidates(rows):
    """Check actual earlier native-control provenance, not its figure filenames."""
    target_texts={r['line_text'] for r in rows};records=[]
    for name in ('9.7native-controls','9.9endpoint-controls','9.9rgb-endpoint-controls'):
        for path in sorted((ROOT/'outputs'/name).rglob('*.json')):
            if not any('flux' in part for part in path.parts):continue
            if path.name not in ('metadata.json','trajectory.json','A_control.json','B_control.json','C_control.json','config.json'):continue
            value=read(path);config=value.get('config',value.get('configuration',{}));direct=value.get('prompt')
            prompt_path=config.get('prompt',{}).get('path') if isinstance(config.get('prompt'),dict) else None
            exact=isinstance(direct,str) and direct in target_texts
            records.append(dict(path=str(path),sha256=sha(path),recorded_prompt=direct,recorded_prompt_path=prompt_path,
                exact_target_line_match=exact,accepted=False,
                reason='Excluded native_control prompt / native-canvas diagnostic; not an exact target line standalone sample'))
            assert not exact,'A potential exact historical line needs a full standalone semantic audit before freezing: '+str(path)
    return records

def prepare():
    assert not (OUT/'manifest.json').exists(),'Refusing to overwrite a production freeze'
    source=sources();rows=[]
    for row in definitions():
        settings=semantic(row);row.update(semantic=settings,semantic_sha256=digest(settings),source_sha256=digest(source));rows.append(row)
    assert len(rows)==len({r['case_id'] for r in rows})==len({r['semantic_sha256'] for r in rows})==120
    prompts={p:prompt_record(p) for p in prompt_names()}
    weights=read(WEIGHTS);assert weights['all_shared_files_identical']
    for record in weights['files']:
        p=Path(record['methods']['diffpano']['path']);assert p.is_file() and p.stat().st_size==record['methods']['diffpano']['bytes']
    inputs={str(p):sha(p) for p in (BASELINE,WEIGHTS)}
    for sub in ('logs/generation','logs/validation','validation','history'): (OUT/sub).mkdir(parents=True,exist_ok=True)
    search_roots=('outputs/all-prompts-erp-cea-seed0','outputs/flux-step-controls-seed0','outputs/camera-patching-seed0',
        'outputs/camera-count-erp-seed0-v2','outputs/10.7gradient-blending-flux','outputs/10.9ournew-flux28','outputs/mean-max-old89-flux90')
    searched={name:dict(exists=(ROOT/name).exists(),resolved=str((ROOT/name).resolve())) for name in search_roots}
    audit=[dict(index=r['index'],case_id=r['case_id'],decision='generate',candidate=None,
        reason='No verified independent exact-line Gaussian native FLUX20 artifact; panorama ERPs/rendered views are ineligible',reused=False) for r in rows]
    native_candidates=historical_native_candidates(rows)
    write(OUT/'reuse_audit.json',dict(task='A',searched=searched,historical_native_candidates=native_candidates,rows=audit,reused=0,generate=120,blocked=0))
    write(OUT/'generation_plan.json',dict(artifact_kind='standalone_perspective',expected=120,reused=[],generate=list(range(120)),blocked=[],
        grouping='assigned index groups share loaded model and cached exact-line embeddings; batch size always one',max_generation_gpu_jobs_across_A_B=2))
    write(OUT/'manifest.json',dict(task='A',artifact_kind='standalone_perspective',expected=120,rows=rows,prompts=prompts,source_hashes=source,
        input_artifacts=inputs,weight_audit=weights,outputs_symlink=os.readlink(ROOT/'outputs'),reused=0,generate=120,
        implementation_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),
        worktree_status=subprocess.check_output(['git','status','--short'],cwd=str(ROOT)).decode(),expected_counts=COUNTS,
        expected_guided_steps=20,metrics='none: standalone views are not panorama metric inputs'))
    with tarfile.open(str(OUT/'history/frozen-source.tar.gz'),'w:gz') as archive:
        for name in source:archive.add(str(ROOT/name),arcname=name)
    write(OUT/'history/freeze.json',dict(source_sha256=digest(source),manifest_sha256=sha(OUT/'manifest.json'),archive_sha256=sha(OUT/'history/frozen-source.tar.gz')))
    print('PREPARED 120 standalone images; zero reused; zero blocked',flush=True)

if __name__=='__main__':prepare()
