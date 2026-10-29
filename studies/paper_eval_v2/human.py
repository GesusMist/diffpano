"""Offline blinded panorama study; preparation never fabricates human ratings."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from .common import ROOT, RUN, read, write, sha, digest, atomic_text, csv_write, lock, update_component

QUESTIONS = ('quality', 'geometry_continuity', 'prompt_adherence')
COMPARISONS = (('select','sphere20'), ('select','spherope'))
FIELDS = ('assignment_id','participant_id','rater_slot','quality','geometry_continuity',
          'prompt_adherence','yaw','pitch','elapsed_seconds','submitted_at_utc','protocol_sha256')
PROTOCOL = dict(version='human_common_v2_v1',cohort='confirmation24',scene_count=24,
    comparisons=2,ratings_per_pair=5,expected_pair_evaluations=240,
    questions=dict(quality='Overall visual quality',geometry_continuity='Geometry and continuity',
                   prompt_adherence='Prompt adherence'),responses=['A','B','tie'],
    display=dict(erp=[2048,1024],projection='synchronized perspective viewport',fov=90,
                 yaw_pitch='shared',presets=['front','seam','north','south'],native_detail=False),
    assignment='five rater slots; each sees48 pairs; alternate side by scene/comparison/slot, then deterministic hash order',
    uncertainty='2000 crossed prompt-and-rater bootstrap replicates; ties retained; no interval with fewer than two scenes or raters',
    bootstrap_seed=20261026,collection='offline; no recruitment, external transmission, or publication')


def assignments(scenes):
    if len(scenes)!=24 or len({row['prompt_id'] for row in scenes})!=24:
        raise ValueError('Human study requires24 unique frozen confirmation scenes')
    protocol_hash=digest(PROTOCOL)
    study_id=digest(dict(protocol=protocol_hash,scenes=[row['prompt_id'] for row in scenes]))[:16]
    public,private=[],{}
    for scene_index,scene in enumerate(scenes):
        scene_token='scene-%03d'%(scene_index+1)
        for comparison_index,(ours,baseline) in enumerate(COMPARISONS):
            pair_id=digest([study_id,scene['prompt_id'],comparison_index])[:16]
            for slot in range(5):
                ours_left=(scene_index+comparison_index+slot)%2==0
                methods=(ours,baseline) if ours_left else (baseline,ours)
                aid=digest([study_id,pair_id,slot])[:20]
                assets=[digest([study_id,scene['prompt_id'],method])[:24] for method in methods]
                public.append(dict(assignment_id=aid,pair_id=pair_id,scene_id=scene_token,rater_slot=slot+1,
                    left_asset=assets[0],right_asset=assets[1],protocol_sha256=protocol_hash))
                private[aid]=dict(prompt_id=scene['prompt_id'],original_scene_id=scene['scene_id'],
                    baseline=baseline,left_method=methods[0],right_method=methods[1],
                    pair_id=pair_id,rater_slot=slot+1,asset_ids=assets)
    public.sort(key=lambda row:(row['rater_slot'],digest([study_id,row['assignment_id']])))
    assert len(public)==240 and sum(private[row['assignment_id']]['left_method']=='select' for row in public)==120
    return study_id,public,private


def _records(run):
    result=[]
    for path in (run/'inventory/records.json',run/'generation/records.json',run/'generation/verified_records.json',run/'generation/inventory_records.json'):
        if not path.exists():continue
        value=read(path)
        result.extend(value.get('records',value.get('rows',[])) if isinstance(value,dict) else value)
    return result


def prepare(run=RUN, records=None):
    run=Path(run);directory=run/'human';participant=directory/'participant';private_dir=directory/'private'
    participant.mkdir(parents=True,exist_ok=True);private_dir.mkdir(exist_ok=True)
    manifest=read(run/'real_dataset_manifest.json')
    scenes=manifest['rows']
    study_id,planned,key=assignments(scenes)
    captions=read(run/'captions.json') if (run/'captions.json').exists() else {}
    caption_rows={row['prompt_id']:row for row in captions.get('rows',[])} if captions.get('state')=='complete' and captions.get('reviewed') else {}
    by_prompt={row['prompt_id']:row for row in scenes}
    candidates={}
    for row in (_records(run) if records is None else records):
        if row.get('state')!='complete' or row.get('prompt_id') not in by_prompt or row.get('method_id') not in ('select','sphere20','spherope'):
            continue
        if row.get('seed',row.get('parameters',{}).get('seed',0))!=0:continue
        identity=(row['prompt_id'],row['method_id'])
        if identity in candidates and candidates[identity]['image_sha256']!=row['image_sha256']:
            raise ValueError('Ambiguous confirmation source for '+str(identity))
        candidates[identity]=row
    previous=read(private_dir/'asset_provenance.json') if (private_dir/'asset_provenance.json').exists() else {}
    assets={};provenance={};asset_dir=participant/'assets';asset_dir.mkdir(exist_ok=True)
    for aid,assignment in key.items():
        for method,token in zip((assignment['left_method'],assignment['right_method']),assignment['asset_ids']):
            if token in assets:continue
            row=candidates.get((assignment['prompt_id'],method))
            if row is None:
                assets[token]=dict(state='pending_generation',image_path=None)
                continue
            target=asset_dir/(token+'.png')
            old=previous.get(token,{})
            if target.exists() and old.get('source_image_sha256')==row['image_sha256'] and sha(target)==old.get('display_sha256'):
                record=old
            else:
                import numpy as np
                from PIL import Image
                from .rendering import resample_common, protocol_definitions
                assert sha(row['image_path'])==row['image_sha256']
                if row.get('metadata_path') and row.get('metadata_sha256'):assert sha(row['metadata_path'])==row['metadata_sha256']
                with Image.open(row['image_path']) as source:
                    source_size=list(source.size)
                    output=resample_common(np.asarray(source.convert('RGB')))
                temporary=target.with_suffix('.tmp')
                Image.fromarray(output).save(temporary,format='PNG',compress_level=6,optimize=False)
                temporary.replace(target)
                record=dict(source_image_path=row['image_path'],source_image_sha256=row['image_sha256'],
                    source_resolution=source_size,display_sha256=sha(target),display_resolution=[2048,1024],
                    renderer_protocol_sha256=digest(protocol_definitions()['common_v2']),case_id=row['case_id'],method_id=method)
            assets[token]=dict(state='ready',image_path='assets/'+target.name,image_sha256=record['display_sha256'])
            provenance[token]=record
    for row in planned:
        private=key[row['assignment_id']]
        caption=caption_rows.get(private['prompt_id'])
        row['prompt_description']=caption['global_generation_description'] if caption else None
        row['state']='ready' if caption and all(assets[row[side]]['state']=='ready' for side in ('left_asset','right_asset')) else 'pending_assets_or_reviewed_caption'
    public=dict(study_id=study_id,protocol_sha256=digest(PROTOCOL),questions=PROTOCOL['questions'],
        display=PROTOCOL['display'],assignments=planned,assets=assets,
        warning='Ratings remain disabled until both images and the reviewed description are available. Do not enter your name or contact details.')
    # Participant files contain only anonymous image identifiers and display text.
    atomic_text(participant/'manifest.js','window.HUMAN_STUDY = '+json.dumps(public,ensure_ascii=True)+';\n')
    for path in (ROOT/'studies/paper_eval_v2/human_assets').iterdir():
        if path.is_file():atomic_text(participant/path.name,path.read_text())
    write(private_dir/'method_key.json',dict(study_id=study_id,protocol=PROTOCOL,assignments=key))
    write(private_dir/'asset_provenance.json',provenance)
    write(directory/'assignments.json',public)
    write(directory/'rating_schema.json',dict(fields=list(FIELDS),choices=['A','B','tie'],
        participant_id='anonymous code only; each participant assigned one unique rater slot',
        no_placeholder_rows=True,method_key='private/method_key.json; do not share with participants'))
    csv_write(directory/'ratings_template.csv',[],FIELDS)
    instructions='''Serve only the participant folder on loopback, then open http://127.0.0.1:8765/index.html in a browser. From this study directory run: python3 -m http.server 8765 --bind 127.0.0.1 --directory participant. This uses no external hosting; some browsers prohibit WebGL textures when opening file:// images directly. Use the anonymous participant code and assigned rater slot provided by the study organizer; enter no name or contact details.
Drag either panorama to move both views together. Inspect front, seam, north, and south presets. Both images use2048x1024 evaluation ERPs and the same90-degree view. Native source resolution may differ; this task is not a native-detail comparison.
For each pair independently choose A, B, or tie for overall visual quality; geometry and continuity; and adherence to the displayed description. Artistic content alone is not a geometric error. Natural object boundaries are not seams. Do not infer a method identity.
Only ready pairs can be rated. Save each pair, then export your CSV. Ratings stay in this browser until you choose Export CSV. Send the file to the study organizer through the agreed local process. No participant recruitment or publication is performed by this software.
The organizer must keep the private method key outside the participant-facing folder. Five distinct participants rate each pair; missing ratings remain missing.
'''
    atomic_text(directory/'participant_instructions.txt',instructions)
    existing=directory/'ratings/observations.json'
    count=len(read(existing)) if existing.exists() else 0
    status=dict(materials_ready=True,viewer=str(participant/'index.html'),expected_pair_evaluations=240,
        ready_assignments=sum(row['state']=='ready' for row in planned),pending_assignment_ids=[row['assignment_id'] for row in planned if row['state']!='ready'],
        ready_anonymous_assets=sum(row['state']=='ready' for row in assets.values()),expected_anonymous_assets=72,
        observed_ratings=count,state='awaiting_collection' if count==0 else 'collected' if count==240 else 'partially_collected',
        no_recruitment=True,no_publication=True,no_fabricated_ratings=True,protocol_sha256=digest(PROTOCOL))
    write(directory/'status.json',status)
    update_component(run,'human_ratings',status['state'],ratings=count,expected=240,materials_ready=True)
    return status


def ingest(run, path):
    run=Path(run);path=Path(path);directory=run/'human/ratings';directory.mkdir(parents=True,exist_ok=True)
    assignments_by_id={row['assignment_id']:row for row in read(run/'human/assignments.json')['assignments']}
    with path.open(newline='') as stream:incoming=list(csv.DictReader(stream))
    with lock(directory/'ingest.lock'):
        previous=read(directory/'observations.json') if (directory/'observations.json').exists() else []
        by_key={(row['participant_id'],row['assignment_id']):row for row in previous}
        participants={row['participant_id']:int(row['rater_slot']) for row in previous}
        slots={int(row['rater_slot']):row['participant_id'] for row in previous}
        for row in incoming:
            if any(field not in row for field in FIELDS):raise ValueError('Rating fields missing')
            assignment=assignments_by_id.get(row['assignment_id'])
            if assignment is None or assignment['state']!='ready':raise ValueError('Unknown or unavailable assignment')
            if row['protocol_sha256']!=assignment['protocol_sha256']:raise ValueError('Protocol identity differs')
            if any(row[key] not in ('A','B','tie') for key in QUESTIONS):raise ValueError('Invalid A/B/tie rating')
            participant=row['participant_id']
            if not participant or len(participant)>64 or any(not(c.isalnum() or c in '-_') for c in participant):raise ValueError('Use an anonymous alphanumeric participant code')
            slot=int(row['rater_slot'])
            if slot!=assignment['rater_slot']:raise ValueError('Rater slot mismatch')
            if participant in participants and participants[participant]!=slot:raise ValueError('Participant assigned multiple slots')
            if slot in slots and slots[slot]!=participant:raise ValueError('Slot assigned multiple participants')
            participants[participant]=slot;slots[slot]=participant
            for field in ('yaw','pitch','elapsed_seconds'):
                number=float(row[field])
                if not __import__('math').isfinite(number):raise ValueError('Nonfinite interaction data')
            if float(row['elapsed_seconds'])<0 or not -90<=float(row['pitch'])<=90:raise ValueError('Invalid viewing record')
            identity=(participant,row['assignment_id'])
            if identity in by_key and by_key[identity]!=row:raise ValueError('Conflicting duplicate rating; preserve both source CSVs for review')
            by_key[identity]=dict(row)
        raw=directory/'raw';raw.mkdir(exist_ok=True)
        target=raw/(sha(path)+'.csv')
        if not target.exists():target.write_bytes(path.read_bytes())
        observations=sorted(by_key.values(),key=lambda row:(row['participant_id'],row['assignment_id']))
        write(directory/'observations.json',observations)
    return analyze(run)


def analyze(run):
    import numpy as np
    run=Path(run);directory=run/'human';path=directory/'ratings/observations.json'
    observations=read(path) if path.exists() else []
    private=read(directory/'private/method_key.json')['assignments']
    results=[]
    for baseline in ('sphere20','spherope'):
        selected=[row for row in observations if private[row['assignment_id']]['baseline']==baseline]
        scenes=sorted({private[row['assignment_id']]['prompt_id'] for row in selected});raters=sorted({row['participant_id'] for row in selected})
        for criterion in QUESTIONS:
            values=[]
            for row in selected:
                key=private[row['assignment_id']];vote=row[criterion]
                winner=None if vote=='tie' else key['left_method' if vote=='A' else 'right_method']
                values.append((key['prompt_id'],row['participant_id'],0 if winner is None else 1 if winner=='select' else -1))
            n=len(values);wins=sum(v==1 for _,_,v in values);losses=sum(v==-1 for _,_,v in values);ties=n-wins-losses
            result=dict(comparison='select versus '+baseline,criterion=criterion,ratings=n,expected=120,
                prompts=len(scenes),raters=len(raters),wins=wins,losses=losses,ties=ties,
                missing_ratings=120-n,net_preference=None if not n else (wins-losses)/n,
                preference_excluding_ties=None if wins+losses==0 else wins/(wins+losses),ci95=None,
                uncertainty='awaiting_collection' if not n else 'insufficient_prompt_or_rater_clusters')
            if len(scenes)>=2 and len(raters)>=2:
                rng=np.random.RandomState(20261026);draws=[]
                for unused in range(2000):
                    sw=np.bincount(rng.randint(len(scenes),size=len(scenes)),minlength=len(scenes));rw=np.bincount(rng.randint(len(raters),size=len(raters)),minlength=len(raters))
                    numerator=denominator=0.
                    for scene,rater,value in values:
                        weight=sw[scenes.index(scene)]*rw[raters.index(rater)];numerator+=weight*value;denominator+=weight
                    if denominator:draws.append(numerator/denominator)
                result.update(ci95=np.percentile(draws,[2.5,97.5]).tolist() if len(draws)==2000 else None,
                    bootstrap_replicates=len(draws),requested_replicates=2000,
                    uncertainty='crossed prompt-and-rater bootstrap' if len(draws)==2000 else 'incomplete_bootstrap_due_to_sparse_cells')
            results.append(result)
    write(directory/'analysis.json',dict(state='awaiting_collection' if not observations else 'complete' if len(observations)==240 else 'partial',
        observations=len(observations),results=results,ties_retained=True,no_fabricated_ratings=True))
    csv_write(directory/'analysis.csv',results,sorted({key for row in results for key in row}))
    return dict(observations=len(observations),results=results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUN)
    sub=parser.add_subparsers(dest='action');sub.add_parser('prepare');sub.add_parser('analyze')
    p=sub.add_parser('ingest');p.add_argument('csv',type=Path)
    args=parser.parse_args()
    result=prepare(args.run) if args.action=='prepare' else ingest(args.run,args.csv) if args.action=='ingest' else analyze(args.run)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
