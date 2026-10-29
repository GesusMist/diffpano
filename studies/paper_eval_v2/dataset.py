"""Inspect official ODI-SR safely and freeze a score-independent real cohort.

RAR members are streamed to memory, never expanded to archive-controlled paths.
Only selected originals and small review sheets are retained on disk.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import urllib.request

from .common import ROOT, RUN, read, sha, digest, write, freeze, now

VERSION = '26.05'
ASSET = '7z2605-linux-x64.tar.xz'
RELEASE_API = 'https://api.github.com/repos/ip7z/7zip/releases/tags/' + VERSION


def safe_member(name):
    value = PurePosixPath(name.replace('\\', '/'))
    if value.is_absolute() or '..' in value.parts or not value.parts or ':' in value.parts[0] or '\n' in name or '\r' in name:
        raise ValueError('Unsafe archive member name')
    return str(value)


def ensure_tool(run=RUN):
    destination = Path(run) / 'dataset/tools/7zip-26.05'
    provenance = destination / 'provenance.json'
    if provenance.exists():
        record = read(provenance)
        for filename, value in record['files'].items():
            if sha(destination / filename) != value:
                raise ValueError('Pinned archive tool changed: ' + filename)
        return destination / '7zz'
    with urllib.request.urlopen(RELEASE_API, timeout=60) as response:
        release = json.load(response)
    matches = [item for item in release['assets'] if item['name'] == ASSET]
    if len(matches) != 1:
        raise ValueError('Expected exact official7zip asset')
    asset = matches[0]
    with urllib.request.urlopen(asset['browser_download_url'], timeout=120) as response:
        raw = response.read()
    value = hashlib.sha256(raw).hexdigest()
    if asset.get('digest') and asset['digest'] != 'sha256:' + value:
        raise ValueError('Publisher release asset hash differs')
    destination.mkdir(parents=True, exist_ok=True)
    saved = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:xz') as archive:
        for name in ('7zz', 'License.txt'):
            candidates = [member for member in archive.getmembers() if PurePosixPath(member.name).name.lower() == name.lower()]
            if len(candidates) != 1 or not candidates[0].isfile():
                raise ValueError('Expected one regular tool/license member: ' + name)
            data = archive.extractfile(candidates[0]).read()
            target = destination / name
            target.write_bytes(data)
            if name == '7zz':
                target.chmod(0o755)
            saved[name] = hashlib.sha256(data).hexdigest()
    freeze(provenance, dict(version=VERSION, source=RELEASE_API, release_tag=release['tag_name'],
        asset_url=asset['browser_download_url'], asset_sha256=value, publisher_digest=asset.get('digest'),
        files=saved, acquired_utc=now(), extraction='Only exact regular7zz executable and license; no archive paths materialized'))
    return destination / '7zz'


def parse_listing(text):
    rows = []
    for block in text.replace('\r\n', '\n').split('\n\n'):
        row = {}
        for line in block.splitlines():
            if ' = ' in line:
                key, value = line.split(' = ', 1)
                row[key] = value
        if 'Path' not in row:
            continue
        name = safe_member(row['Path'])
        if row.get('Folder') == '+' or row.get('Attributes', '').startswith('D'):
            continue
        if 'Size' not in row:
            continue
        rows.append(dict(member=name, bytes=int(row['Size']), crc32=row.get('CRC'),
                         encrypted=row.get('Encrypted'), attributes=row.get('Attributes'),
                         method=row.get('Method'), solid=row.get('Solid')))
    if len({row['member'] for row in rows}) != len(rows):
        raise ValueError('Duplicate archive member paths')
    return rows


def inspect(run=RUN):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Inspect dataset on a CPU allocation')
    run = Path(run)
    archive_record = read(run / 'dataset/odisr-archive.json')
    archive = Path(archive_record['path'])
    if sha(archive) != archive_record['sha256'] or archive.stat().st_size != archive_record['bytes']:
        raise ValueError('Downloaded dataset archive changed')
    tool = ensure_tool(run)
    result = subprocess.run([str(tool), 'l', '-slt', '-ba', str(archive)], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rows = parse_listing(result.stdout)
    from collections import Counter
    record = dict(dataset='ODI-SR', source_release='official LAU-Net public release',
        archive=archive_record, tool_provenance=str(tool.parent/'provenance.json'),
        tool_sha256=sha(tool), members=rows, member_count=len(rows),
        suffix_counts=dict(Counter(PurePosixPath(row['member']).suffix.lower() for row in rows)),
        directory_counts=dict(Counter(str(PurePosixPath(row['member']).parent) for row in rows)),
        source_split_policy='Inspect actual archive names before declaring split; no result-dependent choice',
        inspected_utc=now(), job=os.environ['SLURM_JOB_ID'])
    freeze(run/'dataset/archive-inventory.json', record)
    print(json.dumps({key:value for key,value in record.items() if key not in ('members','archive')}, indent=2), flush=True)
    return record


def stream_members(run, selected):
    """Yield exact selected regular-file bytes in original archive order."""
    import zlib
    run = Path(run)
    inventory = read(run/'dataset/archive-inventory.json')
    lookup = {row['member']: row for row in inventory['members']}
    if not selected or len(set(selected)) != len(selected) or any(name not in lookup for name in selected):
        raise ValueError('Requested members must be a unique inventoried set')
    rows = [row for row in inventory['members'] if row['member'] in set(selected)]
    tool = ensure_tool(run)
    # Literal include switches avoid wildcard interpretation and filesystem writes.
    args = [str(tool), 'x', '-so', '-bd', '-bb0', '-spd', '--', inventory['archive']['path']] + [safe_member(row['member']) for row in rows]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for row in rows:
            chunks, remaining = [], row['bytes']
            while remaining:
                data = process.stdout.read(min(remaining, 1024*1024))
                if not data:
                    raise ValueError('Premature archive output: ' + row['member'])
                chunks.append(data)
                remaining -= len(data)
            data = b''.join(chunks)
            if row['crc32'] and format(zlib.crc32(data)&0xffffffff, '08X') != row['crc32'].upper():
                raise ValueError('Stream order/CRC mismatch: ' + row['member'])
            yield row, data
        if process.stdout.read(1):
            raise ValueError('Unexpected additional archive bytes')
        errors = process.stderr.read().decode('utf-8', 'replace')
        if process.wait() != 0:
            raise ValueError('7zip extraction failed: ' + errors[:1000])
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()



def preview(run=RUN):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Decode real dataset on a CPU allocation')
    from PIL import Image, ImageDraw
    import numpy as np
    run = Path(run)
    inventory = read(run/'dataset/archive-inventory.json')
    members = [row['member'] for row in inventory['members'] if row['member'].startswith('lau_dataset/odisr/testing/HR/')]
    if len(members) != 100:
        raise ValueError('Expected exact100 released ODI-SR testing HR images')
    policy = dict(cohort='confirmation24', label='our ODI-SR-24 subset', source_split='odisr/testing/HR',
        candidate_count=100, target_categories=dict(indoor=6,outdoor=18),
        selection_order='SHA256(selection-v1|archive_sha256|member), ascending within content category',
        category_review='Inspect every candidate contact sheet; indoor/outdoor from visible dominant scene, uncertain excluded.',
        duplicate_policy='Exact pixel duplicates excluded; visually identifiable repeated captures assigned one scene_group before selection; first hash-ranked capture retained.',
        exclusions='Unreadable/nonRGB/non2:1/uncertain category or repeated scene only; no aesthetic or model-score exclusions.',
        second_seed='First2 indoor and6 outdoor selected scenes ranked by SHA256(second-seed-v1|member)',
        independent_of_generated_outputs=True)
    freeze(run/'dataset/selection-protocol.json',policy)
    sheets=[Image.new('RGB',(2000,1180),'white') for _ in range(4)]
    rows=[]
    for index,(member,data) in enumerate(stream_members(run,members)):
        with Image.open(io.BytesIO(data)) as image:
            if image.mode!='RGB' or image.width!=2*image.height:
                raise ValueError('Expected an RGB2:1 testing ERP: '+member['member'])
            image.load()
            pixels=image.tobytes()
            tiny=np.asarray(image.resize((64,32),Image.Resampling.BILINEAR),dtype=np.uint8)
            row=dict(index=index,member=member['member'],image_sha256=hashlib.sha256(data).hexdigest(),
                pixel_sha256=hashlib.sha256(pixels).hexdigest(),width=image.width,height=image.height,mode=image.mode,
                bytes=len(data),selection_key=hashlib.sha256(('selection-v1|'+inventory['archive']['sha256']+'|'+member['member']).encode()).hexdigest())
            thumbnail=image.resize((400,200),Image.Resampling.LANCZOS)
        page=index//25;slot=index%25;x=(slot%5)*400;y=(slot//5)*236
        sheets[page].paste(thumbnail,(x,y+30))
        ImageDraw.Draw(sheets[page]).text((x+8,y+7),'%03d %s'%(index,PurePosixPath(member['member']).name),fill='black')
        row['review_sheet']=page
        row['thumbnail_rgb_64x32']=tiny.tolist()
        rows.append(row)
    from collections import defaultdict
    duplicate=defaultdict(list)
    for row in rows:duplicate[row['pixel_sha256']].append(row['index'])
    nearest=[]
    arrays=[np.asarray(row.pop('thumbnail_rgb_64x32'),dtype=np.float32)/255. for row in rows]
    for i in range(len(rows)):
        candidates=[(float(np.mean((arrays[i]-arrays[j])**2)),j) for j in range(i)]
        if candidates:
            error,j=min(candidates)
            if error<.015:nearest.append(dict(first=j,second=i,thumbnail_mse=error,policy='Review aid only; not automatic scene exclusion'))
    sheet_records=[]
    for index,sheet in enumerate(sheets):
        path=run/'dataset'/('candidate-contact-%02d.png'%index)
        sheet.save(path,compress_level=6)
        sheet_records.append(dict(path=str(path),sha256=sha(path)))
    record=dict(policy_sha256=digest(policy),archive_sha256=inventory['archive']['sha256'],
        source_split='odisr/testing/HR',candidates=rows,contact_sheets=sheet_records,
        exact_pixel_duplicate_groups=[group for group in duplicate.values() if len(group)>1],
        similar_thumbnail_pairs=nearest,classification='pending content review; no image generation or metrics consulted')
    freeze(run/'dataset/candidate-review.json',record)
    print('PREVIEW_COMPLETE',len(rows),sheet_records,flush=True)
    return record


def select_rows(review, annotations):
    lookup={row['member']:row for row in review['candidates']}
    if len(annotations)!=len(lookup) or {row['member'] for row in annotations}!=set(lookup):
        raise ValueError('Content review must cover every candidate exactly once')
    selected=[];seen=set();pixels=set()
    for category,count in (('indoor',6),('outdoor',18)):
        candidates=sorted((a for a in annotations if a['category']==category and not a.get('excluded_reason')),
                          key=lambda a:lookup[a['member']]['selection_key'])
        for annotation in candidates:
            image=lookup[annotation['member']]
            if annotation['image_sha256']!=image['image_sha256']:
                raise ValueError('Content annotation image hash differs')
            if annotation['scene_group'] in seen or image['pixel_sha256'] in pixels:continue
            selected.append(dict(image,**{k:v for k,v in annotation.items() if k not in image}))
            seen.add(annotation['scene_group']);pixels.add(image['pixel_sha256'])
            if sum(row['category']==category for row in selected)==count:break
        if sum(row['category']==category for row in selected)!=count:
            raise ValueError('Not enough independently identified '+category+' scenes')
    second=[]
    for category,count in (('indoor',2),('outdoor',6)):
        ranked=sorted((r for r in selected if r['category']==category),
                      key=lambda r:hashlib.sha256(('second-seed-v1|'+r['member']).encode()).hexdigest())
        second.extend(r['member'] for r in ranked[:count])
    for row in selected:row['second_seed']=row['member'] in second
    return selected


def caption_sheet(rgb, path):
    from PIL import Image,ImageDraw
    from .rendering import render_views
    views=[dict(yaw=float(yaw),pitch=float(pitch)) for pitch in (10,0,-10) for yaw in (0,90,180,270)]
    views += [dict(yaw=0.,pitch=90.),dict(yaw=0.,pitch=-90.)]
    rendered=render_views(rgb,views,size=320,fov=80.)
    sheet=Image.new('RGB',(1280,2064),'white');draw=ImageDraw.Draw(sheet)
    draw.text((10,8),'Full360 ERP overview; lower panels are80-degree perspective crops. Yaw identifies position around the panorama.',fill='black')
    sheet.paste(Image.fromarray(rgb).resize((1280,640),Image.Resampling.LANCZOS),(0,32))
    for index,(view,pixels) in enumerate(zip(views,rendered)):
        x=(index%4)*320;y=672+(index//4)*348
        draw.text((x+6,y+6),'yaw%03d pitch%+03d'%(view['yaw'],view['pitch']),fill='black')
        sheet.paste(Image.fromarray(pixels),(x,y+28))
    sheet.save(path,compress_level=6)
    return dict(path=str(path),sha256=sha(path),width=1280,height=2064,
                views=views,fov=80.,view_size=320,renderer='studies.paper_eval_v2.rendering.render_views',
                renderer_sha256=sha(ROOT/'studies/paper_eval_v2/rendering.py'))


def finalize(run=RUN):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Extract selected panoramas on a CPU allocation')
    from PIL import Image,ImageDraw
    import numpy as np
    run=Path(run)
    review=read(run/'dataset/candidate-review.json')
    annotations=read(run/'dataset/content-review.json')
    if annotations['candidate_review_sha256']!=digest(review):raise ValueError('Review identity differs')
    for sheet in review['contact_sheets']:
        if sha(sheet['path'])!=sheet['sha256']:raise ValueError('Candidate contact sheet changed')
    selected=select_rows(review,annotations['annotations'])
    inventory=read(run/'dataset/archive-inventory.json')
    selection=dict(schema_version=1,archive_sha256=inventory['archive']['sha256'],
        selection_protocol_sha256=sha(run/'dataset/selection-protocol.json'),
        content_review_sha256=sha(run/'dataset/content-review.json'),
        members=[r['member'] for r in selected],second_seed_members=[r['member'] for r in selected if r['second_seed']])
    freeze(run/'dataset/selection.json',selection)
    lookup={r['member']:r for r in selected}
    directory=run/'dataset/images';directory.mkdir(exist_ok=True)
    inputs=run/'dataset/caption-inputs';inputs.mkdir(exist_ok=True)
    overview=Image.new('RGB',(2400,960),'white');draw=ImageDraw.Draw(overview)
    rows={}
    for entry,data in stream_members(run,selection['members']):
        chosen=lookup[entry['member']]
        if hashlib.sha256(data).hexdigest()!=chosen['image_sha256']:raise ValueError('Selected archive bytes differ')
        stem=PurePosixPath(entry['member']).stem
        if not stem.isdigit():raise ValueError('Expected numeric official test member')
        scene_id='odisr-test-'+stem
        target=directory/(scene_id+'.jpg')
        if target.exists() and sha(target)!=chosen['image_sha256']:raise ValueError('Existing selected source differs')
        if not target.exists():target.write_bytes(data)
        with Image.open(io.BytesIO(data)) as image:
            image.load();rgb=np.asarray(image)
            if image.mode!='RGB' or list(image.size)!=[chosen['width'],chosen['height']]:raise ValueError('Selected image dimensions differ')
            display=image.resize((400,200),Image.Resampling.LANCZOS)
        order=selection['members'].index(entry['member']);x=(order%6)*400;y=(order//6)*240
        overview.paste(display,(x,y+36));draw.text((x+8,y+8),scene_id+' | '+chosen['category']+(' | second seed' if chosen['second_seed'] else ''),fill='black')
        rows[entry['member']]=dict(chosen,scene_id=scene_id,
            prompt_id='confirmation24/ODISR/'+entry['member'].removeprefix('lau_dataset/')+'@'+chosen['image_sha256'],
            image_path=str(target),source_split='odisr/testing/HR',source_archive_sha256=inventory['archive']['sha256'],
            caption_input=caption_sheet(rgb,inputs/(scene_id+'.png')))
    overview_path=run/'dataset/confirmation24-contact.png';overview.save(overview_path,compress_level=6)
    ordered=[rows[member] for member in selection['members']]
    sun=read(ROOT/'outputs/metrics-evaluation/seed0-v1/reference_manifest.json')
    overlap=set(r['image_sha256'] for r in ordered)&set(r['sha256'] for r in sun['records'])
    if overlap:raise ValueError('Confirmation source byte duplicates saved SUN calibration/reference images')
    manifest=dict(schema_version=1,state='complete',cohort_id='confirmation24',dataset_label='our ODI-SR-24 subset',
        source='https://github.com/wangh-allen/LAU-Net',source_split='odisr/testing/HR',official_test_count=100,
        claim='Deterministic24-scene subset of the released100-image testing split; not an official24-image benchmark or full-paper replication.',
        rows=ordered,scene_count=24,category_counts=dict(indoor=6,outdoor=18),
        second_seed_scene_ids=[r['scene_id'] for r in ordered if r['second_seed']],
        selection_sha256=sha(run/'dataset/selection.json'),selection_protocol_sha256=sha(run/'dataset/selection-protocol.json'),
        content_review_sha256=sha(run/'dataset/content-review.json'),source_archive=inventory['archive'],
        contact_sheet=dict(path=str(overview_path),sha256=sha(overview_path)),
        disjointness=dict(saved_SUN4260_exact_file_hash_overlap=0,
            limitation='Different declared releases/splits and no exact JPEG hash overlap; exhaustive visual cross-dataset scene deduplication is not established.'),
        selected_before_generation_and_scoring=True,scene_review='All100testing HR contact sheets reviewed; recognizable repeated captures grouped before hash-ranked selection.')
    freeze(run/'real_dataset_manifest.json',manifest)
    print('DATASET_FROZEN',24,manifest['category_counts'],manifest['second_seed_scene_ids'],flush=True)
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('inspect','preview','finalize'))
    parser.add_argument('--run-dir', type=Path, default=RUN)
    args = parser.parse_args()
    {'inspect':inspect,'preview':preview,'finalize':finalize}[args.action](args.run_dir)


if __name__ == '__main__':
    main()
