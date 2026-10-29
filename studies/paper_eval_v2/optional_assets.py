"""Public metadata pinning and explicit CPU-only optional checkpoint provisioning.

This helper never selects a generation tier, submits a job, launches a model, or
changes an existing generation manifest. Provisioning requires a new run already
configured for the trained extension and a separate explicit CLI acknowledgement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request

from .common import RUN, now, read, write, sha, freeze, lock

REPOSITORY = 'Insta360-Research/DiT360-Panorama-Image-Generation'
API = 'https://huggingface.co/api/models/' + REPOSITORY
HEADERS = {'User-Agent': 'DiffPano-research-provenance/1.0'}
FILENAMES = ('adapter_config.json', 'adapter_model.safetensors')


def fetch_small(url, maximum=1024 * 1024):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as response:
        raw = response.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('Public metadata exceeds its bounded size')
    return raw


def artifact_url(revision, filename):
    if not re.fullmatch('[0-9a-f]{40}', revision) or filename not in FILENAMES:
        raise ValueError('Only a pinned revision and exact adapter files are allowed')
    return 'https://huggingface.co/' + REPOSITORY + '/resolve/' + revision + '/' + filename


def metadata(run=RUN):
    run = Path(run)
    initial = json.loads(fetch_small(API + '?blobs=true'))
    revision = initial['sha']
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('HF did not supply an immutable commit revision')
    url = API + '/revision/' + revision + '?blobs=true'
    remote = json.loads(fetch_small(url))
    if remote.get('sha') != revision or remote.get('id') != REPOSITORY or remote.get('private') or remote.get('gated') or remote.get('disabled'):
        raise ValueError('Official checkpoint is not public, ungated and pinned')
    files = {item['rfilename']: item for item in remote['siblings']}
    config_raw = fetch_small(artifact_url(revision, FILENAMES[0]), maximum=64 * 1024)
    git_blob = hashlib.sha1(b'blob ' + str(len(config_raw)).encode() + b'\0' + config_raw).hexdigest()
    if len(config_raw) != files[FILENAMES[0]]['size'] or git_blob != files[FILENAMES[0]]['blobId']:
        raise ValueError('Adapter configuration differs from the pinned HF Git blob')
    config = json.loads(config_raw)
    if config.get('peft_type') != 'LORA':
        raise ValueError('Official adapter is not a LoRA configuration')
    weights = files[FILENAMES[1]]
    lfs = weights['lfs']
    if lfs['size'] != weights['size'] or not re.fullmatch('[0-9a-f]{64}', lfs['sha256']):
        raise ValueError('HF LFS size/SHA256 missing or contradictory')
    directory = run / 'generation/optional_assets' / ('dit360-' + revision)
    directory.mkdir(parents=True, exist_ok=True)
    remote_path = directory / 'model-info.json'
    if not remote_path.exists():
        freeze(remote_path, dict(source_url=url, observed_utc=now(), response=remote))
    config_path = directory / 'adapter_config.json'
    if config_path.exists() and config_path.read_bytes() != config_raw:
        raise ValueError('Pinned local adapter configuration changed')
    if not config_path.exists():
        config_path.write_bytes(config_raw)
    pinned = dict(repository=REPOSITORY, revision=revision, public=True, gated=False,
        metadata_path=str(remote_path.resolve()), metadata_sha256=sha(remote_path),
        metadata_url=url, model_card_license=remote.get('cardData', {}).get('license'),
        adapter_configuration=config, files=[
            dict(filename=FILENAMES[0], bytes=len(config_raw), sha256=sha(config_path),
                 git_blob_id=git_blob, metadata_local_path=str(config_path.resolve()),
                 download_url=artifact_url(revision, FILENAMES[0])),
            dict(filename=FILENAMES[1], bytes=lfs['size'], sha256=lfs['sha256'],
                 git_blob_id=weights['blobId'], hash_kind='HF Git LFS SHA256',
                 download_url=artifact_url(revision, FILENAMES[1]))])
    pin_path = directory / 'checkpoint-metadata.json'
    freeze(pin_path, pinned)
    availability_path = run / 'generation/optional_baseline_availability.json'
    current = read(availability_path) if availability_path.exists() else dict(schema_version=1)
    if availability_path.exists():
        freeze(run / 'generation/optional_assets' / ('availability-before-' + sha(availability_path) + '.json'), current)
    previous = current.get('dit360', {})
    current['dit360'] = dict(previous, selected=False, state='public_metadata_pinned_weights_not_provisioned',
        checkpoint_repository=REPOSITORY, checkpoint_revision=revision,
        checkpoint_metadata_path=str(pin_path.resolve()), checkpoint_metadata_sha256=sha(pin_path),
        public_ungated_metadata_verified=True, weight_files_downloaded=False, files=pinned['files'],
        reason='Official public revision, adapter configuration and weight LFS identity verified; trained tier remains unselected and no weight bytes were downloaded.',
        required_before_execution=[
            'Create a distinct explicitly selected trained extension run before provisioning; do not edit the frozen core manifest.',
            'Provision the exact LoRA in a finite CPU allocation and verify every local file SHA256 and size.',
            'Bind dit360_checkpoint.json and its file hashes into a new extension generation gate; the frozen core gate does not bind this optional asset.',
            'Create and validate a counted DiT360 hardware profile in the extension workflow; core profile indices cover only the three core methods.'],
        preparation_command='srun --account=142684268708 --partition=cpu --cpus-per-task=1 --mem=2G --time=00:20:00 python3 -m studies.paper_eval_v2.optional_assets provision --run /ABSOLUTE/NEW_TRAINED_RUN --select-trained --metadata ' + str(pin_path.resolve()))
    write(availability_path, current)
    return dict(state=current['dit360']['state'], revision=revision, metadata=str(pin_path), weight_bytes=lfs['size'])


def provision(run, metadata_path, select_trained=False):
    run, metadata_path = Path(run).resolve(), Path(metadata_path).resolve()
    if not select_trained or run == RUN.resolve():
        raise ValueError('Provisioning requires --select-trained and a distinct new extension run')
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Optional weight provisioning requires a finite CPU allocation')
    if int(os.environ.get('SLURM_GPUS_ON_NODE', '0') or '0'):
        raise RuntimeError('Provision optional assets on a CPU allocation')
    config = read(run / 'run_config.json')
    if config.get('selected_generation_tier') not in ('trained', 'full'):
        raise ValueError('The new run must already explicitly select the trained extension')
    if (run / 'generation/manifest.json').exists():
        raise ValueError('Provision before freezing the new extension generation manifest')
    pinned = read(metadata_path)
    if pinned['repository'] != REPOSITORY or {item['filename'] for item in pinned['files']} != set(FILENAMES):
        raise ValueError('Unexpected checkpoint repository/file set')
    original_metadata = read(pinned['metadata_path'])
    if sha(pinned['metadata_path']) != pinned['metadata_sha256'] or original_metadata['response']['sha'] != pinned['revision']:
        raise ValueError('Public metadata provenance changed')
    remote_files = {item['rfilename']: item for item in original_metadata['response']['siblings']}
    directory = run / 'external/DiT360/checkpoints' / pinned['revision']
    directory.mkdir(parents=True, exist_ok=True)
    output = []
    with lock(run / 'generation/optional_assets.lock'):
        for item in pinned['files']:
            filename = item['filename']
            url = artifact_url(pinned['revision'], filename)
            remote = remote_files[filename]
            if item['bytes'] != remote['size'] or (remote.get('lfs') and item['sha256'] != remote['lfs']['sha256']):
                raise ValueError('Pinned artifact metadata disagrees with official Git/LFS evidence')
            target = directory / filename
            if target.exists():
                if target.stat().st_size != item['bytes'] or sha(target) != item['sha256']:
                    raise ValueError('Existing optional asset differs; never overwrite it')
            else:
                descriptor, temporary = tempfile.mkstemp(prefix='.' + filename + '-', dir=str(directory))
                try:
                    h = hashlib.sha256(); count = 0
                    with os.fdopen(descriptor, 'wb') as stream:
                        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
                            while True:
                                chunk = response.read(1024 * 1024)
                                if not chunk:break
                                count += len(chunk)
                                if count > item['bytes']:raise ValueError('Download exceeds pinned file size')
                                h.update(chunk);stream.write(chunk)
                        stream.flush();os.fsync(stream.fileno())
                    if count != item['bytes'] or h.hexdigest() != item['sha256']:
                        raise ValueError('Downloaded optional checkpoint hash/size differs')
                    if filename == 'adapter_config.json':
                        raw = Path(temporary).read_bytes()
                        if hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != remote['blobId']:
                            raise ValueError('Downloaded adapter configuration Git blob differs')
                    os.replace(temporary, target)
                finally:
                    if Path(temporary).exists():Path(temporary).unlink()
            output.append(dict(filename=filename, path=str(target), sha256=sha(target), bytes=target.stat().st_size))
        value = dict(schema_version=1, state='verified_local_checkpoint', repository=REPOSITORY,
            revision=pinned['revision'], local_path=str(directory), files=output,
            metadata_path=str(metadata_path), metadata_sha256=sha(metadata_path),
            source_sha256=sha(__file__), source_job_id=os.environ['SLURM_JOB_ID'],
            runtime_ready=False, generation_gate='A new extension manifest must bind this checkpoint and profile DiT360 before generation.')
        freeze(run / 'generation/dit360_checkpoint.json', value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('metadata', 'provision'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--select-trained', action='store_true')
    args = parser.parse_args()
    if args.action == 'metadata':result = metadata(args.run)
    else:
        if args.metadata is None:parser.error('provision requires --metadata')
        result = provision(args.run, args.metadata, args.select_trained)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':main()
