"""Study-local provenance, argument construction and non-mutating instrumentation."""
import collections
import contextlib
import fcntl
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / 'studies/original_spherediff'
ROOT = REPO / 'outputs/original-spherediff/ruins-underwater-seed0-v1'
HIST = REPO / 'outputs/bridge-factorial-ruins/20260918'
SPHERE = Path('/home/shig/diffpano_reference_sources/spherediff-2c8c68b')
GIT_SOURCE = Path('/home/shig/SphereDiff')
COMMIT = '2c8c68ba088f2803b3dce4b52b7b0d68bc996139'
CACHE = Path('/scratch/user/shig/SphereDiff/hf_cache/hub')
BACKENDS = ('sana', 'flux')
PROMPTS = ('ruins', 'underwater')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe(x) for x in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, float) and not math.isfinite(value):
        return '-Infinity' if value < 0 else 'Infinity' if value > 0 else 'NaN'
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'detach'):
        return safe(value.detach().cpu().tolist())
    return value


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temp.open('x') as f:
        json.dump(safe(value), f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def immutable(path, value):
    path = Path(path)
    value = safe(value)
    if path.exists():
        if read(path) != value:
            raise RuntimeError('Existing immutable record differs: ' + str(path))
    else:
        atomic(path, value)


def specs():
    return read(STUDY / 'config.yaml')['models']  # JSON is a YAML subset.


def snapshot(name):
    spec = specs()[name]
    return CACHE / ('models--' + spec['model_source'].replace('/', '--')) / 'snapshots' / spec['revision']


def folder(name, prompt):
    if name not in BACKENDS or prompt not in PROMPTS:
        raise ValueError((name, prompt))
    return ROOT / name / prompt / 'seed0'


def source_hashes():
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(STUDY.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def official_hashes():
    return {str(p.relative_to(SPHERE)): sha(p) for p in sorted(SPHERE.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def import_official():
    sys.dont_write_bytecode = True
    if str(SPHERE) not in sys.path:
        sys.path.insert(0, str(SPHERE))
    module = importlib.import_module('pipelines_ours')
    classes = {n: getattr(module, specs()[n]['pipeline']) for n in BACKENDS}
    paths = {'pipelines_ours': str(Path(module.__file__).resolve())}
    paths.update({specs()[n]['pipeline']: str(Path(inspect.getfile(cls)).resolve()) for n, cls in classes.items()})
    for p in paths.values():
        if SPHERE.resolve() not in Path(p).parents:
            raise RuntimeError('Wrong original import: ' + p)
    return classes, paths


def defaults(cls):
    return safe({k: v.default for k, v in inspect.signature(cls.__call__).parameters.items()
                 if v.default is not inspect.Parameter.empty})


def arguments(name, prompt, generator):
    if prompt not in PROMPTS:
        raise ValueError(prompt)
    return dict(specs()[name]['call'], prompt_txt_path=str(REPO / 'prompts' / (prompt + '.txt')),
                generator=generator, callback_on_step_end=progress)


def seed_global(torch, numpy):
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def fresh_generator(torch):
    return torch.Generator(device='cuda').manual_seed(0)


def static_solver(scheduler):
    original = scheduler.config.get('solver_order', 1)
    if original > 1:
        scheduler.config.solver_order = 1  # Exactly the pinned official static launcher.
    return original


def scheduler_config(config):
    value = safe(dict(config))
    # Diffusers configuration_utils.py builds this bookkeeping list from a set.
    # Its order is not scheduler behavior; retain every name and every value.
    if '_use_default_values' in value:
        value['_use_default_values'] = sorted(value['_use_default_values'])
    return value


def scheduler_record(scheduler, original):
    config = scheduler.config
    effective = getattr(config, 'solver_order', None)
    return safe(dict(class_name=type(scheduler).__name__, config=scheduler_config(config),
        original_serialized_solver_order=original, serialized_solver_order=config.get('solver_order'),
        effective_config_attribute_solver_order=effective,
        scheduler_order_attribute=getattr(scheduler, 'order', None),
        effective_numerical_order=effective if effective is not None else getattr(scheduler, 'order', None),
        timesteps=getattr(scheduler, 'timesteps', None), sigmas=getattr(scheduler, 'sigmas', None)))


def progress(pipeline, step, timestep, callback_kwargs):
    print('Original SphereDiff completed interval', step + 1, flush=True)
    return callback_kwargs


class ForwardLog:
    def __init__(self, original):
        self.original = original
        self.calls = 0
        self.shapes = collections.Counter()

    def __call__(self, *args, **kwargs):
        self.calls += 1
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        if hidden is not None:
            self.shapes[str(list(hidden.shape))] += 1
        result = self.original(*args, **kwargs)
        if self.calls % 89 == 0:
            print('Original transformer forwards', self.calls, flush=True)
        return result


@contextlib.contextmanager
def run_lock(target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            raise FileExistsError('Existing attempt is never overwritten: ' + str(target))
        yield


def require_gate():
    gate = read(ROOT / 'validation.json')
    if not gate['passed'] or gate['source_hashes'] != source_hashes():
        raise RuntimeError('Study source differs from passed validation')
    manifest = read(ROOT / 'manifest.json')
    if gate['manifest_sha256'] != sha(ROOT / 'manifest.json') or official_hashes() != manifest['official_source_hashes']:
        raise RuntimeError('Manifest or original source changed')
    for p, h in manifest['preserved'].items():
        if sha(p) != h:
            raise RuntimeError('Preserved file changed: ' + p)
    return manifest


def image_audit(path):
    from PIL import Image
    with Image.open(path) as im:
        if im.size != (4096, 2048):
            raise AssertionError(im.size)
        im.verify()
    return dict(path=str(Path(path).resolve()), width=4096, height=2048, sha256=sha(path))
