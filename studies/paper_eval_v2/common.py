"""Small shared filesystem/state interfaces; never serialize process environments."""
import contextlib
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/paper-eval-v2/20261026-core-v1'
ACCOUNT = '142684268708'
PRIMARY_METHODS = ('sphere20', 'rgb', 'select', 'meanmax050')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'-', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def csv_write(path, records, fields):
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in records:
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict,list)) else value for key,value in row.items()})
    atomic_text(path, stream.getvalue())


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


@contextlib.contextmanager
def locked_state(path, default=None):
    path = Path(path)
    with lock(path.with_suffix(path.suffix+'.lock')):
        value = read(path) if path.exists() else ({} if default is None else default)
        yield value
        write(path, value)


def update_component(run, name, state, **details):
    with locked_state(Path(run)/'status.json') as status:
        status.setdefault('components', {})[name] = dict(state=state, updated_utc=now(), **details)
        status['updated_utc'] = now()


def command(args):
    # Non-API commands and scheduler submissions never inherit the API secret.
    env = {key:value for key,value in os.environ.items() if key != 'OPENROUTER_API_KEY'}
    return subprocess.check_output(args, cwd=str(ROOT), env=env, universal_newlines=True).strip()


def freeze(path, value):
    path = Path(path)
    with lock(path.with_suffix(path.suffix+'.lock')):
        if path.exists():
            old = read(path)
            if old != value:
                raise ValueError('Frozen record differs; preserve it and create a new protocol/run: '+str(path))
        else:
            write(path, value)
    return digest(value)
