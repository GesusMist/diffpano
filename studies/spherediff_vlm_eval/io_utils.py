"""Small file utilities, independent of all frozen generation studies."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_csv(path, rows, fields):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()})
    atomic_text(path, buffer.getvalue())
