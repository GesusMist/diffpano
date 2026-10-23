"""Read-only panorama projection with verified, resumable content-addressed views."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

from .protocol import CAMERAS, VIEWS, digest, protocol_record, rendering_protocol_record

REQUIRED = ('panorama_id', 'method_id', 'prompt_id', 'prompt_name', 'image_path', 'image_sha256')
VIEW_FIELDS = REQUIRED + ('view_id', 'yaw', 'pitch', 'elevation_group', 'is_seam', 'is_polar',
                          'view_path', 'view_sha256', 'rendering_protocol_sha256')


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(data)
    return result.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def _cached_views(directory, image_sha256, protocol_sha256):
    """A cache hit requires all view file checksums and image dimensions to match."""
    from PIL import Image
    try:
        metadata = json.loads((directory / 'manifest.json').read_text())
        if (metadata['image_sha256'] != image_sha256 or
                metadata['rendering_protocol_sha256'] != protocol_sha256 or
                set(metadata['views']) != {c['view_id'] for c in CAMERAS}):
            return None
        for camera in CAMERAS:
            entry = metadata['views'][camera['view_id']]
            path = directory / (camera['view_id'] + '.png')
            if entry['sha256'] != sha256_file(path):
                return None
            with Image.open(path) as image:
                if image.size != (512, 512) or image.mode != 'RGB':
                    return None
                image.verify()
        return metadata
    except (OSError, ValueError, KeyError, TypeError):
        return None


def render_panorama(record, output):
    """Render or verify 14 views; original bytes are checked before and after.

    ``record`` is an already accepted inventory row. Only its read-only image
    and identifying fields are consumed. Cache identities exclude method labels
    and contain both original image and rendering-protocol hashes.
    """
    import numpy as np
    from PIL import Image
    from studies.panorama_metrics.render import render

    missing = [key for key in REQUIRED if not record.get(key)]
    if missing:
        raise ValueError('Missing inventory fields: ' + ', '.join(missing))
    source = Path(record['image_path']).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output == source.parent or source.is_relative_to(output):
        raise ValueError('Preprocessing output must be separate from historical source images')
    expected = str(record['image_sha256'])
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != expected:
        raise ValueError(f'Historical image checksum mismatch: {source}')
    protocol = rendering_protocol_record()
    protocol_sha256 = digest(protocol)
    directory = output / 'rendered' / protocol_sha256 / expected
    metadata = _cached_views(directory, expected, protocol_sha256)
    if metadata is None:
        with Image.open(io.BytesIO(source_bytes)) as image:
            if image.mode != 'RGB':
                raise ValueError(f'Expected an RGB source image without color conversion: {source}')
            if image.width != image.height * 2:
                raise ValueError(f'Expected a 2:1 ERP image: {source}')
            rgb = np.asarray(image).copy()
        views = render(rgb, VIEWS, size=512)
        if views.shape != (14, 512, 512, 3):
            raise AssertionError('Unexpected rendered dimensions')
        metadata = dict(image_sha256=expected, rendering_protocol_sha256=protocol_sha256,
                        source_size=[rgb.shape[1], rgb.shape[0]], views={})
        for camera, view in zip(CAMERAS, views):
            buffer = io.BytesIO()
            Image.fromarray(view).save(buffer, format='PNG')
            data = buffer.getvalue()
            path = directory / (camera['view_id'] + '.png')
            atomic_bytes(path, data)
            metadata['views'][camera['view_id']] = dict(sha256=hashlib.sha256(data).hexdigest())
        atomic_json(directory / 'manifest.json', metadata)
    if sha256_file(source) != expected:
        raise ValueError(f'Historical image changed during preprocessing: {source}')
    rows = []
    for camera in CAMERAS:
        row = {key: record[key] for key in REQUIRED}
        row.update({key: camera[key] for key in ('view_id', 'yaw', 'pitch', 'elevation_group', 'is_seam', 'is_polar')})
        row.update(view_path=str(directory / (camera['view_id'] + '.png')),
                   view_sha256=metadata['views'][camera['view_id']]['sha256'],
                   rendering_protocol_sha256=protocol_sha256)
        rows.append(row)
    return rows


def preprocess(records, output):
    """Preprocess accepted records and atomically write views.csv/protocol.json."""
    output = Path(output)
    rows = []
    seen = set()
    for record in records:
        if record['panorama_id'] in seen:
            raise ValueError('Duplicate panorama_id in preprocessing records: ' + record['panorama_id'])
        seen.add(record['panorama_id'])
        rows.extend(render_panorama(record, output))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=VIEW_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(output / 'views.csv', stream.getvalue().encode())
    atomic_json(output / 'protocol.json', protocol_record())
    return rows
