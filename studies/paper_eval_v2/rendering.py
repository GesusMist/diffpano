"""Shared periodic-ERP evaluation cameras; generation cameras are independent."""
import hashlib
import json
from pathlib import Path

HORIZONTAL = tuple((float(yaw), 0.) for yaw in range(0, 360, 45))
VIEWS14 = tuple(dict(view_id='p%+03d_y%03d' % (pitch, yaw), yaw=float(yaw), pitch=float(pitch),
    elevation_group={-45:'lower', 0:'horizontal', 45:'upper'}[pitch],
    prompt_band={-45:3, 0:2, 45:1}[pitch], is_seam=yaw == 180)
    for pitch in (-45, 0, 45) for yaw in (0, 90, 180, 270)) + (
    dict(view_id='north', yaw=0., pitch=90., elevation_group='north', prompt_band=0, is_seam=False),
    dict(view_id='south', yaw=0., pitch=-90., elevation_group='south', prompt_band=4, is_seam=False))
NATIVE_FIVE = tuple(dict(view_id='band%d' % i, yaw=0., pitch=float(pitch), prompt_band=i)
                    for i, pitch in enumerate((90, 10, 0, -10, -90)))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def protocol_definitions():
    import cv2
    import PIL
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    legacy_source = hashlib.sha256((Path(__file__).resolve().parents[1]/'panorama_metrics/render.py').read_bytes()).hexdigest()
    common = dict(protocol_id='common_v2', width=2048, height=1024,
        erp_resampling='OpenCV INTER_LINEAR remap, pixel centers, one-column periodic-longitude padding, latitude clamp; RGB uint8',
        source_resolution='recorded per panorama; upsampling creates no native detail',
        fixed_wrap=True, world_axes='x right, y north, z yaw0; yaw180 centers ERP wrap',
        fov_x=90., fov_y=90., view_size=512, horizontal_views=HORIZONTAL,
        views14=VIEWS14, pixel_centers=True, pole_yaw=0., renderer_source_sha256=source,
        perspective_resampling='OpenCV INTER_LINEAR, periodic longitude padding, latitude clamp',
        legacy_renderer_source_sha256=legacy_source,opencv_version=cv2.__version__,pillow_version=PIL.__version__,
        png_encoding=dict(compress_level=6,optimize=False))
    native = dict(protocol_id='native_five_v2', native_erp=True, source_resolution='recorded per panorama',
        fov_x=80., fov_y=80., view_size=1024, views=NATIVE_FIVE, pole_yaw=0.,
        pixel_centers=True, renderer_source_sha256=source,
        perspective_resampling=common['perspective_resampling'],legacy_renderer_source_sha256=legacy_source,opencv_version=cv2.__version__,
        interpretation='native system fidelity including resolution effects; standalone images have no guaranteed physical FOV')
    return dict(common_v2=common, native_five_v2=native)


def resample_common(image):
    import cv2
    import numpy as np
    assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8
    h, w = image.shape[:2]
    if (w, h) == (2048, 1024):
        return image.copy()
    x = ((np.arange(2048, dtype=np.float64)+.5)*w/2048-.5+1).astype(np.float32)
    y = np.clip((np.arange(1024, dtype=np.float64)+.5)*h/1024-.5, 0, h-1).astype(np.float32)
    xx, yy = np.meshgrid(x, y)
    padded = np.concatenate([image[:, -1:], image, image[:, :1]], axis=1)
    return cv2.remap(padded, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def render_views(image, views, size=512, fov=90.):
    import cv2
    import numpy as np
    from studies.panorama_metrics.render import render, rotation
    angles = [(float(v['yaw']), float(v['pitch'])) if isinstance(v, dict) else tuple(v) for v in views]
    if fov == 90.:
        return render(image, angles, size)
    h, w = image.shape[:2]
    t = np.tan(np.deg2rad(fov/2))
    xx, yy = np.meshgrid(((np.arange(size)+.5)*2/size-1)*t,
                         (1-(np.arange(size)+.5)*2/size)*t)
    local = np.stack([xx, yy, np.ones_like(xx)], -1)
    padded = np.concatenate([image[:, -1:], image, image[:, :1]], axis=1)
    outputs = []
    for yaw, pitch in angles:
        rays = local @ rotation(yaw, pitch).T
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        lon, lat = np.arctan2(rays[..., 0], rays[..., 2]), np.arcsin(rays[..., 1])
        x = ((lon/(2*np.pi)+.5)*w-.5+1).astype(np.float32)
        y = np.clip((.5-lat/np.pi)*h-.5, 0, h-1).astype(np.float32)
        outputs.append(cv2.remap(padded, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return np.stack(outputs)


def render_common_views(image, views=VIEWS14):
    return render_views(resample_common(image), views, 512, 90.)


def render_native_five(image):
    return render_views(image, NATIVE_FIVE, 1024, 80.)


def cache_views(run, records):
    """Persist exact shared PNG bytes in one SQLite database, avoiding inode growth."""
    import io
    import sqlite3
    import numpy as np
    from PIL import Image
    from studies.panorama_metrics.common import sha
    database = Path(run) / 'render_cache/views.sqlite3'
    database.parent.mkdir(parents=True, exist_ok=True)
    protocol = protocol_definitions()['common_v2']
    identity = _hash(protocol)
    output = []
    with sqlite3.connect(str(database), timeout=120) as connection:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('CREATE TABLE IF NOT EXISTS views (key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, png BLOB NOT NULL)')
        for record in records:
            if record.get('state', 'complete') != 'complete':
                continue
            keys = [_hash(dict(source=record['image_sha256'], protocol=identity, view=v['view_id'])) for v in VIEWS14]
            cached = [connection.execute('SELECT sha256 FROM views WHERE key=?', (key,)).fetchone() for key in keys]
            if not all(cached):
                assert sha(record['image_path']) == record['image_sha256']
                with Image.open(record['image_path']) as image:
                    native_size = list(image.size)
                    images = render_common_views(np.asarray(image.convert('RGB')))
                for key, image in zip(keys, images):
                    stream = io.BytesIO()
                    Image.fromarray(image).save(stream, format='PNG', compress_level=6, optimize=False)
                    raw = stream.getvalue()
                    checksum = hashlib.sha256(raw).hexdigest()
                    old = connection.execute('SELECT sha256 FROM views WHERE key=?', (key,)).fetchone()
                    if old and old[0] != checksum:
                        raise RuntimeError('Shared renderer byte mismatch')
                    connection.execute('INSERT OR IGNORE INTO views VALUES (?,?,?)', (key, checksum, raw))
                connection.commit()
                cached = [connection.execute('SELECT sha256 FROM views WHERE key=?', (key,)).fetchone() for key in keys]
            else:
                with Image.open(record['image_path']) as image:
                    native_size = list(image.size)
            for view, key, cached_row in zip(VIEWS14, keys, cached):
                output.append(dict(prompt_id=record['prompt_id'], prompt_name=record.get('prompt_name'),
                    case_id=record['case_id'], panorama_id=record.get('panorama_id', record['case_id']),
                    method_id=record['method_id'], seed=record.get('seed', record.get('parameters', {}).get('seed', 0)),
                    view_id=view['view_id'], yaw=view['yaw'], pitch=view['pitch'], prompt_band=view['prompt_band'],
                    image_path='sqlite://'+str(database)+'#'+key, image_sha256=cached_row[0],
                    source_image_sha256=record['image_sha256'], source_resolution=native_size,
                    evaluation_resolution=[2048,1024], upsampled=native_size[0] < 2048 or native_size[1] < 1024,
                    renderer_protocol_sha256=identity, is_polar=abs(view['pitch']) == 90., is_seam=view['is_seam']))
    return output


def read_view_bytes(record):
    import sqlite3
    value = record['image_path']
    if value.startswith('sqlite://'):
        database, key = value[len('sqlite://'):].rsplit('#', 1)
        with sqlite3.connect('file:'+database+'?mode=ro', uri=True) as connection:
            row = connection.execute('SELECT png FROM views WHERE key=?', (key,)).fetchone()
        if row is None:
            raise FileNotFoundError(key)
        raw = bytes(row[0])
    else:
        raw = Path(value).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record['image_sha256']:
        raise ValueError('Rendered image cache checksum mismatch')
    return raw
