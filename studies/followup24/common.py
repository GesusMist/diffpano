"""Immutable prompt scope shared by standalone and panorama follow-ups.

Keep this scientific input module separate from mutable orchestration/reporting.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_A = ROOT / 'outputs/10.15perspective24-flux/seed0-v1'
OUT_B = ROOT / 'outputs/10.15layout24-flux216/seed0-v1'
PROMPTS20 = (
    'air_balloons', 'aurora', 'blossom', 'city', 'clouds_over_grass',
    'desert', 'desert_sandstorm', 'firefly', 'firework', 'forest',
    'grand_canyon', 'lavender', 'moonlit_snowy_village', 'ocean',
    'rock_mountain', 'rock_mountain_with_star', 'ruins', 'snow_mountain',
    'storm', 'underwater',
)
ORIGINAL20 = PROMPTS20
NEW4 = ('bamboo_grove', 'lantern_market', 'sailboat_marina', 'tiled_courtyard')
ORIGINAL = PROMPTS20
NEW = NEW4
PROMPTS24 = PROMPTS20 + NEW4
PROMPTS = PROMPTS24
DIRECTION_LABELS = ('01_up', '02_slightly_up', '03_eye_level', '04_slightly_down', '05_down')
VARIANTS9 = (
    'spherediff_official_20', 'diffpano_old89_rgb', 'diffpano_old70_rgb',
    'diffpano_old50_rgb', 'diffpano_uniform89_rgb', 'diffpano_uniform70_rgb',
    'diffpano_uniform50_rgb', 'diffpano_old89_screened_select',
    'diffpano_uniform89_screened_select',
)

def prompt_path(name):
    if name not in PROMPTS24:
        raise ValueError('Scene outside the exact 24-prompt scope: ' + str(name))
    return ROOT / 'prompts' / ('ournew' if name in NEW4 else '') / (name + '.txt')

def prompt(name):
    """Return a strict source record and unchanged original bytes."""
    path = prompt_path(name)
    raw = path.read_bytes()
    lines = raw.decode('utf-8').splitlines()
    if len(lines) != 5 or any(not line.strip() for line in lines):
        raise ValueError('Expected exactly five nonempty directional lines: ' + str(path))
    digest = hashlib.sha256(raw).hexdigest()
    record = dict(name=name, path=str(path.relative_to(ROOT)),
                  full_path=str(path), filename=path.name,
                  source_group='new4' if name in NEW4 else 'original20',
                  line_count=5, adapted=False, original_sha256=digest,
                  effective_sha256=digest, effective_lines=lines,
                  line_sha256=[hashlib.sha256(line.encode('utf-8')).hexdigest() for line in lines])
    return record, raw
