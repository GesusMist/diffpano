"""Isolated identities for the synthetic-reference FID evaluation."""
from pathlib import Path
from studies.layout24.common import ROOT, sha, digest, read, write

OUT = ROOT / 'outputs/10.23fid-standalone85/seed0-v1'
SOURCE = ROOT / 'studies/standalone_fid'
PLAN = OUT / 'plan.json'
ASSETS = Path('/scratch/user/shig/diffpano/evaluation-assets')
VARIANTS = (
    'spherediff_official_20', 'diffpano_old89_rgb',
    'diffpano_old89_screened_select', 'diffpano_uniform89_rgb',
    'diffpano_uniform70_rgb', 'diffpano_uniform50_rgb',
    'diffpano_uniform89_screened_select',
)
LABELS = {
    'spherediff_official_20': 'SphereDiff',
    'diffpano_old89_rgb': 'Old89 RGB (ours)',
    'diffpano_old89_screened_select': 'Old89 screened-select (ours)',
    'diffpano_uniform89_rgb': 'Uniform89 RGB (ours)',
    'diffpano_uniform70_rgb': 'Uniform70 RGB (ours)',
    'diffpano_uniform50_rgb': 'Uniform50 RGB (ours)',
    'diffpano_uniform89_screened_select': 'Uniform89 screened-select (ours)',
}


def sources():
    paths = sorted(p for p in SOURCE.iterdir()
                   if p.is_file() and p.suffix in ('.py', '.slurm', '.sh'))
    paths += [ROOT / name for name in (
        'studies/panorama_metrics/numerics.py',
        'studies/panorama_metrics/environment.sh',
        'studies/faster_uniform_resume/completion.py',
        'diffpano/conditioning.py',
    )]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def load_plan():
    assert sha(PLAN) == (OUT / 'plan.sha256').read_text().strip(), 'Plan changed'
    plan = read(PLAN)
    assert plan['source_hashes'] == sources(), 'FID evaluator sources changed'
    for name, value in plan['input_hashes'].items():
        assert sha(name) == value, 'FID input changed: ' + name
    return plan
