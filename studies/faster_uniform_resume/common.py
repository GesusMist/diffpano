"""Resume bookkeeping, separate from the frozen scientific implementation."""
from pathlib import Path

from studies.layout24 import common as original
from studies.faster_uniform_resume.completion import (
    INVENTORY, INVENTORY_SHA256, MANIFEST_SHA256,
)

ROOT = original.ROOT
OUT = ROOT / 'outputs/10.19uniform-resume-faster'
PLAN = OUT / 'plan.json'
SOURCE = ROOT / 'studies/faster_uniform_resume'
PROFILE_INDEX = 17
ACCOUNT = '142684268708'


def adapter_sources():
    return {
        str(path.relative_to(ROOT)): original.sha(path)
        for path in sorted(SOURCE.iterdir())
        if path.is_file() and path.suffix in ('.py', '.slurm', '.sh')
    }


def load_plan():
    if original.sha(PLAN) != (OUT / 'plan.sha256').read_text().strip():
        raise ValueError('Resume plan changed')
    plan = original.read(PLAN)
    if plan['adapter_source_hashes'] != adapter_sources():
        raise ValueError('Resume adapter changed after preparation')
    if original.sha(INVENTORY) != INVENTORY_SHA256:
        raise ValueError('Migration evidence changed')
    if original.sha(original.OUT / 'manifest.json') != MANIFEST_SHA256:
        raise ValueError('Original manifest changed')
    original.freeze_check()
    for path, expected in plan['runtime_evidence_hashes'].items():
        if original.sha(path) != expected:
            raise ValueError('FASTER runtime evidence changed: ' + path)
    return plan
