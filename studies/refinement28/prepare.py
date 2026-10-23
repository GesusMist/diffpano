"""CPU-only preparation and immutable source/configuration freeze for 48 runs."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import tarfile
import unittest

from studies.followup24.common import PROMPTS24, prompt
from studies.refinement28.common import ROOT, OUT, definitions, digest, read, sha, sources, write, preserved
from studies.refinement28.configuration import semantic, schedule_evidence


def collect_preservation(evidence):
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    # Include the pre-existing local adapters too; never include our new package.
    for directory in ('faster_uniform_resume', 'perspective_figures', 'spherediff_vlm_eval', 'step_comparison_figures'):
        files += [str(p.relative_to(ROOT)) for p in (ROOT / 'studies' / directory).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / 'prompts').rglob('*.txt')]
    hashes = {name: sha(ROOT / name) for name in sorted(set(files)) if name and (ROOT / name).is_file()}
    hashes.update({path: sha(path) for path in evidence})
    return dict(hashes=hashes, outputs_symlink=os.readlink(ROOT / 'outputs'),
                branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                created_utc=datetime.now(timezone.utc).isoformat())


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run preparation in a Slurm CPU allocation')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise FileExistsError('Manifest already frozen; inspect the existing study instead of overwriting it')
    for name in ('logs/generation', 'logs/report', 'validation', 'cases', 'runtime-groups'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    schedule, schedule_proof = schedule_evidence()
    coverage_path = ROOT / 'outputs/10.9ournew-flux28/seed0-v1/validation/gpu.json'
    coverage_record = read(coverage_path)
    assert coverage_record['passed']
    coverage = coverage_record['coverage']['old89']
    assert coverage['all_pixels_covered'] and coverage['all_supported_edges_connected']
    assert coverage['coverage_resolution'] == [2048, 4096]
    source_hashes = sources()
    source_digest = digest(source_hashes)
    rows = definitions()
    for row in rows:
        meaning = semantic(row, schedule)
        assert meaning['geometry']['angular_sha256'] == coverage['geometry_sha256']
        row.update(semantic=meaning, semantic_sha256=digest(meaning), source_sha256=source_digest)
    assert len(rows) == 48 and len({r['semantic_sha256'] for r in rows}) == 48
    evidence_paths = [schedule_proof['path'], str(coverage_path)]
    # Preserve the source metadata/configuration consulted by the shared adapter.
    from studies.all_prompts.common import BASES
    evidence_paths.append(str(BASES['ruins'] / 'flux' / 'metadata.json'))
    write(OUT / 'startup-preservation.json', collect_preservation(evidence_paths))
    with (OUT / 'validation' / 'tests.log').open('w') as stream:
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'studies/refinement28'),
                                                   pattern='tests_*.py', top_level_dir=str(ROOT))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('CPU tests failed; see validation/tests.log')
    assert source_hashes == sources(), 'Implementation changed during preparation'
    manifest = dict(study='refinement28', version=1, rows=rows,
                    prompts={name: prompt(name)[0] for name in PROMPTS24},
                    source_hashes=source_hashes, source_sha256=source_digest,
                    implementation_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    schedule_evidence=schedule_proof,
                    coverage_evidence=dict(path=str(coverage_path), sha256=sha(coverage_path), record=coverage),
                    profile_indices=[0, 1], bulk_indices=list(range(2, 48)),
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    inference_steps=28, refinement_steps=[5, 8], paired_prompt_count=24,
                    method='Old89 poisson_select screened lambda_blending', seed=0,
                    existing_compatible_results=0, generation_image_count=48)
    write(OUT / 'manifest.json', manifest)
    with tarfile.open(OUT / 'implementation.tar.gz', 'w:gz') as archive:
        for name in source_hashes:
            archive.add(ROOT / name, arcname=name)
    preserved()
    write(OUT / 'validation.json', dict(passed=True, tests_run=result.testsRun,
                                       source_sha256=source_digest, manifest_sha256=sha(OUT / 'manifest.json'),
                                       schedule_verified=True, full_resolution_old89_coverage_verified=True,
                                       gpu_profile_required_indices=[0, 1]))
    print('PREPARED 48 images / 24 prompts / exact refinement5of28 and8of28;', result.testsRun, 'CPU tests passed', flush=True)


if __name__ == '__main__':
    main()
