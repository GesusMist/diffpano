"""Create the immutable 75-case continuation plan; no inference or submission."""
from collections import Counter
from datetime import datetime, timezone
import fcntl

from studies.layout24 import common as original
from studies.faster_uniform_resume.common import (
    ROOT, OUT, PLAN, PROFILE_INDEX, ACCOUNT, adapter_sources, load_plan,
)
from studies.faster_uniform_resume.completion import (
    load_verifier, INVENTORY, INVENTORY_SHA256, MANIFEST_SHA256,
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if PLAN.exists():
            plan = load_plan()
            print('VALID_PLAN', len(plan['pending_indices']), flush=True)
            return
        manifest = original.freeze_check()
        original.gate()
        verifier = load_verifier(strict_complete=original.complete)
        for index in verifier.protected_records:
            assert verifier.protected_complete(manifest['rows'][index]), index
        indices = sorted(verifier.fresh_indices)
        assert len(indices) == 75 and PROFILE_INDEX in indices
        generation = original.read(original.OUT / 'generation_plan.json')
        targets = {row['index']: row for row in generation['rows']}
        pending = [manifest['rows'][index] for index in indices]
        for row in pending:
            assert row['layout'].startswith('uniform')
            assert targets[row['index']]['state'] == 'generate'
            directory = original.folder(row)
            # All 75 are absent at this handoff, so any artifact needs review.
            assert not directory.exists() or not list(directory.iterdir()), directory
        counts = Counter(row['variant'] for row in pending)
        assert dict(counts) == {
            'diffpano_uniform89_rgb': 16,
            'diffpano_uniform70_rgb': 20,
            'diffpano_uniform50_rgb': 20,
            'diffpano_uniform89_screened_select': 19,
        }
        runtime = ROOT / 'outputs/10.16faster-reconstruction-20261016'
        summary = original.read(runtime / 'summary.json')
        assert summary['cluster'] == 'FASTER'
        assert summary['gpu']['all_environments_passed']
        assert summary['source']['head'] == '30580bb79155098e2b0d6dec9b51f80c638260f9'
        evidence = {
            str(runtime / name): original.sha(runtime / name)
            for name in ('summary.json', 'source-freeze-validation.json',
                         'gpu-diffpano.json', 'gpu-evaluation.json')
        }
        for name in ('logs/generation', 'logs/evaluation', 'logs/report', 'execution'):
            (OUT / name).mkdir(parents=True, exist_ok=True)
        plan = dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            cluster='FASTER', account=ACCOUNT,
            scope='Only the 75 genuinely missing Uniform-camera FLUX panoramas',
            manifest_path=str(original.OUT / 'manifest.json'),
            manifest_sha256=MANIFEST_SHA256,
            migration_inventory=str(INVENTORY),
            migration_inventory_sha256=INVENTORY_SHA256,
            original_scientific_source_sha256=original.digest(manifest['source_hashes']),
            adapter_source_hashes=adapter_sources(), runtime_evidence_hashes=evidence,
            protected_indices=sorted(verifier.protected_records),
            pending_indices=indices, pending_per_variant=dict(counts),
            profile_index=PROFILE_INDEX,
            bulk_indices=[index for index in indices if index != PROFILE_INDEX],
            pending_cases=[dict(index=row['index'], case_id=row['case_id'],
                                semantic_sha256=row['semantic_sha256']) for row in pending],
            expected_uniform_count=96, existing_uniform_count=21,
            excluded_missing_old_count=40,
            generation_resources=dict(partition='gpu', gres='gpu:a100:1',
                                      cpus=4, memory='64G', walltime='01:00:00',
                                      maximum_concurrent_generations=2),
            instructions='Run the one screened-select case first; release the 74-case '
                         'array only after its full scientific completion check passes. '
                         'Never overwrite an existing final image or retry automatically.',
        )
        original.write(PLAN, plan)
        (OUT / 'plan.sha256').write_text(original.sha(PLAN) + '\n')
        load_plan()
        print('PREPARED_UNIFORM_RESUME', dict(counts), 'profile', PROFILE_INDEX, flush=True)


if __name__ == '__main__':
    main()
