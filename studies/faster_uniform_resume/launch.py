"""Submit the profile-gated continuation once, recording each Slurm job."""
from datetime import datetime, timezone
import fcntl
import subprocess

from studies.layout24 import common as original
from studies.faster_uniform_resume.common import ROOT, OUT, PLAN, SOURCE, load_plan


def main():
    plan = load_plan()
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record_path = OUT / 'launch.json'
        if record_path.exists():
            raise RuntimeError('Launch already recorded; inspect its jobs before any resubmission')
        queue = subprocess.check_output(
            ['squeue', '-h', '-u', 'shig', '-o', '%j'], text=True)
        assert not any(line.startswith('uniform-faster') for line in queue.splitlines()), (
            'An existing Uniform continuation job must be reviewed first')
        record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                      plan_sha256=original.sha(PLAN), jobs={}, state='submitting')
        original.write(record_path, record)

        def submit(name, script, options=(), arguments=()):
            command = ['sbatch', '--parsable', '--chdir=' + str(ROOT)]
            command += list(options) + [str(SOURCE / script)] + list(arguments)
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            if result.returncode:
                record['state'] = 'submission_failed'
                record['error'] = dict(stage=name, stdout=result.stdout, stderr=result.stderr)
                original.write(record_path, record)
                raise RuntimeError(result.stderr)
            job = result.stdout.strip().split(';')[0]
            assert job.isdigit(), result.stdout
            record['jobs'][name] = dict(job_id=job, command=command, stdout=result.stdout,
                                       stderr=result.stderr)
            original.write(record_path, record)
            print('SUBMITTED', name, job, flush=True)
            return job

        profile = submit('profile', 'generate.slurm',
                         options=['--job-name=uniform-faster-profile'],
                         arguments=['--stage', 'profile', '--index', str(plan['profile_index'])])
        bulk = submit('bulk', 'generate.slurm', options=[
            '--job-name=uniform-faster-bulk',
            '--dependency=afterok:' + profile,
            '--kill-on-invalid-dep=yes',
            '--array=' + ','.join(map(str, plan['bulk_indices'])) + '%2',
        ], arguments=['--stage', 'bulk'])
        evaluation = submit('evaluation', 'evaluate.slurm', options=[
            '--dependency=afterany:' + bulk + ':' + profile,
            '--kill-on-invalid-dep=yes',
        ])
        submit('report', 'report.slurm', options=[
            '--dependency=afterany:' + evaluation, '--kill-on-invalid-dep=yes'])
        record['state'] = 'submitted'
        original.write(record_path, record)


if __name__ == '__main__':
    main()
