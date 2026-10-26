"""Archive a frozen failed attempt, then run its unchanged scientific trajectory."""
import argparse
from contextlib import ExitStack
import ctypes
from datetime import datetime, timezone
import errno
import fcntl
import getpass
import os
from pathlib import Path
import socket
import subprocess
import traceback

from studies.refinement28 import common as original
from studies.refinement28 import run as original_run
from studies.refinement28_resume import common


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require_index(index):
    if type(index) is not int or index not in common.INDICES:
        raise ValueError('Only the ten explicitly authorized retry indices are allowed')


def require_no_original_jobs(query=None):
    """Different authorized resume cases may run concurrently; old jobs may not."""
    query = subprocess.check_output if query is None else query
    text = query(['squeue', '-M', 'faster', '-h', '-u', getpass.getuser(), '-o', '%i|%j'], text=True)
    old_ids = set()
    launch = common.ORIGINAL_OUT/'launch.json'
    if launch.exists():
        old_ids = {str(value['job_id']) for value in common.read(launch)['jobs'].values()}
    active = []
    for line in text.splitlines():
        if not line.strip():
            continue
        job, separator, name = line.partition('|')
        if not separator:
            raise ValueError('Cannot parse scheduler output; refusing archival')
        job, name = job.strip(), name.strip()
        original_id = job.split('_', 1)[0].split('.', 1)[0] in old_ids
        original_name = name.startswith('refinement28-') and not name.startswith('refinement28-resume')
        if original_id or original_name:
            active.append(dict(job_id=job, name=name))
    if active:
        raise RuntimeError('Original refinement28 jobs are still active: ' + repr(active))
    return dict(checked_utc=utc_now(), active_original_jobs=[])


def _exists(path):
    return path.exists() or path.is_symlink()


def _check_snapshot(path, expected):
    if path.is_symlink():
        raise ValueError('A case directory cannot be replaced by a symlink')
    actual = common.snapshot(path)
    if actual != expected:
        raise ValueError('Case differs from its prepared recovery snapshot: ' + str(path))
    return actual


def _rename_no_replace(source, destination):
    """Atomically move a directory without replacing any pre-existing archive.

    Some shared filesystems reject renameat2 flags. On those filesystems an
    exclusive mkdir reserves the absent destination, then POSIX rename replaces
    only that empty reservation. A nonempty destination is never overwritten.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, 'renameat2', None)
    if function is not None:
        function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        function.restype = ctypes.c_int
        if function(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
            return
        error = ctypes.get_errno()
        if error not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            raise OSError(error, os.strerror(error), str(destination))
    destination = Path(destination)
    destination.mkdir(mode=0o700)  # Exclusive: never reuse an existing directory.
    reserved = destination.stat()
    try:
        os.rename(source, destination)
    except BaseException:
        try:
            current = destination.lstat()
            if (current.st_dev, current.st_ino) == (reserved.st_dev, reserved.st_ino):
                # rmdir also refuses any data arriving in the reservation.
                destination.rmdir()
        except OSError:
            pass
        raise


def _event(path, journal, state, **fields):
    stamp = utc_now()
    journal.update(fields, state=state, updated_utc=stamp)
    journal.setdefault('events', []).append(dict(state=state, utc=stamp))
    common.write(path, journal)


def _archive(directory, archive, expected, journal_path, journal):
    """Resume only a provably incomplete archival operation, never generation."""
    if journal['state'] == 'archive_intent':
        if expected['exists']:
            if _exists(directory) and not _exists(archive):
                with ExitStack() as stack:
                    old_lock = directory/'case.lock'
                    if old_lock.exists():
                        if old_lock.is_symlink():
                            raise ValueError('Original case lock cannot be a symlink')
                        lock = stack.enter_context(old_lock.open('r'))
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    _check_snapshot(directory, expected)
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    _rename_no_replace(directory, archive)
            elif _exists(directory) or not _exists(archive):
                raise RuntimeError('Ambiguous archive state; retain both paths for inspection')
            _check_snapshot(archive, expected)
        elif _exists(directory) or _exists(archive):
            raise RuntimeError('A previously absent case appeared after recovery preparation')
        _event(journal_path, journal, 'archived', archive_exists=expected['exists'])
    if journal['state'] != 'archived':
        raise RuntimeError('This recovery attempt already started; refusing duplicate generation')
    if expected['exists']:
        _check_snapshot(archive, expected)
    elif _exists(archive):
        raise RuntimeError('Unexpected archive for an originally absent case')
    if _exists(directory):
        raise RuntimeError('Canonical case directory is not empty after archival')


def _identity(index, row, entry, plan):
    return dict(index=index, case_id=row['case_id'],
                original_manifest_sha256=plan['original_manifest_sha256'],
                original_source_sha256=row['source_sha256'], semantic_sha256=row['semantic_sha256'],
                resume_manifest_sha256=common.sha(common.PLAN),
                resume_source_sha256=plan['source_sha256'],
                prior_snapshot_sha256=entry['snapshot']['digest'])


def _finish(row, directory, journal_path, journal, preservation_after):
    assert original.complete(row, verify=True), 'Fresh trajectory has not passed original completion checks'
    metadata = common.read(directory/'metadata.json')
    result = dict(index=row['index'], case_id=row['case_id'],
                  original_metadata_sha256=common.sha(directory/'metadata.json'),
                  original_status_sha256=common.sha(directory/'status.json'),
                  original_artifacts=metadata['artifacts'], original_execution=metadata.get('execution'),
                  resume_identity=journal['identity'], archive_path=journal.get('archive_path'),
                  preservation_before=journal.get('preservation_before'),
                  preservation_after=preservation_after, recorded_utc=utc_now())
    # Companion evidence leaves the frozen runner's signed metadata/status intact.
    common.write(journal_path.parent/'result.json', result)
    _event(journal_path, journal, 'complete', result_sha256=common.sha(journal_path.parent/'result.json'),
           preservation_after=preservation_after)
    return metadata


def run(index):
    require_index(index)
    original_run.require_allocation()
    plan = common.gate()
    manifest = original.freeze_check()
    original.validation_gate(manifest)
    assert common.sha(common.ORIGINAL_OUT/'manifest.json') == plan['original_manifest_sha256']
    row = manifest['rows'][index]
    candidates = [entry for entry in plan['cases'] if entry['index'] == index]
    assert len(candidates) == 1, 'Missing or ambiguous recovery case'
    entry = candidates[0]
    for key in ('case_id', 'semantic_sha256', 'source_sha256'):
        assert entry[key] == row[key], 'Recovery case does not match original manifest'
    directory = original.folder(row)
    assert str(directory) == entry['directory']
    archive = common.OUT/'attempts'/row['case_id']
    journal_path = common.OUT/'journals'/row['case_id']/'attempt.json'
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with (journal_path.parent/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert common.gate() == plan, 'Recovery plan changed before archival'
        queue = require_no_original_jobs()
        preservation = common.preserve_original()
        identity = _identity(index, row, entry, plan)
        journal = common.read(journal_path) if journal_path.exists() else None
        if journal is not None:
            assert journal['identity'] == identity, 'Recovery journal belongs to a different plan'
        if original.complete(row, verify=True):
            if journal is not None and journal['state'] == 'complete':
                result_path = journal_path.parent/'result.json'
                assert common.sha(result_path) == journal['result_sha256']
                result = common.read(result_path)
                assert common.sha(directory/'metadata.json') == result['original_metadata_sha256']
                assert common.sha(directory/'status.json') == result['original_status_sha256']
                return common.read(directory/'metadata.json')
            if journal is None:
                # Another validated completion wins over an obsolete retry snapshot.
                journal = dict(identity=identity, archive_path=None, state='existing_complete',
                               preservation_before=preservation, events=[])
            return _finish(row, directory, journal_path, journal, preservation)
        if journal is None:
            if _exists(archive):
                raise FileExistsError('Archive already exists without its journal: ' + str(archive))
            _check_snapshot(directory, entry['snapshot'])
            journal = dict(identity=identity, archive_path=str(archive), canonical_path=str(directory),
                           job=os.environ['SLURM_JOB_ID'], array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),
                           hostname=socket.gethostname(), created_utc=utc_now(),
                           preservation_before=preservation, queue_check=queue, events=[])
            _event(journal_path, journal, 'archive_intent')
        _archive(directory, archive, entry['snapshot'], journal_path, journal)
        # Recheck all bytes after the move and before entering the frozen runner.
        assert common.gate() == plan
        preservation = common.preserve_original()
        _event(journal_path, journal, 'generation_started', preservation_before=preservation,
               generation_job=os.environ['SLURM_JOB_ID'], generation_started_utc=utc_now())
        saved_preserved = original.preserved
        try:
            original.preserved = common.preserve_original
            metadata = original_run.run(index, stage='bulk')
            assert original.complete(row, verify=True)
            assert common.gate() == plan, 'Recovery helper changed during generation'
            return _finish(row, directory, journal_path, journal, common.preserve_original())
        except BaseException:
            _event(journal_path, journal, 'failed', error=traceback.format_exc())
            raise
        finally:
            original.preserved = saved_preserved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int)
    args = parser.parse_args(argv)
    index = int(os.environ['SLURM_ARRAY_TASK_ID']) if args.index is None else args.index
    run(index)


if __name__ == '__main__':
    main()
