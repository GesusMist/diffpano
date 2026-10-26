#!/bin/bash
#SBATCH --job-name=refinement28-resume-report
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:20:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the intended checkout}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
bash studies/all_prompts/environment.sh diffpano - <<'PY'
import fcntl
import shutil
from studies.refinement28.common import sha, write
from studies.refinement28_resume.common import ORIGINAL_OUT, OUT, gate, verify_protected_outputs

plan = gate()
verify_protected_outputs(plan)
with (OUT / "prior-report.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    target = OUT / "prior-report"
    if target.exists():
        raise RuntimeError("Prior report archive already exists; inspect before rerunning the report")
    target.mkdir()
    files = {}
    for relative in ("completion.json", "completion.csv", "figures/index.json"):
        source = ORIGINAL_OUT / relative
        if not source.exists():
            files[relative] = {"state": "absent"}
            continue
        if not source.is_file() or source.is_symlink():
            raise RuntimeError("Unexpected prior report file: " + str(source))
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(source)
        shutil.copy2(source, destination)
        if sha(destination) != expected or sha(source) != expected:
            raise RuntimeError("Prior report copy verification failed: " + relative)
        files[relative] = {"state": "archived", "sha256": expected}
    write(target / "manifest.json", {"files": files, "original_output": str(ORIGINAL_OUT)})
PY
exec bash studies/all_prompts/environment.sh diffpano -m studies.refinement28.report
