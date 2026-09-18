#!/bin/bash
#SBATCH --job-name=small-ij-report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
module purge
module load GCC/13.2.0 OpenMPI/4.1.6 Python/3.11.5 PyTorch/2.7.0
source /scratch/user/shig/virtual_envs/diffpano/bin/activate
cd /home/shig/diffpano
export OMP_NUM_THREADS=2
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw
from studies.planar_small_ij.common import ROOT, FACTORS, read, write
rows=[]; thumbs=[]
for backend in FACTORS:
    completed=[]
    for method in ('I','J'):
        p=ROOT/backend/method
        ok=(p/'complete.json').is_file()
        rows.append(dict(backend=backend,method=method,complete=ok,output=str(p/'generation/result.png')))
        if ok:
            completed.append((read(p/'complete.json'),read(p/'generation/metadata.json')))
    if len(completed)==2:
        a,b=completed
        for k in ('initial_native_sha256','conditioning_sha256','guided_predictions'):
            assert a[0][k]==b[0][k],(backend,k)
        for k in ('model','scheduler','scheduler_timesteps','scheduler_sigmas','native_geometry','backend_details'):
            assert a[1][k]==b[1][k],(backend,k)
        strip=Image.new('RGB',(1024,300),'white'); draw=ImageDraw.Draw(strip)
        for col,method in enumerate(('I','J')):
            im=Image.open(ROOT/backend/method/'generation/result.png').convert('RGB')
            assert im.size==((1024,512) if backend=='sd2' else (2048,1024))
            im.thumbnail((512,256));strip.paste(im,(512*col,35));draw.text((512*col+8,10),backend+' '+method,fill='black')
        thumbs.append(strip)
complete=all(r['complete'] for r in rows)
write(ROOT/'summary.json',dict(complete=complete,runs=rows,paired_checks_passed=len(thumbs)))
if thumbs:
    contact=Image.new('RGB',(1024,300*len(thumbs)),'white')
    for i,im in enumerate(thumbs):contact.paste(im,(0,300*i))
    contact.save(ROOT/'comparison.png')
(ROOT/'REPORT.md').write_text('# I/J: 256×256 RGB patches, stride 64\n\n'+
    ('All eight runs completed; pairwise noise, conditioning, schedule and model checks passed.\n' if complete else 'Some runs did not complete; inspect summary.json and Slurm logs.\n')+
    '\n|Backend|Method|Completed|Output|\n|---|---|---|---|\n'+
    ''.join('|{backend}|{method}|{complete}|{output}|\n'.format(**r) for r in rows))
print('Complete:',complete,'paired checks:',len(thumbs),flush=True)
if not complete:raise SystemExit(1)
PY
