"""Apply the reviewed shared change only after earlier frozen jobs finish."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/home/shig/diffpano');STAGE=ROOT/'studies/ournew_comparison/staged'
def main():
    queue=subprocess.check_output(['squeue','-u','shig','-h','-r','-o','%i|%j'],universal_newlines=True)
    assert not any(x.split('|')[1].startswith(('grad-score','gradref')) for x in queue.splitlines()),'Earlier jobs still use frozen sources'
    original=json.loads((STAGE/'original-hashes.json').read_text())
    for name,value in original.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,'Shared source changed since staging: '+name
    for name in original:
        (ROOT/name).write_bytes((STAGE/Path(name).name).read_bytes())
    # Explicit adapters retain old record interpretation, without changing old manifests.
    for name in ['studies/gradient_refinement/run.py','studies/gradient_refinement_retry/run.py','studies/gradient_score_refresh/recover.py']:
        path=ROOT/name;text=path.read_text()
        text=text.replace("GradientSettings(row['gradient_mode'],reference_mode=row['reference_mode'])", "GradientSettings.from_record(row['semantic']['fusion'])")
        text=text.replace("GradientSettings(**row['semantic']['fusion'])", "GradientSettings.from_record(row['semantic']['fusion'])")
        text=text.replace("assert asdict(settings)==row['semantic']['fusion']", "assert settings.matches_record(row['semantic']['fusion'])")
        path.write_text(text)
    path=ROOT/'studies/gradient_refinement/tests/test_poisson.py';text=path.read_text()
    text=text.replace("reference_mode=r,coarse_rows", "reference_mode=r,single_pixel_source='legacy_source_patch',coarse_rows")
    text=text.replace("GradientSettings('poisson_max',reference_mode='single_pixel')", "GradientSettings('poisson_max',reference_mode='single_pixel',single_pixel_source='legacy_source_patch')")
    path.write_text(text)
    print('ANCHOR_UPDATE_APPLIED',flush=True)
if __name__=='__main__':main()
