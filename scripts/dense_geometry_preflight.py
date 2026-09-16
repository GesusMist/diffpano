"""Produce only two shared compact geometry JSONs for L and M."""
import json
from pathlib import Path
from diffpano.dense_geometry import geometry_record, preset_record, search_dense_cover, OVERLAPS

OUTPUT = Path('outputs/vae-residual-controls/20260915-dense-lm/geometry')

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    l = preset_record('L', .6, geometry_record(.6))
    assert l['camera_count'] == 89
    assert all(r['coverage_percent'] == 100 for r in l['full_resolution_verification'])
    l['official_source'] = dict(repository='pmh9960/SphereDiff',commit='2c8c68ba088f2803b3dce4b52b7b0d68bc996139')
    (OUTPUT/'experiment_l.json').write_text(json.dumps(l,indent=2)+'\n')
    overlap, trials, full = search_dense_cover()
    m = preset_record('M', overlap, full)
    m.update(search_overlap_candidates=OVERLAPS, camera_cap=300, tested_candidates=trials,
             selection='smallest tested camera count passing both actual ERP resolutions')
    (OUTPUT/'experiment_m.json').write_text(json.dumps(m,indent=2)+'\n')
    from diffpano.config import load_experiment_config
    from diffpano.camera import spherediff_camera_cover
    from diffpano.erp_local_consensus import camera_digest
    for label,record in (('L',l),('M',m)):
        for backend in ('sd35','flux','sana','sd2','pixeldit'):
            config=load_experiment_config('configs/experiments/erp_later/'+backend+'-'+label.lower()+'.yaml')
            cameras=spherediff_camera_cover(config.view,overlap_fraction=record['overlap_fraction'])
            verified=next(r for r in record['full_resolution_verification'] if r['erp_size']==[config.erp.height,config.erp.width])
            assert camera_digest(cameras)==verified['camera_sha256'], 'Config/preflight camera digest mismatch'
    print('All ten config camera digests match the verified geometry',flush=True)
    print('L',l,flush=True); print('M',m,flush=True)

if __name__ == '__main__': main()
