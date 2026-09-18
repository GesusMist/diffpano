import unittest
from dataclasses import replace
from types import SimpleNamespace
import torch
from diffpano.erp_noise_initialization import native_cameras,primary_noise_size
from diffpano.projection import perspective_world_rays
from studies.bridge_erp4k.common import *

class FollowupTests(unittest.TestCase):
    def test_exact_requested_subset_and_erp_only_changes(self):
        wanted={'flux':6,'pixeldit':4,'sana':5,'sd2':3,'sd35':4};seen=set()
        for name,codes in SELECTED.items():
            self.assertEqual(len(codes),wanted[name])
            for code in codes:
                cell=cell_name(code);seen.add((name,cell));old=read(BASE/'cells'/name/cell/'metadata.json');c=make_config(name,cell)
                self.assertEqual({r['path'] for r in config_diff(old['config'],c.to_dict())},{'erp.height','erp.width'})
                self.assertEqual((c.erp.height,c.erp.width),ERP_SIZE)
                self.assertEqual((c.view.height,c.view.width),tuple(old['local_RGB_resolution']))
                self.assertEqual(c.consensus_transition.mode,'preserve_current_state')
        self.assertEqual(len(seen),22)

    def test_identical_saved_cameras_rays_noise_grids_and_pyramid_settings(self):
        checked=set()
        for name,codes in SELECTED.items():
            c=make_config(name,cell_name(codes[0]));old=baseline_config(name,cell_name(codes[0]));a=saved_cameras(c.view,ERP_SIZE);b=baseline_cameras(old.view,(old.erp.height,old.erp.width))
            self.assertEqual(a,b);self.assertEqual(len(a),89);self.assertEqual(c.warp.lpw,old.warp.lpw)
            self.assertTrue(all(cam.fov_x==cam.fov_y==80 for cam in a))
            p=read(BASE/'preflight'/(name+'.json'))['provenance'];shape=p['local_native_resolution']
            backend=SimpleNamespace(native_spatial_shape_for_rgb=lambda h,w:tuple(shape))
            self.assertEqual(list(primary_noise_size(native_cameras(backend,a))),p['noise_grid'])
            if c.view.height not in checked:
                for slot in (7,59,87):self.assertTrue(torch.equal(perspective_world_rays(a[slot],device=torch.device('cpu')),perspective_world_rays(b[slot],device=torch.device('cpu'))))
                checked.add(c.view.height)

    def test_reject_wrong_resolution_and_invalid_code(self):
        c=make_config('flux',cell_name('0001'))
        with self.assertRaises(AssertionError):saved_cameras(c.view,(1024,2048))
        for code in ('001','00111','0021'):
            with self.assertRaises(ValueError):cell_name(code)

    def test_existing_scientific_sources_remain_frozen(self):
        v=read(BASE/'validation.json');self.assertEqual(baseline_source_hashes(),v['source_hashes'])
        require_baseline()

if __name__=='__main__':unittest.main()
