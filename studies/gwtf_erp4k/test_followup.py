import unittest
from diffpano.erp_noise_initialization import native_cameras,primary_noise_size
from diffpano.bridge_factorial import digest,angular_geometry
from studies.gwtf_erp4k.common import *

class FollowupTests(unittest.TestCase):
    def test_only_erp_dimensions_change(self):
        for n in BACKENDS:
            c,_,_,old,_=geometry(n)
            self.assertEqual((c.erp.height,c.erp.width),ERP_SIZE)
            self.assertEqual({x['path'] for x in differences(old['config'],c.to_dict())},{'erp.height','erp.width'})
    def test_native_noise_grid_and_saved_cameras_unchanged(self):
        for n in BACKENDS:
            c,b,cams,old,_=geometry(n)
            self.assertEqual(len(cams),89)
            self.assertTrue(all(x.fov_x==x.fov_y==80 for x in cams))
            self.assertEqual(digest(angular_geometry(cams)),old['camera_geometry_sha256'])
            self.assertEqual(list(primary_noise_size(native_cameras(b,cams))),old['noise_grid'])
            self.assertEqual(list(b.native_spatial_shape_for_rgb(c.view.height,c.view.width)),old['local_native_resolution'])
    def test_reused_initializer_and_generation_function(self):
        from studies.gwtf_erp4k.run import runner
        from studies.gwtf_noise import run as baseline
        r=runner()
        self.assertIs(r.initialize_gwtf_shared_noise,baseline.initialize_gwtf_shared_noise)
        self.assertEqual(r.generate.__wrapped__.__code__.co_code,baseline.generate.__wrapped__.__code__.co_code)
        self.assertEqual(r.ROOT,ROOT)
        self.assertEqual(baseline.ROOT,BASE)
