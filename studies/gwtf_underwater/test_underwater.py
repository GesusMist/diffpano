import unittest
from diffpano.erp_noise_initialization import native_cameras,primary_noise_size
from diffpano.bridge_factorial import digest,angular_geometry
from studies.gwtf_underwater.common import *

class UnderwaterTests(unittest.TestCase):
    def test_only_prompt_changes(self):
        self.assertEqual(len(PROMPT.read_text().splitlines()),5)
        for n in BACKENDS:
            c,_,_,old,_=geometry(n)
            self.assertEqual({x['path'] for x in differences(old['config'],c.to_dict())},{'prompt.path'})
            self.assertEqual((c.erp.height,c.erp.width),ERP_SIZE)
            with self.assertRaisesRegex(ValueError,'exact ruins prompt'):
                c.validate()  # Historical guard remains intact.
    def test_native_geometry_stays_fixed(self):
        hashes=set()
        for n in BACKENDS:
            c,b,cams,old,_=geometry(n)
            self.assertEqual(len(cams),89)
            self.assertTrue(all(x.fov_x==x.fov_y==80 for x in cams))
            h=digest(angular_geometry(cams));hashes.add(h)
            self.assertEqual(h,old['camera_geometry_sha256'])
            self.assertEqual(list(primary_noise_size(native_cameras(b,cams))),old['noise_grid'])
            self.assertEqual(list(b.native_spatial_shape_for_rgb(c.view.height,c.view.width)),old['local_native_resolution'])
            self.assertNotEqual(expected_hash(n,'shared'),expected_hash(n,'gwtf'))
        self.assertEqual(len(hashes),1)
    def test_frozen_generation_and_both_initializers(self):
        from studies.gwtf_underwater.run import runner
        from studies.gwtf_noise import run as old
        r=runner()
        self.assertIs(r.initialize_gwtf_shared_noise,old.initialize_gwtf_shared_noise)
        self.assertIs(r.initialize_erp_noise,old.initialize_erp_noise)
        self.assertEqual(r.generate.__wrapped__.__code__.co_code,old.generate.__wrapped__.__code__.co_code)
        self.assertEqual(r.ROOT,ROOT)
