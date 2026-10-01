import math
import unittest
from dataclasses import replace
import torch
from diffpano.camera import PerspectiveCamera
from diffpano.bridge_factorial import angular_geometry,digest
from diffpano.erp_noise_initialization import native_cameras,primary_noise_size
from diffpano.gwtf_noise_initialization import initialize_gwtf_shared_noise,GWTFlowNoiseConfig
from studies.gwtf_noise.common import GeometryBackend
from studies.tt_cea.common import base_config,BACKENDS,BASES,read
from studies.tt_cea.cameras import fibonacci_centers,routing

class CameraTests(unittest.TestCase):
    def test_exact_formula_ids_rasters_and_routing(self):
        poses=fibonacci_centers();self.assertEqual(len(poses),89);self.assertEqual(poses,fibonacci_centers())
        digest_set=set()
        for name in BACKENDS:
            c,b,old,m=base_config(name,'ruins')
            cams=[PerspectiveCamera(**p,height=c.view.height,width=c.view.width) for p in poses]
            digest_set.add(digest(angular_geometry(cams)))
            self.assertEqual(primary_noise_size(native_cameras(b,old)),primary_noise_size(native_cameras(b,cams)))
            for j,cam in enumerate(cams):
                self.assertAlmostEqual(math.sin(cam.pitch),1-2*(j+.5)/89,places=14)
                self.assertEqual((cam.fov_x,cam.fov_y,cam.roll),(80.,80.,0.))
            labels=['top','upper','equator','lower','bottom'];slots=routing(cams,labels)['indices']
            from diffpano.conditioning import expand_directional_prompts
            scores=torch.stack([v.forward() for v in cams])@expand_directional_prompts(labels).directions.T
            self.assertEqual(slots,scores.argmax(1).tolist())
            stat=read('outputs/gwtf-noise-comparison/20260922/statistical-preflight/'+name+'.json')
            self.assertEqual(m['initialization']['initial_local_sha256'],stat['exact_seed0']['gwtf']['record']['initial_local_sha256'])
        self.assertEqual(len(digest_set),1)
    def test_projection_independent_noise_binding_and_ids(self):
        c,b,old,m=base_config('sd2','ruins');stub=GeometryBackend(3,1)
        covers={'old89':[replace(v,height=4,width=4) for v in old],
            'ea89':[PerspectiveCamera(**p,height=4,width=4) for p in fibonacci_centers()]}
        hashes={}
        for cover,cams in covers.items():
            ids=list(range(89)) if cover=='old89' else ['ea89:%03d'%i for i in range(89)]
            cfg=GWTFlowNoiseConfig(*primary_noise_size(cams),0)
            # The unchanged initializer is intentionally independent of canvas projection.
            a,ma=initialize_gwtf_shared_noise(stub,cams,cfg,camera_slots=ids)
            z,mz=initialize_gwtf_shared_noise(stub,cams,cfg,camera_slots=ids,execution_order=list(reversed(range(89))))
            self.assertEqual(ma['initial_local_sha256'],mz['initial_local_sha256'])
            self.assertEqual(ma['camera_slots'],ids);hashes[cover]=ma['initial_local_sha256']
        self.assertNotEqual(*hashes.values())
