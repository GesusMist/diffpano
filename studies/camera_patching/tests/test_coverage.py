import unittest
import torch
from diffpano.config import ViewConfig
from diffpano.geometry import erp_world_directions
from studies.tt_cea.canvas import cea_rays,project_rays
from studies.camera_patching.cameras import cover
from studies.camera_patching.coverage import *

class CoverageTests(unittest.TestCase):
    def test_production_rays_and_frustum_agreement(self):
        view=ViewConfig(height=32,width=32,fov_x=80,fov_y=80)
        cams=cover('fibonacci',view,17)
        for kind,reference in [('erp',erp_world_directions(16,32,device=torch.device('cpu'))),('cea',cea_rays(16,32,'cpu'))]:
            rays=target_rays(kind,0,512,16,32,1000,'cpu')
            torch.testing.assert_close(rays,reference.reshape(-1,3),atol=0,rtol=0)
            counts,total,maximum,dist=ray_statistics(cams,rays)
            ref=sum(project_rays(rays[:,None],c)[1].flatten() for c in cams)
            torch.testing.assert_close(counts.float(),ref)
            self.assertTrue(torch.isfinite(total).all());self.assertTrue(torch.isfinite(dist).all())
    def test_uncovered_vs_weakly_weighted(self):
        from diffpano.camera import camera_for_direction
        cam=camera_for_direction(0,0,height=16,width=16)
        rays=torch.tensor([[0.,0.,1.],[0.,0.,-1.],[.6,.6,1.]])
        rays=rays/rays.norm(dim=1,keepdim=True)
        count,total,maxweight,_=ray_statistics([cam],rays)
        self.assertEqual(count.tolist(),[1,0,1]);self.assertGreater(float(total[2]),0);self.assertLess(float(total[2]),1e-4)
    def test_future_N_readiness(self):
        from studies.camera_patching.common import output_path,expected_counts,rows
        from studies.camera_patching.cameras import stable_ids,routing
        for strategy in ('fibonacci','random'):
            hashes=[]
            for n in (64,89,128):
                cams=cover(strategy,ViewConfig(),n);hashes.append(angular_hash(cams))
                self.assertEqual(len(stable_ids(strategy,n)),n)
                self.assertTrue(all(r['num_cameras']==n for r in rows(n)))
                self.assertEqual(len(routing(cams,['a','b','c','d','e'])['indices']),n)
                result=analyze_camera_cover(cams,height=8,width=16,probe_count=257,chunk_size=64)
                self.assertEqual(result['num_cameras'],n)
                self.assertEqual(expected_counts(n,20)['denoiser'],n*20)
                self.assertIn('n'+str(n),str(output_path(strategy,n,'sana','erp','ruins')))
            self.assertEqual(len(set(hashes)),3)
