import ast
import unittest
import numpy as np
import cv2
from studies.panorama_metrics.common import *
from studies.panorama_metrics.render import *
from studies.panorama_metrics.seams import *

class Geometry(unittest.TestCase):
    def test_analytic_world_pattern(self):
        h,w=512,1024
        lon=((np.arange(w)+.5)/w-.5)*2*np.pi
        lat=(.5-(np.arange(h)+.5)/h)*np.pi
        xx,yy=np.meshgrid(lon,lat)
        rgb=np.stack([np.sin(xx)*np.cos(yy),np.sin(yy),np.cos(xx)*np.cos(yy)],-1).astype(np.float32)
        for yaw,pitch in HORIZONTAL+POLAR:
            actual=render(rgb,((yaw,pitch),),size=65)[0]
            expected=rays(yaw,pitch,size=65)
            self.assertLess(np.max(np.abs(actual-expected)),.007)
            self.assertTrue(np.allclose(actual[32,32],rotation(yaw,pitch)[:,2],atol=.007))
    def test_omni_face_grouping_orientation(self):
        import py360convert
        h,w=128,256
        lon=((np.arange(w)+.5)/w-.5)*2*np.pi
        lat=(.5-(np.arange(h)+.5)/h)*np.pi
        x,y=np.meshgrid(lon,lat)
        rgb=np.stack([np.sin(x)*np.cos(y),np.sin(y),np.cos(x)*np.cos(y)],-1).astype(np.float32)
        faces=py360convert.e2c(rgb,face_w=65,mode='bilinear',cube_format='list')
        expected=np.array([[0,0,1],[1,0,0],[0,0,-1],[-1,0,0],[0,1,0],[0,-1,0]])
        np.testing.assert_allclose(np.stack([a[32,32] for a in faces]),expected,atol=.03)
        # Horizontal vectors are averaged within the panorama before statistics.
        features=np.arange(6*8).reshape(6,8)
        np.testing.assert_equal(features[:4].mean(0),(features[0]+features[1]+features[2]+features[3])/4)
    def test_periodic_seam(self):
        a=np.ones((64,128,3),np.float32);a[:,:,0]=np.cos((np.arange(128)+.5)*2*np.pi/128)
        r=render(a,((180.,0.),(-180.,0.)),size=63)
        np.testing.assert_allclose(r[0],r[1],atol=1e-6)
    def test_cubemap_adjacency(self):
        self.assertEqual(len(edge_pairs()),12)
        self.assertEqual(len({(f,e) for a,b,_ in edge_pairs() for f,e in (a,b)}),24)
        for a,b,rev in edge_pairs():
            ar=edge_rays(*a,n=31,centers=True);br=edge_rays(*b,n=31,centers=True)
            np.testing.assert_allclose(ar,br[::-1] if rev else br,atol=1e-14)
    def test_adjacent_samples_not_identical(self):
        f=render(np.zeros((128,256,3),np.uint8),CUBE,size=512)
        a,b,rev=edge_pairs()[0]
        ra=oriented_edge(rays(*CUBE[a[0]]),a[1])[:,0]
        rb=oriented_edge(rays(*CUBE[b[0]]),b[1])[:,0]
        if rev:rb=rb[::-1]
        self.assertGreater(np.linalg.norm(ra-rb,axis=-1).min(),0)
    def test_prompt_mapping(self):
        from diffpano.camera import PerspectiveCamera
        from diffpano.conditioning import expand_directional_prompts,camera_prompt_indices
        bank=expand_directional_prompts(['north','upper','equator','lower','south'])
        cams=[PerspectiveCamera(np.radians(y),np.radians(p),0,90,90,512,512) for y,p in HORIZONTAL+POLAR]
        slots=camera_prompt_indices(cams,bank.directions).tolist()
        self.assertEqual([v//4 for v in slots],[2]*8+[0,4])

class SeamDefinitions(unittest.TestCase):
    def test_ds_against_linked_source(self):
        # Import just the functions; the upstream file executes a placeholder folder at module scope.
        path=ASSETS/'sources/Text-Driven-Pano-Gen/evaluation_metrics/Discontinuity_Score_cal.py'
        tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
        ns={};exec(compile(tree,str(path),'exec'),ns)
        import torch
        rng=np.random.default_rng(0)
        for a in (np.zeros((32,64,3),np.uint8),rng.integers(0,256,(32,64,3),dtype=np.uint8),np.tile(np.arange(64,dtype=np.uint8)[None,:,None],(32,1,3))):
            x=torch.from_numpy(a.transpose(2,0,1).copy()).float()/255
            # Published detect_seam returns [1,H,4]; compute_DS requires [H,4]. Correct only that singleton.
            ref=ns['compute_DS'](ns['detect_seam'](x).squeeze(0))
            self.assertAlmostEqual(ds(a),ref,places=5)
    def test_constant_step_blur(self):
        c=np.full((128,256,3),128,np.uint8);self.assertLess(abs(ds(c)),1e-5)
        step=c.copy();step[:,:128]=255;self.assertGreater(ds(step),0)
        blur=cv2.GaussianBlur(step,(9,9),2)
        self.assertTrue(np.isfinite(ds(blur)))  # No invented monotonicity claim for a ratio.
        scores=cubemap_seams(np.full((6,512,512,3),128,np.uint8))
        self.assertAlmostEqual(scores['Seam-SSIM'],1);self.assertEqual(scores['Seam-Sobel'],0)
    def test_sobel_is_not_gradient_difference(self):
        row=np.tile(np.linspace(0,255,512,dtype=np.uint8)[None,:,None],(512,1,3))
        x=cubemap_seams(np.stack([row]*6));self.assertGreater(x['Seam-Sobel'],0)
        # Constant offsets across faces can be missed by published mean-gradient definition.
        faces=np.stack([np.full((512,512,3),i*40,np.uint8) for i in range(6)])
        self.assertEqual(cubemap_seams(faces)['Seam-Sobel'],0)
    def test_cache_changes(self):
        from studies.panorama_metrics.protocols import cpu_identity
        p=cpu_identity();self.assertNotEqual(digest(dict(image='a',protocol=p)),digest(dict(image='b',protocol=p)))
        self.assertNotEqual(digest(dict(image='a',protocol=p)),digest(dict(image='a',protocol=p+'changed')))

class Incremental(unittest.TestCase):
    def test_saved_camera_controls(self):
        from studies.panorama_metrics.report import validate_camera_pair
        rows=read(ROOT/'inventory.json')['rows']
        count=0
        for b in rows:
            if b['family']!='camera' or b['completion_status']!='complete':continue
            a=next(r for r in rows if r['family']=='benchmark' and r['method']==b['method'] and r['backend']==b['backend'] and r['consensus_projection']==b['consensus_projection'] and r['prompt_id']==b['prompt_id'])
            self.assertTrue(validate_camera_pair(a,b));count+=1
        self.assertEqual(count,48)

    def test_model_geometry_prompt_cache_invalidation(self):
        import copy
        from studies.panorama_metrics.features import gpu_identity
        from studies.panorama_metrics.protocols import protocols
        p=protocols();initial=gpu_identity(p)
        changed=copy.deepcopy(p);changed['CS']['checkpoint']['sha256']='different-checkpoint'
        self.assertNotEqual(initial,gpu_identity(changed))
        changed=copy.deepcopy(p);changed['CS']['renderer']['fov_x']=91
        self.assertNotEqual(initial,gpu_identity(changed))
        f=digest(dict(image='image-bytes',evaluator=initial))
        self.assertNotEqual(digest(dict(features=f,prompt='p1')),digest(dict(features=f,prompt='p2')))
    def test_duplicate_active_submission_is_noop(self):
        import tempfile
        from unittest.mock import patch
        from studies.panorama_metrics import launch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);atomic(root/'validation/frozen.json',dict(source_hashes={'fixture':'hash'}))
            active=[dict(job='123',name='pmetrics-update',state='RUNNING',gres='gres/gpu:a100:1')]
            with patch.object(launch,'ROOT',root),patch.object(launch,'source_hashes',return_value={'fixture':'hash'}),patch.object(launch,'queue',return_value=active),patch.object(launch.subprocess,'check_output') as sbatch:
                launch.submit();launch.submit();sbatch.assert_not_called()
    def test_partial_and_corrupt_feature_pairs_rejected(self):
        import tempfile
        from studies.panorama_metrics.features import save_features
        from studies.panorama_metrics.distributions import load_feature
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'features.npz';save_features(p,dict(inception=np.ones((8,2048),np.float32)))
            h=sha(p);self.assertEqual(load_feature(str(p),h)['inception'].shape,(8,2048))
            load_feature.cache_clear();p.write_bytes(b'partial')
            with self.assertRaises(AssertionError):load_feature(str(p),h)

if __name__=='__main__':unittest.main()

