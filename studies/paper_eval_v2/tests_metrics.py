"""Focused numerical, geometry, weighting, and cache checks; no GPU or generation."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from .rendering import VIEWS14, NATIVE_FIVE, render_views, resample_common
from .statistics import summarize, bootstrap_mean
from .distribution import FixedReferenceFID, FixedReferenceKID, paired_bootstrap
from studies.panorama_metrics.numerics import frechet, kid
from studies.panorama_metrics.seams import ds


class SeamTests(unittest.TestCase):
    def test_constant_and_broken_boundary(self):
        import cv2
        image=np.zeros((64,128,3),dtype=np.uint8)
        self.assertEqual(ds(image),0.)
        image[:,-12:]=255
        broken=ds(image)
        self.assertGreater(broken,10.)
        padded=np.concatenate([image[:,-32:],image,image[:,:32]],axis=1)
        smooth=cv2.GaussianBlur(padded,(15,1),3)[:,32:-32]
        self.assertLess(ds(smooth),broken)

    def test_scharr_reference(self):
        rng=np.random.RandomState(12)
        image=rng.randint(0,256,(17,32,3),dtype=np.uint8)
        rgb=image.astype(np.float32)/255.
        grey=.299*rgb[...,0]+.587*rgb[...,1]+.114*rgb[...,2]
        strip=np.concatenate([grey[:,-3:],grey[:,:3]],axis=1)
        padded=np.pad(strip,((1,1),(0,0)))
        kernel=np.array([[-3,0,3],[-10,0,10],[-3,0,3]],dtype=np.float32)
        response=np.zeros((17,4),dtype=np.float32)
        for y in range(3):
            for x in range(3):response+=kernel[y,x]*padded[y:y+17,x:x+4]
        response=np.abs(response)
        expected=float(np.mean(response[:,1]/(response[:,0]+.1)+response[:,2]/(response[:,3]+.1))/2)
        self.assertAlmostEqual(ds(image),expected,places=5)


class RenderingTests(unittest.TestCase):
    def pattern(self):
        h,w=256,512
        lon=((np.arange(w)+.5)/w-.5)*2*np.pi
        lat=(.5-(np.arange(h)+.5)/h)*np.pi
        image=np.empty((h,w,3),dtype=np.uint8)
        image[:,:,0]=np.round((np.cos(lon)[None,:]+1)*127.5)
        image[:,:,1]=np.round((np.sin(lon)[None,:]+1)*127.5)
        image[:,:,2]=np.round((np.sin(lat)[:,None]+1)*127.5)
        return image

    def test_view_counts_and_bands(self):
        self.assertEqual(len(VIEWS14),14)
        self.assertEqual(sum(v['is_seam'] for v in VIEWS14),3)
        self.assertEqual([sum(v['prompt_band']==i for v in VIEWS14) for i in range(5)],[1,4,4,4,1])
        self.assertEqual([v['pitch'] for v in NATIVE_FIVE],[90.,10.,0.,-10.,-90.])

    def test_seam_poles_periodicity_and_native_fov(self):
        image=self.pattern()
        views=render_views(image,[(0,0),(180,0),(360,0),(0,90),(0,-90)],33)
        self.assertGreater(views[0,16,16,0],250)
        self.assertLess(views[1,16,16,0],5)
        np.testing.assert_array_equal(views[0],views[2])
        self.assertGreater(views[3,16,16,2],250)
        self.assertLess(views[4,16,16,2],5)
        native=render_views(image,[(0,0)],33,80.)
        self.assertGreater(native[0,16,-1,0],views[0,16,-1,0])

    def test_common_periodic_resize(self):
        image=self.pattern()
        output=resample_common(image)
        self.assertEqual(output.shape,(1024,2048,3))
        self.assertLess(np.abs(output[:,0].astype(int)-output[:,-1].astype(int)).max(),5)
        np.testing.assert_array_equal(resample_common(output),output)


class StatisticalTests(unittest.TestCase):
    def test_seed_average_then_equal_prompt(self):
        rows=[dict(case_id=str(i),method_id='m',prompt_id=p,metrics={'x':v})
              for i,(p,v) in enumerate([('p0',0.),('p0',10.),('p1',20.)])]
        groups,_=summarize(rows,'x','c',['p0','p1'],comparisons=())
        self.assertEqual(groups[0]['mean'],12.5)
        self.assertEqual(groups[0]['bootstrap_replicates'],2000)

    def test_paired_same_clusters_and_missing(self):
        rows=[]
        for method,offset in [('a',0),('b',2)]:
            for p,value in enumerate([0,100,1000]):
                rows.append(dict(case_id=method+str(p),method_id=method,prompt_id=str(p),metrics={'x':value+offset}))
        _,pairs=summarize(rows,'x','c',['0','1','2','missing'],comparisons=[('a','b')])
        self.assertEqual(pairs[0]['mean'],2.)
        self.assertEqual(pairs[0]['ci95'],[2.,2.])
        self.assertEqual(pairs[0]['missing_prompt_ids'],['missing'])

    def test_single_prompt_has_no_invented_interval(self):
        self.assertEqual(bootstrap_mean([3.])['ci95'],[None,None])


class DistributionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.RandomState(18)
        self.x=rng.normal(size=(12,7));self.y=rng.normal(size=(25,7))

    def test_exact_full_dimension_cached_fid(self):
        evaluator=FixedReferenceFID(self.x,self.y)
        indices=np.array([0,1,1,2,2,2,4,6,7,8,10,11])
        self.assertAlmostEqual(evaluator(indices),frechet(self.x[indices],self.y),places=7)

    def test_kid_frozen_subsets_and_negative_values(self):
        evaluator=FixedReferenceKID(self.x,self.y,subsets=17,subset_size=8)
        expected=kid(self.x,self.y,subsets=17,subset_size=8)['mean']
        self.assertAlmostEqual(evaluator(np.arange(len(self.x))),expected,places=12)
        value=kid(self.x,self.x,subsets=1,subset_size=len(self.x))['mean']
        self.assertLess(value,0.)

    def test_uncertainty_budget_is_not_fake_completion(self):
        a={str(i):self.x[i*3:(i+1)*3] for i in range(4)}
        b={key:value+.1 for key,value in a.items()}
        result=paired_bootstrap(a,b,self.y,max_seconds=0.)
        self.assertEqual(result['achieved_replicates'],0)
        self.assertIsNone(result['ci95'])
        self.assertEqual(result['state'],'pending_uncertainty_compute')

    def test_omni_horizontal_features_are_averaged_first(self):
        faces=np.arange(3*6*7,dtype=np.float64).reshape(3,6,7)
        horizontal=faces[:,:4].mean(1)
        self.assertEqual(horizontal.shape,(3,7))
        self.assertEqual(frechet(horizontal,horizontal),0.)


class CacheIntegrationTests(unittest.TestCase):
    def test_feature_bytes_and_checksum(self):
        from .metrics import _connect, save_features, load_features
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with _connect(root) as connection:
                record=save_features(connection,'key',{'inception':np.ones((8,7),dtype=np.float32)},root)
            self.assertEqual(load_features(record)['inception'].shape,(8,7))
            record['feature_sha256']='0'*64
            with self.assertRaises(AssertionError):load_features(record)

    def test_aggregate_excludes_markers_and_stale_protocols(self):
        from .metrics import _connect,_store,aggregate
        from studies.panorama_metrics.common import atomic
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            records=[dict(case_id='a',panorama_id='a',method_id='select',prompt_id='p',state='complete',image_sha256='image')]
            cohorts=dict(selected_cells=[dict(case_id='a')],cohorts={'development15':dict(prompt_ids=['p'])})
            atomic(root/'group1/protocol.json',dict(identity='active'))
            value=dict(case_id='a',method_id='select',prompt_id='p',image_sha256='image',namespace='common_v2',protocol_sha256='active',metrics={'DS':2.})
            with _connect(root) as connection:
                _store(connection,'marker',dict(marker=True,case_id='a'))
                _store(connection,'active',value)
                _store(connection,'stale',dict(value,protocol_sha256='old',metrics={'DS':99.}))
            result=aggregate(root,records,cohorts)
            self.assertEqual(len(result['per_panorama']),1)
            self.assertEqual(result['per_panorama'][0]['metrics']['DS'],2.)

if __name__=='__main__':unittest.main()

