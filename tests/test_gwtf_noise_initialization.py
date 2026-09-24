"""Independent Algorithm 1 oracles and spherical/native initialization guards."""
import gc
import hashlib
import json
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from diffpano.camera import camera_for_direction
from diffpano.erp_noise_initialization import initialize_erp_noise, ERPNoiseConfig, states_digest
from diffpano.gwtf_noise_initialization import (GWTFlowNoiseConfig, graph_from_maps, camera_graph,
    initialize_gwtf_shared_noise, transport_values, innovation_seed)
from diffpano.pipelines.native_state import NativeStateMixin


class NativeStub(NativeStateMixin):
    native_channels=7;native_spatial_factor=2;native_initial_noise_sigma=2.5;device=torch.device('cpu')
    def __init__(self):self.scales=0
    def initialize_native_state(self,x):self.scales+=1;return super().initialize_native_state(x)
    def encode_clean(self,*a):raise AssertionError('No VAE')
    def decode_clean(self,*a):raise AssertionError('No VAE')
    def predict_clean_and_endpoint(self,*a):raise AssertionError('No denoiser')
    def add_fixed_noise(self,*a):raise AssertionError('No renoising')


def scalar_oracle(q,graph,z):
    # Independent literal loops implementing Algorithm 1, using explicit graph.
    result=torch.zeros(q.shape[0],graph.target_size)
    for row in range(q.shape[0]):
        variance=torch.zeros(graph.target_size)
        for s in range(graph.source_ids.numel()):
            edges=[e for e,v in enumerate(graph.source_index.tolist()) if v==s]
            d=len(edges);mean=sum(float(z[row,e]) for e in edges)/d
            for e in edges:
                t=int(graph.target_index[e])
                if t<0:continue
                x=float(q[row,s])/d+(float(z[row,e])-mean)/(d**.5)
                alpha=float(graph.density[s])/d
                result[row,t]+=alpha*x;variance[t]+=alpha*alpha/d
        result[row]/=variance.sqrt()
    return result


class GWTFlowTests(unittest.TestCase):
    def cams(self):
        return [camera_for_direction(y,p,roll_degrees=r,height=h,width=w) for y,p,r,h,w in
                [(0,0,0,12,16),(180,0,17,16,12),(33,90,30,12,12),(7,-90,0,12,12)]]

    def test_manual_graph_forward_priority_and_mixed_degrees(self):
        g=graph_from_maps(torch.tensor([0,0,2,-1]),torch.tensor([1,0,2,2,3]),(1,5))
        self.assertEqual(list(zip(g.source_ids[g.source_index].tolist(),g.target_index.tolist())),
                         [(0,0),(0,1),(1,0),(2,2),(2,3),(3,4)])
        self.assertEqual(g.degree.tolist(),[2,1,2,1]);self.assertEqual(g.forward_edges,3)
        q=torch.randn(23,4,generator=torch.Generator().manual_seed(4))
        z=torch.randn(23,6,generator=torch.Generator().manual_seed(8))
        out=transport_values(q,g,torch.Generator(),innovations=z).flatten(1)
        torch.testing.assert_close(out,scalar_oracle(q,g,z),atol=4e-7,rtol=3e-6)

    def test_density_weighted_algorithm_oracle(self):
        g=graph_from_maps(torch.tensor([0,0,2,-1]),torch.tensor([1,0,2,2,3]),(5,),torch.tensor([1.,3.,.25,2.]))
        q=torch.randn(9,4,generator=torch.Generator().manual_seed(7));z=torch.randn(9,6,generator=torch.Generator().manual_seed(8))
        torch.testing.assert_close(transport_values(q,g,torch.Generator(),innovations=z),scalar_oracle(q,g,z),atol=1e-6,rtol=2e-6)
        torch.testing.assert_close(g.target_density,torch.tensor([3.5,.5,.125,.125,2.]))

    def test_identity_projection_is_exact(self):
        ids=torch.arange(20);g=graph_from_maps(ids,ids,(4,5));q=torch.randn(2,7,20,generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(transport_values(q,g,torch.Generator().manual_seed(8)),q.reshape(2,7,4,5)))

    def test_pure_expansion_conditional_sum_and_no_copies(self):
        g=graph_from_maps(torch.tensor([0]),torch.zeros(16,dtype=torch.long),(16,))
        q=torch.randn(12000,1,generator=torch.Generator().manual_seed(11))
        out=transport_values(q,g,torch.Generator().manual_seed(12),channel_chunk=128)
        torch.testing.assert_close(out.sum(1)/4,q[:,0],atol=2e-6,rtol=2e-6)
        self.assertEqual(float((out[:,1:]==out[:,:-1]).float().mean()),0.)
        self.assertLess(float((torch.cov(out.T)-torch.eye(16)).abs().max()),.05)

    def test_pure_contraction_sum(self):
        g=graph_from_maps(torch.zeros(16,dtype=torch.long),torch.tensor([0]),(1,))
        q=torch.randn(12000,16,generator=torch.Generator().manual_seed(12))
        out=transport_values(q,g,torch.Generator().manual_seed(9),channel_chunk=128)
        torch.testing.assert_close(out[:,0],q.sum(1)/4,atol=8e-7,rtol=3e-6)
        self.assertLess(abs(float(out.var())-1),.05)

    def test_mixed_covariance_and_positive_cross_view(self):
        g=graph_from_maps(torch.tensor([0,0,2,-1]),torch.tensor([1,0,2,2,3]),(5,))
        q=torch.randn(24000,4,generator=torch.Generator().manual_seed(10))
        a=transport_values(q,g,torch.Generator().manual_seed(11),channel_chunk=256)
        b=transport_values(q,g,torch.Generator().manual_seed(12),channel_chunk=256)
        self.assertLess(float((torch.cov(a.T)-torch.eye(5)).abs().max()),.04)
        self.assertGreater(float((a*b).mean()),.5)
        self.assertLess(abs(float(a.mean())),.015)

    def test_sparse_marginal_includes_all_sibling_edges(self):
        g=graph_from_maps(torch.tensor([0,0,2,-1]),torch.tensor([1,0,2,2,3]),(5,))
        sub,ids=g.select(torch.tensor([1,3]));self.assertEqual(sub.degree.tolist(),[2.,2.])
        q=torch.randn(3,4,generator=torch.Generator().manual_seed(11));z=torch.randn(3,6,generator=torch.Generator().manual_seed(12))
        a=transport_values(q,g,torch.Generator(),innovations=z)
        edge_mask=torch.isin(g.source_ids[g.source_index],sub.source_ids)
        b=transport_values(q[:,sub.source_ids],sub,torch.Generator(),innovations=z[:,edge_mask])
        self.assertTrue(torch.equal(a[:,ids],b))

    def test_holes_get_white_noise_and_empty_graph(self):
        for f in (torch.tensor([-1,-1]),torch.tensor([0,-1])):
            g=graph_from_maps(f,torch.tensor([-1,-1,-1]),(3,))
            q=torch.randn(12000,g.source_ids.numel(),generator=torch.Generator().manual_seed(44))
            out=transport_values(q,g,torch.Generator().manual_seed(45),channel_chunk=128)
            self.assertTrue(bool(torch.isfinite(out).all()))
            self.assertLess(float((torch.cov(out.T)-torch.eye(3)).abs().max()),.05)

    def test_arbitrary_channels_shapes_batch_and_scaling_once(self):
        for channels in (1,3,7,16,32):
            b=NativeStub();b.native_channels=channels
            states,m=initialize_gwtf_shared_noise(b,self.cams(),GWTFlowNoiseConfig(13,26,5),batch_size=2)
            self.assertEqual(b.scales,4)
            for state,cam in zip(states,self.cams()):self.assertEqual(state.shape,(2,channels,cam.height//2,cam.width//2))
            unit=NativeStub();unit.native_channels=channels;unit.native_initial_noise_sigma=1
            z,_=initialize_gwtf_shared_noise(unit,self.cams(),GWTFlowNoiseConfig(13,26,5),batch_size=2)
            for x,y in zip(states,z):self.assertTrue(torch.equal(x,y*2.5))
            self.assertEqual(m['scaling_applications'],1)

    def test_deterministic_order_and_camera_list_reordering(self):
        b=NativeStub();cams=self.cams();c=GWTFlowNoiseConfig(17,34,8)
        a,_=initialize_gwtf_shared_noise(b,cams,c,camera_slots=[0,1,2,3])
        for order in ([3,2,1,0],[2,0,3,1]):
            z,_=initialize_gwtf_shared_noise(b,cams,c,camera_slots=[0,1,2,3],execution_order=order)
            self.assertEqual(states_digest(a),states_digest(z))
            z,_=initialize_gwtf_shared_noise(b,[cams[i] for i in order],c,camera_slots=order)
            for x,i in zip(z,order):self.assertTrue(torch.equal(x,a[i]))
        a,_=initialize_gwtf_shared_noise(b,cams,c)
        z,_=initialize_gwtf_shared_noise(b,cams[::-1],c)
        self.assertEqual(states_digest(a),states_digest(z[::-1]))

    def test_rng_isolation_and_buffers_released(self):
        refs=[];graphs=[];rand=torch.randn
        def track(*a,**kw):
            x=rand(*a,**kw);refs.append(weakref.ref(x));return x
        import diffpano.gwtf_noise_initialization as module
        build=module.camera_graph
        def track_graph(*a,**kw):
            x=build(*a,**kw);graphs.append(weakref.ref(x));return x
        before=torch.random.get_rng_state().clone()
        with patch.object(module,'camera_graph',side_effect=track_graph),patch('torch.randn',side_effect=track):
            states,m=initialize_gwtf_shared_noise(NativeStub(),self.cams(),GWTFlowNoiseConfig(13,26,9))
        gc.collect()
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertTrue(all(r() is None for r in refs+graphs))
        self.assertTrue(m['source_released'] and m['transport_buffers_released'])
        self.assertEqual(m['denoiser_calls'],0);self.assertEqual(m['vae_calls'],0)
        self.assertFalse(m['fixed_noise_renoising'])

    def test_wrap_poles_roll_and_native_identity_camera(self):
        for y,p,r in ((0,0,0),(180,0,0),(-180,0,0),(11,90,33),(17,-90,21),(7,45,21)):
            for size in (1,5,12):
                c=camera_for_direction(y,p,roll_degrees=r,height=size,width=size)
                g=camera_graph(c,17,34)
                out=transport_values(torch.randn(2,g.source_ids.numel(),generator=torch.Generator().manual_seed(7)),g,torch.Generator().manual_seed(8))
                self.assertEqual(out.shape,(2,size,size));self.assertTrue(bool(torch.isfinite(out).all()))
                self.assertFalse(bool((g.normalizer==0).any()))

    def test_historical_initializer_and_source_unchanged(self):
        v=json.loads(Path('outputs/bridge-factorial-ruins/20260918/validation.json').read_text())
        for p in ('diffpano/erp_noise_initialization.py','diffpano/noise.py','diffpano/dense_consensus.py'):
            self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(),v['source_hashes'][p])
        b=NativeStub();cams=self.cams();before=torch.random.get_rng_state().clone()
        for mode in ('S-direct-local','V-independent-erp','V-shared-erp'):
            a,_=initialize_erp_noise(b,cams,ERPNoiseConfig(mode,17,34,3))
            initialize_gwtf_shared_noise(b,cams,GWTFlowNoiseConfig(17,34,3))
            z,_=initialize_erp_noise(b,cams,ERPNoiseConfig(mode,17,34,3))
            self.assertEqual(states_digest(a),states_digest(z))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_invalid_inputs(self):
        for f,b in ((torch.tensor([3]),torch.tensor([0])),(torch.tensor([0.]),torch.tensor([0]))):
            with self.assertRaises(ValueError):graph_from_maps(f,b,(1,))
        with self.assertRaises(ValueError):GWTFlowNoiseConfig(3,5,0).validate()
        with self.assertRaises(ValueError):initialize_gwtf_shared_noise(NativeStub(),self.cams(),GWTFlowNoiseConfig(3,6,0),execution_order=[0])
        with self.assertRaises(ValueError):initialize_gwtf_shared_noise(NativeStub(),self.cams(),GWTFlowNoiseConfig(3,6,0),camera_slots=[0]*4)

if __name__=='__main__':unittest.main()
