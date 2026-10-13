import importlib.util,sys,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import torch
from diffpano.camera import PerspectiveCamera
from diffpano.config import FusionConfig,WarpConfig
from diffpano.gradient_fusion import GradientSettings,GuidanceAccumulator,edges,reconstruct
from diffpano.projection import ERPContribution
from studies.tt_cea.canvas import CanvasOperator,CanvasSpec
from studies.gradient_blending.operator import StudyCanvas

class AnchorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cameras=[PerspectiveCamera(a,0.,0.,80.,80.,8,16) for a in (0.,.3)]
        self.selected=torch.tensor([[.1,.2,.3],[-.2,.6,.4]])[:,:,None,None].expand(2,3,8,16).clone()
        self.other=torch.tensor([[.7,-.2,.1],[.2,.2,-.4]])[:,:,None,None].expand_as(self.selected).clone()
    def canvas(self,source='fused_rgb_erp'):
        original=CanvasOperator(CanvasSpec('erp',8,16),WarpConfig(mode='standard'),FusionConfig(mode='weighted_average',weight_mode='spherediff_center'),'cpu')
        result=StudyCanvas(original,GradientSettings('poisson_select',reference_mode='single_pixel',single_pixel_source=source),self.cameras,[0,1])
        def project(rgb,camera,size):
            valid=torch.ones(2,1,8,16);weight=valid*(1. if camera==self.cameras[0] else 3.)
            return ERPContribution(rgb,valid,weight)
        result.standard.perspective_to_erp=project
        return result
    def fusion(self,canvas,other,order=(0,1)):
        acc=canvas.make_accumulator(2);images=[self.selected,other]
        for i in order:
            canvas.accumulate(acc,images[i],self.cameras[i])
            if canvas.settings.single_pixel_source=='fused_rgb_erp' and 'anchor' in acc:self.assertNotIn('color',acc['anchor'])
        out=canvas.finalize(acc);a=canvas.records[-1]['anchor'];y,x=a['erp_y'],a['erp_x']
        torch.testing.assert_close(out.rgb[...,y,x],canvas.last_reference[...,y,x] if canvas.settings.single_pixel_source=='fused_rgb_erp' else self.selected[...,y,x],atol=1e-6,rtol=1e-6)
        return out,a
    def test_all_contributions_and_finalization_order(self):
        c=self.canvas();c.stage=1;out,a=self.fusion(c,self.other)
        expected=(self.selected+3*self.other)/4
        torch.testing.assert_close(c.last_reference,expected,atol=1e-7,rtol=1e-7)
        self.assertFalse(torch.equal(out.rgb,self.selected));self.assertEqual(a['source_type'],'fused_rgb_erp')
        self.assertEqual(a['anchor_semantics_version'],2);self.assertEqual(a['location_selection_camera_id'],0)
        other=self.canvas();other.stage=1;rev,b=self.fusion(other,self.other,(1,0))
        torch.testing.assert_close(out.rgb,rev.rgb,atol=1e-6,rtol=1e-6)
        self.assertEqual((a['erp_y'],a['erp_x']),(b['erp_y'],b['erp_x']))
    def test_channels_batches_temporal_and_terminal(self):
        c=self.canvas();anchors=[]
        for step,stage in enumerate((1,2,'terminal')):
            c.stage=stage;_,a=self.fusion(c,self.other+.1*step);anchors.append(a)
            self.assertEqual(a['stage'],stage);self.assertEqual(len(a['c0_per_item_rgb']),2)
            self.assertLess(a['error'],1e-6)
        self.assertEqual(len({(a['erp_y'],a['erp_x']) for a in anchors}),1)
        self.assertNotEqual(anchors[0]['c0'],anchors[1]['c0']);self.assertNotEqual(anchors[1]['c0'],anchors[2]['c0'])
        self.assertTrue(c.records[-1]['terminal'])
    def test_explicit_legacy_and_old_record_semantics(self):
        c=self.canvas('legacy_source_patch');c.stage=1;first,a=self.fusion(c,self.other)
        c.stage='terminal';second,b=self.fusion(c,self.other+1.)
        self.assertTrue(torch.equal(first.rgb,second.rgb));self.assertEqual(a['anchor_semantics_version'],1)
        old=GradientSettings.from_record(dict(mode='poisson_select',reference_mode='single_pixel'))
        self.assertEqual(old.single_pixel_source,'legacy_source_patch');self.assertTrue(old.matches_record(dict(mode='poisson_select',reference_mode='single_pixel')))
        self.assertEqual(GradientSettings(reference_mode='single_pixel').single_pixel_source,'fused_rgb_erp')
    def test_pure_poisson_differs_only_by_channel_offset(self):
        torch.manual_seed(18);ref=torch.randn(2,3,8,16,dtype=torch.float64)
        g=edges(torch.randn_like(ref));m=tuple(torch.ones_like(x[:,:1],dtype=torch.bool) for x in g)
        cfg=GradientSettings('poisson_select',reference_mode='global_mean',relative_tolerance=1e-10,absolute_tolerance=1e-11)
        a,_=reconstruct(ref,g,m,cfg,dtype=torch.float64)
        anchor=dict(erp_y=3,erp_x=7,color=ref[...,3,7].clone(),camera_id=0)
        b,d=reconstruct(ref,g,m,replace(cfg,reference_mode='single_pixel'),anchor=anchor,dtype=torch.float64)
        for x,y in zip(edges(a),edges(b)):torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(b[...,3,7],ref[...,3,7],atol=1e-11,rtol=1e-11)
        self.assertGreater(float((b.mean((-2,-1))-ref.mean((-2,-1))).abs().max()),.01)
        with self.assertRaisesRegex(ValueError,'this fusion reference'):
            reconstruct(ref,g,m,replace(cfg,reference_mode='single_pixel'),anchor=dict(anchor,color=anchor['color']+1.),dtype=torch.float64)
    def test_other_reference_modes_exact_regression(self):
        path=Path('studies/ournew_comparison/staged/before/poisson_reference.py')
        spec=importlib.util.spec_from_file_location('diffpano._before_anchor',path);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
        torch.manual_seed(21);ref=torch.randn(1,3,8,16);g=edges(torch.randn_like(ref));m=tuple(torch.ones_like(x[:,:1],dtype=torch.bool) for x in g)
        for mode in ('global_mean','coarse_color'):
            cfg=GradientSettings('poisson_select',reference_mode=mode,coarse_rows=2,coarse_cols=4,relative_tolerance=1e-4 if mode=='coarse_color' else 1e-5)
            a,da=old.reconstruct_constrained(ref,g,m,cfg);b,db=reconstruct(ref,g,m,cfg)
            self.assertTrue(torch.equal(a,b));self.assertEqual(da['iterations'],db['iterations'])
