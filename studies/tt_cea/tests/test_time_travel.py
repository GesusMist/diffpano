import unittest
from unittest.mock import patch
import torch
from studies.tt_cea.time_travel import backward_original_noise,OriginalNoiseBank

class TimeTravelTests(unittest.TestCase):
    def test_explicit_examples_and_identity(self):
        x=torch.full((1,1,1,1),.2);e=torch.full_like(x,1.4)
        y=backward_original_noise(x,e,alpha_low=.6,sigma_low=.4,alpha_high=.5,sigma_high=.5)
        torch.testing.assert_close(y,torch.full_like(x,.4))
        y=backward_original_noise(x,e,alpha_low=.8,sigma_low=.6,alpha_high=.6,sigma_high=.8)
        torch.testing.assert_close(y,.75*x+.35*e)
        torch.testing.assert_close(backward_original_noise(x,e,alpha_low=.6,sigma_low=.4,alpha_high=.6,sigma_high=.4),x)
    def test_straight_line_oracle_and_perfect_replay(self):
        from diffpano.current_state_transition import interpolate_from_current_state
        g=torch.Generator().manual_seed(42)
        for channels in (1,3,4,7,16,32):
            c=torch.randn(2,channels,3,5,generator=g);e=torch.randn(c.shape,generator=g)
            for ah,sh,al,sl in ((.5,.5,.6,.4),(.1,.9,.7,.3),(.6,.8,.8,.6)):
                low=al*c+sl*e;high=ah*c+sh*e
                before=(low.clone(),e.clone(),torch.random.get_rng_state().clone())
                with patch('torch.randn',side_effect=AssertionError('random draw')):
                    out=backward_original_noise(low,e,alpha_low=al,sigma_low=sl,alpha_high=ah,sigma_high=sh)
                torch.testing.assert_close(out,high,atol=1e-6,rtol=2e-6)
                torch.testing.assert_close(interpolate_from_current_state(out,c,ah,sh,al,sl),low,atol=1e-6,rtol=2e-6)
                self.assertTrue(torch.equal(before[0],low) and torch.equal(before[1],e))
                self.assertTrue(torch.equal(before[2],torch.random.get_rng_state()))
    def test_noise_scale_no_alias_and_hash_guard(self):
        e=torch.randn(1,7,4,5,generator=torch.Generator().manual_seed(4))
        for scale in (1.,2.5):
            state=e*scale;bank=OriginalNoiseBank([state],scale)
            torch.testing.assert_close(bank[0],e,atol=3e-7,rtol=3e-7)
            if scale==1:self.assertTrue(torch.equal(bank[0],state))
            state.add_(2);bank.verify();bank[0].add_(1)
            with self.assertRaises(AssertionError):bank.verify()
    def test_rejections(self):
        x=torch.ones(1,3,2,2);valid=dict(alpha_low=.6,sigma_low=.4,alpha_high=.5,sigma_high=.5)
        for e in (x[:,:,:1],x.double(),torch.empty(x.shape,device='meta')):
            with self.assertRaises(ValueError):backward_original_noise(x,e,**valid)
        for key,value in (('alpha_low',0),('alpha_low',float('nan')),('sigma_high',.2),('alpha_high',.7),('sigma_low',-.1),('sigma_high',torch.ones(2))):
            kwargs=dict(valid);kwargs[key]=value
            with self.assertRaises(ValueError):backward_original_noise(x,x,**kwargs)
