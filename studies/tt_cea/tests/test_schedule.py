import unittest
from types import SimpleNamespace
import torch
from diffusers import DDIMScheduler,FlowMatchEulerDiscreteScheduler,DPMSolverMultistepScheduler
from diffpano.pipelines.pixeldit_solver import PixelDiTFirstOrderSolver
from diffpano.pipelines.endpoints import flow_endpoints,ddim_endpoints,pixel_endpoints,flow_bounds
from studies.tt_cea.schedule import *
from studies.tt_cea.time_travel import backward_original_noise
from studies.tt_cea.common import BASES,BACKENDS,read
from studies.tt_cea.audit import baseline_coefficients

class ScheduleTests(unittest.TestCase):
    def test_exact_windows_and_disabled(self):
        for n,lo,hi in ((20,4,16),(30,6,24),(40,8,32),(50,10,40)):
            self.assertEqual(eligible_indices(n),tuple(range(lo,hi)))
            self.assertEqual(eligible_indices(n,False),())
        self.assertEqual(eligible_indices(1),())
        for n in (2,3,7,11):self.assertEqual(eligible_indices(n),tuple(k for k in range(n) if .2<=k/n<.8))
        with self.assertRaises(ValueError):eligible_indices(0)
    def test_actual_recorded_five_schedules(self):
        x=torch.ones(1,2,2,3)
        for n in BACKENDS:
            m=read(BASES['ruins']/n/'metadata.json');rows=baseline_coefficients(m)
            for k in eligible_indices(len(rows)):
                r=rows[k]
                v=backward_original_noise(x,x,**{key:r[key] for key in ('alpha_high','sigma_high','alpha_low','sigma_low')})
                self.assertTrue(bool(torch.isfinite(v).all()))
    def test_prepared_coefficients_and_no_recursive_replay(self):
        for scheduler,opts in ((DDIMScheduler(clip_sample=False),{}),(FlowMatchEulerDiscreteScheduler(shift=3),{}),
            (DPMSolverMultistepScheduler(solver_order=1,prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=3),{})):
            scheduler.set_timesteps(7,**opts)
            b=SimpleNamespace(pipeline=SimpleNamespace(scheduler=scheduler),device=torch.device('cpu'),timesteps=scheduler.timesteps)
            before=b.timesteps.clone();table=prepare_interval_table(b);x=torch.ones(1,1,2,3)
            for interval,t in zip(table,b.timesteps):
                pair=ddim_endpoints(scheduler,x,x,t) if type(scheduler).__name__=='DDIMScheduler' else flow_endpoints(x,x,*flow_bounds(scheduler,t,x))
                verify_prediction(pair,interval)
            plan=execution_plan(table,True)
            self.assertEqual(len(plan),7+len(eligible_indices(7)))
            self.assertEqual([v.k for v,kind in plan if kind=='replay'],list(eligible_indices(7)))
            self.assertTrue(torch.equal(before,b.timesteps));self.assertEqual(plan[-1][0].sigma_low,table[-1].sigma_low)
        b=SimpleNamespace(device=torch.device('cpu'),solver=PixelDiTFirstOrderSolver())
        b.timesteps=b.solver.prepare(7,device=b.device)
        for v in prepare_interval_table(b):verify_prediction(pixel_endpoints(x,x,*b.solver.bounds_for(v.model_timestep)),v)
    def test_coefficient_mismatch_rejected(self):
        s=FlowMatchEulerDiscreteScheduler();s.set_timesteps(5)
        b=SimpleNamespace(pipeline=SimpleNamespace(scheduler=s),device=torch.device('cpu'),timesteps=s.timesteps)
        v=prepare_interval_table(b)[0];x=torch.ones(1,1,1,1)
        with self.assertRaises(AssertionError):verify_prediction(flow_endpoints(x,x,.8,.7),v)
