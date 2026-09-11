import unittest
import numpy as np
from diffpano.seams import boundary_gradient_metrics

class BoundaryMetricTests(unittest.TestCase):
    def test_constant_and_ramp_have_no_boundary_excess(self):
        patches=[dict(x=0,y=0,size=8),dict(x=4,y=0,size=8),dict(x=8,y=0,size=8)]
        for rgb in (np.zeros((8,16,3)),np.broadcast_to(np.arange(16)[None,:,None]/15,(8,16,3))):
            metric=boundary_gradient_metrics(rgb,patches)
            self.assertAlmostEqual(metric['boundary_excess'],0.)
            self.assertEqual(metric['vertical']['boundary_lines'],[4,8,12])
    def test_boundary_jump_increases_metric_and_rotates_consistently(self):
        rgb=np.zeros((8,16,3));rgb[:,8:]=1
        patches=[dict(x=0,y=0,size=8),dict(x=8,y=0,size=8)]
        metric=boundary_gradient_metrics(rgb,patches)
        self.assertEqual(metric['boundary_gradient'],1.)
        self.assertEqual(metric['nearby_gradient'],0.)
        rotated=boundary_gradient_metrics(rgb.transpose(1,0,2),[dict(x=p['y'],y=p['x'],size=p['size']) for p in patches])
        self.assertEqual(rotated['boundary_gradient'],metric['boundary_gradient'])
