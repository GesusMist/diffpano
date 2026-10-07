import argparse
from studies.gradient_blending.common import PROMPTS
from studies.gradient_blending.run import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args()
    run(PROMPTS[a.index//2],('poisson_mean','poisson_select')[a.index%2])
