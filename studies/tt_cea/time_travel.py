"""The specified original-noise bridge; no model, projection, VAE or RNG call."""
import math
import torch
from diffpano.erp_noise_initialization import states_digest

class OriginalNoiseBank:
    def __init__(self,states,scale):
        if not math.isfinite(float(scale)) or float(scale)<=0:raise ValueError('Invalid initialization scale')
        self._noise=tuple(x.detach().to(device='cpu',dtype=torch.float32).clone()/float(scale) for x in states)
        if not all(bool(torch.isfinite(x).all()) for x in self._noise):raise ValueError('Nonfinite initial noise')
        self.initial_sha256=states_digest(self._noise)
    def __getitem__(self,i):return self._noise[i]
    def __len__(self):return len(self._noise)
    def verify(self):
        value=states_digest(self._noise)
        if value!=self.initial_sha256:raise AssertionError('Original-noise bank mutated')
        return value

def backward_original_noise(x_low,epsilon_init,*,alpha_low,sigma_low,alpha_high,sigma_high):
    if x_low.ndim!=4 or x_low.shape!=epsilon_init.shape or x_low.dtype!=epsilon_init.dtype or x_low.device!=epsilon_init.device:
        raise ValueError('Matching BCHW native shapes, dtype and device required after explicit conversion')
    if not x_low.is_floating_point():raise ValueError('Floating native states required')
    vals=[]
    for value in (alpha_low,sigma_low,alpha_high,sigma_high):
        v=torch.as_tensor(value,device=x_low.device,dtype=x_low.dtype)
        if v.numel()!=1 or not bool(torch.isfinite(v)):raise ValueError('Finite scalar coefficient required')
        vals.append(v)
    al,sl,ah,sh=vals
    if not (float(al)>0 and 0<=float(ah)<=float(al) and 0<=float(sl)<=float(sh)):
        raise ValueError('Invalid high/low noise ordering or alpha_low')
    if not bool(torch.isfinite(x_low).all()) or not bool(torch.isfinite(epsilon_init).all()):raise ValueError('Nonfinite input')
    ratio=ah/al
    result=ratio*x_low+(sh-ratio*sl)*epsilon_init
    if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite backward result')
    return result
