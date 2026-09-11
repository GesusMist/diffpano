"""Small timestep-conditioned residual correction of roundtripped CLEAN latents."""
from pathlib import Path

import torch
from torch import nn


class LatentBridge(nn.Module):
    def __init__(self, channels, width=48, blocks=2, time_scale=1000.):
        super().__init__()
        self.architecture = dict(channels=channels, width=width, blocks=blocks, time_scale=time_scale)
        self.input = nn.Conv2d(channels, width, 3, padding=1)
        self.time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Conv2d(width,width,3,padding=1),
                                                   nn.SiLU(), nn.Conv2d(width,width,3,padding=1)) for _ in range(blocks)])
        self.output = nn.Conv2d(width, channels, 3, padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, latent, timestep):
        if latent.ndim != 4 or latent.shape[1] != self.architecture['channels']:
            raise ValueError('Bridge expects raw BCHW clean latents with its trained channel count')
        x = latent.float()
        t = torch.as_tensor(timestep, device=x.device, dtype=x.dtype).reshape(-1,1)
        if t.shape[0] == 1: t = t.expand(x.shape[0],1)
        if t.shape[0] != x.shape[0]: raise ValueError('Bridge timestep batch mismatch')
        hidden = self.input(x) + self.time(t/self.architecture['time_scale'])[:,:,None,None]
        for block in self.blocks: hidden = hidden + block(hidden)
        return x + self.output(hidden)


def save_bridge(path, bridge, metadata):
    path = Path(path)
    torch.save(dict(architecture=bridge.architecture, state_dict=bridge.state_dict(), metadata=metadata),path)


def load_bridge(path, *, device='cpu', expected_backend=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if expected_backend is not None and checkpoint['metadata']['backend'] != expected_backend:
        raise ValueError('Bridge checkpoint belongs to a different backend')
    bridge = LatentBridge(**checkpoint['architecture'])
    bridge.load_state_dict(checkpoint['state_dict'])
    return bridge.to(device).eval(), checkpoint['metadata']


def projection_metrics(original, roundtrip, corrected=None, epsilon=1e-8):
    def error(value):
        delta = (value-original).float()
        return float(delta.abs().mean()), float(delta.square().mean().sqrt())
    mae, rmse = error(roundtrip)
    std = float(original.float().std(unbiased=False))
    result = dict(roundtrip_mae=mae, roundtrip_rmse=rmse,
        normalized_roundtrip_mae=mae/max(std,epsilon), normalization_epsilon=epsilon,
        latent_before_mean=float(original.float().mean()), latent_before_std=std,
        latent_after_mean=float(roundtrip.float().mean()),latent_after_std=float(roundtrip.float().std(unbiased=False)))
    if corrected is not None:
        bm,br = error(corrected)
        result.update(bridge_mae=bm,bridge_rmse=br,normalized_bridge_mae=bm/max(std,epsilon),
            bridge_correction_mae=float((corrected-roundtrip).abs().mean()),
            error_reduction_percent=100*(mae-bm)/max(mae,epsilon))
    return result
