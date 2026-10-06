import numpy as np
from .common import *
from .numerics import frechet,kid,inception_score

def validate(ex):
    import torch
    import clip
    from pytorch_fid.fid_score import calculate_frechet_distance
    from torch_fidelity.metric_kid import kernel_mmd
    torch.set_num_threads(4)
    rng=np.random.default_rng(6);a=rng.normal(size=(17,8));b=rng.normal(size=(19,8))
    ours=frechet(a,b);official=calculate_frechet_distance(a.mean(0),np.cov(a,rowvar=False),b.mean(0),np.cov(b,rowvar=False))
    assert abs(ours-official)<1e-8,(ours,official)
    assert frechet(a,a)<1e-8
    from .distributions import fid_stats
    assert abs(fid_stats(a,b.mean(0),np.cov(b,rowvar=False))-official)<1e-7
    assert abs(frechet(a[::-1],b[::-1])-ours)<1e-8
    singular=rng.normal(size=(3,2048));assert frechet(singular,singular)<1e-7
    try:kid(a[:1],b)
    except ValueError:pass
    else:raise AssertionError('KID accepted one sample')
    # Compare unbiased kernel estimator with the official torch-fidelity primitive.
    x=a[:9];y=b[:9]
    ours_k=kid(x,y,subsets=1,subset_size=9)['mean']
    official_k=kernel_mmd(x,y,kid_kernel='poly',kid_kernel_poly_degree=3,kid_kernel_poly_gamma=None,kid_kernel_poly_coef0=1)
    assert abs(ours_k-float(official_k))<1e-10,(ours_k,official_k)
    logits=rng.normal(size=(16,1008)).astype(np.float32)
    assert np.isfinite(inception_score(logits))
    checks=dict(fid_fixture=ours,fid_official=official,kid_fixture=ours_k,kid_official=float(official_k),singular_feature_dimension=2048)
    if ex.clip is not None:
        text='A coral reef underwater.'
        ours,record=ex.text_embedding(text)
        with torch.inference_mode():
            direct=ex.clip.encode_text(clip.tokenize([text]).to(ex.device)).float();direct/=direct.norm(dim=-1,keepdim=True)
        np.testing.assert_allclose(ours,direct[0].cpu().numpy(),atol=2e-6)
        long='coral reef fish colorful water '*100
        _,meta=ex.text_embedding(long)
        assert len(meta['chunks'])>1 and sum(meta['chunks'])==meta['payload_tokens'] and not meta['truncated']
        checks.update(short_text_exact=True,long_text_tokens=meta['payload_tokens'],long_text_chunks=meta['chunks'])
    if ex.inception is not None:
        inputs=torch.from_numpy(rng.integers(0,256,size=(2,3,512,512),dtype=np.uint8)).to(ex.device)
        with torch.inference_mode():
            f,l=ex.inception(inputs);f2,l2=ex.inception(inputs)
        assert f.shape==(2,2048) and l.shape==(2,1008) and torch.equal(f,f2) and torch.equal(l,l2)
        assert torch.isfinite(f).all() and torch.isfinite(l).all()
        checks.update(inception_feature_dimensions=2048,logits_dimensions=1008,deterministic_fixture=True)
    atomic(ROOT/'validation/gpu.json',dict(passed=True,protocol_sha256=ex.identity,checks=checks,
        source_hashes=source_hashes(),job=os.environ['SLURM_JOB_ID'],time=now()))
    print('GPU_EVALUATOR_VALIDATED',checks,flush=True)
