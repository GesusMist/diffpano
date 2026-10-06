from .common import *
from .render import PROTOCOL
from .seams import DS_PROTOCOL,SEAM_PROTOCOL

def protocols():
    revisions=read(ASSETS/'source_revisions.json') if (ASSETS/'source_revisions.json').exists() else {}
    assets=read(ASSETS/'asset_status.json') if (ASSETS/'asset_status.json').exists() else {}
    p={k:dict(name=k,direction='higher' if k in ('Seam-SSIM','IS','CS') else 'lower') for k in METRICS}
    p['DS'].update(DS_PROTOCOL,code_revision=revisions.get('Text-Driven-Pano-Gen'),checkpoint='not applicable',reference='not required')
    for k in ('Seam-SSIM','Seam-Sobel'):p[k].update(SEAM_PROTOCOL,code_revision=revisions.get('JoPano'),checkpoint='not applicable',reference='not required')
    inception=dict(code='https://github.com/toshas/torch-fidelity',code_revision=revisions.get('torch-fidelity'),
       checkpoint=assets.get('inception'),extractor='inception-v3-compat',layer='2048',
       preprocessing='8 fixed 512x512 horizontal 90-degree perspective views per panorama; RGB uint8; torch-fidelity internal TensorFlow-compatible bilinear resize to299; internal normalization to[-1,1]',
       representation='common perspective-view adaptation',renderer=PROTOCOL,precision='float32 inference, float64 statistics')
    for k in ('FID','KID','IS'):p[k].update(inception)
    p['FID'].update(paper='https://arxiv.org/abs/1706.08500',reference='frozen real SUN360 test collection',
        definition='squared mean distance + traces of covariances - 2*trace geometric covariance mean; feature dimension fixed2048',units='unscaled FID',status='blocked_missing_reference')
    p['KID'].update(paper='https://arxiv.org/abs/1801.01401',reference='same real collection as FID',
        definition='unbiased polynomial MMD: kernel=(x dot y/2048+1)^3; subsets100, subset_size=min(1000,n_real,n_generated), seed2020',
        units='raw MMD squared, no multiplication by1000; negative finite estimates valid',status='blocked_missing_reference')
    p['IS'].update(paper='https://arxiv.org/abs/1606.03498',reference='not required',layer='logits_unbiased',
        definition='official torch-fidelity exp(mean KL(p(class|view)||mean p)); one full-collection split; no crop independence inference',
        units='unscaled Inception Score',status='pending_feature_extraction')
    p['OmniFID'].update(paper='https://arxiv.org/html/2407.18207',code='https://github.com/Anderschri/OmniFID',
        code_revision=revisions.get('OmniFID'),checkpoint=assets.get('inception'),implementation='paper-faithful Eq4 grouping; official author code unreleased',
        preprocessing='ERP resized to1024x512 with PIL bilinear; py360convert e2c face_w256 bilinear; F/R/B/L/U/D; Inception-v3-compat2048',
        definition='average the four horizontal feature vectors within EACH panorama, compute three FIDs (horizontal,up,down), arithmetic mean of three scores',
        reference='same real panorama collection, same preprocessing',status='blocked_missing_reference')
    p['FAED'].update(paper='https://github.com/chang9711/BIPS',code='https://github.com/chengzhag/PanFusion',code_revision=revisions.get('PanFusion'),
        checkpoint=assets.get('faed'),implementation='released PanFusion RGB-only autoencoder variant; no synthetic depth',
        preprocessing='512x1024 ERP bilinear; RGB mapped to[-1,1]; original encoder; longitude mean; cos(linspace(pi/2,-pi/2,H_feature)) latitude weights; flatten',
        reference='same real panorama collection; FAED checkpoint trained on Matterport3D indoors, domain mismatch with mixed prompts/SUN360',
        definition='Frechet distance in pretrained latitude-weighted encoder feature space',status='blocked_missing_checkpoint',
        blocker='Author-linked PanFusion RGB FAED SharePoint download returned HTTP 403; no verified compatible local checkpoint. No untrained or substitute features were used.')
    p['Distort-FID'].update(paper='https://arxiv.org/abs/2503.18420',code='https://github.com/iSEE-Laboratory/PanoDecouple',
        code_revision=revisions.get('PanoDecouple'),checkpoint=None,implementation='not substituted',preprocessing='unverified: evaluator not released',
        reference='real reference required',definition='Frechet distance in distortion-specific trained feature space',status='blocked_missing_checkpoint',
        blocker='Official repository contains README/LICENSE only; no Distort-CLIP evaluator or checkpoint. Ordinary CLIP is not a substitute.')
    p['CS'].update(paper='https://arxiv.org/abs/2103.00020',code='https://github.com/openai/CLIP',code_revision=revisions.get('CLIP'),
        checkpoint=assets.get('clip'),implementation='official CLIP ViT-B/32; fixed-view directional adaptation',renderer=PROTOCOL,
        preprocessing='official224 bicubic Resize/CenterCrop, RGB, official CLIP normalization',
        definition='100*raw cosine (no clamp) between normalized view and normalized text embeddings, mean over eight horizontal views; poles separately',
        long_text_policy='No truncation: tokenize all BPE tokens, contiguous chunks of at most75 payload tokens plus SOT/EOT; token-count-weighted average of normalized chunk embeddings, renormalize. Identical policy across methods.',
        prompt_assignment='world pitch0 -> prompt line2; north -> line0; south -> line4 (zero based); single original line applies to every view',
        reference='prompt text only',status='pending_feature_extraction')
    return p

def cpu_identity():
    return digest(dict(ds=DS_PROTOCOL,seam=SEAM_PROTOCOL,renderer=PROTOCOL,
        sources={n:sha(REPO/'studies/panorama_metrics'/n) for n in ('seams.py','render.py')}))

def save_protocols():
    p=protocols();atomic(ROOT/'metric_protocols.json',p);return p
