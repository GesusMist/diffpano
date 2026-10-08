"""Assemble a compact factual report; manual visual observations can be appended separately."""
from studies.gradient_blending.common import *

def number(v):return 'not recorded' if v is None else '%.5g'%v

def main():
    baseline=read(OUT/'baseline.json');validation=read(GATES/'validation.json')
    evaluation=read(OUT/'evaluation/summary.json');replay=read(GATES/'offline/replay.json')
    steps=baseline['config']['generation']['num_inference_steps']
    calls=expected_counts(BACKEND,steps)
    lines=['GRADIENT-DOMAIN BLENDING: '+BACKEND.upper()+' / ERP / Old89 / seed 0 / %d steps'%steps,
        '', 'Conclusion: inconclusive pending visual review. This %d-prompt study is descriptive only.'%len(PROMPTS),
        '', 'IMPLEMENTATION AND BASELINE',
        'Active path: '+' -> '.join(baseline['path']),
        'StudyPipeline inherits the original loop, bridge, current-state interpolation, terminal decoding and full-coverage checks.',
        'StudyCanvas changes only clean-RGB consensus, at all %d intervals and terminal assembly. rgb delegates to the original reducer.'%steps,
        'Reference R uses the original RGBFusionAccumulator, weighted_average with bilinearly projected spherediff_center weights (temperature 0.1).',
        'Fusion input: '+baseline['fusion_input']['meaning']+'. Shape [1,3,1024,1024] FP32, projected to [1,3,2048,4096]. Nominal range [-1,1]; no evolving-canvas clamping.',
        'Camera geometry SHA256: '+baseline['camera_geometry_sha256'],
        'Initial local state SHA256: '+baseline['initialization']['initial_local_sha256'],
        'Prepared schedule SHA256: '+baseline['schedule_sha256'],
        'Original revision: '+baseline['source_revision'],
        '', 'OBJECTIVE AND GUIDANCE',
        'I=R+u; (lambda I + D^T M D)u = D^T M(g-D R), lambda=0.1 shared by both variants.',
        'Forward pixel differences; horizontal modulo-W adjacency, vertical H-1 edges. No top-to-bottom connection or equality constraint on the two edge columns.',
        'This is an ERP-grid objective, not a rotation-invariant spherical gradient objective; no latitude weighting.',
        'M is binary same-view edge support. Source confidence is min(endpoint baseline weights). Missing edges are omitted.',
        'poisson_mean: weighted average of source gradients. poisson_select: max geometric confidence with lowest stable camera ID breaking ties; one source for all RGB channels.',
        'FP32 matrix-free Jacobi PCG, 200 iterations, rtol=1e-5, atol=1e-7; true-residual checks; nonconvergence/nonfinite output fails the run.',
        'Storage is streaming O(BCHW), independent of number of cameras. No full N-view ERP stack.',
        '', 'VALIDATION',
        'Tests passed: '+str(sum(r['tests'] or 0 for r in validation['runs']))+'; logs: validation/'+validation['job'],
        'Includes RGB regression, adjoint/SPD/dense FP64 reference, zero RHS/identity, masks/NaNs, ERP boundaries, deterministic ties/order, duplicate/weight scale/tiny weights, color offsets, solver failure, pixel/latent bridge mocks.',
        'Known-target synthetic comparisons: synthetic/metrics.json and synthetic/known-target-comparisons.png.',
        '', 'OFFLINE REPLAY',
        'Instrumented SANA ruins RGB trajectory captures steps 1,10,20 and views 7,22,59,75. Lossless sufficient statistics replay both variants at lambda 0.01,0.1,1.0.',
        '18 single-fusion solves: '+str(sum(r['diagnostics']['converged'] for r in replay['records']))+' converged. Pilot lambda fixed to 0.1. Full details: offline/replay.json.',
        'These compare identical input proposals and do not feed back into the captured baseline trajectory.',
        '', 'MATCHED STUDY (per-prompt metrics; no per-image IS)',
        'prompt | mode | DS | CS | Seam-SSIM | Seam-Sobel | generation seconds | solver seconds | peak allocated GiB']
    for row in evaluation['rows']:
        lines.append(' | '.join([row['prompt'],row['mode']]+[number(row['metrics'][k]) for k in ['DS','CS','Seam-SSIM','Seam-Sobel']]+
                              [number(row.get(k)) for k in ['generation_seconds','solver_seconds','peak_allocated_gib']]))
    lines+=['','COLLECTION SCORES (exact same %d prompts / %d dependent horizontal views)'%(len(PROMPTS),evaluation['collections']['rgb']['view_count']),'mode | DS mean | CS mean | IS (one full collection split)']
    for mode,row in evaluation['collections'].items():lines.append(' | '.join([mode,number(row['DS']),number(row['CS']),number(row['IS'])]))
    for group,modes in evaluation.get('collection_groups',{}).items():
        lines+=['','SUBSET COLLECTION: '+group,'mode | DS mean | CS mean | IS (one full collection split)']
        for mode,row in modes.items():lines.append(' | '.join([mode,number(row['DS']),number(row['CS']),number(row['IS'])]))
    lines+=['','KNOWN-TARGET SYNTHETICS','case | mode | reconstruction MSE | gradient MSE | texture std / target std']
    for case in read(GATES/'synthetic/metrics.json')['cases']:
        for mode,m in case['metrics'].items():
            lines.append(' | '.join([case['case'],mode,number(m['reconstruction_mse']),number(m['gradient_mse']),number(m['texture_std_ratio'])]))
    lines.append('The one-pixel misalignment case demonstrates why energy is insufficient: selection increases texture amplitude while worsening target reconstruction and gradient error.')
    lines+=['','RUNTIME, MEMORY, AND COUNTS']
    for prompt in PROMPTS:
        for mode in MODES:
            meta=read(OUT/'cases'/prompt/mode/'metadata.json');fusions=meta.get('fusions',[])
            solved=[r for r in fusions if 'iterations' in r]
            detail=('iterations min/mean/max %d/%.1f/%d'%(min(r['iterations'] for r in solved),sum(r['iterations'] for r in solved)/len(solved),max(r['iterations'] for r in solved))) if solved else 'no solver'
            lines.append('%s/%s: %s; calls %s; reused=%s'%(prompt,mode,detail,meta['counts'],meta.get('reused',False)))
    lines+=['Generation-only timing excludes study diagnostic callbacks and instrument-only statistics. Combined projection/fusion and per-stage timings are in each metadata.json; solver timing is also separate.',
        'Cached baseline timing is reported only when explicitly recorded; broader historical runtime is not substituted. Peak allocation on the instrumented baseline includes its diagnostics.',
        'Every fresh run checks %d denoiser calls, %d VAE encodes, %d VAE decodes and identical initialization/camera/schedule/conditioning for the matched prompt.'%(calls['denoiser'],calls['encode'],calls['decode']),
        'terminal-rgb-reference.png is labeled RGB terminal assembly of gradient-run terminal states; it is not an independent RGB baseline trajectory.',
        '', 'VISUALS AND LIMITATIONS',
        'figures/*final-erp.png, *fixed-views.png, *detail-wrap.png: full trajectories. SANA root figures additionally contain selected intermediate comparisons.',
        'Shared SANA root figures/replay-*.png: identical-proposal offline comparisons. ownership-step-*.png: geometry-selected owner boundaries.',
        'Weighted-average gradients can reproduce averaging behavior. Confidence selection can retain source detail but introduce ownership-transition artifacts.',
        'Poisson reconstruction cannot guarantee inconsistent source content becomes geometrically correct. Higher edge energy or lower DS alone is not evidence of success.',
        'Known-target reconstruction and gradient errors, color corrections, output range, and solver objective/residuals are retained even when unfavorable.',
        'Frozen evaluator definitions and historical caches were not changed; %d prompts at one seed do not establish generalization or significance.'%len(PROMPTS),
        '', 'REPRODUCTION',
        'sbatch studies/gradient_blending/validate.slurm',
        'sbatch studies/gradient_blending/gpu.slurm studies.gradient_blending.gpu_smoke',
        'sbatch studies/gradient_blending/gpu.slurm studies.gradient_blending.run --prompt ruins --mode rgb --instrument',
        'sbatch studies/gradient_blending/gpu.slurm studies.gradient_blending.replay',
        'python3 -m studies.gradient_blending.launch --pilot  # after validation, smoke and replay gates pass; at most two generation jobs',
        'sbatch studies/gradient_blending/evaluate.slurm',
        'sbatch studies/gradient_blending/figures.slurm',
        'python -m studies.gradient_blending.report',
        'PixelDiT/SD3.5 extension: sbatch studies/gradient_blending/prepare_extension.slurm; then python3 -m studies.gradient_blending.launch_extension',
        'For per-backend evaluation/figures/report, set DIFFPANO_GRADIENT_BACKEND to '+BACKEND+' and DIFFPANO_GRADIENT_SUITE to '+SUITE+'.',
        'FLUX scene extension: sbatch studies/gradient_blending/prepare_scenes.slurm; then DIFFPANO_GRADIENT_BACKEND=flux DIFFPANO_GRADIENT_SUITE=flux-scenes20 python3 -m studies.gradient_blending.launch_scenes',
        '', 'Changed/new implementation files: diffpano/gradient_fusion.py and studies/gradient_blending/. All 519 pre-existing preserved files are byte-identical.']
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    preserved()
if __name__=='__main__':main()
