"""Weights-free audit of pinned original source, prompts, caches and reusable references."""
import ast
import inspect
import subprocess
from studies.original_spherediff.common import *


def match_reference(metadata, name, prompt, imports, current_defaults, expected_scheduler):
    """Reject any scientific mismatch; old scheduler mapping is not effective order."""
    reasons = []
    checks = dict(source_commit=metadata.get('source_commit') == COMMIT,
        original_source=Path(metadata.get('original_source', '/missing')).resolve() == SPHERE.resolve(),
        backend=metadata.get('backend') == name, spec=metadata.get('spec') == specs()[name],
        seed=metadata.get('seed') == 0, generator_device=metadata.get('generator_device') == 'cuda',
        prompt=metadata.get('prompt', {}).get('sha256') == prompt['sha256'] and metadata.get('prompt', {}).get('lines') == prompt['lines'],
        resolution=metadata.get('output_resolution') == [2048, 4096],
        representation=metadata.get('representation') == 'original persistent spherical native latent state')
    checks['source_hashes'] = bool(metadata.get('source_hashes')) and all(
        official_hashes().get(p) == h for p, h in metadata.get('source_hashes', {}).items())
    wrapper = REPO / 'scripts/bridge_factorial_reference.py'
    checks['preserved_wrapper'] = metadata.get('wrapper_source_hashes', {}).get('scripts/bridge_factorial_reference.py') == sha(wrapper)
    effective = metadata.get('effective_call', {})
    checks['explicit_arguments'] = all(effective.get(k) == v for k, v in specs()[name]['call'].items())
    ignored = set(specs()[name]['call']) | {'prompt_txt_path', 'generator', 'callback_on_step_end'}
    checks['other_official_defaults'] = all(safe(effective.get(k)) == v for k, v in current_defaults.items() if k not in ignored)
    schedule = safe(metadata.get('prepared_schedule', {}))
    checks['scheduler_config'] = scheduler_config(schedule.get('config', {})) == expected_scheduler['config']
    checks['scheduler_class'] = schedule.get('class_name') == expected_scheduler['class_name']
    checks['scheduler_timesteps'] = schedule.get('timesteps') == expected_scheduler['timesteps']
    checks['scheduler_sigmas'] = schedule.get('sigmas') == expected_scheduler['sigmas']
    actual = schedule.get('effective_config_attribute_overrides', {}).get('solver_order', metadata.get('effective_solver_order'))
    checks['effective_scheduler'] = actual == expected_scheduler['effective_numerical_order']
    checks['forward_count'] = metadata.get('transformer_forwards') == 89 * specs()[name]['call']['num_inference_steps']
    checks['shape_counts'] = sum(metadata.get('transformer_input_shape_histogram', {}).values()) == metadata.get('transformer_forwards')
    for key, ok in checks.items():
        if not ok:
            reasons.append(key)
    return dict(matched=not reasons, checks=checks, mismatches=reasons,
                import_paths_verified_now=imports, historical_import_evidence='Preserved fresh-process wrapper inserts the verified snapshot before importing pipelines_ours; no historic inspect.getfile record was captured.')


def prompt_record(prompt):
    path = REPO / 'prompts' / (prompt + '.txt')
    raw = path.read_bytes()
    data = dict(path=str(path), sha256=sha(path), text=path.read_text(), lines=path.read_text().splitlines(), seed=0)
    assert len(data['lines']) == 5
    evidence = []
    manifest_path = HIST / 'manifest.json' if prompt == 'ruins' else REPO / 'outputs/gwtf-underwater/20260923/manifest.json'
    original = read(manifest_path)
    record = original['prompt'] if prompt == 'ruins' else dict(sha256=original['prompt_sha256'], lines=original['prompt_lines'])
    assert record['sha256'] == data['sha256'] and record['lines'] == data['lines']
    root = REPO / 'outputs' / ('gwtf-erp4k' if prompt == 'ruins' else 'gwtf-underwater') / '20260923'
    for name in BACKENDS:
        p = root / name / 'metadata.json'; m = read(p)
        assert m['config']['experiment']['seed'] == 0
        assert m['config']['prompt']['path'] == 'prompts/' + prompt + '.txt'
        if prompt == 'underwater':
            assert m['prompt_sha256'] == data['sha256']
        evidence.append(dict(metadata=str(p), sha256=sha(p), seed=0))
    q = SPHERE / 'data/prompts' / (prompt + '.txt')
    committed = subprocess.check_output(['git', '-C', str(GIT_SOURCE), 'show', COMMIT + ':data/prompts/' + prompt + '.txt'])
    data.update(completed_run_evidence=evidence, prompt_manifest=str(manifest_path), prompt_manifest_sha256=sha(manifest_path),
        snapshot_prompt_exists=q.exists(), snapshot_byte_identical=q.exists() and q.read_bytes() == raw,
        pinned_git_prompt_sha256=hashlib.sha256(committed).hexdigest(), pinned_git_byte_identical=committed == raw,
        selection='Pass verified local file through prompt_txt_path; preserve all five lines in their original order.')
    return data


def prepared_scheduler(name):
    from diffusers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift
    cls = DPMSolverMultistepScheduler if name == 'sana' else FlowMatchEulerDiscreteScheduler
    scheduler = cls.from_config(read(snapshot(name) / 'scheduler/scheduler_config.json'))
    original = static_solver(scheduler)
    steps = specs()[name]['call']['num_inference_steps']
    if name == 'sana':
        scheduler.set_timesteps(steps, device='cpu')
    else:
        import numpy as np
        # Official default 1024 RGB -> 128 latent -> 4096 packed tokens,
        # before the separate spherical Gaussian draw (not forced patch sizing).
        mu = calculate_shift(4096, scheduler.config.get('base_image_seq_len', 256), scheduler.config.get('max_image_seq_len', 4096),
                             scheduler.config.get('base_shift', .5), scheduler.config.get('max_shift', 1.15))
        scheduler.set_timesteps(sigmas=np.linspace(1., 1. / steps, steps), device='cpu', mu=mu)
    record = scheduler_record(scheduler, original)
    record['implementation_file'] = inspect.getfile(cls)
    record['implementation_sha256'] = sha(inspect.getfile(cls))
    return record


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / '.audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start_path = ROOT / 'starting_checkout.json'
        if not start_path.exists():
            atomic(start_path, dict(head=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                branch=subprocess.check_output(['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip(),
                status=subprocess.check_output(['git', '-C', str(REPO), 'status', '--short'], text=True)))
        official = official_hashes()
        for rel, h in official.items():
            committed = subprocess.check_output(['git', '-C', str(GIT_SOURCE), 'show', COMMIT + ':' + rel])
            assert hashlib.sha256(committed).hexdigest() == h, rel
        prior = read(HIST / 'execution.json')
        assert prior['spherediff_commit'] == COMMIT
        assert all(official[p] == h for p, h in prior['spherediff_source_hashes'].items())
        classes, imports = import_official()
        prompts = {p: prompt_record(p) for p in PROMPTS}
        schedules = {n: prepared_scheduler(n) for n in BACKENDS}
        cache_records = {}
        for n in BACKENDS:
            p = snapshot(n); assert (p / 'model_index.json').is_file(), p
            files = {str(x.relative_to(p)): dict(bytes=x.stat().st_size, resolved_path=str(x.resolve()))
                     for x in sorted(p.rglob('*')) if x.is_file()}
            assert any(k.endswith('.safetensors') for k in files)
            assert not any(x.is_symlink() and not x.exists() for x in p.rglob('*'))
            cache_records[n] = dict(path=str(p), files=files, local_files_only=True)
        from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        correction = prior['original_sana_scheduler_metadata_correction']
        assert sha(inspect.getfile(DPMSolverMultistepScheduler)) == correction['scheduler_source_sha256']
        assert 'self.config.solver_order == 1' in inspect.getsource(DPMSolverMultistepScheduler.step)
        preserved = {}
        for base in ['diffpano', 'scripts', 'tests', 'studies/tt_cea']:
            for p in sorted((REPO / base).rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts:
                    preserved[str(p)] = sha(p)
        for p in [HIST / 'execution.json', HIST / 'manifest.json', HIST / 'validation.json']:
            preserved[str(p)] = sha(p)
        rows = []; matches = {}
        for n in BACKENDS:
            oldfolder = HIST / 'references' / n
            m = read(oldfolder / 'metadata.json')
            match = match_reference(m, n, prompts['ruins'], imports, defaults(classes[n]), schedules[n])
            matches[n] = match
            for p in oldfolder.iterdir():
                if p.is_file():preserved[str(p)] = sha(p)
            for prompt in PROMPTS:
                reuse = prompt == 'ruins' and match['matched']
                if prompt == 'ruins' and not reuse:
                    raise AssertionError('Historical reference mismatch requires explicit diagnosis: ' + repr((n, match['mismatches'], safe(m['prepared_schedule']['config']), schedules[n]['config'])))
                image = image_audit(oldfolder / 'final_result.png') if reuse else None
                rows.append(dict(backend=n, prompt=prompt, seed=0, spec=specs()[n], official_defaults=defaults(classes[n]),
                    reuse=reuse, output=image['path'] if reuse else str(folder(n, prompt) / 'final_result.png'),
                    metadata=str(oldfolder / 'metadata.json') if reuse else str(folder(n, prompt) / 'metadata.json'),
                    reused_image=image, reused_metadata_sha256=sha(oldfolder / 'metadata.json') if reuse else None,
                    expected_scheduler=schedules[n], reuse_evidence=match if reuse else None))
        for p in prompts.values():
            preserved[p['path']] = p['sha256']; preserved[p['prompt_manifest']] = p['prompt_manifest_sha256']
            for e in p['completed_run_evidence']:preserved[e['metadata']] = e['sha256']
        plan = dict(study=read(STUDY / 'config.yaml')['study'], source_commit=COMMIT, official_source=str(SPHERE),
            official_source_hashes=official, import_paths=imports, prompts=prompts, rows=rows, cache=cache_records,
            preserved=preserved, starting_checkout=read(start_path), max_concurrent_gpu_jobs=2,
            bounded_search='Inspected historical original-reference directories plus manifests/execution records in gwtf-underwater, gwtf-erp4k and gwtf-noise-comparison; no newer original underwater entries found.',
            rng_evidence=dict(sana='pipeline_spherical_sana.py:342 draws spherical latents using the supplied generator.',
                flux='pipeline_spherical_flux.py:342-350 first calls prepare_latents with the generator; line 386 then draws spherical latents with the same advanced generator. Both draws are preserved.',
                global_order='random, numpy, torch, cuda_all seeded before from_pretrained; fresh CUDA generator created after model loading, as in the preserved reference wrapper.'),
            scheduler_correction_evidence=correction, scheduler_bookkeeping_comparison='Only _use_default_values ordering is canonicalized: installed Diffusers constructs this name list using list(set(...)); all names, parameter values, timesteps and sigmas still match exactly.')
        immutable(ROOT / 'manifest.json', plan)
        immutable(ROOT / 'audit.json', dict(passed=True, manifest_sha256=sha(ROOT / 'manifest.json'), source_commit_verified_locally=True,
            import_paths=imports, references=matches, prompts=prompts, scheduler=schedules,
            new_cases=[(x['backend'], x['prompt']) for x in rows if not x['reuse']]))
        print('AUDIT PASSED: reuse 2 ruins originals; generate 2 underwater originals', flush=True)


if __name__ == '__main__':
    main()
