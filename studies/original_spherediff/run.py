"""Run a missing case with the unmodified pinned official SphereDiff class."""
import argparse
import importlib.metadata
import platform
import resource
import time
import traceback
from studies.original_spherediff.common import *

PROCESS_START = time.perf_counter()


def run(name, prompt):
    plan = require_gate()
    row = next(x for x in plan['rows'] if (x['backend'], x['prompt']) == (name, prompt))
    if row['reuse']:
        assert image_audit(row['output']) == row['reused_image']
        print('REUSED; no generation:', row['output'], flush=True)
        return
    import numpy as np
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('A GPU Slurm allocation is required')
    target = folder(name, prompt)
    with run_lock(target):
        target.mkdir()
        status = dict(state='running', backend=name, prompt=prompt, seed=0, job=os.environ['SLURM_JOB_ID'])
        atomic(target / 'status.json', status)
        try:
            classes, imports = import_official()
            assert imports == plan['import_paths']
            atomic(target / 'import_verification.json', dict(paths=imports, verified_before_loading_weights=True,
                source_commit=COMMIT, official_source_hashes=official_hashes()))
            spec = specs()[name]
            seed_global(torch, np)
            load_started = time.perf_counter()
            pipe = classes[name].from_pretrained(spec['model_source'], revision=spec['revision'], variant=spec['variant'],
                torch_dtype=torch.bfloat16, local_files_only=True, cache_dir=str(CACHE))
            pipe.to(torch.device('cuda'), dtype=torch.bfloat16)
            original_order = static_solver(pipe.scheduler)
            model_loading_seconds = time.perf_counter() - load_started
            assert not getattr(pipe.vae, 'use_tiling', False)
            forward = pipe.transformer.forward
            counter = ForwardLog(forward)
            pipe.transformer.forward = counter
            generator = fresh_generator(torch)
            call = arguments(name, prompt, generator)
            resolved = defaults(classes[name]); resolved.update(spec['call'])
            resolved.update(prompt_txt_path=call['prompt_txt_path'], generator='fresh torch.Generator(cuda), seed=0',
                callback_on_step_end='logging-only; returns the identical callback state')
            defaults_runtime = dict(default_sample_size=getattr(pipe, 'default_sample_size', None),
                vae_scale_factor=getattr(pipe, 'vae_scale_factor', None))
            if name == 'flux':
                defaults_runtime['resolved_nominal_height'] = pipe.default_sample_size * pipe.vae_scale_factor
                defaults_runtime['resolved_nominal_width'] = pipe.default_sample_size * pipe.vae_scale_factor
            else:
                defaults_runtime.update(resolved_nominal_height=1024, resolved_nominal_width=1024)
            setup_peak = torch.cuda.max_memory_allocated()
            setup_reserved = torch.cuda.max_memory_reserved()
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                with torch.no_grad():
                    result = pipe(**call)
                torch.cuda.synchronize()
                generation_seconds = time.perf_counter() - started
            finally:
                pipe.transformer.forward = forward
            schedule = scheduler_record(pipe.scheduler, original_order)
            expected = row['expected_scheduler']
            for key in ['class_name', 'config', 'timesteps', 'sigmas', 'effective_numerical_order']:
                assert schedule[key] == expected[key], (key, schedule[key], expected[key])
            assert counter.calls == spec['call']['num_inference_steps'] * 89
            assert counter.calls == sum(counter.shapes.values())
            image = result.images[0]
            assert image.size == (4096, 2048)
            image.save(target / 'final_result.png')
            image_record = image_audit(target / 'final_result.png')
            require_gate()
            metadata = dict(backend=name, prompt_name=prompt, pipeline_class=spec['pipeline'], actual_import_paths=imports,
                source_commit=COMMIT, original_source=str(SPHERE), official_source_hashes=official_hashes(),
                checkpoint=spec['model_source'], revision=spec['revision'], variant=spec['variant'], precision='torch.bfloat16',
                model_cpu_offload=False, vae_tiling=False, local_files_only=True, checkpoint_cache=str(snapshot(name)),
                explicit_call_arguments=spec['call'], resolved_call_arguments=resolved, runtime_defaults=defaults_runtime,
                prompt=plan['prompts'][prompt], seed=0, generator_device=str(generator.device), rng_order=plan['rng_evidence']['global_order'],
                scheduler=schedule, output=image_record, transformer_forwards=counter.calls,
                model_input_shape_histogram=dict(counter.shapes), generation_seconds=generation_seconds,
                model_loading_seconds=model_loading_seconds, process_seconds_since_module_start=time.perf_counter()-PROCESS_START,
                timing_scope='Generation timer brackets only the original pipeline call with CUDA synchronization; process timer starts before NumPy/Torch import inside this module and excludes shell activation.',
                generation_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_allocated_gib=max(setup_peak,torch.cuda.max_memory_allocated())/2**30,
                peak_reserved_gib=max(setup_reserved,torch.cuda.max_memory_reserved())/2**30,
                generation_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                node=platform.node(), gpu=torch.cuda.get_device_name(), job=os.environ['SLURM_JOB_ID'],
                environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                    diffusers=importlib.metadata.version('diffusers'), transformers=importlib.metadata.version('transformers'),
                    accelerate=importlib.metadata.version('accelerate'), cudnn=torch.backends.cudnn.version(),
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
                    attention_processors=sorted({type(x).__name__ for x in pipe.transformer.attn_processors.values()}) if hasattr(pipe.transformer,'attn_processors') else None),
                newly_generated=True, reused=False, representation='official persistent spherical latent state',
                final_output_semantics='Unmodified official terminal decoding and ERP assembly; images[0].',
                source_hashes=source_hashes(), manifest_sha256=sha(ROOT/'manifest.json'))
            atomic(target / 'metadata.json', metadata)
            assert read(target/'metadata.json')['output']['sha256'] == sha(target/'final_result.png')
            atomic(target/'status.json', dict(status, state='complete', metadata_sha256=sha(target/'metadata.json'),
                image_sha256=image_record['sha256'], generation_seconds=generation_seconds))
            print('COMPLETED ORIGINAL', name, prompt, generation_seconds, counter.calls, flush=True)
        except BaseException:
            atomic(target/'status.json', dict(status, state='failed', error=traceback.format_exc()))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=BACKENDS, required=True)
    p.add_argument('--prompt', choices=PROMPTS, required=True)
    a = p.parse_args(); run(a.backend, a.prompt)
