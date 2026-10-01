import ast
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from studies.original_spherediff.common import *
from studies.original_spherediff.audit import match_reference, prepared_scheduler, prompt_record


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classes, cls.imports = import_official()

    def test_official_class_imports(self):
        self.assertEqual(set(self.classes), {'sana', 'flux'})
        for n, cls in self.classes.items():
            self.assertEqual(Path(inspect.getfile(cls)).resolve().parent, SPHERE/'pipelines_ours')
            self.assertEqual(cls.__name__, specs()[n]['pipeline'])

    def test_wrong_import_rejected(self):
        fake = types.SimpleNamespace(__file__='/tmp/wrong/pipelines_ours.py',
            SphericalSanaPipeline=self.classes['sana'], SphericalFluxPipeline=self.classes['flux'])
        with patch('studies.original_spherediff.common.importlib.import_module', return_value=fake):
            with self.assertRaises(RuntimeError):import_official()

    def test_exact_sana_arguments(self):
        token = object(); a = arguments('sana', 'ruins', token)
        expected = dict(num_inference_steps=20, guidance_scale=4.5, height=1024, width=1024,
            n_spherical_points=2600, weighted_average_temperature=.1, erp_height=2048, erp_width=4096)
        self.assertEqual({k:v for k,v in a.items() if k not in ['generator','prompt_txt_path','callback_on_step_end']}, expected)
        self.assertIs(a['generator'], token)
        s = specs()['sana']; self.assertEqual(s['revision'],'e2b3c0cbffebcd09d83805e88b9f5f106afc74ac')
        self.assertEqual(s['variant'],'bf16'); self.assertFalse(s['enable_model_cpu_offload']); self.assertFalse(s['enable_vae_tiling'])

    def test_exact_flux_arguments_and_defaults(self):
        a = arguments('flux', 'underwater', object())
        expected = dict(num_inference_steps=28, guidance_scale=3.5, true_cfg_scale=1., n_spherical_points=26500,
            weighted_average_temperature=.1, erp_height=2048, erp_width=4096)
        self.assertEqual({k:v for k,v in a.items() if k not in ['generator','prompt_txt_path','callback_on_step_end']}, expected)
        self.assertNotIn('height', a); self.assertNotIn('width', a)
        s=specs()['flux']; self.assertEqual(s['model_source'],'black-forest-labs/FLUX.1-dev')
        self.assertEqual(s['revision'],'3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'); self.assertIsNone(s['variant'])
        self.assertIsNone(defaults(self.classes['flux'])['height'])

    def test_exact_prompt_selection(self):
        for p in PROMPTS:
            a=arguments('sana',p,object()); self.assertEqual(a['prompt_txt_path'],str(REPO/'prompts'/(p+'.txt')))
            record=prompt_record(p); self.assertEqual(record['lines'],Path(a['prompt_txt_path']).read_text().splitlines())
            self.assertEqual(len(record['lines']),5); self.assertTrue(record['pinned_git_byte_identical'])
        with self.assertRaises(ValueError):arguments('sana','other',object())

    def test_seed_order_and_fresh_generators(self):
        events=[]
        class Generator:
            def __init__(self,device):self.device=device; events.append(('generator',device))
            def manual_seed(self,seed):self.seed=seed; events.append(('generator_seed',seed)); return self
        torch=types.SimpleNamespace(manual_seed=lambda s:events.append(('torch',s)),
            cuda=types.SimpleNamespace(manual_seed_all=lambda s:events.append(('cuda_all',s))),Generator=Generator)
        numpy=types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda s:events.append(('numpy',s))))
        with patch('studies.original_spherediff.common.random.seed', side_effect=lambda s:events.append(('random',s))):seed_global(torch,numpy)
        self.assertEqual(events,[('random',0),('numpy',0),('torch',0),('cuda_all',0)])
        a,b=fresh_generator(torch),fresh_generator(torch)
        self.assertIsNot(a,b);self.assertEqual((a.device,a.seed,b.device,b.seed),('cuda',0,'cuda',0))

    def test_official_generator_reaches_initialization(self):
        for n in BACKENDS:
            tree=ast.parse(Path(inspect.getfile(self.classes[n])).read_text())
            draws=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='randn_tensor']
            self.assertEqual(len(draws),1)
            self.assertIsInstance(draws[0].args[1],ast.Name);self.assertEqual(draws[0].args[1].id,'generator')
        text=Path(inspect.getfile(self.classes['flux'])).read_text()
        self.assertLess(text.index('self.prepare_latents('),text.index('latents = randn_tensor('))

    def test_reference_reuse_and_mismatch_rejection(self):
        for n in BACKENDS:
            m=read(HIST/'references'/n/'metadata.json'); p=prompt_record('ruins'); d=defaults(self.classes[n]);s=prepared_scheduler(n)
            check=lambda x:match_reference(x,n,p,self.imports,d,s)['matched']
            self.assertTrue(check(m))
            for key,value in [('seed',1),('generator_device','cpu'),('source_commit','other'),('output_resolution',[1024,2048])]:
                bad=copy.deepcopy(m);bad[key]=value;self.assertFalse(check(bad),key)
            for key,value in [('num_inference_steps',19),('n_spherical_points',42),('guidance_scale',0)]:
                bad=copy.deepcopy(m);bad['spec']['call'][key]=value;self.assertFalse(check(bad),key)
            bad=copy.deepcopy(m);bad['prompt']['lines'].reverse();self.assertFalse(check(bad))
            bad=copy.deepcopy(m);bad['spec']['revision']='other';self.assertFalse(check(bad))

    def test_output_separation_and_overwrite_protection(self):
        self.assertEqual(len({folder(n,p) for n in BACKENDS for p in PROMPTS}),4)
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'seed0'
            with run_lock(target):target.mkdir()
            with self.assertRaises(FileExistsError):
                with run_lock(target):pass

    def test_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'seed0'
            with run_lock(target):
                with self.assertRaises(BlockingIOError):
                    with run_lock(target):pass

    def test_scheduler_default_name_order_is_not_behavior(self):
        original={'_use_default_values':['shift_terminal','invert_sigmas'],'invert_sigmas':False}
        reordered={'_use_default_values':['invert_sigmas','shift_terminal'],'invert_sigmas':False}
        changed=dict(reordered,invert_sigmas=True)
        self.assertEqual(scheduler_config(original),scheduler_config(reordered))
        self.assertNotEqual(scheduler_config(original),scheduler_config(changed))
        self.assertEqual(original['_use_default_values'],['shift_terminal','invert_sigmas'])

    def test_scheduler_mapping_attribute_trap(self):
        from diffusers import DPMSolverMultistepScheduler
        scheduler=DPMSolverMultistepScheduler(solver_order=2)
        original=static_solver(scheduler);r=scheduler_record(scheduler,original)
        self.assertEqual((r['original_serialized_solver_order'],r['serialized_solver_order'],r['effective_config_attribute_solver_order']),(2,2,1))
        self.assertEqual(r['effective_numerical_order'],1)
        self.assertIn('self.config.solver_order == 1',inspect.getsource(scheduler.step))
        json.dumps(r,allow_nan=False)

    def test_logging_preserves_values_and_arguments(self):
        state={'latents':object()};self.assertIs(progress(None,0,0,state),state)
        result=object();hidden=types.SimpleNamespace(shape=(2,32,21,21));seen=[]
        def original(*a,**k):seen.append((a,k));return result
        count=ForwardLog(original)
        self.assertIs(count(hidden,flag=state),result);self.assertIs(seen[0][0][0],hidden);self.assertIs(seen[0][1]['flag'],state)
        self.assertEqual(count.calls,1);self.assertEqual(dict(count.shapes),{'[2, 32, 21, 21]':1})


if __name__=='__main__':unittest.main()
