import inspect,unittest
from types import SimpleNamespace
import torch
from diffusers import FlowMatchEulerDiscreteScheduler,FluxPipeline
from diffpano.pipelines.flux import FluxViewDenoiser
from studies.perspective24.common import definitions,prompt_names,prompt_record,LABELS
from studies.perspective24.configuration import configuration
from studies.perspective24.runtime import sample,tensor_hash,encode_exact_lines,schedule_record,reset_schedule
from studies.perspective24.run import parse_indices

class Transformer:
    dtype=torch.float32
    config=SimpleNamespace(in_channels=64,guidance_embeds=True)
    def __call__(self,**kwargs):return self.forward(**kwargs)
    def forward(self,**kwargs):return (torch.ones_like(kwargs['hidden_states'])*kwargs['encoder_hidden_states'].flatten()[0]*.001,)
class VAE:
    dtype=torch.float32
    config=SimpleNamespace(scaling_factor=.3611,shift_factor=.1159)
    def encode(self,*a,**kw):raise AssertionError('Not an img2img trajectory')
    def decode(self,x,return_dict=False):return (x[:,:3].repeat_interleave(8,-2).repeat_interleave(8,-1),)
class Pipeline:
    _execution_device='cpu'
    vae_scale_factor=8
    _pack_latents=staticmethod(FluxPipeline._pack_latents)
    _unpack_latents=staticmethod(FluxPipeline._unpack_latents)
    _prepare_latent_image_ids=staticmethod(FluxPipeline._prepare_latent_image_ids)
    def __init__(self):
        _,expected,_=configuration();self.scheduler=FlowMatchEulerDiscreteScheduler.from_config(expected['config'])
        self.vae=VAE();self.transformer=Transformer();self.text_encoder=object();self.text_encoder_2=object();self.encoded=[]
    def encode_prompt(self,**kw):
        assert self.text_encoder is not None and self.text_encoder_2 is not None
        self.encoded.append(kw);n=sum(kw['prompt'].encode())%97
        return torch.full((1,512,4),float(n)),torch.full((1,3),float(n)),torch.zeros(512,3)

class PerspectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def backend(self):return FluxViewDenoiser(Pipeline(),guidance_scale=3.5,true_cfg_scale=1.)

    def test_exact_matrix_and_directional_lines(self):
        rows=definitions();self.assertEqual(len(rows),120);self.assertEqual(len({r['case_id'] for r in rows}),120)
        self.assertEqual(len(prompt_names()),24);self.assertNotIn('native_control',prompt_names())
        for i,name in enumerate(prompt_names()):
            record=prompt_record(name);subset=rows[5*i:5*i+5]
            self.assertEqual([r['direction_label'] for r in subset],list(LABELS))
            self.assertEqual([r['line_text'] for r in subset],record['effective_lines'])
            self.assertEqual([r['line_index'] for r in subset],[1,2,3,4,5])
            self.assertEqual({r['seed'] for r in subset},{0})
            self.assertEqual({(r['height'],r['width'],r['steps'],r['batch_size']) for r in subset},{(1024,1024,20,1)})

    def test_exact_encoding_lifecycle(self):
        b=self.backend();rows=definitions()[:5]+definitions()[-5:];result=encode_exact_lines(b,rows,'')
        self.assertEqual(len(result),10);self.assertIsNone(b.pipeline.text_encoder);self.assertIsNone(b.pipeline.text_encoder_2)
        self.assertEqual([k['prompt'] for k in b.pipeline.encoded],[r['line_text'] for r in rows])
        self.assertTrue(all(k['prompt_2'] is None and k['num_images_per_prompt']==1 and k['max_sequence_length']==512 for k in b.pipeline.encoded))
        self.assertEqual(len(b.pipeline.encoded),10)

    def test_native_only_counts_fresh_rng_schedule_order(self):
        b=self.backend();rows=definitions()[:2];conditions=encode_exact_lines(b,rows,'');expected=configuration()[1]
        first,one=sample(b,rows[0],conditions[0],expected)
        # An intervening case consumes global RNG and scheduler state; fresh seed
        # and native scheduler preparation must make replay exactly independent.
        torch.randn(31);middle,two=sample(b,rows[1],conditions[1],expected)
        again,three=sample(b,rows[0],conditions[0],expected)
        self.assertTrue(torch.equal(first,again));self.assertEqual(one['initialization'],three['initialization'])
        self.assertFalse(torch.equal(first,middle),'Each case must receive its own exact-line conditioning')
        self.assertEqual(one['initialization']['initial_native_sha256'],two['initialization']['initial_native_sha256'])
        self.assertEqual(one['counts'],dict(denoiser=20,encode=0,decode=1,initialize=1))
        self.assertEqual(one['guided_steps'],20);self.assertEqual(one['schedule'],expected)
        self.assertEqual(first.shape,(1,3,1024,1024))
        forbidden=('ordinary_run','StudyPipeline','erp_to_perspective','perspective_to_erp','prepare_prompt_conditioning','encode_clean','predict_clean')
        body=inspect.getsource(sample)
        self.assertTrue(all(name not in body for name in forbidden))

    def test_scheduler_is_new_instance_each_case(self):
        b=self.backend();old=b.pipeline.scheduler;row=definitions()[0]
        reset_schedule(b,row,configuration()[1]);self.assertIsNot(b.pipeline.scheduler,old)
        self.assertIsNone(b.pipeline.scheduler.step_index);self.assertEqual(schedule_record(b),configuration()[1])

    def test_index_parse_rejects_duplicates_and_out_of_matrix(self):
        self.assertEqual(parse_indices('0,2-4'),[0,2,3,4])
        for value in ('0,0','0-2,2','-1','120','3-1',''):
            with self.assertRaises(ValueError):parse_indices(value)

if __name__=='__main__':unittest.main()
