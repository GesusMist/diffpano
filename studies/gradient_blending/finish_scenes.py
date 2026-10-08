"""Final audit of all 60 matched scene outputs, metrics and figure artifacts."""
from studies.gradient_blending.common import *

def main():
    assert BACKEND=='flux' and SUITE=='flux-scenes20'
    preserved();preserve_prior_results()
    gate=read(OUT/'validation.json');assert gate['passed'] and gate['source_hashes']==source_hashes()
    statuses=[]
    for prompt in PROMPTS:
        matched=[]
        for mode in MODES:
            folder=OUT/'cases'/prompt/mode;status=read(folder/'status.json');m=read(folder/'metadata.json')
            assert status['state']=='complete' and sha(folder/'final.png')==m['artifacts']['final.png']
            assert sha(folder/'metadata.json')==status['metadata_sha256']
            assert m['counts']==expected_counts('flux',20) and m['configuration']['generation']['num_inference_steps']==20
            if mode!='rgb':
                assert len(m['fusions'])==21 and m['fusions'][-1]['terminal']
                assert all(v['mode']==mode and v['converged'] for v in m['fusions'])
                assert m['fusion']['lambda_color']==.1 and m['terminal_fusion_mode']==mode
                if prompt in NEW_FLUX_PROMPTS:assert m['source_hashes']==gate['source_hashes']
            matched.append(m)
            statuses.append(dict(prompt=prompt,mode=mode,state='complete',job=status.get('job'),image_sha256=m['artifacts']['final.png']))
        for field in ('camera_geometry_sha256','schedule_sha256'):
            assert len({m[field] for m in matched})==1,(prompt,field)
        assert len({m['initialization']['initial_local_sha256'] for m in matched})==1
        for suffix in ('final-erp','fixed-views','detail-wrap'):
            assert (OUT/'figures'/(prompt+'-'+suffix+'.png')).is_file()
    evaluation=read(OUT/'evaluation/summary.json');assert len(evaluation['rows'])==60
    for mode in MODES:
        assert evaluation['collections'][mode]['panorama_count']==20
        assert evaluation['collections'][mode]['view_count']==160
        assert evaluation['collection_groups']['new_17'][mode]['panorama_count']==17
        assert evaluation['collection_groups']['new_17'][mode]['view_count']==136
        assert evaluation['collection_groups']['pilot_3'][mode]['panorama_count']==3
    assert (OUT/'report.txt').is_file()
    write(OUT/'completion.json',dict(passed=True,rows=statuses,logical_outputs=60,new_gradient_runs=34,
        new_prompts=NEW_FLUX_PROMPTS,reused_pilot_prompts=PILOT_PROMPTS,excluded_prompt='native_control',
        source_preserved=True,prior_results_preserved=True,report_sha256=sha(OUT/'report.txt'),
        evaluation_sha256=sha(OUT/'evaluation/summary.json')))
    from studies.gradient_blending.scenes_status import main as status
    status();print('FLUX_SCENE_STUDY_COMPLETE',60,flush=True)
if __name__=='__main__':main()
