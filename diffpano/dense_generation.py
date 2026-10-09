"""Normal CLI construction for dense and bridge pipelines, without study output gates."""
from diffpano.erp_pipeline import ERPGenerationResult
from diffpano.refinement import resolved_refinement


def generate_dense(config,backend):
    from scripts.dense_erp_experiment import build_dense_pipeline,prepare
    from diffpano.bridge_factorial import BridgeFactorialPipeline,saved_cameras,make_operator
    if config.global_pipeline.mode=='erp_bridge_factorial':
        cameras=saved_cameras(config.view,(config.erp.height,config.erp.width))
        pipe=BridgeFactorialPipeline(backend=backend,cameras=cameras,erp_size=(config.erp.height,config.erp.width),
            warp_operator=make_operator(config),backend_name=config.model.pipeline,refinement_config=config.global_pipeline.refinement)
        minimum=1
    else:pipe,_,minimum=build_dense_pipeline(config,backend)
    conditions,_=prepare(config,backend,pipe)
    pipe.precompute_geometry(minimum)
    states=pipe.initialize_local_states(config.experiment.seed,config.generation.batch_size)
    dense=pipe.run_dense(states,conditions,required_minimum=minimum)
    result=ERPGenerationResult(dense.erp_rgb,[])
    result.audit=dense.audit;result.dense_metrics=dense.metrics;result.stage_seconds=dense.stage_seconds
    return result
