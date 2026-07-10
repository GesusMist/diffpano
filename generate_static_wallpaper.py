import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf
from tools_mpark.dictaction import DictAction

import pipelines_ours
from output_metadata import save_output_metadata


@dataclass
class TestConfig:
    # load pipeline
    pipeline_cls: str = "SanaPipeline"
    pretrained_model_name_or_path: str = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
    revision: Optional[str] = None
    variant: Optional[str] = 'bf16'
    mixed_precision: str = "bf16"
    additional_pipeline_kwargs: Optional[dict] = None

    # modify pipeline
    modify_method_functions: Optional[List[str]] = None

    # enable options
    enable_vae_tiling: bool = False
    enable_model_cpu_offload: bool = False

    # generation options  # integrated to call_kwargs
    prompt_to_log: Optional[List[Dict[str, Any]]] = None
    call_kwargs: Optional[Dict[str, Any]] = None

    # save options
    save_path: str = "./outputs/test"


def main(args: TestConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "CUDA is not available. Please install CUDA and cuDNN."

    dtype = (
        torch.float16
        if args.mixed_precision == "fp16"
        else torch.bfloat16
        if args.mixed_precision == "bf16"
        else torch.float32
    )

    # Create pipeline
    pipeline_cls = getattr(pipelines_ours, args.pipeline_cls)
    additional_pipeline_kwargs = args.additional_pipeline_kwargs if args.additional_pipeline_kwargs else {}
    pipe = pipeline_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=dtype,
        **additional_pipeline_kwargs,
    )

    assert not args.enable_vae_tiling, "enable_vae_tiling is not supported in `SanaPipeline`."
    if args.enable_vae_tiling:
        pipe.enable_vae_tiling() if hasattr(pipe, "enable_vae_tiling") else pipe.vae.enable_tiling()
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device, dtype=dtype)

    if pipe.scheduler.config.get('solver_order', 1) > 1:  # (added) cannot use multi-step solver
        print("Warning: solver_order > 1 is not supported. Setting solver_order to 1.")
        pipe.scheduler.config.solver_order = 1

    call_kwargs = args.call_kwargs if args.call_kwargs else {}

    with open(call_kwargs['prompt_txt_path'], 'r') as f:
        lines = f.readlines()
    args.prompt_to_log = [line.strip() for line in lines]

    output = pipe(**call_kwargs)

    filename = args.save_path + f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image = output.images[0]
    image.save(filename)
    metadata_filename = save_output_metadata(args, pipe, call_kwargs, filename)
    print(f"Saved image to {filename}")
    print(f"Saved parameters to {metadata_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_add', action=DictAction, default=dict(), nargs='*')
    args = parser.parse_args()

    args = TestConfig(**args.config_add)
    print(OmegaConf.to_yaml(args))

    main(args)
