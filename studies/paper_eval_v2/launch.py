"""Thin finite launch dispatch through the one shared scheduler journal."""
from .common import *
from .resources import submit


def launch(run,component):
    folder=ROOT/'studies/paper_eval_v2'
    if component=='inventory':return submit(run,'inventory-v1',folder/'cpu.slurm',script_args=('--run',str(run),'inventory'),reserve_su=8)
    if component in ('cpu','gpu','views'):
        prepare=run/'protocol_registry.json'
        if not prepare.exists():raise RuntimeError('Freeze protocols before evaluation')
        group='views' if component=='views' else 'group1'
        stage='gpu' if component=='gpu' else 'cpu'
        role='evaluation' if component=='gpu' else 'cpu'
        return submit(run,component+'-v1',folder/('gpu.slurm' if component=='gpu' else 'cpu.slurm'),role=role,
            script_args=('--run',str(run),'evaluate','--group',group,'--stage',stage),reserve_su=584 if component=='gpu' else 8)
    if component=='native':
        from .native import launch as execute
        return execute(run)
    if component=='confirmation':
        from .generation import launch as execute
        return execute(run)
    if component=='api':
        from .api_launch import launch as execute
        return execute(run)
    raise ValueError(component)
