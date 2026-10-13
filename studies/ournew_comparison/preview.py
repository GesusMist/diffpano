"""Development-only isolated import of candidate modules while legacy jobs run.

This never edits shared modules or claims to be a production validation gate.
"""
import importlib.util,sys,unittest
from pathlib import Path
from studies.ournew_comparison.common import OUT,write

def main():
    import diffpano
    for name,filename in (('diffpano.gradient_fusion','gradient_fusion.py'),('diffpano.poisson_reference','poisson_reference.py')):
        path=Path('studies/ournew_comparison/staged')/filename
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module);setattr(diffpano,name.split('.')[-1],module)
    name='studies.gradient_blending.operator';spec=importlib.util.spec_from_file_location(name,'studies/ournew_comparison/staged/operator.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    suite=unittest.defaultTestLoader.discover('studies/ournew_comparison/tests')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(OUT/'development-preview.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,production_gate=False))
    raise SystemExit(not result.wasSuccessful())
if __name__=='__main__':main()
