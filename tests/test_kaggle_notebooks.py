"""Execute notebook launch logic; isolate only the GPU subprocess boundary."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def test_changed_compile_setting_runs_a_fresh_gpu_preflight(tmp_path):
    notebook=json.loads((Path(__file__).resolve().parents[1]/'kaggle_t5.ipynb').read_text())
    source=notebook['cells'][3]['source']
    commands=[]
    def run(command,**kwargs):
        assert kwargs['check'] is True
        commands.append(command)
        destination=Path(command[command.index('--output-dir')+1])
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'complete.json').write_text('{}')
    # Only redirect the platform's absolute working mount to this test fixture.
    def platform_path(value):
        return tmp_path/'working' if str(value)=='/kaggle/working' else Path(value)
    namespace={'Path':platform_path,'json':json,'os':os,'sys':sys,
        'subprocess':SimpleNamespace(run=run),'ARCHITECTURE':'t5','BUNDLE_ROOT':tmp_path,
        'EXPECTED_MANIFEST_SHA':'abc123','MICRO_BATCH':16,'NUM_GPUS':2,
        'GLOBAL_BATCH':256,'LEARNING_RATE':2e-4,'PRECISION':'fp16','USE_COMPILE':False,
        'GRADIENT_CHECKPOINTING':True,'RESUME_FROM':None,'OUTPUT':tmp_path/'full'}
    exec(compile(source,'notebook launch cell','exec'),namespace)
    namespace['USE_COMPILE']=True
    namespace['launch'](preflight=True)
    assert len(commands)==2, 'Changed numerical/compile settings must not reuse a previous preflight.'
    assert '--torch-compile' not in commands[0] and '--torch-compile' in commands[1]
    outputs=[c[c.index('--output-dir')+1] for c in commands]
    assert outputs[0]!=outputs[1]
