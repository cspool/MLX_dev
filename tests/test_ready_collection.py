"""Many persistent inputs plus aliases must preserve physical collection semantics."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from mlxsim.model_ready_evidence import verify_ready_execution
from test_block_pipeline import matrix_pair

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('assets',[0,32,512])
def test_persistent_asset_regions_do_not_change_lifetimes_or_results(tmp_path,assets):
    program,expected,tokens=matrix_pair(m=3,n=19,k=5,pes=16,precision='f32')
    for i in range(assets):program['assets'][f'persistent-{i:04d}']={'kind':'literal','dtype':'f32','shape':[1],'values':[i/16]}
    for kind in ('matrix','vector'):program[kind+'_schedule_options']['trace']=False
    options={'base':2**32,'bytes':1048576,'max_cycles':10000000,'tile_pipeline':True,'template_load_timing':True,'memory':{'latency':8,'trace_limit':0}}
    file=tmp_path/'program.json';file.write_text(json.dumps(program));config=tmp_path/'options.json';config.write_text(json.dumps(options));target=tmp_path/'out'
    p=subprocess.run([str(ROOT/'build/collection-candidate/mlx-ready-graph'),str(file),str(config),str(target)],capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stdout+p.stderr
    result=json.loads((target/'result.json').read_text());verify_ready_execution(program,result,options)
    assert result['preloaded_assets']==len(program['assets']) and result['memory']['live_payload_bindings']==0
    assert np.fromfile(result['outputs'][0]['logits_file'],dtype=np.float32).tobytes()==expected.tobytes() and result['outputs'][0]['tokens']==tokens
