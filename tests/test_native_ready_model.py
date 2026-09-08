import copy
import json
from pathlib import Path
import subprocess
import struct

import pytest

from mlxsim.model_ready_evidence import verify_ready_execution
from scripts.run_mlx_ready_model import compare_outputs,parameter_extents
from test_pair_graph import separated_pair,batch_pair
from test_block_pipeline import matrix_pair

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module",params=["separated","batch","tail","generation","perturb"])
def attempt(request,tmp_path_factory):
    path=tmp_path_factory.mktemp('ready-model-'+request.param)
    if request.param in ('generation','perturb'):
        source=ROOT/'artifacts/tagged/pair-graph-003/pytest'/('test_generated_rv64_dispatch_r'+str(int(request.param=='perturb')))
        program=json.loads((source/'program.json').read_text());reference=json.loads((source/'reference.json').read_text())
    else:
        if request.param=="tail":program,expected,tokens=matrix_pair(m=5,n=19);program.pop('block_pipeline_plan')
        else:program,expected=separated_pair() if request.param=="separated" else batch_pair();tokens=expected.argmax(-1).reshape(-1).tolist()
        (path/'logits.bin').write_bytes(expected.tobytes())
        reference={'outputs':[{'forward_id':0,'dtype':'f16' if expected.dtype.itemsize==2 else 'f32','shape':list(expected.shape),'tokens':tokens,'logits_file':str(path/'logits.bin')}]}
    (path/'program.json').write_text(json.dumps(program))
    (path/'reference.json').write_text(json.dumps(reference))
    options={'base':2**32,'bytes':1048576,'max_cycles':10000000,'max_active_nodes':32,'tile_pipeline':True,'template_load_timing':True,'operator_progress':True,'memory':{'trace_limit':0,'latency':3,'nack_every':5}}
    (path/'options.json').write_text(json.dumps(options))
    command=[str(ROOT/'.venv/bin/python'),'-m','scripts.run_mlx_ready_model','--program',str(path/'program.json'),'--reference',str(path/'reference.json'),'--options',str(path/'options.json'),'--event-slots','2']
    process=subprocess.run([*command,'--output',str(path/'run')],cwd=ROOT,capture_output=True,text=True,timeout=300)
    (path/'run.log').write_text(process.stdout+process.stderr)
    assert process.returncode==0,process.stdout+process.stderr
    return path,command


def load(path):return json.loads(path.read_text())


def test_actual_native_model_attempt_covers_every_source_and_reads_outputs(attempt):
    path,_=attempt;report=load(path/'run/comparison.json');state=load(path/'run/execution.json')
    assert state['status']=='exited' and state['exit_code']==0 and state['validation']=='native_source_work_checked_numeric_pass'
    assert report['numeric_reference_bitwise_passed'] and report['coverage']['pipeline_groups']==(3 if len(report['comparison'])==3 else 1)
    assert report['coverage']['all_buffers_released'] and report['coverage']['all_events_drained']
    assert not report['actual_cpu_execution'] and not report['full_model_execution_verified'] and not report['inference_performance_eligible']


@pytest.mark.parametrize('damage',['source','window','macs','dependency','resource','pair','epoch','template','readback','memory','leak','fallback','control','template_period'])
def test_native_completion_evidence_rejects_omissions_and_false_work(attempt,damage):
    path,_=attempt;program=load(path/'run/program.json');result=load(path/'run/native/result.json');options=load(path/'options.json')
    if damage=='source':result['events'].pop()
    elif damage=='window':result['windows']['matrix'].pop()
    elif damage=='macs':result['windows']['matrix'][0]['numeric_instructions']['mul_active_lanes']-=1
    elif damage=='dependency':
        next(e for e in result['events'] if e['kind']=='argmax')['start_cycle']=0
    elif damage=='resource':result['array']['peak_rf_vectors_per_pe']=17
    elif damage=='pair':result['pipeline_groups'].pop()
    elif damage=='epoch':result['pipeline_groups'][0]['epoch']+=1
    elif damage=='template':result['array']['pending_template_words']=1
    elif damage=='readback':result['host_readback_requests']-=1
    elif damage=='memory':result['memory']['read_bytes']-=1
    elif damage=='leak':result['arena_drained']['reserved_bytes']=64
    elif damage=='fallback':result['blas_calls']=1
    elif damage=='control':result['windows']['control'][0]['instructions']=0
    else:result['array']['template_word_period']+=1
    with pytest.raises(RuntimeError):verify_ready_execution(program,result,options)


def test_numeric_failure_preserves_real_process_and_target_outputs(attempt,tmp_path):
    path,command=attempt;reference=load(path/'reference.json');bad=tmp_path/'wrong.bin';bad.write_bytes(bytes(Path(reference['outputs'][0]['logits_file']).stat().st_size))
    reference['outputs'][0]['logits_file']=str(bad);file=tmp_path/'reference.json';file.write_text(json.dumps(reference))
    command=list(command);command[command.index('--reference')+1]=str(file);out=tmp_path/'run'
    process=subprocess.run([*command,'--output',str(out)],cwd=ROOT,capture_output=True,text=True,timeout=300)
    assert process.returncode!=0 and (out/'native/result.json').exists()
    assert not load(out/'comparison.json')['numeric_reference_bitwise_passed']
    assert load(out/'execution.json')['exit_code']==0 and load(out/'execution.json')['validation']=='native_source_work_checked_numeric_failure'


def test_full_scope_requires_complete_bound_model_before_simulation(tmp_path):
    program,_=separated_pair();file=tmp_path/'program.json';file.write_text(json.dumps(program));options=tmp_path/'options.json';options.write_text(json.dumps({'tile_pipeline':True,'template_load_timing':True}))
    reference=tmp_path/'reference.json';reference.write_text('{}');source=tmp_path/'source.json';source.write_text(json.dumps({'model_identity':{'family':'toy'}}));out=tmp_path/'run'
    process=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.run_mlx_ready_model','--program',str(file),'--reference',str(reference),'--options',str(options),'--source-inventory',str(source),'--scope','public-dense-llama2','--output',str(out)],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert process.returncode!=0 and 'complete public dense checkpoint' in process.stderr and not (out/'execution.json').exists()


@pytest.mark.parametrize('damage',[None,'shape','offset','bytes','dtype','shard','duplicate'])
def test_parameter_extents_are_bound_to_actual_checkpoint_headers(tmp_path,damage):
    header=json.dumps({'weight':{'dtype':'F16','shape':[2,3],'data_offsets':[0,12]}}).encode();file=tmp_path/'model.safetensors';file.write_bytes(struct.pack('<Q',len(header))+header+bytes(12))
    asset={'kind':'mapped_file','parameter_name':'weight','path':str(file),'dtype':'f16','shape':[2,3],'byte_offset':8+len(header),'bytes':12};program={'assets':{'w':asset}};index={'weight_map':{'weight':file.name}}
    if damage=='shape':asset['shape']=[1,6]
    elif damage=='offset':asset['byte_offset']+=2
    elif damage=='bytes':asset['bytes']=10
    elif damage=='dtype':asset['dtype']='f32'
    elif damage=='shard':asset['path']=str(tmp_path/'other.safetensors')
    elif damage=='duplicate':program['assets']['copy']=dict(asset)
    if damage:
        with pytest.raises(RuntimeError):parameter_extents(program,index,tmp_path,{'weight'})
    else:assert parameter_extents(program,index,tmp_path,{'weight'})==6


def test_cycle_limit_preserves_failed_process_without_output_certificate(attempt,tmp_path):
    path,command=attempt;options=load(path/'options.json');options['max_cycles']=1;file=tmp_path/'options.json';file.write_text(json.dumps(options));command=list(command);command[command.index('--options')+1]=str(file);out=tmp_path/'limited'
    process=subprocess.run([*command,'--output',str(out)],cwd=ROOT,capture_output=True,text=True,timeout=300)
    assert process.returncode!=0 and load(out/'execution.json')['exit_code']!=0
    assert not (out/'comparison.json').exists() and not (out/'native/result.json').exists()


def test_repeated_shard_bindings_share_collection_hash_but_not_final_checks(tmp_path,monkeypatch):
    import scripts.run_mlx_ready_model as runner
    file=tmp_path/'shard';file.write_bytes(b'weights');original=runner.sha;calls=[]
    def counted(path):calls.append(path);return original(path)
    monkeypatch.setattr(runner,'sha',counted);files={};digest=runner.bind(file,files)
    assert runner.bind(file,files,digest)==digest and len(calls)==1
    with pytest.raises(RuntimeError):runner.bind(file,files,'0'*64)
    file.write_bytes(b'changed')
    assert not all(runner.sha(Path(p))==h for p,h in files.items())
