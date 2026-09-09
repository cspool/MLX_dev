import json
from pathlib import Path
import subprocess
import sys
import copy

import pytest

from mlxsim.model_readback_timing import readback_job
from test_model_event_linker import fixture_program
from test_ready_graph import binary as native_binary, execute
from test_qa_result_contract import capture
from mlxsim.model_block_pipeline import compile_block_pipelines
from test_event_schedule import binary as event_binary
from test_model_event_linker import catalogue
from mlxsim.model_event_linker import ModelEventLinker
from scripts.verify_mlx_full_event_link import audit_graph
from scripts.mlx_system_attempt import digest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def readback_binary():
    build=ROOT/'build/readback-timing'
    for command in (['cmake','-S',str(ROOT/'simulator_ext/readback_timing'),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release'],
                    ['cmake','--build',str(build),'-j4']):
        subprocess.run(command,check=True,capture_output=True,timeout=120)
    return build/'mlx-readback-timing'


def run(binary,path,job):
    path.mkdir();source=path/'job.json';target=path/'result.json'
    source.write_text(json.dumps(job))
    process=subprocess.run([str(binary),str(source),str(target)],capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stderr
    return json.loads(target.read_text())


@pytest.mark.parametrize('latency,period',[(1,1),(8,1),(3,2),(8,4)])
@pytest.mark.parametrize('kind',['batched','neg'])
def test_readback_matches_actual_native_loop(readback_binary,native_binary,tmp_path,latency,period,kind):
    p,_=fixture_program(kind)
    options=dict(tile_pipeline=True,template_load_timing=True,memory=dict(latency=latency,accept_period=period,nack_every=0))
    native=execute(native_binary,tmp_path/'native',p,options)
    job=readback_job(p,options,native['graph_cycles'])
    result=run(readback_binary,tmp_path/'event-readback',job)
    for field in ('graph_cycles','host_readback_cycles','host_readback_requests','shared_elapsed_cycles'):
        assert result[field]==native[field],field


@pytest.mark.parametrize('damage',['retry','period','negative','width','overflow'])
def test_invalid_readback_rejected(readback_binary,tmp_path,damage):
    job=dict(schema='mlx_post_graph_readback_v1',graph_cycles=0,memory=dict(latency=8,accept_period=1,nack_every=0),
             reads=[dict(elements=1,element_bytes=4)])
    if damage=='retry':job['memory']['nack_every']=2
    elif damage=='period':job['memory']['accept_period']=0
    elif damage=='negative':job['graph_cycles']=-1
    elif damage=='width':job['reads'][0]['element_bytes']=3
    else:job['graph_cycles']=2**64-1
    source=tmp_path/'job.json';source.write_text(json.dumps(job))
    p=subprocess.run([str(readback_binary),str(source),str(tmp_path/'result.json')],capture_output=True,text=True,timeout=30)
    assert p.returncode!=0 and not (tmp_path/'result.json').exists()


@pytest.mark.parametrize('length',[28,64])
@pytest.mark.parametrize('period',[1,4])
def test_qa_reads_logits_mask_and_both_offsets(readback_binary,native_binary,tmp_path,length,period):
    p,_,_,_=capture(tmp_path,length)
    p=compile_block_pipelines(p,event_slots=4)
    options=dict(tile_pipeline=True,template_load_timing=True,memory=dict(latency=8,accept_period=period,nack_every=0))
    native=execute(native_binary,tmp_path/'native',p,options)
    job=readback_job(p,options,native['graph_cycles'])
    assert [r['role'] for r in job['reads']]==['start_logits','end_logits','context_mask','offsets_utf8']
    result=run(readback_binary,tmp_path/'readback',job)
    assert result['host_readback_requests']==native['host_readback_requests']==5*length
    assert result['readback_bytes']==25*length
    assert result['host_readback_cycles']==native['host_readback_cycles']
    assert result['shared_elapsed_cycles']==native['shared_elapsed_cycles']


def test_actual_graph_runner_then_readback_cli(readback_binary,event_binary,native_binary,tmp_path):
    # Genuine execution and audit of a complete *fixture*, never a BERT result.
    p,_=fixture_program('batched')
    options=dict(tile_pipeline=True,template_load_timing=True,memory=dict(latency=3,accept_period=4,nack_every=0))
    program_path=tmp_path/'program.json';program_path.write_text(json.dumps(p))
    options_path=tmp_path/'options.json';options_path.write_text(json.dumps(options))
    manifest,patterns=catalogue(p)
    linked=tmp_path/'linked';linked.mkdir()
    graph=ModelEventLinker(p,manifest,lambda k:copy.deepcopy(patterns[k]),linked/'patterns',options).link()
    graph_path=linked/'graph.json';graph_path.write_text(json.dumps(graph))
    inputs={str(path):digest(path) for path in (program_path,options_path,graph_path)}
    def read_block(meta):
        path=Path(meta['path']);inputs[str(path)]=digest(path)
        return json.loads(path.read_text())
    audit=audit_graph(graph,manifest,p,lambda k:copy.deepcopy(patterns[k]),read_block)
    audit.update(classification='independent_full_event_link_audit_not_execution',inputs=inputs,
                 auditor_sha256=digest(ROOT/'scripts/verify_mlx_full_event_link.py'))
    audit_path=tmp_path/'audit.json';audit_path.write_text(json.dumps(audit))
    graph_run=tmp_path/'graph-run';readback=tmp_path/'readback'
    first=[sys.executable,'-m','scripts.run_mlx_full_event_graph','--linked',str(linked),'--audit',str(audit_path),
           '--binary',str(event_binary),'--output',str(graph_run),'--host-timeout','120']
    result=subprocess.run(first,capture_output=True,text=True,timeout=150,cwd=ROOT)
    assert result.returncode==0,result.stdout+result.stderr
    second=[sys.executable,'-m','scripts.complete_mlx_event_readback','--event-run',str(graph_run),
            '--program',str(program_path),'--runtime-options',str(options_path),'--binary',str(readback_binary),'--output',str(readback)]
    result=subprocess.run(second,capture_output=True,text=True,timeout=60,cwd=ROOT)
    assert result.returncode==0,result.stdout+result.stderr
    native=execute(native_binary,tmp_path/'native',p,options)
    report=json.loads((readback/'report.json').read_text())
    assert report['event_total_cycles']==native['shared_elapsed_cycles']
    assert report['event_graph_cycles']==native['graph_cycles']
    assert report['event_readback_cycles']==native['host_readback_cycles']
    assert not report['model_performance_error_available'] and not report['mlx_system_verified']
    # Changed target options must not be accepted by reusing the successful run.
    options['memory']['latency']+=1;options_path.write_text(json.dumps(options))
    second[-1]=str(tmp_path/'wrong-options')
    result=subprocess.run(second,capture_output=True,text=True,timeout=60,cwd=ROOT)
    assert result.returncode!=0 and 'program/options differ' in result.stderr
    assert not (tmp_path/'wrong-options').exists()
