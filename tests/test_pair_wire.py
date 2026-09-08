"""Pair descriptors execute through the real external-edge device controller."""
import copy
import struct

import numpy as np
import pytest
import torch

from mlxsim.model_memory_program import Planner
from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_vector_program import vector_program
from system_sim.physical_device.pair_lowering import lower_pair,BYTES
from test_clocked_device import binary,run,BASE
from test_block_pipeline import matrix_pair
from test_model_tensor_semantics import literal,node,ref


def pair_job(precision="f16", *, batches=False, vector_producer=False):
    if batches:
        a=torch.arange(30,dtype=torch.float32).reshape(2,1,3,5)/16;w=torch.arange(140,dtype=torch.float32).reshape(1,4,7,5)/32;b=w.transpose(-2,-1)
        first=a@b;second=-first
        nodes=[node(0,"transpose",[ref("w"),-2,-1],b),node(1,"matmul",[ref("a"),ref("v0")],first),node(2,"neg",[ref("v1")],second)]
        nodes[1]["matrix_program"]=matrix_program("f32","f32");nodes[2]["vector_program"]=vector_program("neg",["f32"],"f32")
        assets={"a":literal(a),"w":literal(w)};producer,consumer=nodes[1:]
    elif vector_producer:
        x=torch.tensor([1,4,16]*11,dtype=torch.float16).reshape(33,1).expand(33,16).contiguous();first=x.mean(-1,keepdim=True);second=first.rsqrt()
        nodes=[node(0,"mean",[ref("x"),[-1],True],first),node(1,"rsqrt",[ref("v0")],second)]
        nodes[0]["vector_program"]=vector_program("mean",["f16"],"f16",width=16);nodes[1]["vector_program"]=vector_program("rsqrt",["f16"],"f16")
        assets={"x":literal(x)};producer,consumer=nodes
    else:
        program,_,_=matrix_pair(m=5,n=19,precision=precision);assets=program["assets"];nodes=program["nodes"][:2];producer,consumer=nodes
        dtype=torch.float16 if precision=="f16" else torch.float32
        a=torch.tensor(assets["a"]["values"],dtype=dtype);w=torch.tensor(assets["w"]["values"],dtype=dtype);first=torch.nn.functional.linear(a,w);second=first.rsqrt()
    planner=Planner()
    for name,spec in assets.items():planner.add_asset(name,spec)
    for item in nodes:planner.register(item,planned=True)
    bindings={};next_address=BASE+65536
    for name,layout in planner.layouts.items():
        root=layout["root"]
        if root in bindings:continue
        size=layout["storage_elements"]*{"f16":2,"f32":4,"i64":8,"bool":1}[layout["dtype"]]
        bindings[root]={"base":next_address,"bytes":size,"writable":root not in assets};next_address+=max(64,(size+63)//64*64)
    wire,route=lower_pair(producer,consumer,planner.layouts,bindings,event_slots=2)
    initial=[]
    for name,spec in assets.items():
        data=np.asarray(spec["values"],dtype={"f16":np.float16,"f32":np.float32}[spec["dtype"]]).tobytes()
        initial.append({"address":bindings[name]["base"],"data":list(data)})
    outputs=[{"address":bindings[planner.layouts[n["id"]]["root"]]["base"],"bytes":len(value.numpy().tobytes())} for n,value in ((producer,first),(consumer,second))]
    job={"base":BASE,"bytes":4*1024*1024,"matrix_options":{"rows":1,"columns":1,"trace":True},"vector_options":{"rows":1,"columns":1,"trace":True},"memory_options":{"trace":False},"initial":initial,
         "commands":[{"address":BASE+4096,"data":list(wire),"source_id":0}],"outputs":outputs}
    return job,[list(first.numpy().tobytes()),list(second.numpy().tobytes())],route


@pytest.mark.parametrize("precision,batches,vector_producer",[("f16",False,False),("f32",False,False),("f32",True,False),("f16",False,True)])
def test_pair_wire_fetches_both_programs_and_executes_actual_values(binary,tmp_path,precision,batches,vector_producer):
    job,expected,route=pair_job(precision,batches=batches,vector_producer=vector_producer)
    result=run(binary,tmp_path/"pair",job)
    assert result["outputs"]==expected
    window=result["launches"][0];kernel=window["kernel"]
    assert window["done"] and not window["error"] and window["backend"]=="pair"
    assert window["descriptor_bytes_fetched"]==BYTES==8832 and window["run_cycles"]==kernel["cycles"]
    assert kernel["producer_source"]==route["producer_source"] and kernel["consumer_source"]==route["consumer_source"]
    assert kernel["producer_batches"]==(8 if batches else 1)==len(kernel["producer_windows"])
    assert kernel["array"]["template_configuration_cycles_modeled"] and kernel["block_events"]["finished"] and kernel["physical_mux"]["idle"]
    transactions=sum(p["dma_requests"] for p in kernel["producer_windows"])+kernel["consumer_window"]["dma_requests"]
    assert window["transport"]["submitted"]==transactions+BYTES//8
    assert kernel["array"]["peak_rf_vectors_per_pe"]<=16 and kernel["array"]["peak_spm_vectors"]<=128 and kernel["array"]["peak_rom_words_per_pe"]<=32


def test_pair_status_polling_and_bus_retries_do_not_change_outputs(binary,tmp_path):
    job,expected,_=pair_job();plain=run(binary,tmp_path/"plain",job)
    job["status_polls"]=5;observed=run(binary,tmp_path/"observed",job);assert observed==plain
    job.update(latency=5,accept_period=3,nack_every=3);retried=run(binary,tmp_path/"retry",job)
    assert retried["outputs"]==expected and retried["launches"][0]["transport"]["nacks"]>0


@pytest.mark.parametrize("word,value,message",[(1,2,"header/version"),(4,99,"identity/event"),(6,33,"identity/event"),(7,0,"identity/event"),(8,1,"header/version"),(9,1,"header/version")])
def test_pair_descriptor_corruption_fails_before_tensor_execution(binary,tmp_path,word,value,message):
    job,_,_=pair_job();raw=bytearray(job["commands"][0]["data"]);struct.pack_into("<Q",raw,word*8,value);job["commands"][0]["data"]=list(raw)
    result=run(binary,tmp_path/"bad",job);window=result["launches"][0]
    assert not window["done"] and message in window["error"] and window["transport"]["idle"]
    assert all(c["address"]<BASE+65536 for c in result["commits"])


def test_pair_output_alias_is_rejected(binary,tmp_path):
    job,_,_=pair_job();raw=bytearray(job["commands"][0]["data"])
    struct.pack_into("<Q",raw,256+4288+544,job["outputs"][0]["address"])
    job["commands"][0]["data"]=list(raw);result=run(binary,tmp_path/"alias",job)
    assert not result["launches"][0]["done"] and "output buffers overlap" in result["launches"][0]["error"]


def test_pair_failure_drains_and_next_launch_uses_a_new_event_epoch(binary,tmp_path):
    job,expected,_=pair_job();bad=copy.deepcopy(job["commands"][0]);struct_bytes=bytearray(bad["data"]);struct.pack_into("<Q",struct_bytes,8,2);bad["data"]=list(struct_bytes)
    job["commands"].insert(0,bad);job["recover"]=True;job["reset_between"]=True
    result=run(binary,tmp_path/"recover",job)
    assert not result["launches"][0]["done"] and result["launches"][1]["done"] and result["outputs"]==expected
    assert result["launches"][1]["kernel"]["block_events"]["epoch"]==2


@pytest.mark.parametrize("fault",["descriptor_error","write_error"])
def test_pair_bus_faults_never_become_successful_outputs(binary,tmp_path,fault):
    job,_,_=pair_job();job["fault"]=fault
    result=run(binary,tmp_path/fault,job);window=result["launches"][0]
    assert not window["done"] and window["error"] and window["transport"]["idle"]
    assert not result["outputs"] and not window["busy"]


def test_pair_cycle_limit_drains_an_owned_bus_request(binary,tmp_path):
    job,_,_=pair_job();job.update(latency=8)
    baseline=run(binary,tmp_path/"baseline",job)
    accepted=min(a["cycle"] for a in baseline["attempts"] if a["address"]>=BASE+65536)
    job["max_busy_cycles"]=accepted+2
    result=run(binary,tmp_path/"limit",job);window=result["launches"][0]
    assert not window["done"] and "busy cycle limit" in window["error"] and window["transport"]["idle"]
    assert window["descriptor_bytes_fetched"]==BYTES and window["run_cycles"]>0 and window["drain_cycles"]>0 and not result["outputs"]
