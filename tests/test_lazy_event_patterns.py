import copy
import json
from pathlib import Path

import pytest

from mlxsim.model_lazy_event_patterns import externalize,materialize
from mlxsim.model_array_graph_events import array_graph_events
from scripts.verify_mlx_event_schedule import audit_trace
from test_event_source_graph import graph_job
from test_event_source_order import two_sources
from test_control_events import base_request
from mlxsim.model_control_events import control_events
from test_event_schedule import binary,event
from test_event_loops import run


def compare(binary,tmp_path,p,cache=8*1024*1024):
    lazy=externalize(p,tmp_path/"patterns",cache);eager=materialize(lazy)
    expected=run(binary,tmp_path/"eager",eager);actual=run(binary,tmp_path/"lazy",lazy)
    extra=actual.pop("lazy_patterns")
    assert actual==expected
    assert extra["event_state_drained"] and extra["loaded_blocks"]==len(p["blocks"])+len(p.get("controllers",[]))
    assert extra["peak_serialized_cache_bytes"]<=cache and extra["peak_live_sequence_nodes"]<=1000000
    audit_trace(eager,actual)
    return lazy,actual,extra


@pytest.mark.parametrize("precision",["f16","f32"])
@pytest.mark.parametrize("slow",[False,True])
@pytest.mark.parametrize("limit",[2,4])
def test_real_source_graph_patterns_match_eager_calendar(binary,tmp_path,precision,slow,limit):
    compare(binary,tmp_path,array_graph_events(graph_job(precision,slow,limit)))


def test_recycled_metadata_is_bounded_by_residency_not_total_blocks(binary,tmp_path):
    p=two_sources();p["blocks"]=[]
    for i in range(2000):p["blocks"].append(dict(id=f"b{i}",source_operator_id=0,pe=0,template="t",admission_dependencies=[],events=[event("one"),event("two","add")]))
    _,r,extra=compare(binary,tmp_path,p)
    assert extra["allocated_event_slots"]==extra["peak_live_event_leaves"]==4
    assert extra["peak_live_sequence_nodes"]==4 and r["stored_event_leaves"]==4000
    assert extra["file_reads"]==1 and extra["cache_hits"]==1999


def test_control_and_memory_zero_work_and_private_units_match(binary,tmp_path):
    p=control_events(base_request());p["source_tick_order"]=True
    p["controllers"].append(dict(id="memory",domain="memory",source_operator_id=1,admission_dependencies=[],events=[dict(id="zero",op="memory_complete",dependencies=[])]))
    compare(binary,tmp_path,p)


def test_evicted_pattern_is_rehashed_when_loaded_again(binary,tmp_path):
    p=two_sources();p["hardware"]["contexts"]=1;p["blocks"]=[]
    for i in range(6):p["blocks"].append(dict(id=str(i),source_operator_id=0,pe=0,template="t",admission_dependencies=[],events=[event("a","mul" if i%2 else "add")]))
    _,_,extra=compare(binary,tmp_path,p,cache=150)
    assert extra["file_reads"]==6 and extra["cache_hits"]==0


def test_future_pattern_is_not_read_before_its_source_is_ready(binary,tmp_path):
    p=two_sources()
    for i,b in enumerate(p["blocks"]):b["events"]=[event("a","mul" if i else "add")]
    p["source_graph"]=[dict(source_operator_id=i,family="matrix",parents=[0] if i else [],windows=[[str(i)]]) for i in range(2)]
    lazy=externalize(p,tmp_path/"patterns");second=lazy["blocks"][1]["event_pattern"];lazy["event_patterns"][second]["path"]=str(tmp_path/"missing.json")
    lazy["max_cycles"]=1
    # The first source reaches the simulation budget before the future file is needed.
    run(binary,tmp_path/"budget",lazy,"cycle limit")


@pytest.mark.parametrize("damage",["hash","count","zero","domain","cache","unknown","dependency"])
def test_invalid_lazy_pattern_contract_rejected(binary,tmp_path,damage):
    p=two_sources()
    for b in p["blocks"]:b["events"]=[event("op")]
    lazy=externalize(p,tmp_path/"patterns");key=lazy["blocks"][0]["event_pattern"];spec=lazy["event_patterns"][key]
    error=""
    if damage=="hash":spec["sha256"]="0"*64;error="SHA256"
    elif damage=="count":spec["dynamic_events"]+=1;error="dynamic work"
    elif damage=="zero":spec["zero_work"]=True;error="zero-work array"
    elif damage=="cache":lazy["pattern_cache_bytes"]=1;error="cache byte"
    elif damage=="unknown":lazy["blocks"][0]["event_pattern"]="missing";error="known pattern"
    elif damage=="dependency":lazy["blocks"][1]["admission_dependencies"]=["0"];error="source-level"
    else:
        lazy["blocks"][0].pop("pe");lazy["blocks"][0].pop("template");lazy["blocks"][0]["domain"]="control"
        lazy["controllers"].append(lazy["blocks"].pop(0));error="resource domain"
    run(binary,tmp_path/"bad",lazy,error)


def test_externalizer_refuses_cross_block_leaf_edges(tmp_path):
    p=two_sources();p["blocks"][0]["events"]=[event("a")];p["blocks"][1]["events"]=[event("b",deps=["a"])]
    with pytest.raises(ValueError,match="cross-block"):externalize(p,tmp_path/"patterns")


def test_host_pattern_loading_does_not_remove_timed_rom_programming(binary,tmp_path):
    p=array_graph_events(graph_job());p["hardware"]["template_load_timing"]=True
    _,r,_=compare(binary,tmp_path,p)
    assert r["counts"]["template_program_pe_cycles"]>0


def test_cumulative_ir_exceeds_old_limit_without_resident_growth(binary,tmp_path):
    p=control_events(base_request());p["source_tick_order"]=True;p["trace_limit"]=0
    p["controllers"][0]["events"]=[dict(id=f"e{i}",op="control_literal",dependencies=[]) for i in range(600)]
    lazy=externalize(p,tmp_path/"patterns");first=lazy["controllers"][0]
    lazy["controllers"]=[dict(first,id=f"c{i}") for i in range(1667)]
    r=run(binary,tmp_path/"large",lazy)
    assert r["events"]==r["cycles"]==r["stored_event_leaves"]==1000200
    assert r["lazy_patterns"]["peak_live_sequence_nodes"]==r["lazy_patterns"]["allocated_event_slots"]==600
    assert r["lazy_patterns"]["event_state_drained"] and r["lazy_patterns"]["file_reads"]==1
