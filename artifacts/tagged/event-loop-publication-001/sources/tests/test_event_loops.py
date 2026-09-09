"""Compact loop IR preserves every concurrent event; no native timing claim."""
from collections import Counter
import copy
import json
import subprocess

import pytest

from test_event_schedule import binary,event,program


def expanded(p):
    result=copy.deepcopy(p);result["schema"]="mlx_event_schedule_v1";occurrences=Counter()
    def visit(items):
        for item in items:
            if "repeat" in item:
                for _ in range(item["repeat"]):yield from visit(item["body"])
            else:
                row=copy.deepcopy(item);row["id"]=f'{item["id"]}@{occurrences[item["id"]]}'
                occurrences[item["id"]]+=1;yield row
    for block in result["blocks"]:block["events"]=list(visit(block["events"]))
    for block in result["blocks"]:
        for item in block["events"]:item["dependencies"]=[f'{name}@{occurrences[name]-1}' for name in item["dependencies"]]
    return result


def run(binary,path,p,error=None):
    path.mkdir();(path/"program.json").write_text(json.dumps(p))
    result=subprocess.run([str(binary),str(path/"program.json"),str(path/"result.json")],capture_output=True,text=True,timeout=30)
    (path/"run.log").write_text(result.stdout+result.stderr)
    if error:
        assert result.returncode==1 and error in result.stderr,result.stdout+result.stderr
        assert not (path/"result.json").exists();return
    assert result.returncode==0,result.stdout+result.stderr
    return json.loads((path/"result.json").read_text())


def normalize(result):
    result=copy.deepcopy(result)
    for key in ("schema","stored_event_leaves","stored_sequence_nodes","loop_execution","named_dependency_scope","timing_semantics"):result.pop(key,None)
    for row in result["trace"]:
        if "instance" in row:row["operation"]+=f'@{row.pop("instance")}'
    return result


@pytest.mark.parametrize("policy",["source_priority","round_robin"])
@pytest.mark.parametrize("timed",[False,True])
@pytest.mark.parametrize("same_pe",[False,True])
def test_nested_loops_equal_full_unrolling_with_concurrent_resources(binary,tmp_path,policy,timed,same_pe):
    p=program(count=3,same_pe=same_pe,timed=timed);p.update(schema="mlx_event_schedule_v2",policy=policy)
    for i,block in enumerate(p["blocks"]):
        block["events"]=[event(f"init{i}","zero"),{"repeat":3,"body":[event(f"read{i}","dma_read"),
                           {"repeat":2,"body":[event(f"mul{i}"),event(f"add{i}","add")]}]},event(f"write{i}","dma_write")]
    p["blocks"][2]["events"][0]["dependencies"]=["add0"]
    loop=run(binary,tmp_path/"loop",p);flat=run(binary,tmp_path/"flat",expanded(p))
    assert normalize(loop)==flat
    assert loop["stored_event_leaves"]==15 and loop["events"]==51
    assert loop["peak"]["contexts"]>1 and loop["all_resources_drained"]
    assert not loop["tensor_values_executed"] and not loop["full_model_verified"]


def test_late_leaf_dependency_waits_for_its_final_loop_occurrence(binary,tmp_path):
    p=program();p["schema"]="mlx_event_schedule_v2"
    p["blocks"][0]["events"]=[{"repeat":4,"body":[event("step","add")]},event("tail","dma_read")]
    p["blocks"][1]["events"]=[event("consumer","mul",["step"])]
    result=run(binary,tmp_path/"last",p)
    trace=result["trace"]
    last=next(r["cycle"] for r in trace if r.get("operation")=="step" and r.get("instance")==3 and r["event"]=="complete")
    consumer=next(r["cycle"] for r in trace if r.get("operation")=="consumer" and r["event"]=="issue")
    assert consumer==last+1
    assert normalize(result)==run(binary,tmp_path/"flat",expanded(p))


def test_large_repeat_executes_all_instances_with_bounded_ir(binary,tmp_path):
    p=program(count=1);p.update(schema="mlx_event_schedule_v2",trace_limit=0)
    p["blocks"][0]["events"]=[{"repeat":10000,"body":[event("load","dma_read"),event("compute","mul")]}]
    result=run(binary,tmp_path/"large",p)
    assert result["events"]==result["counts"]["events_completed"]==20000
    assert result["stored_event_leaves"]==2 and result["stored_sequence_nodes"]==3
    assert result["declared_work"]["dma_read_bytes"]==40000
    assert result["cycles"]==140000 and result["calendar_transitions"]==20001


def test_billion_repeat_reaches_cycle_budget_without_allocating_instances(binary,tmp_path):
    p=program(count=1);p.update(schema="mlx_event_schedule_v2",max_cycles=100)
    p["blocks"][0]["events"]=[{"repeat":10**9,"body":[event("step")]}]
    run(binary,tmp_path/"budget",p,"cycle limit")


@pytest.mark.parametrize("damage",["zero","negative","bool","empty","field","overflow","local_dependency","depth","legacy"])
def test_invalid_or_ambiguous_loops_rejected(binary,tmp_path,damage):
    p=program(count=1);p["schema"]="mlx_event_schedule_v2"
    loop={"repeat":2,"body":[event("a"),event("b")]};p["blocks"][0]["events"]=[loop]
    expected="loop"
    if damage in {"zero","negative","bool"}:loop["repeat"]={"zero":0,"negative":-1,"bool":True}[damage]
    elif damage=="empty":loop["body"]=[]
    elif damage=="field":loop["extra"]=1;expected="unknown"
    elif damage=="overflow":loop["repeat"]=2**61;loop["body"]=[{"repeat":32,"body":[event("a")]}];expected="overflow"
    elif damage=="local_dependency":loop["body"][1]["dependencies"]=["a"];expected="same-block"
    elif damage=="depth":
        child=loop
        for _ in range(17):child["body"]=[{"repeat":1,"body":[]}];child=child["body"][0]
        child["body"]=[event("deep")]
    elif damage=="legacy":p["schema"]="mlx_event_schedule_v1";expected="schema v2"
    run(binary,tmp_path/damage,p,expected)
