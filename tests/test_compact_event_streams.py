import copy

import pytest

from mlxsim.model_compact_event_streams import compact_streams,expand_streams
from mlxsim.model_lazy_event_patterns import externalize
from mlxsim.model_array_graph_events import array_graph_events
from test_event_source_graph import graph_job,marker_graph
from test_event_schedule import binary
from test_event_loops import run
from scripts.verify_mlx_event_schedule import audit_trace
from mlxsim.model_control_events import control_events
from test_control_events import base_request


def normalize(result):
    r=copy.deepcopy(result);r.pop("compact_blocks",None);r["lazy_patterns"].pop("block_descriptors_still_eager")
    r["block_intervals"].sort(key=lambda x:x["id"])
    r["trace"].sort(key=lambda x:(x["cycle"],x["event"],x["source_operator_id"],x["block"],x.get("operation",""),x.get("instance",0)))
    return r


def compare(binary,tmp_path,p,intervals=1000000):
    lazy=externalize(p,tmp_path/"patterns");compact=compact_streams(lazy,intervals);eager=expand_streams(compact)
    expected=run(binary,tmp_path/"expanded",eager);actual=run(binary,tmp_path/"compact",compact)
    assert normalize(actual)==normalize(expected)
    assert actual["compact_blocks"]["descriptors_drained"] and not actual["lazy_patterns"]["block_descriptors_still_eager"]
    assert actual["compact_blocks"]["peak_live_or_pending_descriptors"]<=2*p["hardware"]["source_window_limit"]+p["hardware"]["rows"]*p["hardware"]["columns"]*p["hardware"]["contexts"]+2
    audit_trace(eager,expected)
    return compact,actual


@pytest.mark.parametrize("precision",["f16","f32"])
@pytest.mark.parametrize("slow",[False,True])
@pytest.mark.parametrize("limit",[2,4])
def test_compact_real_dependency_graph_preserves_all_instances_and_timing(binary,tmp_path,precision,slow,limit):
    compare(binary,tmp_path,array_graph_events(graph_job(precision,slow,limit)))


def stress(tmp_path,count,cycles=10000000,history=0):
    p=marker_graph();p["blocks"]=p["blocks"][:1];p["source_graph"]=p["source_graph"][:1];p["trace_limit"]=0
    p["blocks"][0]["events"]=[dict(id="skip",op="predicate_skip",dependencies=[])]
    p["hardware"]["contexts"]=2;p["max_cycles"]=cycles
    lazy=externalize(p,tmp_path/"patterns");compact=compact_streams(lazy,history)
    compact["source_streams"][0]["windows"][0]["runs"][0]["count"]=count
    return compact


def test_large_block_run_has_constant_live_descriptors_and_complete_work(binary,tmp_path):
    p=stress(tmp_path,100000);r=run(binary,tmp_path/"large",p)
    assert r["blocks"]==r["events"]==r["counts"]["events_completed"]==100000
    assert r["cycles"]==100001
    assert r["compact_blocks"]["peak_live_or_pending_descriptors"]<=3
    assert r["compact_blocks"]["run_descriptors"]==1 and r["compact_blocks"]["descriptors_drained"]
    assert r["block_intervals"]==[] and r["compact_blocks"]["block_intervals_truncated"]


def test_billion_blocks_reach_budget_without_expanding_descriptors(binary,tmp_path):
    run(binary,tmp_path/"budget",stress(tmp_path,1000000000,cycles=100),"cycle limit")


def test_history_limit_does_not_limit_execution(binary,tmp_path):
    p=stress(tmp_path,30,history=3);r=run(binary,tmp_path/"bounded",p)
    assert r["events"]==30 and len(r["block_intervals"])==3
    assert r["compact_blocks"]["block_intervals_truncated"]


@pytest.mark.parametrize("damage",["zero","negative","bool","overflow","pe","pattern","parent","mixed","private"])
def test_compact_stream_invalid_contract_is_rejected(binary,tmp_path,damage):
    p=stress(tmp_path,3);s=p["source_streams"][0];item=s["windows"][0]["runs"][0];error="compact"
    if damage in {"zero","negative","bool","overflow"}:item["count"]={"zero":0,"negative":-1,"bool":True,"overflow":2**31}[damage]
    elif damage=="pe":item["pe_base"]=100
    elif damage=="pattern":item["event_pattern"]="missing"
    elif damage=="parent":s["parents"]=[0]
    elif damage=="mixed":p["source_graph"]=[]
    else:s["family"]="control";item.pop("template");item.pop("pe_base");item.pop("pe_stride")
    run(binary,tmp_path/"bad",p,error)


def test_compact_audit_does_not_accidentally_expand_a_billion_blocks(tmp_path):
    p=stress(tmp_path,1000000000)
    with pytest.raises(ValueError,match="audit expansion"):expand_streams(p)


def test_compact_private_control_and_view_sources_keep_their_domains(binary,tmp_path):
    p=control_events(base_request());p["source_tick_order"]=True
    p["controllers"].append(dict(id="view",domain="memory",source_operator_id=1,admission_dependencies=[],events=[dict(id="empty",op="memory_complete",dependencies=[])]))
    p["source_graph"]=[dict(source_operator_id=i,family=family,parents=[],windows=[[name]]) for i,family,name in [(0,"control","control"),(1,"memory","view")]]
    _,r=compare(binary,tmp_path,p)
    assert r["memory_controller_resources"]["peak_active"]==0 and r["control_controller_resources"]["peak_active"]==1


def test_compact_timed_templates_and_generation_reuse_match(binary,tmp_path):
    p=array_graph_events(graph_job());p["hardware"]["template_load_timing"]=True
    _,r=compare(binary,tmp_path,p)
    assert r["counts"]["template_program_pe_cycles"]>0


@pytest.mark.parametrize("stride",[0,3])
def test_compact_pe_wrap_and_pattern_tail_keep_logical_block_order(binary,tmp_path,stride):
    p=marker_graph();first=copy.deepcopy(p["blocks"][0]);p["hardware"]["columns"]=4;p["blocks"]=[]
    for i in range(20):
        b=copy.deepcopy(first);b.update(id=str(i),pe=(1+i*stride)%4)
        b["events"][0]["op"]="add" if i==15 else "mul";p["blocks"].append(b)
    p["source_graph"]=p["source_graph"][:1];p["source_graph"][0]["windows"]=[[b["id"] for b in p["blocks"]]]
    c,_=compare(binary,tmp_path,p)
    runs=c["source_streams"][0]["windows"][0]["runs"]
    assert [r["count"] for r in runs]==[15,1,4] and runs[0]["pe_stride"]==stride
