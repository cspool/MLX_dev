"""Compose complete concurrent array windows with native logical-source order.

This is not a source DAG, batch iterator, streaming pair, or timed template model.
"""
import copy
import json

from .model_matrix_events import matrix_events
from .model_vector_events import vector_events


def array_group_events(job):
    if set(job)!={"schema","windows"} or job["schema"]!="mlx_array_event_group_v1":raise ValueError("expected independent array event group")
    windows=job["windows"]
    if not isinstance(windows,list) or not 2<=len(windows)<=32:raise ValueError("array group requires 2..32 complete windows")
    programs=[]
    for window in windows:
        family=window["schema"]
        if family not in {"mlx_matrix_window_job_v1","mlx_vector_window_job_v1"}:raise ValueError("array group supports only independent matrix/vector windows")
        programs.append((matrix_events if family=="mlx_matrix_window_job_v1" else vector_events)(window))
    def leaves(items):
        for item in items:
            if "repeat" in item:yield from leaves(item["body"])
            else:yield item
    hardware=copy.deepcopy(programs[0]["hardware"]);latencies={};used=set()
    for p in programs:
        h=p["hardware"]
        if any(h.get(key)!=value for key,value in hardware.items() if key not in {"latencies","sfu_ii"}):raise ValueError("group windows disagree on shared array resources/ports")
        ops={e["op"] for b in p["blocks"] for e in leaves(b["events"])}|{"mul","add","convert","spm_read","spm_write","dma_read","dma_write"}
        if "sfu_ii" in h:ops|={"exp","div","sqrt"}
        for op in ops:
            value=h["latencies"][op]
            if op in used and latencies[op]!=value:raise ValueError("group windows disagree on shared execution latency")
            latencies[op]=value
        used|=ops
    for p in programs:
        for op,value in p["hardware"]["latencies"].items():latencies.setdefault(op,value)
    sfu_ii={p["hardware"]["sfu_ii"] for p in programs if "sfu_ii" in p["hardware"]}
    if len(sfu_ii)>1:raise ValueError("group windows disagree on shared SFU issue interval")
    hardware["sfu_ii"]=next(iter(sfu_ii),1)
    for op,value in dict(operand_prepare=0,spm_initialize=0,neg=1,sub=2,maximum=2,constant=1,move=1,broadcast=1,shuffle=1,cos=8,sin=8,
                         memory_literal=0,memory_convert=2,memory_complete=0,memory_index_read=8,memory_predicate_read=8,
                         control_alu=1,control_multiply=3,control_float=3,control_branch=1,control_literal=0,control_complete=0).items():latencies.setdefault(op,value)
    hardware["latencies"]=latencies
    hardware["memory_controller"]=dict(data_register_bytes=32,staging_bytes=128,conversion_result_latch_bytes=8,request_data_latch_bytes=8,max_active=1,request_period=1,response_period=1)
    hardware["control_controller"]=dict(register_bytes=512,rom_words=32,max_active=1,request_period=1,response_period=1)
    result=dict(schema="mlx_event_schedule_v6",source_tick_order=True,hardware=hardware,templates=[],blocks=[],controllers=[],policy="source_priority",
                max_cycles=sum(p["max_cycles"] for p in programs),trace_limit=min(1000000,sum(p["trace_limit"] for p in programs)))
    codes={}
    for source,p in enumerate(programs):
        prefix=f"s{source}:"
        def rename(items):
            for item in items:
                if "repeat" in item:rename(item["body"])
                else:item["id"]=prefix+item["id"];item["dependencies"]=[prefix+x for x in item["dependencies"]]
        code_names={}
        window=windows[source]
        if window["schema"]=="mlx_matrix_window_job_v1":key=("matrix",tuple(len(window["program"][part]) for part in ("prologue","body","epilogue")))
        else:key=("vector",json.dumps(window["node"]["vector_program"]["phases"],sort_keys=True))
        for template in p["templates"]:
            identity=(key,tuple(template["words"]));old_name=template["id"]
            if identity in codes:
                old=codes[identity]
                if any(old[k]!=template[k] for k in ("rf_vectors","spm_vectors")):raise ValueError("shared code with different per-context demands requires separate resource descriptors")
                code_names[old_name]=old["id"]
            else:
                template["id"]=prefix+old_name;codes[identity]=template;code_names[old_name]=template["id"];result["templates"].append(template)
        for block in p["blocks"]:
            block["id"]=prefix+block["id"];block["source_operator_id"]=source;block["template"]=code_names[block["template"]]
            block["admission_dependencies"]=[prefix+x for x in block["admission_dependencies"]];rename(block["events"]);result["blocks"].append(block)
    return result


def native_group_job(job):
    """Bind the same architecture to the independent native validation driver."""
    p=array_group_events(job);h=p["hardware"];l=h["latencies"]
    hardware={key:h[key] for key in ("rows","columns","contexts","compute_ii","sfu_ii","dma_request_period","dma_response_period")}
    hardware.update(spm_period=h["spm_port_period"],writeback_period=h["writeback_period"],multiply_latency=l["mul"],add_latency=l["add"],convert_latency=l["convert"],spm_latency=l["spm_read"],exp_latency=l["exp"],div_latency=l["div"],sqrt_latency=l["sqrt"])
    return dict(schema="mlx_native_array_group_v1",windows=copy.deepcopy(job["windows"]),hardware=hardware,max_cycles=p["max_cycles"])
