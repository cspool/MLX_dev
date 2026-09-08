"""Structural/accounting gates for actual shared-native physical execution."""
import math
from .model_value_outputs import value_outputs, require_value_contract
from .model_source_groups import verify_source_groups

CLASSIFICATION = "shared_native_physical_model_execution_not_chipyard_system_validation"
EXECUTION_CLASSIFICATION = "full_model_native_physical_not_chipyard_validation"
BYTES = {"f16":2,"f32":4,"i64":8,"bool":1}


def require(value, message):
    if not value: raise RuntimeError(message)


def scheduled_compile_options(program):
    return {"matrix_backend":program["matrix_backend"],"schedule_options":program.get("matrix_schedule_options"),
            "vector_backend":program["vector_backend"],"vector_schedule_options":program.get("vector_schedule_options"),
            "memory_backend":program.get("memory_backend","functional"),"memory_schedule_options":program.get("memory_schedule_options"),
            "control_backend":program.get("control_backend","functional"),"control_schedule_options":program.get("control_schedule_options")}


def verify_physical_execution(program, native, system_options=None):
    require_value_contract(program)
    verify_source_groups(program,native)
    require(native["classification"]==CLASSIFICATION,"unexpected physical execution evidence class")
    require(all(program.get(name+"_backend")=="scheduled" for name in ("matrix","vector","memory","control")),"physical program permits a functional backend")
    require(native["virtual_tensor_backing_used"] is True and native["functional_entry_calls"]==native["blas_calls"]==native["python_or_gpu_execution_fallbacks"]==0,"physical execution used a prohibited data/compute fallback")
    nodes=program["nodes"];events=native["events"];groups=native["windows"]
    require(len({node["source_operator_id"] for node in nodes})==len(nodes),"physical source IDs are duplicated")
    require(native.get("executed_lowered_calls",native["executed_source_calls"])==len(nodes)==len(events),"incomplete physical model execution")
    require(set(groups)=={"matrix","vector","memory","control"},"physical backend groups missing or unknown")
    observed={name:[] for name in groups};cursor=0
    diagnostics=native.get("observations",[]);by_id={row["source_operator_id"]:row for row in diagnostics}
    require(len(by_id)==len(diagnostics),"physical observations duplicated")
    if system_options is not None:
        requested=system_options.get("observe_operators",[])
        require(len(set(requested))==len(requested) and set(requested)==set(by_id),"physical observation selection mismatch")
    diagnostic_cycles=diagnostic_requests=diagnostic_bytes=0
    for node,event in zip(nodes,events,strict=True):
        kinds=[name for name in groups if name+"_program" in node];require(len(kinds)==1,"physical source has ambiguous or missing route")
        for field in ("source_operator_id","kind","forward_id","layer_idx"):
            require(event[field]==node[field],"physical event differs from source operator")
        require(event["entry"]=="tensor_model::"+node["kind"],"physical entry routing mismatch")
        if node["kind"] == "split":
            require(event.get("produced_values") == [name for name,_ in value_outputs(node)],"physical split did not publish every output")
        require(event["shared_start_cycle"]==cursor and event["shared_end_cycle"]>=cursor,"physical graph clock is discontinuous")
        cursor=event["shared_end_cycle"];name=kinds[0]
        batches=math.prod(node["output"]["shape"][:-2]) if node["kind"]=="matmul" else 1
        require(batches>0,"unregistered empty physical matrix batch")
        observed[name].extend((node,event,batch) for batch in range(batches))
        if node["source_operator_id"] in by_id:
            row=by_id[node["source_operator_id"]]
            require(row["producer_completed"] is True and row["forward_id"]==node["forward_id"] and row["layer_idx"]==node["layer_idx"] and row.get("phase")==node.get("phase"),"physical observation producer mismatch")
            count=math.prod(node["output"]["shape"]);size=count*BYTES[node["output"]["dtype"]]
            require(row["shape"]==node["output"]["shape"] and row["dtype"]==node["output"]["dtype"] and row["bytes"]==size and size<=1048576,"physical observation shape/byte limit mismatch")
            require(len(row["sha256"])==64 and all(c in "0123456789abcdef" for c in row["sha256"]),"invalid physical output digest")
            require(row["shared_start_cycle"]==cursor and row["shared_end_cycle"]>=cursor,"physical observation clock mismatch")
            diagnostic_cycles+=row["shared_end_cycle"]-cursor;cursor=row["shared_end_cycle"];diagnostic_requests+=count;diagnostic_bytes+=size
    require(set(by_id).issubset({node["source_operator_id"] for node in nodes}) and diagnostic_bytes<=33554432,"physical observation outside source/byte budget")
    require(diagnostic_cycles==native.get("diagnostic_readback_cycles",0) and diagnostic_requests==native.get("diagnostic_readback_requests",0) and diagnostic_bytes==native.get("diagnostic_readback_bytes",0),"physical diagnostic accounting mismatch")
    request_count=read_bytes=write_bytes=window_cycles=0
    for name,expected in observed.items():
        require(len(groups[name])==len(expected),"physical window coverage mismatch")
        for window,(node,event,batch) in zip(groups[name],expected,strict=True):
            require(window["source_operator_id"]==node["source_operator_id"] and window["batch_index"]==batch,"physical window identity/batch mismatch")
            require(window["forward_id"]==node["forward_id"] and window["layer_idx"]==node["layer_idx"],"physical window layer/forward mismatch")
            require(window["done"] is True and window["external_memory_port"] is True and window["dma_requests"]==window["dma_responses"],"physical backend window did not drain")
            if node["kind"] == "split":
                require(window["view_elided"] and window["numeric_instructions"].get("split_view_outputs") == len(value_outputs(node)),"physical split view count differs")
            start,end=window["shared_start_cycle"],window["shared_end_cycle"]
            require(event["shared_start_cycle"]<=start<=end<=event["shared_end_cycle"] and end-start==window["cycles"],"physical window clock mismatch")
            window_cycles+=window["cycles"];request_count+=window["dma_requests"]
            if name=="vector":
                read_bytes+=window["numeric_instructions"]["global_read_bytes"]
                write_bytes+=window["numeric_instructions"]["global_write_bytes"]
            else:
                read_bytes+=window["read_bytes" if name=="control" else "dma_read_bytes"]
                write_bytes+=window["write_bytes" if name=="control" else "dma_write_bytes"]
            if name=="matrix":
                require(window["active_contexts"]==window["allocated_spm_vectors"]==window["pending_compute"]==0 and not window["pending_dma"] and not window["pending_spm"],"matrix resources remained live")
                require(window["spm_capacity_vectors"]==128 and window["peak_spm_vectors"]<=128 and window["rf_frame_vectors"]==6 and window["rom_words"]<=32,"physical matrix resource contract changed")
            if name=="memory":
                stats=window["numeric_instructions"]
                require(stats["staging_bytes"]==128 and stats["register_bytes_total"]==32 and window["inflight_transactions"]==0,"physical transfer capacity/lifetime mismatch")
        # Each parent interval is tiled exactly by its windows, including
        # zero-cycle checked views; no gaps or overlapping batch replays.
        position=0
        while position<len(expected):
            source=expected[position][0]["source_operator_id"];parent=expected[position][1];at=parent["shared_start_cycle"]
            while position<len(expected) and expected[position][0]["source_operator_id"]==source:
                w=groups[name][position];require(w["shared_start_cycle"]==at,"physical batch windows overlap or leave gaps");at=w["shared_end_cycle"];position+=1
            require(at==parent["shared_end_cycle"],"physical parent completion lacks a window")
    require(cursor==window_cycles+diagnostic_cycles and window_cycles==native["device_component_cycles"],"physical component cycle accounting mismatch")
    require(native["shared_elapsed_cycles"]==cursor+native["host_readback_cycles"],"physical readback clock accounting mismatch")
    shapes={name:asset["shape"] for name,asset in program["assets"].items()}
    shapes.update({identifier:spec["shape"] for n in nodes for identifier,spec in value_outputs(n)})
    macs=sum(math.prod(n["output"]["shape"])*shapes[n["args"][0]["value"]][-1] for n in nodes if "matrix_program" in n)
    require((native["matrix_microcode"] or {}).get("mul_active_lanes",0)==macs,"physical matrix did not execute every model MAC")
    require((native["vector_microcode"] or {}).get("calls",0)==len(observed["vector"]),"physical vector execution coverage mismatch")
    require((native["memory_programs"] or {}).get("calls",0)==len(observed["memory"]),"physical memory execution coverage mismatch")
    require((native["control_programs"] or {}).get("calls",0)==len(observed["control"]),"physical control execution coverage mismatch")
    for name,summary,fields in (("matrix","matrix_microcode",("global_read_bytes","global_write_bytes")),
        ("vector","vector_microcode",("global_read_bytes","global_write_bytes")),
        ("memory","memory_programs",("read_bytes","write_bytes")),("control","control_programs",("read_bytes","write_bytes"))):
        for field in (*fields,"instructions"):
            require(sum((w if name=="control" else w["numeric_instructions"])[field] for w in groups[name])==(native[summary] or {}).get(field,0),"physical numeric byte/instruction accounting mismatch")
    expected_outputs=program["outputs"];outputs=native["outputs"]
    require([o["forward_id"] for o in outputs]==[o["forward_id"] for o in expected_outputs],"physical output steps missing/reordered")
    readback_count=readback_bytes=0
    dtypes={name:asset["dtype"] for name,asset in program["assets"].items()};dtypes.update({n["id"]:n["output"]["dtype"] for n in nodes})
    for spec,actual in zip(expected_outputs,outputs,strict=True):
        require(actual["shape"]==shapes[spec["logits"]] and actual["dtype"]==dtypes[spec["logits"]],"physical logits shape/dtype mismatch")
        require(len(actual["tokens"])==math.prod(shapes[spec["token"]]),"physical token readback size mismatch")
        for key in ("logits","token"):
            count=math.prod(shapes[spec[key]]);readback_count+=count;readback_bytes+=count*BYTES[dtypes[spec[key]]]
    memory=native["memory"]
    require(memory["idle"] is True and memory["submitted"]==memory["responses"]==memory["consumed"]==memory["reads"]+memory["writes"],"physical transactions did not drain")
    require(native["host_readback_requests"]==readback_count and memory["submitted"]==request_count+readback_count+diagnostic_requests,"physical request/readback accounting mismatch")
    require(memory["read_bytes"]==read_bytes+readback_bytes+diagnostic_bytes and memory["write_bytes"]==write_bytes and memory["accepted"]==memory["submitted"]+memory["nacks"],"physical byte/retry accounting mismatch")
    arena=native["arena_drained"]
    require(arena["reserved_bytes"]==0 and not arena["allocations"] and arena["total_allocations"]==arena["total_frees"] and memory["live_payload_bindings"]==0,"physical model retained allocations after readback")
    initial_bytes=sum(math.prod(a["shape"])*BYTES[a["dtype"]] for a in program["assets"].values())
    require(native["preloaded_assets"]==len(program["assets"]) and native["preloaded_bytes"]==initial_bytes,"physical asset preload coverage mismatch")
    require(native["weight_loading"]=="preloaded_not_cpu_or_dma_loader" and native["mlx_system_verified"] is False and native["inference_performance_eligible"] is False,"physical component improperly claims system/performance validation")
    coverage={"source_calls":native["executed_source_calls"],"matrix_mac_lanes":macs,"backend_windows":{name:len(w) for name,w in groups.items()},
              "physical_requests":memory["submitted"],"all_requests_drained":True,"all_buffers_released":True}
    if "source_groups" in program:coverage["lowered_calls"]=len(nodes)
    return coverage
