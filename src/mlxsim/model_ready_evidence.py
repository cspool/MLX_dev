"""Accounting/dependency checks for completed native shared-array graphs.

This validates actual source work, not CPU/cache timing or all-model scope.
"""
from collections import Counter
import math

from .model_block_pipeline import compile_block_pipelines,references
from .model_system_evidence import require,count,check_kernel,WIDTH

CLASSIFICATION="ready_graph_bounded_pair_events_not_general_cdc_or_system_acceptance"


def verify_ready_execution(program,result,options):
    require(result["classification"]==CLASSIFICATION and options.get("tile_pipeline") is True,"wrong native paired execution mode")
    require(all(program.get(f+"_backend")=="scheduled" for f in ("matrix","vector","memory","control")),"native graph permits a functional backend")
    require(result["virtual_tensor_backing_used"] is True and all(result[k]==0 for k in ("functional_entry_calls","blas_calls","python_or_gpu_execution_fallbacks")),"native graph used a forbidden numerical fallback")
    require(result["control_execution"]=="scheduled_rv64_leaf_not_actual_cpu" and result["weight_loading"]=="preloaded_not_cpu_or_dma_loader","native execution scope was relabelled")
    nodes=program["nodes"];ids=[n["source_operator_id"] for n in nodes]
    require(nodes and len(set(ids))==len(ids) and len({n["id"] for n in nodes})==len(nodes),"duplicate or empty source identities")
    events=result["events"];require(len(events)==result["executed_source_calls"]==len(nodes) and {e["source_operator_id"] for e in events}==set(ids),"not every source completed exactly once")
    by_id={e["source_operator_id"]:e for e in events};by_value={n["id"]:by_id[n["source_operator_id"]] for n in nodes}
    plan=program["block_pipeline_plan"]
    require(compile_block_pipelines(program,event_slots=plan["event_slots"])["block_pipeline_plan"]==plan,"pipeline plan no longer matches the complete graph")
    parents={p["consumer_source"]:p["producer_source"] for p in plan["pairs"]}
    values={**program["assets"],**{n["id"]:n["output"] for n in nodes}}
    profile={f+"_options":{**({"rows":4,"columns":4,"contexts":2} if f!="memory" else {}),**program[f+"_schedule_options"]} for f in ("matrix","vector","memory")}
    groups=result["windows"];require(set(groups)=={"matrix","vector","memory","control"},"missing or unknown native backend groups")
    indexed={};totals=Counter();admitted=0
    for family,windows in groups.items():
        for w in windows:
            key=(w["source_operator_id"],w["batch_index"])
            require(key not in indexed,"duplicate source batch execution")
            indexed[key]=(family,w)
    expected_keys=set()
    for node in nodes:
        sid=node["source_operator_id"];e=by_id[sid]
        families=[f for f in groups if f+"_program" in node];require(len(families)==1,"ambiguous source lowering")
        family=families[0];batches=math.prod(node["output"]["shape"][:-2]) if node["kind"]=="matmul" else 1
        require(batches>0 and e["batches"]==batches and e["family"]==family,"source batch/family coverage differs")
        for key in ("kind","forward_id","layer_idx"):require(e[key]==node[key],"source event identity differs")
        start=count(e["start_cycle"],"source start");end=count(e["publish_cycle"],"source publish")
        require(start<end<=result["graph_cycles"],"source interval is invalid")
        for dep in set(references(node["args"]))|set(references(node.get("kwargs",{}))):
            if dep not in by_value:continue
            parent=by_value[dep]
            if parents.get(sid)==parent["source_operator_id"]:
                require(start==parent["start_cycle"] and end>=parent["publish_cycle"],"paired source admission/completion order differs")
            else:require(start>=parent["publish_cycle"],"source executed before a whole-source dependency was visible")
        cursor=start
        for batch in range(batches):
            key=(sid,batch);expected_keys.add(key);require(key in indexed,"missing complete-source batch")
            actual,w=indexed[key];require(actual==family and w["forward_id"]==node["forward_id"] and w["layer_idx"]==node["layer_idx"],"batch route identity differs")
            require(w["shared_start_cycle"]==cursor and w["shared_end_cycle"]-cursor==max(1,count(w["cycles"],"window cycles")),"batch windows overlap or lose external edges")
            cursor=w["shared_end_cycle"]
            require(w["done"] is True and w["external_memory_port"] is True and w["dma_requests"]==w["dma_responses"],"backend did not drain through actual physical responses")
            elements=math.prod(node["output"]["shape"]);size=elements*WIDTH[node["output"]["dtype"]]
            view=family=="memory" and node["memory_program"]["mode"]=="view"
            if family=="control":
                require(w["classification"]=="response_driven_controller_leaf_component_not_riscv_cpu" and w["register_bytes"]==512
                        and w["max_inflight_instructions"]<=1 and w["max_inflight_transactions"]<=1 and w["write_bytes"]==size
                        and count(w["instructions"],"control instructions")>0,"control leaf work/capacity differs")
                work={"requests":w["dma_requests"],"read_bytes":count(w["read_bytes"],"control reads"),"write_bytes":size,"matrix_mac_lanes":0}
            elif view:
                n=w["numeric_instructions"]
                require(w["view_elided"] and not w["inflight_transactions"] and n["calls"]==n["view_elisions"]==1
                        and n["allocations"]==n["read_bytes"]==n["write_bytes"]==w["dma_requests"]==0,"view was materialized or reported numerical work")
                work={"requests":0,"read_bytes":0,"write_bytes":0,"matrix_mac_lanes":0}
            else:work=check_kernel(node,{"family":family,"batch_count":batches},w,profile,values)
            if family in ("matrix","vector"):
                require(w["external_shared_array"] is True,"array frontend used private physical resources")
                admitted+=w["admitted"]
            totals.update(work)
        require(cursor==end,"source published before all its batches completed")
    require(set(indexed)==expected_keys,"extra source/batch windows")
    observed=result["pipeline_groups"];require(len(observed)==len(plan["pairs"]),"not all selected pipeline groups executed")
    expected={(p["producer_source"],p["consumer_source"]):p for p in plan["pairs"]};seen=set();previous_end=0
    for epoch,p in enumerate(observed,1):
        key=(p["producer_source"],p["consumer_source"]);require(key in expected and key not in seen,"missing/duplicate pipeline identities");seen.add(key)
        spec=expected[key];pe,ce=by_id[key[0]],by_id[key[1]]
        require(p["epoch"]==epoch and p["event_slots"]==plan["event_slots"] and p["peak_event_slots"]<=p["event_slots"],"pipeline event identity/capacity differs")
        require(p["finished"] and all(p[k]==spec["producer_blocks"] for k in ("blocks","admitted","completed","frontier"))
                and not any(p[k] for k in ("active","pending_visibility","out_of_order_done")),"pipeline block events did not drain")
        require(p["begin_cycle"]==pe["start_cycle"]==ce["start_cycle"] and p["end_cycle"]==max(pe["publish_cycle"],ce["publish_cycle"])
                and p["begin_cycle"]>=previous_end,"closed pipeline reservations overlap or misreport their lifetime")
        previous_end=p["end_cycle"]
    require(result["pipeline_whole_source_barrier"]==options.get("pipeline_whole_source_barrier",False),"pipeline comparison mode changed")
    a=result["array"];m=profile["matrix_options"]
    require(a["idle"] and not a["poisoned"] and a["admitted"]==a["retired"]==admitted and a["rf_vectors_per_pe"]==16
            and a["rom_words_per_pe"]==32 and a["spm_vectors_total"]==128 and a["context_slots_per_pe"]==m.get("contexts",2)
            and a["peak_contexts"]<=m.get("rows",4)*m.get("columns",4)*m.get("contexts",2)
            and a["peak_rf_vectors_per_pe"]<=16 and a["peak_rom_words_per_pe"]<=32 and a["peak_spm_vectors"]<=128,"shared array resource conservation/capacity differs")
    require(a["template_configuration_cycles_modeled"]==options.get("template_load_timing",False) and not a["pending_template_words"]
            and a["template_words_requested"]==a["template_words_loaded"] and a["template_word_period"]==options.get("template_word_period",1)
            and (not admitted or not options.get("template_load_timing",False) or a["template_words_loaded"]>0),"local template programming is incomplete")
    outputs=result["outputs"];require([r["forward_id"] for r in outputs]==[o["forward_id"] for o in program["outputs"]],"final forward outputs are missing or reordered")
    readback=readback_bytes=0
    for spec,actual in zip(program["outputs"],outputs,strict=True):
        shape=values[spec["logits"]]["shape"];dtype=values[spec["logits"]]["dtype"]
        require(actual["shape"]==shape and actual["dtype"]==dtype and len(actual["tokens"])==math.prod(values[spec["token"]]["shape"]),"final output shape/type/readback differs")
        for role in ("logits","token"):
            value=values[spec[role]];n=math.prod(value["shape"]);readback+=n;readback_bytes+=n*WIDTH[value["dtype"]]
    memory=result["memory"];mux=result["physical_mux"];arena=result["arena_drained"]
    require(result["host_readback_requests"]==readback and memory["idle"] and memory["submitted"]==memory["responses"]==memory["consumed"]==memory["reads"]+memory["writes"]==totals["requests"]+readback,"physical transactions/readback did not conserve")
    require(memory["read_bytes"]==totals["read_bytes"]+readback_bytes and memory["write_bytes"]==totals["write_bytes"]
            and memory["accepted"]==memory["submitted"]+memory["nacks"],"physical byte/retry accounting differs")
    require(mux["idle"] and not mux["poisoned"] and mux["active_channels"]==0 and mux["owner_channel"] is None
            and all(not c["open"] and c["submitted"]==c["arrived"]==c["consumed"] for c in mux["channels"])
            and sum(c["submitted"] for c in mux["channels"])==memory["submitted"],"shared response ownership did not drain")
    require(arena["reserved_bytes"]==0 and not arena["allocations"] and arena["total_allocations"]==arena["total_frees"] and memory["live_payload_bindings"]==0,"native graph leaked live storage")
    require(result["preloaded_assets"]==len(program["assets"]) and result["preloaded_bytes"]==sum(math.prod(v["shape"])*WIDTH[v["dtype"]] for v in program["assets"].values()),"native graph did not initialize every input asset")
    require(result["shared_elapsed_cycles"]==result["graph_cycles"]+result["host_readback_cycles"] and result["overlap"]==options.get("overlap",True)
            and result["max_active_nodes"]==options.get("max_active_nodes",32) and result["peak_active_nodes"]<=result["max_active_nodes"],"native clock/active-source accounting differs")
    require(not result["mlx_system_verified"] and not result["inference_performance_eligible"],"native evidence was relabelled as system/performance acceptance")
    return {"source_calls":len(nodes),"backend_windows":{k:len(v) for k,v in groups.items()},"pipeline_groups":len(observed),"numerical_work":dict(totals),
            "physical_requests":memory["submitted"],"all_buffers_released":True,"all_events_drained":True,"actual_cpu_execution":False,"inference_performance_eligible":False}
