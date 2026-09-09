"""Full compiled-program source/batch/data contracts, not executed event timing."""
from collections import Counter
import hashlib
import json
import math

from .model_block_pipeline import references
from .model_event_resources import resource_contract
from .model_value_outputs import value_outputs
from .model_result_contract import result_roles


def graph_plan(program):
    resources=resource_contract(program);metadata={name:{k:a[k] for k in ("shape","dtype")} for name,a in program["assets"].items()}
    producer={};ids=set();sources=[];families=Counter();windows=Counter();witnesses=[]
    for index,node in enumerate(program["nodes"]):
        identity=node["source_operator_id"]
        if type(identity) is not int or identity<0 or identity in ids:raise ValueError("invalid full graph source identity")
        ids.add(identity);inputs=sorted(set(name for key in ("args","kwargs","control_dependencies") for name in references(node.get(key))))
        if any(name not in metadata for name in inputs):raise ValueError("full event graph has missing or forward data/control input")
        family=resources["sources"][index]["family"];families[family]+=1
        row=dict(source_operator_id=identity,origin_source_operator_id=node.get("origin_source_operator_id",identity),value=node["id"],kind=node["kind"],family=family,
                 forward_id=node["forward_id"],layer_idx=node.get("layer_idx"),inputs=inputs,parents=sorted({producer[name] for name in inputs if name in producer}),
                 control_inputs=sorted(set(references(node.get("control_dependencies")))),window_count=1,
                 backend_program_sha256=hashlib.sha256(json.dumps(node[family+"_program"],sort_keys=True).encode()).hexdigest())
        args=node["args"]
        if family=="matrix":
            a,b=(metadata[args[i]["value"]]["shape"] for i in (0,1));linear=node["kind"]=="linear"
            if len(a)<2 or len(b)<2 or node["kind"] not in {"linear","matmul"}:raise ValueError("full graph matrix rank/kind invalid")
            m=math.prod(a[:-1]) if linear else a[-2];n=b[0] if linear else b[-1];k=a[-1]
            if k!=(b[1] if linear else b[-2]):raise ValueError("full graph matrix contraction differs")
            ab,bb=a[:-2],b[:-2];batch=[]
            if linear:
                if len(b)!=2:raise ValueError("linear weight rank differs")
                expected=a[:-1]+[n]
            else:
                rank=max(len(ab),len(bb));left=[1]*(rank-len(ab))+ab;right=[1]*(rank-len(bb))+bb
                if any(x!=y and x!=1 and y!=1 for x,y in zip(left,right)):raise ValueError("full graph matrix batch broadcast differs")
                batch=[y if x==1 else x for x,y in zip(left,right)];expected=batch+[m,n]
            count=math.prod(batch)
            if not count or expected!=node["output"]["shape"]:raise ValueError("full graph matrix output/batch scope differs")
            row.update(window_count=count,matrix=dict(m=m,n=n,k=k,transposed_b=linear,a_batch_shape=[] if linear else ab,b_batch_shape=[] if linear else bb,output_batch_shape=batch,
                input_batch_selection="right_aligned_broadcast_flat_index",output_batch_selection="contiguous_batch_index"))
        if family=="control" and node["kind"] in {"guard","argmax"}:
            count=1 if node["kind"]=="guard" else math.prod(node["output"]["shape"])*(metadata[args[0]["value"]]["shape"][-1]-1)
            if count:
                witness=dict(source_operator_id=identity,kind="actual_guard_boolean" if node["kind"]=="guard" else "actual_argmax_branch_taken",count=count)
                witnesses.append(witness);row["required_timing_witness"]=witness
        if family=="memory" and node["kind"]=="where":
            def signature(arg):return metadata[arg["value"]]["dtype"] if isinstance(arg,dict) and "value" in arg else "literal"
            if signature(args[1])!=signature(args[2]):
                witness=dict(source_operator_id=identity,kind="actual_where_predicate",count=math.prod(node["output"]["shape"]));witnesses.append(witness);row["required_timing_witness"]=witness
        outputs=value_outputs(node);row["produced_values"]=[name for name,_ in outputs]
        for name,meta in outputs:
            if name in metadata:raise ValueError("full graph redefines a value")
            metadata[name]=meta;producer[name]=identity
        windows[family]+=row["window_count"];sources.append(row)
    results=[]
    for output in program["outputs"]:
        for role in result_roles(program):
            name=output[role]
            if name not in metadata:raise ValueError("full graph result value unbound")
            results.append(dict(forward_id=output["forward_id"],role=role,value=name,producer=producer.get(name),layout=metadata[name]))
    return dict(classification="complete_source_batch_dependency_plan_not_executed_event_graph",program_sha256=resources["canonical_program_sha256"],
        source_calls=resources["source_calls"],lowered_calls=len(sources),family_sources=dict(families),family_windows=dict(windows),total_windows=sum(windows.values()),
        dependency_edges=sum(len(s["parents"]) for s in sources),sources=sources,results=results,timing_witness_requirements=witnesses,
        graph_contract=dict(source_frontend_slots_hold_while_waiting_for_pe=True,new_source_launch_requires_shared_memory_quiescence=True,
            all_batches_complete_before_source_publication=True,source_group_completion="all_lowered_stages",streaming_pairs=program.get("block_pipeline_plan")),
        full_event_lowering_complete=False,event_simulated_cycles=None,performance_error_available=False,
        remaining_execution_inputs=["lazy_window_IR","bound_dynamic_witnesses","streaming_block_events","physical_memory_and_template_timing"])
