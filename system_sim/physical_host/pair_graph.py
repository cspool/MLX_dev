"""Compile closed pairs into dependency-checked CPU tasks with new lifetimes.

Only compilation and address planning happen here. All numerical execution,
block admission, context issue and response timing remain in C/C++.
"""
from mlxsim.model_block_pipeline import compile_block_pipelines,references

from .address_plan import iter_bindings
from .lowering import BYTES,lower_control
from ..physical_device.lowering import geometry,lower_matrix_window
from ..physical_device.vector_lowering import lower_vector
from ..physical_device.memory_lowering import lower_memory
from ..physical_device.pair_lowering import lower_pair


def schedule(program,event_slots):
    nodes=program["nodes"];ordinal={n["id"]:i for i,n in enumerate(nodes)}
    ids=[n["source_operator_id"] for n in nodes]
    if not nodes or len(ordinal)!=len(nodes) or len(set(ids))!=len(ids) or any(type(i) is not int or not 0<=i<2**64-1 for i in ids):
        raise ValueError("pair graph has invalid or duplicate source identities")
    dependencies=[]
    for i,n in enumerate(nodes):
        names=set(references(n["args"]))|set(references(n.get("kwargs",{})))
        if any(name not in program["assets"] and (name not in ordinal or ordinal[name]>=i) for name in names):
            raise ValueError("pair graph input order is not a complete SSA DAG")
        dependencies.append(sorted(ordinal[name] for name in names if name in ordinal))
    selected=compile_block_pipelines(program,event_slots=event_slots)["block_pipeline_plan"]
    producers={p["producer"] for p in selected["pairs"]};consumers={p["consumer"]:p["producer"] for p in selected["pairs"]}
    # A closed producer has no intervening users. Delay it until its consumer,
    # retaining its inputs through the group; do not execute other inputs early.
    groups=[[consumers[i],i] if i in consumers else [i] for i in range(len(nodes)) if i not in producers]
    complete=set()
    for group in groups:
        for i in group:
            if not set(dependencies[i])<=complete:raise ValueError("pair grouping violates a source dependency")
            complete.add(i)
    if len(complete)!=len(nodes):raise ValueError("pair grouping lost source calls")
    return groups,dependencies,selected


def storage(program,layouts,groups,*,base,limit):
    """Closed live intervals: every group input and both outputs coexist."""
    nodes=program["nodes"];at={i:step for step,g in enumerate(groups) for i in g}
    owners={n["id"]:i for i,n in enumerate(nodes)};roots={l["root"] for l in layouts.values()}
    intervals={root:{"first":-1 if root in program["assets"] else at[owners[root]],
                     "last":len(groups) if root in program["assets"] else at[owners[root]]} for root in roots}
    for i,node in enumerate(nodes):
        names={node["id"]}|set(references(node["args"]))|set(references(node.get("kwargs",{})))
        for name in names:
            record=intervals[layouts[name]["root"]]
            if at[i]<record["first"]:raise ValueError("pair storage used before its producer group")
            record["last"]=max(record["last"],at[i])
    for output in program["outputs"]:
        for role in ("logits","token"):intervals[layouts[output[role]]["root"]]["last"]=len(groups)
    active=[];holes=[];end=base;bindings={};records=[]
    for root in sorted(roots,key=lambda name:(intervals[name]["first"],name)):
        interval=intervals[root];live=[]
        for last,address,size in active:
            if last<interval["first"]:holes.append((address,size))
            else:live.append((last,address,size))
        active=live;merged=[]
        for address,size in sorted(holes):
            if merged and merged[-1][0]+merged[-1][1]==address:merged[-1]=(merged[-1][0],merged[-1][1]+size)
            else:merged.append((address,size))
        holes=merged;layout=layouts[root];size=layout["storage_elements"]*BYTES[layout["dtype"]];reserved=max(64,(size+63)//64*64)
        chosen=next((i for i,(_,length) in enumerate(holes) if length>=reserved),None)
        if chosen is None:address=end;end+=reserved
        else:
            address,length=holes.pop(chosen)
            if length>reserved:holes.append((address+reserved,length-reserved))
        if end>limit:raise ValueError(f"paired graph storage exceeds mapped limit: {end} > {limit}")
        bindings[root]={"base":address,"bytes":size,"writable":root not in program["assets"]}
        records.append({"root":root,**interval,**bindings[root],"reserved_bytes":reserved})
        active.append((interval["last"],address,reserved))
    return bindings,records,end


def compile_pair_graph(program,life,*,device_base,device_bytes,data_offset,scratch_offset,scratch_bytes,event_slots):
    # Validate the original graph/lifetime identities before making a different
    # physical plan. Never reuse a serial free-list across overlapping sources.
    layouts=None
    for _,layouts,_ in iter_bindings(program,life):pass
    if layouts is None:raise ValueError("empty pair graph")
    groups,dependencies,selected=schedule(program,event_slots)
    bindings,intervals,end=storage(program,layouts,groups,base=device_base+data_offset,limit=device_base+device_bytes)
    nodes=program["nodes"];blob=bytearray();tasks=[];sources=[];counts={name:0 for name in ("matrix","vector","memory","control","view")}
    for i,node in enumerate(nodes):
        routes=[name for name in ("matrix","vector","memory","control") if name+"_program" in node]
        if len(routes)!=1:raise ValueError("pair graph source has missing/ambiguous route")
        family=routes[0];view=family=="memory" and node["memory_program"]["mode"]=="view"
        counts[family]+=1;counts["view"]+=int(view)
        sources.append({"source_ordinal":i,"source_operator_id":node["source_operator_id"],"kind":node["kind"],"family":family,"view_elided":view,
                        "task_count":0,"forward_id":node["forward_id"],"layer_idx":node["layer_idx"]})
    for group in groups:
        i=group[0];node=nodes[i];source=sources[i];family=source["family"];extra={}
        if len(group)==2:
            j=group[1];command,_=lower_pair(node,nodes[j],layouts,bindings,event_slots=event_slots);commands=[command];family="pair"
            extra={"consumer_ordinal":j,"consumer_source_id":nodes[j]["source_operator_id"]}
            sources[j]["task_count"]=1
        elif source["view_elided"]:lower_memory(node,layouts,bindings);commands=[None]
        elif family=="matrix":commands=[lower_matrix_window(node,layouts,bindings,b)[0] for b in range(geometry(node,layouts)["batches"])]
        elif family=="vector":commands=[lower_vector(node,layouts,bindings)[0]]
        elif family=="memory":commands=[lower_memory(node,layouts,bindings)[0]]
        else:commands=[lower_control(node,layouts,bindings)[0]]
        source["task_count"]=len(commands)
        for batch,command in enumerate(commands):
            offset=len(blob) if command is not None else 0
            if command is not None:blob.extend(command)
            tasks.append({"kind":3 if family=="pair" else 0 if source["view_elided"] else 1 if family=="control" else 2,
                          "source_ordinal":i,"source_id":node["source_operator_id"],"batch_index":batch,"batch_count":len(commands),
                          "command_offset":offset,"bytes":len(command) if command else 0,"family":family,**extra})
    outputs=[{"forward_id":o["forward_id"],"role":role,"value":o[role],"layout":layouts[o[role]],"binding":bindings[layouts[o[role]]["root"]]}
             for o in program["outputs"] for role in ("logits","token")]
    return bytes(blob),{"classification":"compiled_rv64_graph_dispatch_plan_not_execution","host_abi_version":2,"pair_event_slots":event_slots,
        "pair_count":len(selected["pairs"]),"execution_groups":groups,"source_dependencies":dependencies,"storage_intervals":intervals,
        "device_base":device_base,"device_bytes":device_bytes,"required_mapped_bytes":end-device_base,"data_offset":data_offset,"scratch_offset":scratch_offset,"scratch_bytes":scratch_bytes,
        "sources":sources,"tasks":tasks,"assets":{name:bindings[name] for name in program["assets"]},"outputs":outputs,"family_source_calls":counts,
        "command_bytes":len(blob),"source_calls":len(sources),"task_count":len(tasks),"model_data_executed":False,"mlx_system_verified":False,"inference_performance_eligible":False}
