"""Explicit original-source groups over executable lowered-stage CPU slots."""
import copy

from mlxsim.model_block_pipeline import references
from mlxsim.model_value_outputs import require_value_contract,value_outputs
from mlxsim.model_result_contract import result_roles


def dependencies(program):
    require_value_contract(program)
    owners={};result=[];guards=[]
    assets=set(program["assets"])
    for i,node in enumerate(program["nodes"]):
        if node.get("control_dependencies",[])!=[{"value":name} for name in guards]:
            raise ValueError("missing or foreign CPU graph guard dependency")
        names=set().union(*(set(references(node.get(field,[]))) for field in ("args","kwargs","control_dependencies")))
        if any(name not in assets and name not in owners for name in names):raise ValueError("CPU graph has an unbound or forward SSA edge")
        result.append(sorted({owners[name] for name in names if name not in assets}))
        for name,_ in value_outputs(node):
            if name in assets or name in owners:raise ValueError("CPU graph redefines a value")
            owners[name]=i
        if node["kind"]=="guard":guards.append(node["id"])
    return result


def attach(program,plan):
    if program.get("schema","mlx_tensor_semantics_v1")=="mlx_tensor_semantics_v1":return plan
    groups=program.get("source_groups")
    if groups is None:
        groups=[{"source_operator_id":node["source_operator_id"],"source_operator":node["source_operator"],
                 "lowered_ids":[i],"lowering_profile":"direct","output_values":[name for name,_ in value_outputs(node)]}
                for i,node in enumerate(program["nodes"])]
    table=[];stage_to_group=[];cursor=0;last=-1
    for index,group in enumerate(groups):
        ids=group["lowered_ids"];source=group["source_operator_id"]
        if ids!=list(range(cursor,cursor+len(ids))) or not ids or type(source) is not int or not last<source<2**64-1:
            raise ValueError("system source groups must partition stages with ordered source identities")
        table.append({"source_id":source,"stage_begin":cursor,"stage_count":len(ids)})
        stage_to_group.extend([index]*len(ids));cursor+=len(ids);last=source
    if cursor!=len(program["nodes"]):raise ValueError("system source groups omit lowered work")
    plan.update(host_abi_version=3,source_count_basis="lowered_stage_slots",lowered_calls=cursor,
                original_source_calls=len(groups),source_groups=copy.deepcopy(groups),source_group_table=table,
                stage_to_source_group=stage_to_group,source_dependencies=dependencies(program),
                program_schema=program["schema"],result_roles=list(result_roles(program)))
    if "output_contract" in program:
        plan["output_contract"]=program["output_contract"]
        plan["qa_results"]=copy.deepcopy(program["outputs"])
        plan["qa_metadata_values"]={out[role]:copy.deepcopy(program["assets"][out[role]]["values"])
                                    for out in program["outputs"] for role in ("context_mask","offsets_utf8")}
    for row,node in zip(plan["sources"],program["nodes"],strict=True):
        for key in ("origin_source_operator_id","lowering_stage","lowering_stage_name"):
            if key in node:row[key]=node[key]
    return plan
