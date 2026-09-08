"""Original source identity is distinct from executed lowered-stage identity."""
import math

def validate_source_groups(program):
    groups=program.get("source_groups")
    if groups is None:
        if program.get("schema")=="mlx_tensor_semantics_v3":raise ValueError("v3 source groups missing")
        return
    if not isinstance(groups,list) or not groups or program["schema"]!="mlx_tensor_semantics_v3" or program.get("value_contract")!="source_groups_v1":
        raise ValueError("invalid source-group program contract")
    cursor=0;nodes=program["nodes"]
    for source,group in enumerate(groups):
        ids=group["lowered_ids"];names=group["stage_names"];kinds=group["lowered_kinds"]
        if group["source_operator_id"]!=source or not ids or ids!=list(range(cursor,cursor+len(ids))) or len(names)!=len(ids) or len(kinds)!=len(ids):
            raise ValueError("source groups must partition every lowered stage exactly once")
        if group["lowering_profile"]=="direct":
            if names!=["direct"] or group["source_operator"]=="aten.layer_norm.default":raise ValueError("invalid direct source group")
        elif group["lowering_profile"]=="mlx-layernorm-shifted-fp32-v1":
            cfg=group["layer_norm_config"]
            if cfg.get("input_dtype") not in {"torch.float16","torch.float32"} or cfg.get("output_dtype")!=cfg["input_dtype"]:
                raise ValueError("LayerNorm source precision contract differs")
            if not isinstance(cfg.get("shape"),list) or not cfg["shape"] or any(type(n) is not int or n<0 for n in cfg["shape"]) or not 1<=cfg["shape"][-1]<=2**31:
                raise ValueError("LayerNorm source shape contract differs")
            if type(cfg.get("epsilon")) not in (int,float) or not math.isfinite(cfg["epsilon"]) or cfg["epsilon"]<0 or type(cfg.get("cudnn_enable")) is not bool:
                raise ValueError("LayerNorm source epsilon/flag contract differs")
            expected=["input_f32"] if cfg["input_dtype"]!="torch.float32" else []
            expected += ["anchor_select","anchor_expand","shift","mean_shift","center","square","variance","epsilon","rsqrt","normalize"]
            for parameter in ("weight","bias"):
                dtype=cfg[parameter+"_dtype"]
                if dtype not in (None,cfg["input_dtype"],"torch.float32"):raise ValueError("LayerNorm source affine precision contract differs")
                if dtype is not None:
                    if dtype!="torch.float32":expected.append(parameter+"_f32")
                    expected.append("affine_"+parameter)
            if cfg["output_dtype"]!="torch.float32":expected.append("output_cast")
            if group["source_operator"]!="aten.layer_norm.default" or names!=expected:raise ValueError("LayerNorm recipe stages are incomplete")
        else:raise ValueError("unknown source-group lowering profile")
        for stage,identifier in enumerate(ids):
            if identifier>=len(nodes):raise ValueError("source group references a missing stage")
            node=nodes[identifier]
            if group["lowering_profile"]=="direct" and node["source_operator"]!=group["source_operator"]:raise ValueError("direct source operator label differs")
            if (node["source_operator_id"],node.get("origin_source_operator_id"),node.get("lowering_stage"),node.get("lowering_stage_name"),node["kind"])!=(identifier,source,stage,names[stage],kinds[stage]):
                raise ValueError("lowered node source/stage identity mismatch")
            if group["lowering_profile"]!="direct":
                required={"input_f32":"cast","anchor_select":"select","anchor_expand":"unsqueeze","shift":"sub","mean_shift":"mean","center":"sub","square":"pow","variance":"mean","epsilon":"add","rsqrt":"rsqrt","normalize":"mul","weight_f32":"cast","affine_weight":"mul","bias_f32":"cast","affine_bias":"add","output_cast":"cast"}
                if node["kind"]!=required[names[stage]]:raise ValueError("LayerNorm primitive kind differs from the recipe")
                if names[stage]=="epsilon" and node["args"][1]!=group["layer_norm_config"]["epsilon"]:raise ValueError("LayerNorm source epsilon differs from its primitive")
        last=nodes[ids[-1]]
        if group["lowering_profile"]!="direct":
            cfg=group["layer_norm_config"]
            if last["output"]!={"shape":cfg["shape"],"dtype":"f16" if cfg["output_dtype"]=="torch.float16" else "f32"}:raise ValueError("LayerNorm source output contract differs")
        outputs=[o["id"] for o in last["split_outputs"]] if last["kind"]=="split" else [last["id"]]
        if group["output_values"]!=outputs:raise ValueError("source output binding mismatch")
        cursor+=len(ids)
    if cursor!=len(nodes):raise ValueError("ungrouped lowered stages")


def verify_source_groups(program,result):
    validate_source_groups(program)
    if "source_groups" not in program:return
    groups=program["source_groups"];events=result["events"]
    if result.get("executed_lowered_calls")!=len(program["nodes"]) or result["executed_source_calls"]!=len(groups):raise RuntimeError("original/lowered source accounting mismatch")
    by_id={e["source_operator_id"]:e for e in events}
    if len(by_id)!=len(program["nodes"]):raise RuntimeError("lowered source events incomplete")
    records=result.get("source_group_events",[])
    if len(records)!=len(groups):raise RuntimeError("original source groups incomplete")
    for group,record in zip(groups,records,strict=True):
        ids=group["lowered_ids"]
        if record["source_operator_id"]!=group["source_operator_id"] or record["source_operator"]!=group["source_operator"] or record["lowered_ids"]!=ids or not record["complete"]:raise RuntimeError("source group completion differs")
        for stage,identifier in enumerate(ids):
            event=by_id[identifier]
            if event.get("lowered_operator_id")!=identifier or event.get("origin_source_operator_id")!=group["source_operator_id"] or event.get("lowering_stage")!=stage or event.get("lowering_stage_name")!=group["stage_names"][stage]:raise RuntimeError("event lost original source identity")
        if record.get("timing_observed"):
            starts=[by_id[i].get("start_cycle",by_id[i].get("shared_start_cycle")) for i in ids]
            ends=[by_id[i].get("publish_cycle",by_id[i].get("shared_end_cycle")) for i in ids]
            if record["start_cycle"]!=min(starts) or record["complete_cycle"]!=max(ends):raise RuntimeError("source group interval differs from actual stages")
