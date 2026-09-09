"""Encode checked memory plans, retaining aliases and real transfer templates."""
import copy
import struct

from mlxsim.model_memory_program import Planner
from system_sim.physical_host.lowering import tensor_descriptor,require,MAX
from .vector_lowering import double_bits

MAGIC=0x4D4C584D454D3031
KINDS=("embedding","where","cat","cast","cast_device","contiguous","reshape","transpose","slice","select","unsqueeze","expand","alias","dropout_inference","advanced_index","new_ones","squeeze")
SELECTORS=("linear","concat","indexed_rows","predicate_select","indexed_nd","constant_one")


def lower_view(node,layouts,bindings):
    """Validate all tuple aliases without inventing a device data transfer."""
    if node["kind"]!="split":
        _,route=lower_memory(node,layouts,bindings)
        require(route["mode"]=="view","view elision requires a checked affine view")
        return route
    from mlxsim.model_value_outputs import value_outputs
    p=node.get("memory_program",{});candidate=copy.deepcopy(node);planner=Planner()
    require(p.get("mode")=="view" and p.get("words")==[],"split requires a nonexecuting affine plan")
    planner.layouts={name:copy.deepcopy(layouts[name]) for name in p["input_layouts"]}
    planner.register(candidate,planned=True,same_device=p["same_reference_device"])
    require(candidate["memory_program"]==p and layouts[node["id"]]==p["output_layout"],"split view plan was modified")
    source=node["args"][0]["value"];root=layouts[source]["root"]
    tensor_descriptor(source,layouts,bindings)
    definitions=[]
    for name,spec in value_outputs(node):
        require(layouts[name]==p["split_layouts"][name] and layouts[name]["root"]==root,"split output layout lost its root")
        _,info=tensor_descriptor(name,layouts,bindings)
        definitions.append({"value":name,"output":spec,"layout":layouts[name],"binding":info})
    return {"kind":"split","entry":"checked_split_affine_aliases","mode":"view","definitions":definitions,
            "device_commands":0,"model_execution_verified":False,"mlx_system_verified":False}


def signed(value):
    require(type(value) is int and -2**63<=value<2**63,"memory wire integer parameter out of range");return value&MAX


def lower_memory(node,layouts,bindings):
    kind=node["kind"];require(kind in KINDS and "memory_program" in node,"memory wire requires a checked memory plan")
    p=node["memory_program"];candidate=copy.deepcopy(node);planner=Planner();planner.layouts={name:copy.deepcopy(layouts[name]) for name in p["input_layouts"]}
    require(layouts.get(node["id"])==p["output_layout"],"memory wire output layout differs from plan")
    planner.register(candidate,planned=True,same_device=p["same_reference_device"])
    require(candidate["memory_program"]==p,"memory wire cannot replace a modified layout/transfer program")
    args=node["args"];flags=int(p["same_reference_device"]);params=[]
    operands=list(args[0]) if kind=="cat" else [args[0],*args[1]] if kind=="advanced_index" else list(args[:3]) if kind=="where" else list(args[:2]) if kind=="embedding" else [args[0]]
    if kind=="embedding":
        require(2<=len(args)<=5,"memory wire embedding arity invalid");params=[signed(args[2] if len(args)>2 else -1),int(bool(args[3])) if len(args)>3 else 0,int(bool(args[4])) if len(args)>4 else 0]
    elif kind=="where":require(len(args)==3,"memory wire where arity invalid")
    elif kind=="cat":require(1<=len(args)<=2,"memory wire cat arity invalid");params=[signed(args[1] if len(args)>1 else 0)]
    elif kind in {"cast","cast_device"}:
        shift=0 if kind=="cast" else 1;require(2+shift<=len(args)<=4+shift,"memory wire cast arity invalid")
        dtype_name={"f16":"torch.float16","f32":"torch.float32","i64":"torch.int64","bool":"torch.bool"}[node["output"]["dtype"]]
        require(args[1+shift]==dtype_name,"memory wire cast dtype argument disagrees with output")
        if len(args)>2+shift and args[2+shift]:flags|=32
        if len(args)>3+shift and args[3+shift]:flags|=2
    elif kind=="contiguous":require(len(args)==1,"memory wire contiguous arity invalid")
    elif kind=="advanced_index":
        require(len(args)==2 and isinstance(args[1],list) and 1<=len(args[1])<=8,"memory wire advanced index arguments invalid")
    elif kind in {"reshape","expand","new_ones"}:
        require(len(args)==2 and len(args[1])<=8,"memory wire shape argument invalid");params=[signed(n) for n in args[1]]
        if kind=="reshape" and node.get("source_operator")=="aten.view.default":flags|=16
        if kind=="new_ones":
            kw=node.get("kwargs",{})
            require(kw.get("layout") in {None,"torch.strided"} and (kw.get("pin_memory") is None or kw.get("pin_memory") is False),"memory wire new_ones layout/pinning unsupported")
            if kw.get("dtype") is not None:flags|=64
    elif kind in {"transpose","select"}:require(len(args)==3,"memory wire axis/index arity invalid");params=[signed(args[1]),signed(args[2])]
    elif kind in {"unsqueeze","squeeze"}:require(len(args)==2,"memory wire insertion/squeeze arity invalid");params=[signed(args[1])]
    elif kind=="slice":
        require(1<=len(args)<=5,"memory wire slice arity invalid");start=args[2] if len(args)>2 else None;end=args[3] if len(args)>3 else None
        params=[signed(args[1] if len(args)>1 else 0),signed(start or 0),signed(end or 0),signed(args[4] if len(args)>4 else 1),int(start is None)|(int(end is None)<<1)]
    elif kind=="dropout_inference":require(len(args)==3 and args[2] is False,"memory wire training dropout is unsupported");params=[double_bits(args[1])]
    else:require(len(args)==1,"memory wire alias arity invalid")
    fmt=node.get("kwargs",{}).get("memory_format")
    require(fmt in {None,"torch.contiguous_format","torch.preserve_format"},"memory wire format unsupported")
    if fmt=="torch.contiguous_format":flags|=4
    if fmt=="torch.preserve_format":flags|=8
    require(len(operands)<=256,"memory wire argument list exceeds control capacity")
    table=[];indices=[];seen={};roots={};used=[];tensor_slots=0
    for arg in operands:
        if isinstance(arg,dict) and "value" in arg:
            identifier=arg["value"]
            if identifier in seen:indices.append(seen[identifier]);continue
            descriptor,info=tensor_descriptor(identifier,layouts,bindings);root=layouts[identifier]["root"]
            if root not in roots:roots[root]=len(roots)+1
            seen[identifier]=len(table);indices.append(len(table));table.append([1,0,0,roots[root],*descriptor]);used.append(info);tensor_slots+=1
        else:
            require(kind=="where","memory wire only where accepts data literals")
            dtype,bits=(3,int(arg)) if type(arg) is bool else (2,signed(arg)) if type(arg) is int else (4,double_bits(arg))
            indices.append(len(table));table.append([2,dtype,bits,0,*([0]*22)])
    require(len(table)<=64 and tensor_slots<=63,"memory wire operand/region capacity exceeded")
    view=p["mode"]=="view";require(p["mode"] in {"view","transfer"},"memory wire mode invalid")
    output,info=tensor_descriptor(node["id"],layouts,bindings,not view);out_root=layouts[node["id"]]["root"]
    if view:
        require(out_root in roots and isinstance(operands[0],dict) and layouts[operands[0]["value"]]["root"]==out_root,"memory wire view lost its root")
        output_root=roots[out_root]
    else:
        output_root=0
        for operand in used:require(not (operand["bytes"] and info["bytes"] and operand["base"]<info["base"]+info["bytes"] and info["base"]<operand["base"]+operand["bytes"]),"memory wire output overlaps input storage")
    version=2 if kind in {"advanced_index","new_ones","squeeze"} else 1;capacity=12 if version==2 else 4
    used.append(info);words=p["words"];require(len(words)<=capacity,"memory wire transfer template too long")
    raw=[MAGIC,version,KINDS.index(kind)+1,int(not view),SELECTORS.index(p["selector"])+1,flags,len(table),len(indices),len(words),output_root,len(params),0,0,0,0,0,
         *params,*([0]*(16-len(params))),*indices,*([0]*(256-len(indices))),*output,
         *[word for entry in table for word in entry],*([0]*((64-len(table))*26)),*words,*([0]*(capacity-len(words))),*([0]*6)]
    require(len(raw)==(1992 if version==2 else 1984),"memory wire ABI size changed");blob=struct.pack(f"<{len(raw)}Q",*raw)
    return blob,{"source_operator_id":node["source_operator_id"],"kind":kind,"mode":p["mode"],"selector":p["selector"],"wire_version":version,"command_bytes":len(blob),"operand_slots":len(table),"argument_count":len(indices),"input_slots":seen,"storage_root_ids":roots,"bindings":used,"entry":"mlx::memory_model::Simulator","model_execution_verified":False,"mlx_system_verified":False}
