"""Encode canonical vector/normalization programs without discarding phases."""
import copy
import math
import struct

from mlxsim.model_vector_program import vector_program,FLOAT_KINDS
from mlxsim.model_dtype_lowering import lower_softmax_input_cast
from mlxsim.model_memory_program import contiguous
from system_sim.physical_host.lowering import tensor_descriptor,require

MAGIC=0x4D4C585645433031
KINDS=("add","mul","pow","rsqrt","silu","cos","sin","neg","mean","softmax")
PHASES=("body","to_carry","save_carry","root","merge_sum","sum_tile","final","max_tile","merge_max","save_max","save_sum","output_tile")


def double_bits(value):
    if isinstance(value,dict):value=value["float_literal"]
    return struct.unpack("<Q",struct.pack("<d",float(value)))[0]


def lower_vector(node,layouts,bindings):
    kind=node["kind"];require(kind in FLOAT_KINDS,"vector wire kind is not registered")
    require(node["id"] in layouts,"vector wire output layout missing");out=layouts[node["id"]]
    require(out["dtype"] in {"f16","f32"} and out["shape"]==node["output"]["shape"] and out["dtype"]==node["output"]["dtype"] and out["offset"]==0 and contiguous(out),"vector wire output contract mismatch")
    args=node["args"];count=2 if kind in {"add","mul"} else 1;require(len(args)>=count,"vector wire operands missing")
    if kind in {"add","mul"}:require(len(args)==2,"vector wire unexpected binary arguments")
    descriptors=[];used=[];types=[];shapes=[]
    for index in range(2):
        if index>=count:descriptors.extend([0]*26);continue
        arg=args[index]
        if isinstance(arg,dict) and "value" in arg:
            require(arg["value"] in layouts and layouts[arg["value"]]["dtype"] in {"f16","f32"},"vector wire input dtype/layout invalid")
            words,info=tensor_descriptor(arg["value"],layouts,bindings);descriptors.extend([1,0,0,0,*words]);used.append(info);types.append(layouts[arg["value"]]["dtype"]);shapes.append(layouts[arg["value"]]["shape"])
        else:
            descriptors.extend([2,4,double_bits(arg),0,*([0]*22)]);types.append("f32");shapes.append([])
    width=None;flags=0
    if kind in {"mean","softmax"}:
        require(isinstance(args[0],dict) and "value" in args[0] and shapes[0],"vector wire reduction requires a tensor")
        width=shapes[0][-1];require(1<=width<=2**31,"vector wire reduction width invalid")
        if kind=="mean":
            require(len(args)>=2 and isinstance(args[1],list) and len(args[1])==1 and args[1][0] in {-1,len(shapes[0])-1},"vector wire mean requires final axis")
            require(len(args)<=3,"vector wire unexpected mean argument")
            keep=bool(args[2]) if len(args)>2 else False;flags=int(keep);expected=shapes[0][:-1]+([1] if keep else [])
        else:
            require(2<=len(args)<=3 and args[1] in {-1,len(shapes[0])-1},"vector wire softmax requires final axis");expected=shapes[0]
        require(expected==out["shape"],"vector wire reduction output shape mismatch")
    else:
        if kind=="pow":require(len(args)==2 and args[1]==2,"vector wire only supports pow2")
        elif kind not in {"add","mul"}:require(len(args)==1,"vector wire unexpected unary arguments")
        expected=[]
        for shape in shapes:
            rank=max(len(shape),len(expected));left=[1]*(rank-len(expected))+expected;right=[1]*(rank-len(shape))+shape;expected=[]
            for a,b in zip(left,right,strict=True):require(a==b or a==1 or b==1,"vector wire broadcast mismatch");expected.append(b if a==1 else a)
        require(expected==out["shape"],"vector wire broadcast output mismatch")
    candidate=copy.deepcopy(node);candidate["vector_program"]=vector_program(kind,types,out["dtype"],width=width,alpha=node.get("kwargs",{}).get("alpha",1));lower_softmax_input_cast(candidate)
    p=node.get("vector_program");require(p==candidate["vector_program"],"vector wire cannot replace a modified microprogram or phase map")
    if p.get("softmax_input_cast"):flags|=2
    output,info=tensor_descriptor(node["id"],layouts,bindings,True)
    for operand in used:require(not (operand["bytes"] and info["bytes"] and operand["base"]<info["base"]+info["bytes"] and info["base"]<operand["base"]+operand["bytes"]),"vector wire writable output overlaps input")
    used.append(info);require(len(p["rom"])<=32 and len(p["constants"])<=16,"vector wire table capacity exceeded")
    lengths=[];indices=[]
    for phase in PHASES:
        sequence=p["phases"].get(phase,[]);require(len(sequence)<=32 and all(type(i) is int and 0<=i<len(p["rom"]) for i in sequence),"vector wire phase length/index invalid")
        lengths.append(len(sequence));indices.extend(sequence+[0]*(32-len(sequence)))
    require(set(p["phases"])<=set(PHASES),"vector wire unknown phase")
    words=[MAGIC,1,KINDS.index(kind)+1,flags,width or 0,len(p["rom"]),len(p["constants"]),len(p["phases"]),*([0]*8),*descriptors,*output,
        *p["rom"],*([0]*(32-len(p["rom"]))),*[double_bits(v) for v in p["constants"]],*([0]*(16-len(p["constants"]))),*lengths,*indices,0,0]
    require(len(words)==536,"vector wire ABI size changed")
    blob=struct.pack("<536Q",*words)
    return blob,{"source_operator_id":node["source_operator_id"],"kind":kind,"entry":"mlx::vector_schedule::Simulator","wire_version":1,"command_bytes":len(blob),"rom_words":len(p["rom"]),"phase_index_entries":sum(lengths),"bindings":used,"model_execution_verified":False,"mlx_system_verified":False}
