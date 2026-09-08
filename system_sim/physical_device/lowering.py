"""Encode model matrix windows into the project matrix wire ABI."""
import math
import struct

from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_memory_program import contiguous
from system_sim.physical_host.lowering import tensor_descriptor,require

MAGIC=0x4D4C584D41543031


def geometry(node,layouts):
    kind=node["kind"];args=node["args"]
    require(kind in {"linear","matmul"},"matrix wire lowering requires a matrix operator")
    require(len(args) in ({2,3} if kind=="linear" else {2}),"matrix wire operand count invalid")
    require(all(isinstance(arg,dict) and "value" in arg and arg["value"] in layouts for arg in args[:2]),"matrix wire has missing operands")
    require(node["id"] in layouts,"matrix wire output layout missing")
    a=layouts[args[0]["value"]];b=layouts[args[1]["value"]];out=layouts[node["id"]]
    require(len(a["shape"])>=2 and len(b["shape"])>=2,"matrix wire requires rank-two-or-higher operands")
    require(a["dtype"] in {"f16","f32"} and b["dtype"]==a["dtype"] and out["dtype"] in {"f16","f32"},"matrix wire dtype contract invalid")
    linear=kind=="linear";k=a["shape"][-1];ba=a["shape"][:-2];bb=b["shape"][:-2]
    if linear:
        require(len(b["shape"])==2 and b["shape"][1]==k,"linear wire weight shape mismatch")
        m=math.prod(a["shape"][:-1]);n=b["shape"][0];batch=[];expected=a["shape"][:-1]+[n]
    else:
        require(b["shape"][-2]==k,"matrix wire contracted dimension mismatch")
        m=a["shape"][-2];n=b["shape"][-1];rank=max(len(ba),len(bb));batch=[]
        for av,bv in zip([1]*(rank-len(ba))+ba,[1]*(rank-len(bb))+bb,strict=True):
            require(av==bv or av==1 or bv==1,"matrix wire batch broadcast mismatch");batch.append(bv if av==1 else av)
        expected=batch+[m,n]
    require(expected==out["shape"]==node["output"]["shape"] and out["dtype"]==node["output"]["dtype"] and contiguous(out),"matrix wire output shape/layout mismatch")
    require(all(0<=x<=2**31-1 for x in (m,n,k)),"matrix wire dimensions exceed execution interface")
    batches=1 if linear else math.prod(batch);require(batches>0,"empty matrix batch is not registered")
    if linear and len(args)==3 and args[2] is not None:require(isinstance(args[2],dict) and "value" in args[2],"matrix wire bias is not a tensor reference")
    bias=args[2]["value"] if linear and len(args)==3 and args[2] is not None else None
    if bias is not None:require(bias in layouts and layouts[bias]["dtype"]==a["dtype"] and math.prod(layouts[bias]["shape"])==n,"matrix wire bias mismatch")
    require(node.get("matrix_program")==matrix_program(a["dtype"],out["dtype"],bias is not None),"matrix wire lowering cannot replace a modified microprogram")
    return {"m":m,"n":n,"k":k,"batch_shape":batch,"a_batch_shape":ba,"b_batch_shape":bb,"batches":batches,"linear":linear,"bias":bias}


def broadcast_index(index,output_shape,input_shape):
    result=0;stride=1;shift=len(output_shape)-len(input_shape)
    for dim in range(len(output_shape)-1,-1,-1):
        coordinate=index%output_shape[dim];index//=output_shape[dim]
        if dim>=shift:
            size=input_shape[dim-shift]
            if size!=1:result+=coordinate*stride
            stride*=size
    return result


def lower_matrix_window(node,layouts,bindings,output_batch=0):
    g=geometry(node,layouts);require(type(output_batch) is int and 0<=output_batch<g["batches"],"matrix wire output batch out of range")
    ab=0 if g["linear"] else broadcast_index(output_batch,g["batch_shape"],g["a_batch_shape"])
    bb=0 if g["linear"] else broadcast_index(output_batch,g["batch_shape"],g["b_batch_shape"])
    descriptors=[];used=[]
    for identifier,write in ((node["args"][0]["value"],False),(node["args"][1]["value"],False),(g["bias"],False),(node["id"],True)):
        if identifier is None:descriptors.extend([0]*22);continue
        words,info=tensor_descriptor(identifier,layouts,bindings,write);descriptors.extend(words);used.append(info)
    output=used[-1]
    for operand in used[:-1]:require(not (operand["bytes"] and output["bytes"] and operand["base"]<output["base"]+output["bytes"] and output["base"]<operand["base"]+operand["bytes"]),"matrix wire writable output overlaps input storage")
    p=node["matrix_program"];prologue=p["prologue"];body=p["body"];epilogue=p["epilogue"];code=prologue+body+epilogue
    words=[MAGIC,1,int(g["linear"])|(2 if g["bias"] else 0),g["m"],g["n"],g["k"],ab,bb,output_batch,len(prologue),len(body),len(epilogue),0,0,0,0,*descriptors,*code,*([0]*(32-len(code)))]
    require(len(words)==136,"matrix wire ABI size changed")
    blob=struct.pack("<136Q",*words)
    return blob,{"source_operator_id":node["source_operator_id"],"kind":node["kind"],"output_batch":output_batch,"a_batch":ab,"b_batch":bb,"m":g["m"],"n":g["n"],"k":g["k"],"transpose_b":g["linear"],"has_bias":g["bias"] is not None,
        "profile":p["profile"],"wire_version":1,"command_bytes":len(blob),"bindings":used,"entry":"mlx::matrix_schedule::Simulator","model_execution_verified":False,"mlx_system_verified":False}
