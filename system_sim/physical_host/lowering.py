"""Lower canonical control nodes to the host-memory ABI, not PE/RoCC opcodes."""
import copy
import math
import struct

from mlxsim.model_control_program import control_program
from mlxsim.model_memory_program import Planner,KINDS

MAGIC=0x4D4C584843545231
DTYPE={"f16":0,"f32":1,"i64":2,"bool":3}
BYTES={"f16":2,"f32":4,"i64":8,"bool":1}
OP={"arange":1,"add":2,"mul":3,"le":4,"argmax":5}
MAX=2**64-1


def require(value,message):
    if not value:raise ValueError(message)


def collect_layouts(program):
    planner=Planner()
    for name,asset in program["assets"].items():planner.add_asset(name,asset)
    for node in program["nodes"]:
        require(node["id"] not in planner.layouts,"host layout SSA redefinition")
        copied=copy.deepcopy(node)
        if node["kind"] in KINDS:
            require("memory_program" in node,"host address lowering requires checked memory layouts")
            planner.register(copied,planned=True,same_device=node["memory_program"].get("same_reference_device",True))
            require(copied["memory_program"]==node["memory_program"],"host address lowering found a noncanonical memory layout")
        else:planner.register(copied,planned=True)
    return planner.layouts


def tensor_descriptor(identifier,layouts,bindings,write=False):
    require(identifier in layouts,"missing host tensor layout")
    layout=layouts[identifier];require(layout["root"] in bindings,"missing host physical binding")
    binding=bindings[layout["root"]];shape=layout["shape"];stride=layout["strides"];dtype=layout["dtype"]
    require(dtype in DTYPE and len(shape)==len(stride)<=8,"invalid host tensor rank/type")
    require(all(type(v) is int and 0<=v<2**63 for v in shape+stride) and type(layout["offset"]) is int and 0<=layout["offset"]<2**63,"host layout is outside the unsigned descriptor contract")
    base=binding["base"];size=binding["bytes"];width=BYTES[dtype]
    require(type(base) is int and type(size) is int and 0<=base<=MAX and 0<=size<=MAX-base and (not size or base) and base%width==size%width==0,"host physical address/extent invalid")
    require(size==layout["storage_elements"]*width,"host binding changed storage capacity")
    require(not write or binding["writable"] is True,"host output binding is read-only")
    count=math.prod(shape);require(count<=MAX,"host logical element count overflow")
    if count:
        last=layout["offset"]+sum((n-1)*s for n,s in zip(shape,stride,strict=True))
        require(last<size//width,"host view exceeds physical storage")
    info={"value":identifier,"root":layout["root"],"base":base,"bytes":size,"write":write}
    return [base,size,layout["offset"],len(shape),DTYPE[dtype],3 if write else 1,*shape,*([0]*(8-len(shape))),*stride,*([0]*(8-len(stride)))],info


def lower_control(node,layouts,bindings):
    kind=node["kind"];program=node.get("control_program",{});domain=program.get("input_dtype")
    require(kind in OP and domain in {"i64","f16","f32"},"host controller kind/domain is not registered")
    require(program==control_program(kind,domain),"host C lowering cannot replace a modified RV64 leaf program")
    require(node.get("kwargs",{}).get("alpha",1)==1,"host integer add requires unit alpha")
    used=[]
    def tensor(identifier,write=False):
        words,info=tensor_descriptor(identifier,layouts,bindings,write);used.append(info);return words
    def operand(arg):
        if isinstance(arg,dict) and "value" in arg:
            layout=layouts.get(arg["value"],{});dtype=layout.get("dtype")
            require(dtype in ({"i64","bool"} if domain=="i64" else {"f16","f32"}),"host operand dtype disagrees with control domain")
            return [1,0,0,0,*tensor(arg["value"])]
        if domain=="i64":
            require(type(arg) in {bool,int} and -2**63<=arg<2**63,"host scalar cannot silently narrow an integer")
            dtype=3 if type(arg) is bool else 2;bits=int(arg)&MAX
        else:
            raw=arg.get("float_literal") if isinstance(arg,dict) else arg
            value=float(raw)
            try:packed=struct.pack("<f",value)
            except OverflowError:packed=struct.pack("<f",math.copysign(math.inf,value))
            dtype=1;bits=int.from_bytes(packed,"little")
        return [2,dtype,bits,0,*([0]*22)]
    args=node["args"];flags=extent=0
    if kind=="arange":
        require(len(args)==1 and type(args[0]) is int and 0<=args[0]<2**63,"host arange extent invalid")
        extent=args[0];a=b=[0]*26
    elif kind=="argmax":
        require(2<=len(args)<=3 and isinstance(args[0],dict) and "value" in args[0],"host argmax arguments invalid")
        rank=len(layouts[args[0]["value"]]["shape"]);require(args[1] in {-1,rank-1},"host argmax requires final axis")
        flags=int(bool(args[2])) if len(args)==3 else 0;a=operand(args[0]);b=[0]*26
    else:
        require(len(args)==2,"host control binary arity mismatch");a=operand(args[0]);b=operand(args[1])
    output=tensor(node["id"],True);words=[MAGIC,1,OP[kind],flags,extent,0,*a,*b,*output]
    require(len(words)==80,"host command ABI size changed")
    blob=struct.pack("<80Q",*words)
    return blob,{"source_operator_id":node["source_operator_id"],"kind":kind,"entry":"mlx_host_control_execute","abi_version":1,"command_bytes":len(blob),"bindings":used,
        "host_isa":"RV64IMAFD","pe_opcode":False,"rocket_execution_verified":False,"mlx_system_verified":False}
