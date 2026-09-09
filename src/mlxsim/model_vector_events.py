"""Complete vector-window timing lowering, including reduction control flow."""
import copy
import math

from .model_vector_program import vector_program,FLOAT_KINDS
from .model_dtype_lowering import lower_softmax_input_cast


def vector_events(job):
    if job.get("schema")!="mlx_vector_window_job_v1":raise ValueError("expected a complete vector window job")
    node=job["node"];kind=node["kind"];p=node["vector_program"];options=job.get("options",{})
    if kind not in FLOAT_KINDS:raise ValueError("unregistered vector kind")
    allowed={"rows","columns","contexts","dma_latency","spm_latency","multiply_latency","add_latency","convert_latency","exp_latency","div_latency","sqrt_latency","dma_request_period","dma_response_period","spm_period","writeback_period","vector_ii","trans_ii","trace_limit","max_cycles","overlap","trace","inject_stale_response"}
    if set(options)-allowed or options.get("overlap",True) is not True or options.get("inject_stale_response",False) is not False:raise ValueError("vector events require concurrent, non-fault-injected options")
    def option(name,default,maximum=1024):
        value=options.get(name,default)
        if type(value) is not int or not 1<=value<=maximum:raise ValueError("invalid vector event option: "+name)
        return value
    def shape(dims):
        if not isinstance(dims,list) or len(dims)>8 or any(type(n) is not int or not 0<=n<2**31 for n in dims):raise ValueError("invalid vector event shape")
        return dims
    out_shape=shape(node["output"]["shape"]);out_dtype=node["output"]["dtype"]
    if out_dtype not in {"f16","f32"} or not math.prod(out_shape):raise ValueError("vector window requires nonempty FP output")
    binary=kind in {"add","sub","mul","div","maximum"};count=2 if binary else 1
    args=node["args"];operands=[]
    if len(args)<count:raise ValueError("missing vector event operand")
    for arg in args[:count]:
        literal=not (isinstance(arg,dict) and "value" in arg)
        spec={"dtype":"f32","shape":[]} if literal else job["assets"][arg["value"]]
        if spec["dtype"] not in {"f16","f32"}:raise ValueError("vector event input precision unsupported")
        operands.append(dict(literal=literal,dtype=spec["dtype"],shape=shape(spec["shape"])))
    reduction=kind in {"mean","softmax"};width=None
    if reduction:
        if operands[0]["literal"] or not operands[0]["shape"]:raise ValueError("vector reduction requires a tensor")
        width=operands[0]["shape"][-1]
        if width<=0:raise ValueError("vector reduction width is empty")
        if kind=="mean":
            if len(args) not in (2,3) or args[1] not in ([-1],[len(operands[0]["shape"])-1]):raise ValueError("mean axis is not the last dimension")
            expected=operands[0]["shape"][:-1]+([1] if len(args)==3 and args[2] else [])
        else:
            if len(args) not in (2,3) or args[1] not in (-1,len(operands[0]["shape"])-1):raise ValueError("softmax axis is not the last dimension")
            expected=operands[0]["shape"]
    else:
        if binary and len(args)!=2:raise ValueError("binary vector arity differs")
        if kind=="pow" and (len(args)!=2 or args[1]!=2):raise ValueError("only pow2 is registered")
        if not binary and kind!="pow" and len(args)!=1:raise ValueError("unary vector arity differs")
        expected=[]
        for operand in operands:
            dims=operand["shape"];rank=max(len(expected),len(dims));left=[1]*(rank-len(expected))+expected;right=[1]*(rank-len(dims))+dims
            if any(a!=b and a!=1 and b!=1 for a,b in zip(left,right)):raise ValueError("vector event broadcast mismatch")
            expected=[b if a==1 else a for a,b in zip(left,right)]
    if expected!=out_shape:raise ValueError("vector event output geometry mismatch")
    canonical=copy.deepcopy(node);canonical["vector_program"]=vector_program(kind,[o["dtype"] for o in operands],out_dtype,width=width,alpha=node.get("kwargs",{}).get("alpha",1));lower_softmax_input_cast(canonical)
    if p!=canonical["vector_program"]:raise ValueError("vector event lowering cannot replace modified microinstructions")
    rows=option("rows",4,4);columns=option("columns",4,4)
    spm=option("spm_latency",3);add=option("add_latency",2);dma=option("dma_latency",8);exp=option("exp_latency",8)
    latencies=dict(zero=1,mul=option("multiply_latency",4),add=add,convert=option("convert_latency",2),exp=exp,div=option("div_latency",12),sqrt=option("sqrt_latency",12),
                   spm_read=spm,spm_write=spm,dma_read=dma,dma_write=dma,predicate_skip=0,operand_prepare=0,spm_initialize=0,
                   neg=1,sub=add,maximum=add,constant=1,move=1,broadcast=1,shuffle=1,cos=exp,sin=exp)
    hardware=dict(rows=rows,columns=columns,contexts=option("contexts",2,2),rf_vectors_per_pe=16,spm_vectors_total=128,rom_words_per_pe=32,
                  source_window_limit=32,template_load_timing=False,latencies=latencies,spm_port_period=option("spm_period",1),writeback_period=option("writeback_period",1),
                  dma_request_period=option("dma_request_period",1),dma_response_period=option("dma_response_period",1),compute_ii=option("vector_ii",1),sfu_ii=option("trans_ii",1))
    total=math.prod(out_shape);block_count=math.prod(operands[0]["shape"])//width if reduction else (total+15)//16
    blocks=[];metadata=0;out_width=2 if out_dtype=="f16" else 4
    for block in range(block_count):
        serial=0
        def leaf(op,**fields):
            nonlocal serial,metadata
            metadata+=1
            if metadata>900000:raise ValueError("vector event window exceeds bounded IR metadata; no work was truncated")
            row=dict(id=f"b{block}:e{serial}",op=op,dependencies=[],**fields);serial+=1;return row
        def phase(name,lanes,valid):
            result=[]
            for index in p["phases"][name]:
                word=p["rom"][index];op=word&255;imm=(word>>20)&15
                active=max(0,min(lanes-imm*4,4)) if 17<=op<=21 else 1 if op in (24,26,27) else lanes
                if not active:result.append(leaf("predicate_skip"));continue
                if op in (2,3,9,10):
                    operand=operands[1 if op in (3,10) else 0];size=2 if operand["dtype"]=="f16" else 4
                    result.extend([leaf("operand_prepare"),leaf("spm_initialize",bytes=lanes*size)])
                    if not operand["literal"] and valid:result.append(dict(repeat=valid,body=[leaf("dma_read",bytes=size)]))
                    result.append(leaf("spm_read",bytes=lanes*size))
                elif op==26:result.append(leaf("spm_read",bytes=4))
                elif op in (8,11,27):result.append(leaf("spm_write",bytes=4 if op==27 else valid*out_width))
                else:
                    names={4:"convert",5:"mul",6:"add",7:"convert",14:"constant",15:"neg",16:"sub",17:"exp",18:"div",19:"sqrt",20:"cos",21:"sin",22:"shuffle",23:"maximum",24:"move",25:"broadcast"}
                    result.append(leaf(names[op],active_lanes=active))
            return result
        def drain(count):return dict(repeat=count,body=[leaf("dma_write",bytes=out_width)])
        if not reduction:
            valid=min(16,total-block*16);events=phase("body",valid,valid)+[drain(valid)]
        else:
            chunks=(width+15)//16;padded=1<<(chunks-1).bit_length();lanes=1<<(min(width,16)-1).bit_length();events=[]
            for maximum in ([True,False] if kind=="softmax" else [False]):
                for tile in range(padded):
                    valid=max(0,min(16,width-tile*16))
                    events+=phase("max_tile" if maximum else "sum_tile",lanes,valid)
                    events+=phase("to_carry",1,valid)
                    merge=(tile^(tile+1)).bit_length()-1
                    if merge:events.append(dict(repeat=merge,body=phase("merge_max" if maximum else "merge_sum",1,valid)))
                    events+=phase("save_carry",1,valid)
                events+=phase("root",1,0)
                if kind=="softmax":events+=phase("save_max" if maximum else "save_sum",1,0)
            if kind=="mean":events+=phase("final",1,1)+[drain(1)]
            else:
                for tile in range(chunks):
                    valid=min(16,width-tile*16);events+=phase("output_tile",valid,valid)+[drain(valid)]
        blocks.append(dict(id=f"b{block}",source_operator_id=0,pe=block%(rows*columns),template="vector",admission_dependencies=[],events=events))
    return dict(schema="mlx_event_schedule_v4",policy="source_priority",hardware=hardware,max_cycles=options.get("max_cycles",10000000),
                trace_limit=options.get("trace_limit",200000) if options.get("trace",True) else 0,
                templates=[dict(id="vector",words=p["rom"],rf_vectors=8,spm_vectors=5)],blocks=blocks)
