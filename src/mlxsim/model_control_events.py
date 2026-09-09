"""Lower complete descriptor-bound RV64 control windows, not Rocket CPU timing.

Data-dependent branch/guard witnesses contain values only. Durations and native
timestamps are never accepted as event inputs.
"""
import math
from itertools import groupby

from .model_control_program import control_program

BYTES={"i64":8,"f16":2,"f32":4,"bool":1}


def control_events(job):
    if job.get("schema")!="mlx_control_event_window_job_v1":raise ValueError("expected control event window job")
    if set(job)-{"schema","node","input_layouts","options","branch_taken","guard_value"}:raise ValueError("unknown control event input")
    node=job["node"];p=node["control_program"];kind=node["kind"];args=node["args"];out=node["output"]
    if p!=control_program(kind,p["input_dtype"]):raise ValueError("control event requires canonical RV64 phases")
    options=job.get("options",{});layouts=job["input_layouts"];integer=p["input_dtype"]=="i64"
    if set(options)-{"dma_latency","alu_latency","multiply_latency","float_latency","branch_latency","request_period","response_period","trace_limit","max_cycles","trace"}:raise ValueError("unknown control event option")
    def option(name,default):
        v=options.get(name,default)
        if type(v) is not int or not 1<=v<=1024:raise ValueError("invalid control event option: "+name)
        return v
    def shape(value):
        if not isinstance(value,list) or any(type(x) is not int or x<0 for x in value):raise ValueError("invalid control shape")
        return value
    def is_ref(a):return isinstance(a,dict) and set(a)=={"value"}
    def layout(a):
        if not is_ref(a) or a["value"] not in layouts:raise ValueError("missing control input layout")
        l=layouts[a["value"]];shape(l["shape"])
        if l["dtype"] not in BYTES:raise ValueError("unsupported control layout dtype")
        return l
    count=math.prod(shape(out["shape"]));dtype=out["dtype"]
    if dtype not in BYTES:raise ValueError("unsupported control output dtype")
    if node.get("kwargs",{}).get("alpha",1)!=1:raise ValueError("control add alpha must be one")
    arity=0 if kind=="arange" else 1 if kind in {"argmax","all"} else 2
    for arg in args[:arity]:
        if is_ref(arg):
            t=layout(arg)["dtype"]
            if t not in ({"i64","bool"} if integer else {"f16","f32"}):raise ValueError("control input dtype mismatch")
        elif integer:
            if type(arg) not in {int,bool} or not -(2**63)<=arg<2**63:raise ValueError("invalid integer control literal")
        elif type(arg) not in {int,float}:raise ValueError("invalid floating control literal")
    width=0
    if kind=="arange":
        if len(args)!=1 or type(args[0]) is not int or not 0<=args[0]<2**63 or out["shape"]!=[args[0]] or dtype!="i64":raise ValueError("invalid control arange contract")
    elif kind=="argmax":
        l=layout(args[0]);s=l["shape"]
        if not s or s[-1]<=0 or len(args) not in {2,3} or args[1] not in {-1,len(s)-1} or (len(args)==3 and type(args[2]) is not bool):raise ValueError("invalid argmax axis/shape")
        expected=s[:-1]+([1] if len(args)==3 and args[2] else [])
        if out["shape"]!=expected or dtype!="i64" or l["dtype"]!=p["input_dtype"]:raise ValueError("argmax layout/dtype mismatch")
        width=s[-1]
    elif kind in {"all","guard"}:
        l=layout(args[0])
        if dtype!="bool" or out["shape"] or l["dtype"]!="bool" or len(args)!=(1 if kind=="all" else 2):raise ValueError("invalid Boolean control contract")
        width=math.prod(l["shape"])
        if kind=="guard":
            if width!=1 or type(args[1]) is not bool or type(job.get("guard_value")) is not bool:raise ValueError("guard requires actual Boolean input witness")
            if job["guard_value"]!=args[1]:raise ValueError("control-flow guard mismatch")
    else:
        if len(args)!=2 or dtype!=("bool" if kind in {"le","ge","bitwise_and"} else "i64"):raise ValueError("invalid control elementwise contract")
        expected=[]
        for arg in args:
            if not is_ref(arg):continue
            s=layout(arg)["shape"];rank=max(len(expected),len(s));a=[1]*(rank-len(expected))+expected;b=[1]*(rank-len(s))+s
            if any(x!=y and x!=1 and y!=1 for x,y in zip(a,b)):raise ValueError("control broadcast mismatch")
            expected=[y if x==1 else x for x,y in zip(a,b)]
        if expected!=out["shape"]:raise ValueError("control output broadcast mismatch")
        if kind=="bitwise_and" and any(not is_ref(a) or layout(a)["dtype"]!="bool" for a in args):raise ValueError("Boolean and requires tensors")
    branches=job.get("branch_taken",[])
    if not isinstance(branches,list) or len(branches)!=(count*(width-1) if kind=="argmax" else 0) or any(type(v) is not bool for v in branches):raise ValueError("argmax requires actual branch outcome witness")
    if "guard_value" in job and kind!="guard":raise ValueError("unexpected guard witness")
    serial=0
    def leaf(op,bytes=0):
        nonlocal serial
        serial+=1
        if serial>900000:raise ValueError("control event IR capacity exceeded")
        r=dict(id=f"control:e{serial}",op=op,dependencies=[])
        if bytes:r["bytes"]=bytes
        return r
    def phase(name,taken=False):
        words=p["phases"][name];result=[];pc=0
        while pc<len(words):
            word=words[pc];opcode=word&127
            op="control_branch" if opcode==0x63 else "control_float" if opcode==0x53 else "control_multiply" if opcode==0x33 and word>>25==1 else "control_alu"
            result.append(leaf(op))
            # Canonical select is BEQ x12,x0,+12. A taken branch skips both moves.
            pc+=3 if opcode==0x63 and taken else 1
        return result
    def load(a):return [leaf("dma_read",BYTES[layout(a)["dtype"]])] if is_ref(a) else [leaf("control_literal")]
    def store():return [leaf("dma_write",BYTES[dtype])]
    def repeat(n,body):return [dict(repeat=n,body=body)] if n else []
    if kind=="arange":events=phase("init")+repeat(count,phase("body")+store()+phase("advance"))
    elif kind=="all":events=phase("init")+repeat(width,load(args[0])+phase("body"))+store()
    elif count==0:events=[leaf("control_complete")]
    elif kind=="argmax":
        events=[]
        for row in range(count):
            events+=load(args[0])+phase("init")
            for taken,run in groupby(branches[row*(width-1):(row+1)*(width-1)]):
                events+=repeat(sum(1 for _ in run),load(args[0])+phase("advance")+phase("compare")+phase("select",taken))
            events+=store()
    else:events=repeat(count,load(args[0])+load(args[1])+phase("body")+store())
    dma=option("dma_latency",8)
    latencies=dict(zero=1,mul=4,add=2,convert=2,exp=8,div=12,sqrt=12,spm_read=3,spm_write=3,dma_read=dma,dma_write=dma,predicate_skip=0,
                   operand_prepare=0,spm_initialize=0,neg=1,sub=2,maximum=2,constant=1,move=1,broadcast=1,shuffle=1,cos=8,sin=8,
                   memory_literal=0,memory_convert=2,memory_complete=0,memory_index_read=dma,memory_predicate_read=dma,
                   control_alu=option("alu_latency",1),control_multiply=option("multiply_latency",3),control_float=option("float_latency",3),
                   control_branch=option("branch_latency",1),control_literal=0,control_complete=0)
    hardware=dict(rows=1,columns=1,contexts=2,rf_vectors_per_pe=16,spm_vectors_total=128,rom_words_per_pe=32,source_window_limit=32,
                  template_load_timing=False,latencies=latencies,spm_port_period=1,writeback_period=1,dma_request_period=1,dma_response_period=1,compute_ii=1,sfu_ii=1,
                  memory_controller=dict(data_register_bytes=32,staging_bytes=128,conversion_result_latch_bytes=8,request_data_latch_bytes=8,max_active=1,request_period=1,response_period=1),
                  control_controller=dict(register_bytes=512,rom_words=32,max_active=1,request_period=option("request_period",1),response_period=option("response_period",1)))
    return dict(schema="mlx_event_schedule_v6",hardware=hardware,templates=[],blocks=[],policy="source_priority",max_cycles=options.get("max_cycles",10000000),
                trace_limit=options.get("trace_limit",200000) if options.get("trace",True) else 0,
                controllers=[dict(id="control",domain="control",source_operator_id=0,admission_dependencies=[],events=events)])
