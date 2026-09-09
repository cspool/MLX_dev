"""Memory-controller event lowering; private staging is not PE RF/SPM.

Data-dependent where choices must come from actual predicate values, not a
reference result. No source timestamps or durations are consumed.
"""
import copy
import math
from itertools import groupby

from .model_memory_program import Planner,BYTES


def memory_events(job):
    if job.get("schema")!="mlx_memory_event_window_job_v1":raise ValueError("expected a memory event window job")
    node=job["node"];p=node["memory_program"];options=job.get("options",{})
    if set(options)-{"dma_latency","convert_latency","request_period","response_period","trace_limit","max_cycles","trace"}:raise ValueError("unknown memory event option")
    def option(name,default):
        value=options.get(name,default)
        if type(value) is not int or not 1<=value<=1024:raise ValueError("invalid memory event option: "+name)
        return value
    planner=Planner();planner.layouts=copy.deepcopy(p["input_layouts"]);candidate=copy.deepcopy(node)
    planner.register(candidate,planned=True,same_device=p["same_reference_device"])
    if candidate["memory_program"]!=p:raise ValueError("memory event lowering requires the complete canonical plan")
    output=p["output_layout"];count=math.prod(output["shape"]);kind=node["kind"];args=node["args"];selector=p["selector"]
    if kind=="dropout_inference" and (len(args)!=3 or args[2] is not False):raise ValueError("training dropout cannot be an event view")
    if kind=="where" and len(args)!=3:raise ValueError("where needs exactly three source operands")
    if kind in {"cast","cast_device"}:
        index=1 if kind=="cast" else 2
        declared={"f16":"torch.float16","f32":"torch.float32","i64":"torch.int64","bool":"torch.bool"}[output["dtype"]]
        if not index<len(args)<=index+3 or args[index]!=declared:raise ValueError("memory cast argument differs from actual output dtype")
    if kind=="new_ones":
        kw=node.get("kwargs",{})
        if kw.get("layout") not in {None,"torch.strided"} or (kw.get("pin_memory") is not None and kw.get("pin_memory") is not False):raise ValueError("new_ones layout/pinning is not registered")
    def layout(arg):return p["input_layouts"][arg["value"]]
    def signature(arg):
        return ("dma_read",BYTES[layout(arg)["dtype"]]) if isinstance(arg,dict) and "value" in arg else ("memory_literal",0)
    empty_embedding=kind=="embedding" and layout(args[0])["shape"][1]==0
    if empty_embedding:count=math.prod(layout(args[1])["shape"])
    choices=job.get("predicate_choices")
    if choices is not None and (selector!="predicate_select" or not isinstance(choices,list) or len(choices)!=count or any(type(x) is not bool for x in choices)):
        raise ValueError("predicate choice witness has wrong scope/shape/type")
    serial=0
    def leaf(op,bytes=0):
        nonlocal serial
        row=dict(id=f"memory:e{serial}",op=op,dependencies=[]);serial+=1
        if bytes:row["bytes"]=bytes
        return row
    def body(data=None):
        result=[]
        if selector=="indexed_nd":result=[leaf("memory_index_read",8) for _ in layout(args[0])["shape"]]
        elif selector=="indexed_rows":result=[leaf("memory_index_read",8)]
        elif selector=="predicate_select":result=[leaf("memory_predicate_read",1)]
        if empty_embedding:return result
        if data is None:
            if selector=="constant_one":data=("memory_literal",0)
            elif selector=="concat":data=("dma_read",BYTES[output["dtype"]])
            else:data=signature(args[0])
        result.extend([leaf(data[0],data[1]),leaf("memory_convert"),leaf("dma_write",BYTES[output["dtype"]])])
        return result
    if p["mode"]=="view" or count==0:events=[leaf("memory_complete")]
    elif selector=="predicate_select":
        first,last=signature(args[1]),signature(args[2])
        if choices is None:
            if first!=last:raise ValueError("data-dependent memory timing requires actual predicate choices")
            events=[dict(repeat=count,body=body(first))]
        else:events=[dict(repeat=sum(1 for _ in run),body=body(first if value else last)) for value,run in groupby(choices)]
    else:events=[dict(repeat=count,body=body())]
    dma=option("dma_latency",8)
    latencies=dict(zero=1,mul=4,add=2,convert=2,exp=8,div=12,sqrt=12,spm_read=3,spm_write=3,dma_read=dma,dma_write=dma,predicate_skip=0,
                   operand_prepare=0,spm_initialize=0,neg=1,sub=2,maximum=2,constant=1,move=1,broadcast=1,shuffle=1,cos=8,sin=8,
                   memory_literal=0,memory_convert=option("convert_latency",2),memory_complete=0,memory_index_read=dma,memory_predicate_read=dma)
    hardware=dict(rows=1,columns=1,contexts=2,rf_vectors_per_pe=16,spm_vectors_total=128,rom_words_per_pe=32,source_window_limit=32,
                  template_load_timing=False,latencies=latencies,spm_port_period=1,writeback_period=1,dma_request_period=1,dma_response_period=1,compute_ii=1,sfu_ii=1,
                  memory_controller=dict(data_register_bytes=32,staging_bytes=128,conversion_result_latch_bytes=8,request_data_latch_bytes=8,max_active=1,
                                         request_period=option("request_period",1),response_period=option("response_period",1)))
    return dict(schema="mlx_event_schedule_v5",hardware=hardware,templates=[],blocks=[],policy="source_priority",
                max_cycles=options.get("max_cycles",10000000),trace_limit=options.get("trace_limit",200000) if options.get("trace",True) else 0,
                controllers=[dict(id="memory",source_operator_id=0,admission_dependencies=[],events=events)])
