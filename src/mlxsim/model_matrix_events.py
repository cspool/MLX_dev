"""Lower one complete matrix window to explicit, compact timing events.

No numerical input/output values or measured native durations enter this IR.
Memory timing is the declared fixed-latency endpoint, not a cache substitute.
"""
from .model_matrix_program import matrix_program


def matrix_events(job):
    if job.get("schema")!="mlx_matrix_window_job_v1":raise ValueError("expected a complete matrix window job")
    m,n,k=(job[name] for name in ("m","n","k"))
    if any(type(v) is not int or not 0<=v<2**31 for v in (m,n,k)) or not m or not n:raise ValueError("matrix event dimensions require nonempty output")
    p=job["program"];bias=bool(p["has_bias"])
    if p!=matrix_program(p["input_dtype"],p["output_dtype"],bias) or bias!=("bias" in job):raise ValueError("matrix event lowering requires the canonical actual microprogram")
    transposed=job.get("transposed_b",True)
    if type(transposed) is not bool or job["a"]["shape"]!=[m,k] or job["b"]["shape"]!=([n,k] if transposed else [k,n]):raise ValueError("matrix event operand geometry differs from the window")
    if any(job[key]["dtype"]!=p["input_dtype"] for key in (("a","b","bias") if bias else ("a","b"))) or (bias and job["bias"]["shape"]!=[n]):raise ValueError("matrix event operand dtype/bias differs")
    options=job.get("options",{})
    allowed={"rows","columns","contexts","dma_latency","spm_latency","multiply_latency","add_latency","convert_latency","dma_request_period","dma_response_period","spm_period","writeback_period","compute_ii","trace_limit","max_cycles","overlap","trace","inject_stale_dma_epoch","cache_control"}
    if set(options)-allowed or options.get("overlap",True) is not True or options.get("inject_stale_dma_epoch",False) is not False:
        raise ValueError("matrix events require concurrent, non-fault-injected native options")
    def option(name,default,maximum=1024):
        value=options.get(name,default)
        if type(value) is not int or not 1<=value<=maximum:raise ValueError("invalid matrix event option: "+name)
        return value
    rows=option("rows",4,4);columns=option("columns",4,4);contexts=option("contexts",2,2)
    width=2 if p["input_dtype"]=="f16" else 4;out_width=2 if p["output_dtype"]=="f16" else 4
    latencies=dict(zero=1,mul=option("multiply_latency",4),add=option("add_latency",2),convert=option("convert_latency",2),
                   exp=8,div=12,sqrt=12,spm_read=option("spm_latency",3),spm_write=option("spm_latency",3),
                   dma_read=option("dma_latency",8),dma_write=option("dma_latency",8),predicate_skip=0)
    hardware=dict(rows=rows,columns=columns,contexts=contexts,rf_vectors_per_pe=16,spm_vectors_total=128,rom_words_per_pe=32,
                  source_window_limit=32,template_load_timing=False,latencies=latencies,spm_port_period=option("spm_period",1),
                  writeback_period=option("writeback_period",1),dma_request_period=option("dma_request_period",1),
                  dma_response_period=option("dma_response_period",1),compute_ii=option("compute_ii",1))
    blocks=[]
    for row in range(0,m,2):
        for col in range(0,n,16):
            block=len(blocks);actual_rows=min(2,m-row);lanes=min(16,n-col);serial=0
            def leaf(op,**fields):
                nonlocal serial
                result=dict(id=f"b{block}:e{serial}",op=op,dependencies=[],**fields);serial+=1;return result
            def code(words):
                result=[]
                for word in words:
                    op=word&255;selected_row=(word>>20)&3
                    if selected_row<2 and selected_row>=actual_rows:result.append(leaf("predicate_skip"));continue
                    if op in (2,3,9,10,12,13):result.append(leaf("spm_read",bytes=width if op in (2,9) else lanes*width))
                    elif op in (8,11):result.append(leaf("spm_write",bytes=lanes*out_width))
                    else:result.append(leaf({1:"zero",4:"convert",5:"mul",6:"add",7:"convert"}[op],active_lanes=lanes))
                return result
            def chunk(count):
                return [{"repeat":(actual_rows+lanes)*count,"body":[leaf("dma_read",bytes=width)]},
                        {"repeat":count,"body":code(p["body"])}]
            events=code(p["prologue"])
            if k//64:events.append({"repeat":k//64,"body":chunk(64)})
            if k%64:events.extend(chunk(k%64))
            if bias:events.append({"repeat":lanes,"body":[leaf("dma_read",bytes=width)]})
            events.extend(code(p["epilogue"]))
            events.append({"repeat":actual_rows*lanes,"body":[leaf("dma_write",bytes=out_width)]})
            blocks.append(dict(id=f"b{block}",source_operator_id=0,pe=block%(rows*columns),template="matrix",admission_dependencies=[],events=events))
    return dict(schema="mlx_event_schedule_v3",policy="source_priority",hardware=hardware,max_cycles=options.get("max_cycles",10000000),
                trace_limit=options.get("trace_limit",100000) if options.get("trace",True) else 0,
                templates=[dict(id="matrix",words=p["prologue"]+p["body"]+p["epilogue"],rf_vectors=p["rf_vectors_used"],spm_vectors=(p["spm_bytes_used"]+63)//64)],blocks=blocks)
