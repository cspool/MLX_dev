"""Emit actual CPU QA readback/postprocessing; reference bytes are check-only."""
import math
import hashlib
from pathlib import Path
import re
import struct

from .lowering import tensor_descriptor


def reference_files(reference):
    for row in reference["outputs"]:
        if row.get("output_contract")=="mlx-qa-result-v1":
            for role in ("start_logits","end_logits"):yield Path(row["outputs"][role]["file"])
        else:yield Path(row["logits_file"])


def emit(plan,reference):
    rows={row["forward_id"]:row for row in reference["outputs"]}
    if len(rows)!=len(reference["outputs"]) or set(rows)!={row["forward_id"] for row in plan["qa_results"]}:raise ValueError("QA reference forward set differs")
    decl=['#include "qa_output.h"','static int qa_same(const void *a,const unsigned char *b,uint64_t n){const unsigned char *p=a;for(uint64_t i=0;i<n;++i)if(p[i]!=b[i])return 0;return 1;}']
    execute=[];check=[];publish=[]
    for index,spec in enumerate(plan["qa_results"]):
        forward=spec["forward_id"];expected=rows[forward]
        if expected.get("output_contract")!="mlx-qa-result-v1":raise ValueError("QA requires a completed native QA reference")
        outputs={row["role"]:row for row in plan["outputs"] if row["forward_id"]==forward}
        if set(outputs)!={"start_logits","end_logits","context_mask","offsets_utf8"}:raise ValueError("QA system output roles incomplete")
        n=outputs["start_logits"]["layout"]["shape"][1];views=[]
        for role in ("start_logits","end_logits","context_mask","offsets_utf8"):
            row=outputs[role];layout=row["layout"];binding=row["binding"]
            words,_=tensor_descriptor(row["value"],{row["value"]:layout},{layout["root"]:binding})
            views.append('{'+','.join(f'UINT64_C({v})' for v in words[:6])+',{'+','.join(map(str,words[6:14]))+'},{'+','.join(map(str,words[14:22]))+'}}')
        decl.extend([f'static const volatile mlx_host_tensor qa_views_{index}[]={{'+','.join(views)+'};',
                     f'static float qa_start_{index}[{n}],qa_end_{index}[{n}];',f'static uint8_t qa_mask_{index}[{n}];',
                     f'static int64_t qa_offsets_{index}[{2*n}];',f'static mlx_qa_span qa_span_{index};'])
        execute.append(f'if(mlx_host_qa_capture(qa_views_{index},{n},qa_start_{index},qa_end_{index},qa_mask_{index},qa_offsets_{index},&qa_span_{index}))return 20;')
        for role,buffer in (("start_logits","start"),("end_logits","end"),("context_mask","mask"),("offsets_utf8","offsets")):
            row=outputs[role];layout=row["layout"]
            if role in ("start_logits","end_logits"):
                ref=expected["outputs"][role]
                if ref["dtype"]!="f32" or ref["shape"]!=[1,n]:raise ValueError("QA reference output shape/dtype differs")
                raw=Path(ref["file"]).read_bytes()
                if len(raw)!=4*n or not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',raw)):raise ValueError("QA reference bytes are missing or nonfinite")
            else:
                values=plan["qa_metadata_values"][spec[role]]
                raw=bytes(values) if role=="context_mask" else struct.pack('<'+'q'*(2*n),*(v for pair in values for v in pair))
            decl.append(f'static const unsigned char qa_expected_{buffer}_{index}[]={{'+','.join(map(str,raw))+'};')
            check.append(f'if(!qa_same(qa_{buffer}_{index},qa_expected_{buffer}_{index},{len(raw)}))return 21;')
        span=expected["span"];a,b=span["start"],span["end"]
        if type(a) is not int or type(b) is not int or not 0<=a<=b<n or b-a>=30:raise ValueError("QA reference span invalid")
        offsets=plan["qa_metadata_values"][spec["offsets_utf8"]];first,last=offsets[a][0],offsets[b][1]
        if spec["context_utf8"].encode()[first:last].decode()!=span["text"]:raise ValueError("QA reference span text differs from tokenizer input")
        bits=int.from_bytes(struct.pack('<d',span["score_f64"]),'little')
        decl.append(f'static const uint64_t qa_score_bits_{index}=UINT64_C({bits});')
        check.extend([f'if(qa_span_{index}.start!={a}||qa_span_{index}.end!={b}||qa_offsets_{index}[2*qa_span_{index}.start]!={first}||qa_offsets_{index}[2*qa_span_{index}.end+1]!={last})return 22;',
                      f'if(!qa_same(&qa_span_{index}.score,(const unsigned char *)&qa_score_bits_{index},8))return 23;'])
        publish.append(f'mlx_host_qa_emit({forward},{n},qa_start_{index},qa_end_{index},qa_offsets_{index},&qa_span_{index});')
    return decl,execute,check,publish


def audit_output(log,plan,reference,directory):
    """Recompute spans from CPU-emitted logits, not from expected span fields."""
    logits={};results={};timing=[]
    number=r"(0|[1-9][0-9]*)"
    for line in log.splitlines():
        if line.startswith("MLX_QA_LOGITS"):
            match=re.fullmatch(r"MLX_QA_LOGITS forward="+number+r" role=(start_logits|end_logits) words=([0-9a-f]{8}(?:,[0-9a-f]{8})*)",line)
            if not match:raise ValueError("malformed actual QA logits record")
            key=int(match[1]),match[2]
            if key in logits:raise ValueError("duplicate actual QA logits")
            logits[key]=b''.join(struct.pack('<I',int(word,16)) for word in match[3].split(','))
        elif line.startswith("MLX_QA_RESULT"):
            match=re.fullmatch("MLX_QA_RESULT"+''.join(" "+key+"="+number for key in ("forward","start","end","start_byte","end_byte","score_bits","candidates")),line)
            if not match:raise ValueError("malformed actual QA span record")
            values=list(map(int,match.groups()))
            if any(v>=2**64 for v in values) or values[0] in results:raise ValueError("invalid/duplicate actual QA span")
            results[values[0]]=dict(zip(("start","end","start_byte","end_byte","score_bits","candidates"),values[1:],strict=True))
        elif line.startswith("MLX_QA_CPU_CYCLES"):
            match=re.fullmatch("MLX_QA_CPU_CYCLES"+''.join(" "+key+"="+number for key in ("begin","graph_end","post_end")),line)
            if not match:raise ValueError("malformed QA CPU cycle record")
            timing.append(list(map(int,match.groups())))
    forwards={row["forward_id"] for row in plan["qa_results"]}
    if set(results)!=forwards or set(logits)!={(f,r) for f in forwards for r in ("start_logits","end_logits")}:
        raise ValueError("actual QA output coverage incomplete")
    if len(timing)!=1 or not 0<=timing[0][0]<timing[0][1]<timing[0][2]<2**64:raise ValueError("actual QA CPU cycle boundaries invalid")
    expected={row["forward_id"]:row for row in reference["outputs"]};rows=[]
    for spec in plan["qa_results"]:
        f=spec["forward_id"];mask=plan["qa_metadata_values"][spec["context_mask"]];offsets=plan["qa_metadata_values"][spec["offsets_utf8"]];n=len(mask)
        data={role:[v[0] for v in struct.iter_unpack('<f',logits[f,role])] for role in ("start_logits","end_logits")}
        if any(len(v)!=n or not all(map(math.isfinite,v)) for v in data.values()):raise ValueError("actual QA logits nonfinite or wrong size")
        best=None;count=0
        for i in range(n):
            if not mask[i]:continue
            for j in range(i,min(n,i+30)):
                if not mask[j]:break
                score=data["start_logits"][i]+data["end_logits"][j];count+=1
                if best is None or score>best[0]:best=(score,i,j)
        if best is None:raise ValueError("actual QA has no eligible span")
        score,i,j=best
        reconstructed=dict(start=i,end=j,start_byte=offsets[i][0],end_byte=offsets[j][1],score_bits=int.from_bytes(struct.pack('<d',score),'little'),candidates=count)
        if results[f]!=reconstructed:raise ValueError("CPU span differs from actual-logits recomputation")
        text=spec["context_utf8"].encode()[offsets[i][0]:offsets[j][1]].decode()
        if text!=expected[f]["span"]["text"]:raise ValueError("actual QA answer differs from bound reference")
        output={"forward_id":f,"span":{**reconstructed,"text":text},"outputs":{}}
        for role in ("start_logits","end_logits"):
            raw=logits[f,role]
            if raw!=Path(expected[f]["outputs"][role]["file"]).read_bytes():raise ValueError("actual CPU logits differ from native execution reference")
            path=Path(directory)/f"cpu-{role}-{f}.f32.bin"
            if path.exists() and path.read_bytes()!=raw:raise ValueError("refusing to overwrite different CPU output")
            if not path.exists():path.write_bytes(raw)
            output["outputs"][role]={"file":str(path.resolve()),"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}
        rows.append(output)
    a,b,c=timing[0]
    return {"outputs":rows,"actual_cpu_readback_and_span_execution":True,"independent_span_recomputation":True,
            "cycle_boundaries":{"begin":a,"graph_end":b,"post_end":c},"graph_cycle_span":b-a,"cpu_postprocess_cycle_span":c-b,
            "cycle_basis":"guest_rdcycle_counter_requires_owning_CPU_timing_model","reference_checks_and_output_logging_excluded":True,
            "preload_time_included":False,"inference_performance_eligible":False}
