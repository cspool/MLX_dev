"""Audit erf-form GELU primitive execution while preserving framework failures."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from mlxsim.model_tensor_compiler import compile_inventory,validate_event
from mlxsim.model_composites import expand_composites
from mlxsim.model_source_groups import verify_source_groups
from scripts.run_mlx_ready_model import ROOT,sources
from scripts.run_mlx_tensor_semantics import sha
from scripts.run_mlx_physical_model import runtime_libraries
from scripts.verify_mlx_ready_graph import normalized


def require(value,message):
    if not value:raise RuntimeError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("tests","asan-tensor","asan-physical","asan-graph","legacy-program","legacy-inventory","bert-inventory","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--case-timeout",type=int,default=3600,help="host watchdog seconds per case; not a simulated cycle limit")
    args=parser.parse_args();test=args.tests.resolve();out=args.output.resolve()
    require(args.case_timeout>0,"case timeout must be positive")
    require(not out.exists(),"choose a fresh GELU verification directory")
    suite=ET.parse(test/"regression.xml").getroot().find("testsuite")
    require(suite is not None and all(suite.get(k)=="0" for k in ("failures","errors","skipped")),"GELU regression suite did not pass")
    before=sources()
    for file in (Path(__file__).resolve(),ROOT/"tests/test_gelu_source_groups.py",ROOT/"src/mlxsim/model_numeric_reference.py",ROOT/"scripts/verify_mlx_ready_graph.py"):
        before[str(file.relative_to(ROOT))]=sha(file)
    binaries={"tensor":args.asan_tensor.resolve(),"physical":args.asan_physical.resolve(),"graph":args.asan_graph.resolve()}
    files={str(p):sha(p) for p in binaries.values()}
    for p in binaries.values():files.update(runtime_libraries(p))
    for p in (test/"regression.xml",args.legacy_program.resolve(),args.legacy_inventory.resolve(),args.bert_inventory.resolve()):files[str(p)]=sha(p)
    selected=[]
    prefixes=("test_gelu_tail_boundaries_","test_every_finite_fp16_","test_fp32_dense_and_bitpattern","test_gelu_actual_operator_","test_gelu_shared_pair_","test_gelu_recipe_cannot_")
    for p in sorted((test/"pytest").rglob("program.json")):
        if any(a.is_symlink() for a in p.parents):continue
        if not p.relative_to(test/"pytest").parts[0].startswith(prefixes):continue
        kind="physical" if p.parent.name=="physical" else "graph" if p.parent.name in {"paired","bad"} else "tensor"
        selected.append((kind,p));files[str(p)]=sha(p);files[str(p.parent/"options.json")]=sha(p.parent/"options.json")
        for f in (p.parent/"out").glob("*"):
            if f.is_file():files[str(f)]=sha(f)
    require(len(selected)==14,f"expected 14 GELU safety replays, found {len(selected)}")
    domain_files=[]
    for p in (test/"pytest").rglob("numeric-domain.json"):
        if not any(a.is_symlink() for a in p.parents):domain_files.append(p);files[str(p)]=sha(p)
    require(len(domain_files)==2,"numeric domain evidence missing")
    domains=[json.loads(p.read_text()) for p in domain_files]
    require(any(d.get("domain")=="all_63488_finite_fp16_encodings" and d["elements"]==63488 for d in domains),"complete finite FP16 domain missing")
    out.mkdir(parents=True)
    attempt={"classification":"gelu_sanitizer_attempt_not_acceptance","status":"running","sources":before,"files":files,"case_timeout_seconds":args.case_timeout,"completed_replays":[]}
    (out/"attempt.json").write_text(json.dumps(attempt,indent=2)+"\n")
    env=os.environ.copy();env["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";env["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    replays=[]
    for index,(kind,p) in enumerate(selected):
        dst=out/f"asan-{index}";log=out/f"asan-{index}.log"
        command=[binaries[kind],p,dst,"none","1"] if kind=="tensor" else [binaries[kind],p,p.parent/"options.json",dst]
        expected=0 if (p.parent/"out/result.json").exists() else 1
        attempt["active_case"]={"index":index,"kind":kind,"command":list(map(str,command)),"expected_exit":expected}
        (out/"attempt.json").write_text(json.dumps(attempt,indent=2)+"\n")
        print(f"GELU_SAFETY_REPLAY {index} {kind} expected_exit={expected}",flush=True)
        try:
            with log.open("w") as stream:
                run=subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=args.case_timeout)
        except subprocess.TimeoutExpired:
            attempt["status"]="host_watchdog_expired"
            (out/"attempt.json").write_text(json.dumps(attempt,indent=2)+"\n")
            raise
        require(run.returncode==expected,f"GELU sanitizer outcome differs: {log}")
        require(not any(s in log.read_text() for s in ("ERROR: AddressSanitizer","runtime error:","LeakSanitizer","DEADLYSIGNAL")),f"GELU sanitizer failed: {log}")
        if expected:require(not (dst/"result.json").exists(),"rejected GELU produced success evidence")
        else:
            old=json.loads((p.parent/"out/result.json").read_text());new=json.loads((dst/"result.json").read_text())
            require(normalized(old)==normalized(new),"GELU sanitizer changed complete execution report")
            verify_source_groups(json.loads(p.read_text()),new)
            for a,b in zip(old["outputs"],new["outputs"],strict=True):
                require(Path(a["logits_file"]).read_bytes()==Path(b["logits_file"]).read_bytes(),"GELU sanitizer changed actual values")
                values=np.fromfile(b["logits_file"],dtype=np.float16 if b["dtype"]=="f16" else np.float32).reshape(b["shape"])
                require(np.isfinite(values).all() and values.argmax(-1).reshape(-1).tolist()==b["tokens"],"GELU token differs from actual output")
        replays.append({"case":str(p.relative_to(test)),"kind":kind,"expected_exit":expected,"output":str(dst),"log_sha256":sha(log),"complete_values_and_reports_equal":not expected})
        attempt["completed_replays"]=list(replays)
        (out/"attempt.json").write_text(json.dumps(attempt,indent=2)+"\n")
    prior=json.loads(args.legacy_program.read_text());inventory=json.loads(args.legacy_inventory.read_text())
    rebuilt,_=compile_inventory(inventory,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled",
        schedule_options=prior["matrix_schedule_options"],vector_schedule_options=prior["vector_schedule_options"],memory_schedule_options=prior["memory_schedule_options"],control_schedule_options=prior["control_schedule_options"])
    require(len(prior["nodes"])==6181 and rebuilt==prior,"legacy full Llama2 program changed")
    bert=json.loads(args.bert_inventory.read_text());rejected=Counter()
    for event in bert["operations"]:
        try:validate_event(event)
        except ValueError:rejected[event["operator"]]+=1
    low,groups=expand_composites(bert)
    require(low["bindings"]==bert["bindings"],"GELU/LayerNorm expansion introduced data bindings")
    try:compile_inventory(bert,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled")
    except ValueError as error:
        reason=str(error);require(reason=="not every generation step has its own computed logits/token output","unexpected remaining BERT compilation gap")
    else:raise RuntimeError("QA output protocol gate was unexpectedly bypassed")
    require(all(sha(ROOT/p)==h for p,h in before.items()) and all(sha(Path(p))==h for p,h in files.items()),"GELU verification sources/inputs changed")
    report={"classification":"gelu_erf_primitive_numerical_and_safety_validation_not_full_model_or_system",
            "sources":before,"files":files,"regression_tests":int(suite.get("tests")),"replays":replays,"numeric_domains":domains,
            "formula_source":"https://personal.math.ubc.ca/~cbm/aands/page_299.htm","formula":"Abramowitz-Stegun 7.1.26",
            "printed_real_erf_bound_not_fp32_execution_bound":1.5e-7,"numeric_profile":"mlx-gelu-erf-as7126-fp32-v1",
            "legacy_full_model_compilation_equal":True,"bert_name_attribute_rejections":dict(rejected),
            "bert_source_calls":len(groups),"bert_lowered_stages":len(low["operations"]),"bert_bindings_unchanged":True,"bert_remaining_gate":reason,
            "full_model_execution_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    attempt["status"]="completed";attempt.pop("active_case",None)
    (out/"attempt.json").write_text(json.dumps(attempt,indent=2)+"\n")
    print(f"GELU_NATIVE_SAFETY_PASS replays={len(replays)} {out/'report.json'}")


if __name__=="__main__":main()
