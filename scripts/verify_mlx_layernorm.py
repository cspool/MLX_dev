"""Audit real primitive-stage execution, numerical scope and closed model gates."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from mlxsim.model_tensor_compiler import compile_inventory,validate_event
from mlxsim.model_source_groups import verify_source_groups
from scripts.run_mlx_ready_model import ROOT,sources
from scripts.run_mlx_tensor_semantics import sha
from scripts.run_mlx_physical_model import runtime_libraries
from scripts.verify_mlx_ready_graph import normalized


def require(value,message):
    if not value:raise RuntimeError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("tests","asan-physical","asan-graph","asan-tensor","legacy-program","legacy-inventory","bert-inventory","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();test=args.tests.resolve();out=args.output.resolve()
    require(not out.exists(),"choose a fresh LayerNorm verification directory")
    suite=ET.parse(test/"regression.xml").getroot().find("testsuite")
    require(suite is not None and all(suite.get(k)=="0" for k in ("failures","errors","skipped")),"LayerNorm regressions failed")
    before=sources()
    for file in (Path(__file__).resolve(),ROOT/"tests/test_layernorm_source_groups.py",ROOT/"src/mlxsim/model_numeric_reference.py",ROOT/"scripts/verify_mlx_ready_graph.py"):
        before[str(file.relative_to(ROOT))]=sha(file)
    binaries={"physical":args.asan_physical.resolve(),"graph":args.asan_graph.resolve(),"tensor":args.asan_tensor.resolve()}
    files={str(p):sha(p) for p in binaries.values()}
    for p in binaries.values():files.update(runtime_libraries(p))
    for p in (test/"regression.xml",args.legacy_program.resolve(),args.legacy_inventory.resolve(),args.bert_inventory.resolve()):files[str(p)]=sha(p)
    def bind(value):
        if isinstance(value,dict):
            if value.get("kind")=="mapped_file":
                p=Path(value["path"]).resolve();files[str(p)]=sha(p)
            for child in value.values():bind(child)
        elif isinstance(value,list):
            for child in value:bind(child)
    selected=[]
    prefixes=("test_layernorm_full_shapes_","test_shifted_statistics_","test_grouped_recipe_","test_composite_source_groups_","test_source_group_cannot_")
    for p in sorted((test/"pytest").rglob("program.json")):
        if any(a.is_symlink() for a in p.parents):continue
        if not p.relative_to(test/"pytest").parts[0].startswith(prefixes):continue
        kind="tensor" if p.parent.name=="tensor" else "physical" if p.parent.name in {"physical","shifted"} else "graph"
        selected.append((kind,p));files[str(p)]=sha(p);files[str(p.parent/"options.json")]=sha(p.parent/"options.json");bind(json.loads(p.read_text()))
        for f in (p.parent/"out").glob("*"):
            if f.is_file():files[str(f)]=sha(f)
    require(len(selected)==19,f"expected 19 LayerNorm/source-group safety replays, found {len(selected)}")
    offsets=list((test/"pytest").glob("test_shifted_statistics_*/large-offset-comparison.json"))
    offsets=[p for p in offsets if not p.parent.is_symlink()];require(len(offsets)==1,"large-offset diagnostic missing")
    files[str(offsets[0])]=sha(offsets[0]);large_offset=json.loads(offsets[0].read_text())
    out.mkdir(parents=True);env=os.environ.copy();env["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";env["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    replays=[]
    for index,(kind,p) in enumerate(selected):
        dst=out/f"asan-{index}";log=out/f"asan-{index}.log"
        command=[binaries[kind],p,dst,"none","1"] if kind=="tensor" else [binaries[kind],p,p.parent/"options.json",dst]
        expected=0 if (p.parent/"out/result.json").exists() else 1
        with log.open("w") as stream:
            run=subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        require(run.returncode==expected,f"LayerNorm sanitizer outcome differs: {log}")
        require(not any(s in log.read_text() for s in ("ERROR: AddressSanitizer","runtime error:","LeakSanitizer","DEADLYSIGNAL")),f"sanitizer failure: {log}")
        item={"kind":kind,"case":str(p.relative_to(test)),"expected_exit":expected,"output":str(dst),"log_sha256":sha(log)}
        if expected:require(not (dst/"result.json").exists(),"rejected source group produced success")
        else:
            old=json.loads((p.parent/"out/result.json").read_text());new=json.loads((dst/"result.json").read_text())
            require(normalized(old)==normalized(new),"sanitizer changed the complete original/lowered execution report")
            verify_source_groups(json.loads(p.read_text()),new)
            for a,b in zip(old["outputs"],new["outputs"],strict=True):
                require(Path(a["logits_file"]).read_bytes()==Path(b["logits_file"]).read_bytes(),"sanitizer changed actual output bytes")
                values=np.fromfile(b["logits_file"],dtype=np.float16 if b["dtype"]=="f16" else np.float32).reshape(b["shape"])
                require(np.isfinite(values).all() and values.argmax(-1).reshape(-1).tolist()==b["tokens"],"reported token differs from actual logits")
            item.update(original_sources=new["executed_source_calls"],lowered_stages=new["executed_lowered_calls"],complete_reports_and_values_equal=True,actual_argmax_checked=True)
        replays.append(item)
    prior=json.loads(args.legacy_program.read_text());inventory=json.loads(args.legacy_inventory.read_text())
    program,_=compile_inventory(inventory,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled",
        schedule_options=prior["matrix_schedule_options"],vector_schedule_options=prior["vector_schedule_options"],
        memory_schedule_options=prior["memory_schedule_options"],control_schedule_options=prior["control_schedule_options"])
    require(len(prior["nodes"])==6181 and program==prior,"legacy Llama2 program changed")
    bert=json.loads(args.bert_inventory.read_text());rejected=Counter()
    for event in bert["operations"]:
        try:validate_event(event)
        except ValueError:rejected[event["operator"]]+=1
    try:compile_inventory(bert,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled")
    except ValueError as error:
        compile_error=str(error);require(compile_error=="missing native semantic lowering: aten.gelu.default","BERT did not reach the expected remaining GELU gap")
    else:raise RuntimeError("unexpected complete BERT compilation")
    require(all(sha(ROOT/p)==h for p,h in before.items()) and all(sha(Path(p))==h for p,h in files.items()),"LayerNorm verification sources/inputs changed")
    report={"classification":"layernorm_primitive_source_group_validation_not_full_model_or_system_acceptance","sources":before,"files":files,
            "regression_tests":int(suite.get("tests")),"replays":replays,"large_offset_framework_difference":large_offset,
            "numeric_profile":"mlx-layernorm-shifted-fp32-v1","legacy_full_model_compilation_equal":True,"legacy_source_calls":6181,
            "bert_name_attribute_preflight":{"source_calls":len(bert["operations"]),"rejected":dict(rejected),"compiled_and_executed_calls":0},
            "bert_full_compile_rejection":compile_error,"full_model_execution_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"LAYERNORM_NATIVE_SAFETY_PASS replays={len(replays)} {out/'report.json'}")


if __name__=="__main__":main()
