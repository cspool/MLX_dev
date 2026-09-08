"""Bind tuple/checkpoint component safety, real BERT gaps and legacy compatibility."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from mlxsim.model_tensor_compiler import compile_inventory, validate_event
from scripts.run_mlx_ready_model import ROOT, sources
from scripts.run_mlx_tensor_semantics import sha
from scripts.run_mlx_physical_model import runtime_libraries
from scripts.verify_mlx_ready_graph import normalized


def require(value,message):
    if not value:raise RuntimeError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("tests","asan-memory","asan-graph","asan-physical","asan-tensor","bert-inventory","legacy-program","legacy-inventory","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();test=args.tests.resolve();out=args.output.resolve()
    require(not out.exists(),"choose a fresh tuple verification directory")
    suite=ET.parse(test/"regression.xml").getroot().find("testsuite")
    require(suite is not None and all(suite.get(k)=="0" for k in ("failures","errors","skipped")),"tuple regressions did not pass")
    before=sources()
    for file in (Path(__file__).resolve(),ROOT/"tests/test_split_value_outputs.py",ROOT/"tests/test_single_checkpoint_binding.py",
                 ROOT/"tests/memory_external_memory.cc",ROOT/"scripts/verify_mlx_ready_graph.py"):
        before[str(file.relative_to(ROOT))]=sha(file)
    binaries={"memory":args.asan_memory.resolve(),"graph":args.asan_graph.resolve(),"physical":args.asan_physical.resolve(),"tensor":args.asan_tensor.resolve()}
    files={str(p):sha(p) for p in binaries.values()}
    for p in binaries.values():files.update(runtime_libraries(p))
    for p in (test/"regression.xml",args.bert_inventory.resolve(),args.legacy_program.resolve(),args.legacy_inventory.resolve()):files[str(p)]=sha(p)
    def bind_assets(value):
        if isinstance(value,dict):
            if value.get("kind")=="mapped_file":
                p=Path(value["path"]).resolve();files[str(p)]=sha(p)
            for child in value.values():bind_assets(child)
        elif isinstance(value,list):
            for child in value:bind_assets(child)
    selected=[]
    for p in sorted((test/"pytest").rglob("job.json")):
        if any(a.is_symlink() for a in p.parents):continue
        if p.relative_to(test/"pytest").parts[0].startswith(("test_all_split_parts_","test_noncontiguous_split_","test_empty_or_single_split_","test_split_canonical_layout_")):
            selected.append(("memory",p,None,p.parent/"out",None))
    for p in sorted((test/"pytest").rglob("program.json")):
        if any(a.is_symlink() for a in p.parents):continue
        name=p.relative_to(test/"pytest").parts[0]
        if name.startswith(("test_complete_source_compil","test_tuple_program_cannot_","test_single_checkpoint_exact_")):
            kind="physical" if p.parent.name in {"serial","native"} else "graph"
            options=p.parent/("system-options.json" if kind=="physical" else "options.json")
            selected.append((kind,p,options,p.parent/"out",None))
        elif name.startswith("test_plain_native_consumes_"):
            selected.append(("tensor",p,None,p.parent/"plain",None))
            selected.append(("tensor",p,None,p.parent/"observe",p.parent/"observe.json"))
    require(len(selected)==32,f"expected 32 tuple/checkpoint safety replays, found {len(selected)}")
    for kind,p,options,original,observation in selected:
        files[str(p)]=sha(p);bind_assets(json.loads(p.read_text()))
        for file in (options,observation):
            if file is not None:files[str(file)]=sha(file)
        for f in original.glob("*"):
            if f.is_file():files[str(f)]=sha(f)
    out.mkdir(parents=True)
    env=os.environ.copy();env["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";env["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    replays=[]
    for index,(kind,p,options,original,observation) in enumerate(selected):
        dst=out/f"asan-{index}";log=out/f"asan-{index}.log";command=[binaries[kind],p]
        if options is not None:command.append(options)
        command.append(dst)
        if kind=="tensor":command += ["none","1"]
        if observation is not None:command.append(observation)
        expected=0 if (original/"result.json").exists() else 1
        with log.open("w") as stream:
            run=subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        require(run.returncode==expected,f"tuple sanitizer outcome differs: {log}")
        require(not any(s in log.read_text() for s in ("ERROR: AddressSanitizer","runtime error:","LeakSanitizer","DEADLYSIGNAL")),f"sanitizer failure: {log}")
        if expected:require(not (dst/"result.json").exists(),"rejected tuple produced success evidence")
        else:
            old=json.loads((original/"result.json").read_text());new=json.loads((dst/"result.json").read_text())
            if kind=="memory":require(old==new and (original/"output.bin").read_bytes()==(dst/"output.bin").read_bytes(),"sanitizer changed memory output/report")
            else:
                require(normalized(old)==normalized(new),"sanitizer changed complete graph report")
                for a,b in zip(old["outputs"],new["outputs"],strict=True):
                    require(Path(a["logits_file"]).read_bytes()==Path(b["logits_file"]).read_bytes(),"sanitizer changed actual output")
        replays.append({"kind":kind,"case":str(p.relative_to(test)),"original_output":str(original),"expected_exit":expected,
                        "output":str(dst),"log_sha256":sha(log),"complete_output_and_report_equal":not expected})
    prior=json.loads(args.legacy_program.read_text());inventory=json.loads(args.legacy_inventory.read_text())
    rebuilt,_=compile_inventory(inventory,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled",
        schedule_options=prior["matrix_schedule_options"],vector_schedule_options=prior["vector_schedule_options"],
        memory_schedule_options=prior["memory_schedule_options"],control_schedule_options=prior["control_schedule_options"])
    require(len(prior["nodes"])==6181 and rebuilt==prior,"legacy full Llama2 compilation changed")
    bert=json.loads(args.bert_inventory.read_text());rejected=Counter()
    weight=Path(bert["model_identity"]["path"])/"model.safetensors";files[str(weight)]=sha(weight)
    require(files[str(weight)]==bert["files"][str(weight)]["sha256"],"actual BERT checkpoint changed")
    for operation in bert["operations"]:
        try:validate_event(operation)
        except ValueError:rejected[operation["operator"]]+=1
    try:compile_inventory(bert,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled")
    except ValueError as error:
        compile_error=str(error)
        require(compile_error=="missing native semantic lowering: aten.layer_norm.default","BERT checkpoint binding did not reach the expected remaining kernel gap")
    else:raise RuntimeError("unexpected full BERT compilation")
    require(all(sha(ROOT/p)==h for p,h in before.items()) and all(sha(Path(p))==h for p,h in files.items()),"tuple verification source/input changed")
    report={"classification":"tuple_view_value_and_single_checkpoint_components_not_full_model_or_system_validation",
            "sources":before,"files":files,"regression_tests":int(suite.get("tests")),"replays":replays,
            "legacy_full_model_compilation_equal":True,"legacy_source_calls":6181,
            "bert_name_attribute_preflight":{"source_calls":len(bert["operations"]),"rejected":dict(rejected),"compiled_and_executed_calls":0},
            "bert_full_compile_rejection":compile_error,"full_model_execution_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"TUPLE_VALUES_NATIVE_SAFETY_PASS replays={len(replays)} {out/'report.json'}")


if __name__=="__main__":main()
