"""Replay BERT-required memory components under sanitizers; not model acceptance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from mlxsim.model_tensor_compiler import compile_inventory
from scripts.run_mlx_ready_model import ROOT, sources
from scripts.run_mlx_tensor_semantics import sha
from scripts.run_mlx_physical_model import runtime_libraries
from scripts.verify_mlx_ready_graph import normalized


def require(value,message):
    if not value:raise RuntimeError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("tests","asan-memory","asan-graph","asan-physical","legacy-program","legacy-inventory","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();test=args.tests.resolve();out=args.output.resolve()
    require(not out.exists(),"choose a fresh memory verification directory")
    suite=ET.parse(test/"regression.xml").getroot().find("testsuite")
    require(suite is not None and all(suite.get(k)=="0" for k in ("failures","errors","skipped")),"memory regressions did not pass")
    before=sources()
    for file in (Path(__file__).resolve(),ROOT/"tests/test_bert_memory_paths.py",ROOT/"tests/memory_external_memory.cc",ROOT/"scripts/verify_mlx_ready_graph.py"):
        before[str(file.relative_to(ROOT))]=sha(file)
    binaries={"memory":args.asan_memory.resolve(),"graph":args.asan_graph.resolve(),"physical":args.asan_physical.resolve()}
    files={str(p):sha(p) for p in binaries.values()}
    for p in binaries.values():files.update(runtime_libraries(p))
    for p in (test/"regression.xml",args.legacy_program.resolve(),args.legacy_inventory.resolve()):files[str(p)]=sha(p)
    def bind_assets(value):
        if isinstance(value,dict):
            if value.get("kind")=="mapped_file":
                p=Path(value["path"]).resolve();files[str(p)]=sha(p)
            for child in value.values():bind_assets(child)
        elif isinstance(value,list):
            for child in value:bind_assets(child)
    prefixes=("test_nd_index_reads_actual_","test_nd_index_uses_strides_","test_bad_nd_index_responses_",
              "test_nd_index_rank_capacity_","test_empty_and_scalar_index_","test_new_ones_generates_",
              "test_new_ones_contract_","test_squeeze_proves_","test_index_plan_rejects_")
    selected=[]
    for p in sorted((test/"pytest").rglob("job.json")):
        if any(a.is_symlink() for a in p.parents):continue
        if p.relative_to(test/"pytest").parts[0].startswith(prefixes):selected.append(("memory",p))
    for p in sorted((test/"pytest").rglob("program.json")):
        if any(a.is_symlink() for a in p.parents):continue
        if p.relative_to(test/"pytest").parts[0].startswith("test_actual_operator_capture_"):
            selected.append(("physical" if p.parent.name=="serial" else "graph",p))
    require(len(selected)==46,f"expected 46 memory safety replays, found {len(selected)}")
    for kind,p in selected:
        files[str(p)]=sha(p);bind_assets(json.loads(p.read_text()))
        if kind!="memory":
            options=p.parent/("system-options.json" if kind=="physical" else "options.json")
            files[str(options)]=sha(options)
        for f in (p.parent/"out").glob("*"):
            if f.is_file():files[str(f)]=sha(f)
    out.mkdir(parents=True)
    env=os.environ.copy();env["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";env["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    replays=[]
    for index,(kind,p) in enumerate(selected):
        dst=out/f"asan-{index}";log=out/f"asan-{index}.log";command=[binaries[kind],p]
        if kind!="memory":command.append(p.parent/("system-options.json" if kind=="physical" else "options.json"))
        command.append(dst);expected=0 if (p.parent/"out/result.json").exists() else 1
        with log.open("w") as stream:
            run=subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        require(run.returncode==expected,f"memory sanitizer outcome differs: {log}")
        require(not any(s in log.read_text() for s in ("ERROR: AddressSanitizer","runtime error:","LeakSanitizer","DEADLYSIGNAL")),f"sanitizer failure: {log}")
        if expected:
            require(not (dst/"result.json").exists(),"rejected memory operation produced a success report")
        else:
            old=json.loads((p.parent/"out/result.json").read_text());new=json.loads((dst/"result.json").read_text())
            if kind=="memory":
                require(old==new and (p.parent/"out/output.bin").read_bytes()==(dst/"output.bin").read_bytes(),"sanitizer changed memory values/events/cycles")
            else:
                require(normalized(old)==normalized(new),"sanitizer changed the whole graph report")
                for a,b in zip(old["outputs"],new["outputs"],strict=True):
                    require(Path(a["logits_file"]).read_bytes()==Path(b["logits_file"]).read_bytes(),"sanitizer changed actual graph outputs")
        replays.append({"case":str(p.relative_to(test)),"kind":kind,"expected_exit":expected,
                        "output":str(dst),"log_sha256":sha(log),"complete_values_and_reports_equal":not expected})
    prior=json.loads(args.legacy_program.read_text());inventory=json.loads(args.legacy_inventory.read_text())
    rebuilt,_=compile_inventory(inventory,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled",
        schedule_options=prior["matrix_schedule_options"],vector_schedule_options=prior["vector_schedule_options"],
        memory_schedule_options=prior["memory_schedule_options"],control_schedule_options=prior["control_schedule_options"])
    require(len(prior["nodes"])==6181 and rebuilt==prior,"legacy full Llama2 compilation changed")
    require(all(sha(ROOT/p)==h for p,h in before.items()) and all(sha(Path(p))==h for p,h in files.items()),"verification source/input changed")
    report={"classification":"native_bert_required_memory_component_safety_not_full_model_or_system_validation",
            "regression_tests":int(suite.get("tests")),"sources":before,"files":files,"replays":replays,
            "legacy_full_model_compilation_equal":True,"legacy_source_calls":6181,
            "full_model_execution_verified":False,"actual_rocket_execution_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"BERT_MEMORY_NATIVE_SAFETY_PASS replays={len(replays)} {out/'report.json'}")


if __name__=="__main__":main()
