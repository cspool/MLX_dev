"""Audit bounded pair-event execution, not general CDC or full-model acceptance."""
import argparse
import json
import os
from pathlib import Path
import sys

from mlxsim.model_block_pipeline import compile_block_pipelines
from scripts.run_mlx_tensor_semantics import ROOT,sha
from scripts.verify_mlx_physical_model import identity as model_sources
from scripts.verify_mlx_ready_graph import run,normalized


def sources():
    result=model_sources()
    for file in (Path(__file__).resolve(),ROOT/"scripts/compile_mlx_block_pipeline.py",ROOT/"scripts/verify_mlx_ready_graph.py",
                 ROOT/"tests/test_block_pipeline.py",ROOT/"tests/completion_window_contract.cc",ROOT/"tests/test_ready_graph.py"):
        result[str(file.relative_to(ROOT))]=sha(file)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("output","asan-build","full-program"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh bounded-pipeline validation directory")
    out.mkdir(parents=True);before=sources();full_sha=sha(args.full_program)
    tests=["tests/test_block_pipeline.py","tests/test_ready_graph.py","tests/test_shared_array_scheduler.py",
           "tests/test_physical_model.py","tests/test_model_storage.py","tests/test_model_tensor_semantics.py",
           "tests/test_model_numeric_gate.py","tests/test_matrix_window_scheduler.py","tests/test_matrix_external_memory.py",
           "tests/test_matrix_control_cache.py","tests/test_vector_window_scheduler.py"]
    run([sys.executable,"-m","pytest","-q",*tests,f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],out/"pytest.log",timeout=900)
    asan=args.asan_build.resolve()/"mlx-ready-graph";contract=args.asan_build.resolve()/"completion-window-contract";release=ROOT/"build/mlx-ready-graph/mlx-ready-graph"
    binaries={str(p):sha(p) for p in (asan,contract,release)}
    env=os.environ.copy();env["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";env["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    run([contract],out/"asan-window.log",environment=env)
    cases=[]
    for file in sorted((out/"pytest").rglob("out/result.json")):
        if any(p.is_symlink() for p in file.parents):continue
        result=json.loads(file.read_text())
        if result.get("classification")!="ready_graph_bounded_pair_events_not_general_cdc_or_system_acceptance":continue
        directory=file.parent.parent;destination=out/f"asan-{len(cases):02d}"
        run([asan,directory/"program.json",directory/"options.json",destination],out/f"asan-{len(cases):02d}.log",environment=env)
        other=json.loads((destination/"result.json").read_text())
        if normalized(result)!=normalized(other):raise RuntimeError("sanitizer changed pipeline events, cycles or ownership")
        for a,b in zip(result["outputs"],other["outputs"],strict=True):
            if Path(a["logits_file"]).read_bytes()!=Path(b["logits_file"]).read_bytes():raise RuntimeError("sanitizer changed pipeline values")
        program=json.loads((directory/"program.json").read_text());groups=result["pipeline_groups"]
        if result["executed_source_calls"]!=len(program["nodes"]) or len(groups)!=len(program["block_pipeline_plan"]["pairs"]):raise RuntimeError("source or pipeline coverage is incomplete")
        if any(not g["finished"] or g["frontier"]!=g["blocks"] or g["admitted"]!=g["completed"] or g["peak_event_slots"]>g["event_slots"] or g["pending_visibility"] for g in groups):raise RuntimeError("pipeline event bank did not drain")
        cases.append({"case":str(directory.relative_to(out)),"sources":result["executed_source_calls"],"pairs":len(groups),
                      "program_sha256":sha(directory/"program.json"),"options_sha256":sha(directory/"options.json"),
                      "report_sha256":sha(file),"sanitizer_report_sha256":sha(destination/"result.json")})
    if len(cases)<20:raise RuntimeError("missing tile/barrier, tail, reuse, batch, reduction or generation cases")
    original=json.loads(args.full_program.read_text());compiled=compile_block_pipelines(original)
    stripped=dict(compiled);plan=stripped.pop("block_pipeline_plan")
    if stripped!=original:raise RuntimeError("pipeline compiler changed the model numerical program")
    (out/"full-program.json").write_text(json.dumps(compiled)+"\n")
    options={"base":2**32,"bytes":16*2**30,"max_cycles":1,"max_active_nodes":32,"tile_pipeline":True,"operator_progress":True,"memory":{"trace_limit":0}}
    (out/"startup-options.json").write_text(json.dumps(options)+"\n")
    run([release,out/"full-program.json",out/"startup-options.json",out/"startup"],out/"startup.log",expected=1)
    if "ready graph exceeded global cycle budget" not in (out/"startup.log").read_text() or (out/"startup/result.json").exists():raise RuntimeError("full model startup stopped for an unexpected reason")
    if sources()!=before or sha(args.full_program)!=full_sha or any(sha(Path(p))!=h for p,h in binaries.items()):raise RuntimeError("pipeline validation sources or binaries changed")
    report={"classification":"bounded_closed_pair_event_component_validation_not_general_cdc_or_full_model_acceptance",
            "sources":before,"binaries":binaries,"regression_xml_sha256":sha(out/"regression.xml"),"cases":cases,"asan_ubsan_lsan_cases":len(cases),
            "full_model_compilation":{"input_sha256":full_sha,"pipeline_program_sha256":sha(out/"full-program.json"),"source_calls":len(original["nodes"]),
                                      "pairs":len(plan["pairs"]),"rejected_pairs":len(plan["rejected_pairs"]),"numerical_program_unchanged":True,
                                      "startup_cycle_limit":1,"startup_exit_code":1,"startup_log_sha256":sha(out/"startup.log"),"full_inference_result":False},
            "full_model_execution_verified":False,"complete_cdc_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False,"rtl_verified":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"BOUNDED_PIPELINE_COMPONENT_CHECKS_PASS {out/'report.json'}")


if __name__=="__main__":main()
