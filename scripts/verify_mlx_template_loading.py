"""Validate timed local template programming, not end-to-end system loading."""
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
    for file in (Path(__file__).resolve(),ROOT/"scripts/verify_mlx_ready_graph.py",ROOT/"tests/test_template_loading.py",
                 ROOT/"tests/template_loader_contract.cc",ROOT/"tests/test_block_pipeline.py",ROOT/"tests/test_ready_graph.py"):
        result[str(file.relative_to(ROOT))]=sha(file)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("output","asan-build","full-program"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh template-loading verification directory")
    out.mkdir(parents=True);before=sources();full_hash=sha(args.full_program)
    tests=["tests/test_template_loading.py","tests/test_block_pipeline.py","tests/test_ready_graph.py","tests/test_shared_array_scheduler.py",
           "tests/test_physical_model.py","tests/test_model_storage.py","tests/test_model_tensor_semantics.py","tests/test_model_numeric_gate.py",
           "tests/test_matrix_window_scheduler.py","tests/test_matrix_external_memory.py","tests/test_matrix_control_cache.py","tests/test_vector_window_scheduler.py"]
    run([sys.executable,"-m","pytest","-q",*tests,f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],out/"pytest.log",timeout=900)
    asan=args.asan_build.resolve()/"mlx-ready-graph";contract=args.asan_build.resolve()/"template-loader-contract";release=ROOT/"build/mlx-ready-graph/mlx-ready-graph"
    binaries={str(p):sha(p) for p in (asan,contract,release)}
    env=os.environ.copy();env["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";env["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    run([contract],out/"asan-contract.log",environment=env)
    cases=[]
    for file in sorted((out/"pytest").rglob("out/result.json")):
        if any(p.is_symlink() for p in file.parents):continue
        result=json.loads(file.read_text())
        if not result.get("array",{}).get("template_configuration_cycles_modeled",False):continue
        directory=file.parent.parent;destination=out/f"asan-{len(cases):02d}"
        run([asan,directory/"program.json",directory/"options.json",destination],out/f"asan-{len(cases):02d}.log",environment=env)
        other=json.loads((destination/"result.json").read_text())
        if normalized(result)!=normalized(other):raise RuntimeError("sanitizer changed template/target events, clocks or resources")
        for a,b in zip(result["outputs"],other["outputs"],strict=True):
            if Path(a["logits_file"]).read_bytes()!=Path(b["logits_file"]).read_bytes():raise RuntimeError("sanitizer changed timed-template numerical results")
        array=result["array"]
        if array["pending_template_words"] or array["template_words_loaded"]!=array["template_words_requested"] or array["template_config_pe_cycles"]!=array["template_words_loaded"]:raise RuntimeError("template word requests/writes did not drain")
        cases.append({"case":str(directory.relative_to(out)),"report_sha256":sha(file),"program_sha256":sha(directory/"program.json"),
                      "options_sha256":sha(directory/"options.json"),"sanitizer_report_sha256":sha(destination/"result.json"),
                      "words":array["template_words_loaded"],"configuration_wall_cycles":array["template_config_wall_cycles"],"configuration_pe_cycles":array["template_config_pe_cycles"]})
    if len(cases)<9:raise RuntimeError("missing timing, logging or multi-PE executions")
    full=json.loads(args.full_program.read_text());program=compile_block_pipelines(full);stripped=dict(program);plan=stripped.pop("block_pipeline_plan")
    if stripped!=full:raise RuntimeError("template mode changed the full model numerical program")
    (out/"full-program.json").write_text(json.dumps(program)+"\n")
    options={"base":2**32,"bytes":16*2**30,"max_cycles":1,"max_active_nodes":32,"tile_pipeline":True,"template_load_timing":True,"template_trace_limit":0,"memory":{"trace_limit":0}}
    (out/"startup-options.json").write_text(json.dumps(options)+"\n")
    run([release,out/"full-program.json",out/"startup-options.json",out/"startup"],out/"startup.log",expected=1)
    if "ready graph exceeded global cycle budget" not in (out/"startup.log").read_text() or (out/"startup/result.json").exists():raise RuntimeError("full-program startup did not respect the explicit limit")
    if sources()!=before or sha(args.full_program)!=full_hash or any(sha(Path(p))!=h for p,h in binaries.items()):raise RuntimeError("template verification inputs changed")
    report={"classification":"local_template_programming_component_validation_not_complete_system_loading_or_model_acceptance",
            "sources":before,"binaries":binaries,"regression_xml_sha256":sha(out/"regression.xml"),"cases":cases,"asan_ubsan_lsan_cases":len(cases),
            "full_program_startup":{"input_sha256":full_hash,"pipeline_program_sha256":sha(out/"full-program.json"),"sources":len(full["nodes"]),"pairs":len(plan["pairs"]),"cycle_limit":1,"exit_code":1,"full_inference_result":False},
            "descriptor_host_or_cache_transfer_verified":False,"full_model_execution_verified":False,"complete_cdc_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False,"rtl_verified":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"LOCAL_TEMPLATE_COMPONENT_CHECKS_PASS {out/'report.json'}")


if __name__=="__main__":main()
