"""Audit externally clocked pair descriptor execution and failure draining."""
import argparse
import json
import os
from pathlib import Path
import sys

from scripts.run_mlx_clocked_chipyard import ROOT,source_identity
from scripts.run_mlx_spike_graph import sha
from scripts.verify_mlx_ready_graph import run


def sources():
    result=source_identity()
    for file in (Path(__file__).resolve(),ROOT/"tests/test_pair_wire.py",ROOT/"tests/clocked_device_driver.cc",ROOT/"scripts/verify_mlx_ready_graph.py"):
        result[str(file.relative_to(ROOT))]=sha(file)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--asan-binary",type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh pair-wire validation directory")
    out.mkdir(parents=True);before=sources();asan=args.asan_binary.resolve()
    tests=["tests/test_pair_wire.py","tests/test_clocked_device.py","tests/test_clocked_rocc.py","tests/test_system_progress.py",
           "tests/test_ready_graph.py","tests/test_template_loading.py","tests/test_block_pipeline.py","tests/test_shared_array_scheduler.py",
           "tests/test_matrix_wire_lowering.py","tests/test_vector_wire_lowering.py","tests/test_memory_wire_lowering.py"]
    run([sys.executable,"-m","pytest","-q",*tests,f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],out/"pytest.log",timeout=900)
    hashes={str(asan):sha(asan),str(ROOT/"build/mlx-clocked-device/clocked-device-driver"):sha(ROOT/"build/mlx-clocked-device/clocked-device-driver")}
    environment=os.environ.copy();environment["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";environment["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    cases=[]
    for file in sorted((out/"pytest").rglob("job.json")):
        if any(p.is_symlink() for p in file.parents):continue
        result_file=file.parent/"report.json"
        if not result_file.exists():continue
        job=json.loads(file.read_text())
        if not any(len(c.get("data",[]))==8832 for c in job.get("commands",[])):continue
        expected=json.loads(result_file.read_text());destination=out/f"asan-{len(cases):02d}.json"
        run([asan,file,destination],out/f"asan-{len(cases):02d}.log",environment=environment)
        if json.loads(destination.read_text())!=expected:raise RuntimeError("sanitizer changed pair outputs, target events or error/drain state")
        cases.append({"case":str(file.parent.relative_to(out)),"job_sha256":sha(file),"report_sha256":sha(result_file),"sanitizer_report_sha256":sha(destination),"launches":len(expected["launches"]),"all_launches_done":all(w["done"] for w in expected["launches"])})
    if len(cases)<19:raise RuntimeError("missing pair numerical, retry, error or recovery executions")
    if sources()!=before or any(sha(Path(p))!=h for p,h in hashes.items()):raise RuntimeError("pair verification sources/binaries changed")
    report={"classification":"pair_wire_external_clock_and_bus_component_validation_not_actual_cpu_or_full_model_acceptance",
            "sources":before,"binaries":hashes,"cases":cases,"asan_ubsan_lsan_cases":len(cases),"regression_xml_sha256":sha(out/"regression.xml"),
            "actual_cpu_execution":False,"full_model_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False,"rtl_verified":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"PAIR_WIRE_COMPONENT_CHECKS_PASS {out/'report.json'}")


if __name__=="__main__":main()
