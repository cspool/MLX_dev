"""Audit sparse C++ backing, real high-offset CPU graphs and dense regressions."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from scripts.run_mlx_spike_graph import ROOT, sha
from scripts.verify_mlx_spike_matrix_chain import source_identity as bridge_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--dense-baseline",type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();baseline=args.dense_baseline.resolve()
    if out.exists():raise RuntimeError("choose a fresh sparse-memory evidence directory")
    baseline_report=json.loads((baseline/"report.json").read_text())
    if len(baseline_report["cases"])!=29:raise RuntimeError("expected 29 dense bridge baseline cases")
    baseline_hashes={str(baseline/"report.json"):sha(baseline/"report.json")}
    for case in baseline_report["cases"]:
        file=baseline/case["case"]/"device.json"
        if sha(file)!=case["device_report_sha256"]:raise RuntimeError("dense baseline report binding changed")
        baseline_hashes[str(file)]=sha(file)
    out.mkdir(parents=True)

    def identity():
        result=bridge_sources()
        for path in (Path(__file__).resolve(),ROOT/"scripts/run_mlx_spike_graph.py",ROOT/"tests/mapped_memory_contract.cc",
                     ROOT/"tests/test_mapped_memory.py",ROOT/"tests/test_spike_sparse_graph.py",ROOT/"tests/test_spike_graph_runtime.py",
                     ROOT/"tests/test_model_memory_program.py",ROOT/"tests/test_model_tensor_semantics.py"):
            result[str(path.relative_to(ROOT))]=sha(path)
        return result

    sources=identity();python=str(ROOT/".venv/bin/python")

    def run(command,log,timeout=600,env=None):
        with (out/log).open("w") as output:
            subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=timeout,env=env)

    run([python,"-m","pytest","-q","tests/test_mapped_memory.py","tests/test_spike_sparse_graph.py",
         f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],"pytest.log")
    release=ROOT/"build/mlx-spike-matrix"
    run([str(release/"mapped-memory-contract")],"contract.json")
    san=ROOT/"build/mlx-spike-matrix-asan"
    run(["cmake","-S",str(ROOT/"system_sim/physical_device"),"-B",str(san),"-DCMAKE_BUILD_TYPE=Debug",
         "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer"],"san-configure.log")
    run(["cmake","--build",str(san),"--target","mapped-memory-contract","-j4"],"san-build.log")
    environment=os.environ.copy();environment["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";environment["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    run([str(san/"mapped-memory-contract")],"contract-asan.json",env=environment)
    if (out/"contract.json").read_bytes()!=(out/"contract-asan.json").read_bytes():raise RuntimeError("native sanitizer contract differs")

    graphs=[];devices=[]
    for file in sorted((out/"pytest").rglob("run/report.json")):
        if any(p.is_symlink() for p in file.parents):continue
        report=json.loads(file.read_text());program=file.parents[2]/"dispatch-graph0"
        label="high" if report["data_offset"]>2**32 else "low"
        run([python,"-m","scripts.run_mlx_spike_graph","--asan","--program",str(program/"program.json"),
             "--lifetimes",str(program/"life.json"),"--reference",str(program/"out/result.json"),
             "--device-bytes",str(report["device_capacity_bytes"]),"--data-offset",str(report["data_offset"]),
             "--output",str(out/f"graph-asan-{label}")],f"graph-asan-{label}.log")
        device=json.loads((file.parent/"device.json").read_text());devices.append(device)
        if device!=json.loads((out/f"graph-asan-{label}/device.json").read_text()):raise RuntimeError("sanitizer graph events/counters differ")
        graphs.append({"case":label,"report":str(file.relative_to(out)),"report_sha256":sha(file),
                       "asan_report_sha256":sha(out/f"graph-asan-{label}/report.json"),"host_memory_backing":device["host_memory_backing"]})
    if len(graphs)!=2 or devices[0]!=devices[1]:raise RuntimeError("low/high offset graph events/counters differ")

    run([python,"-m","scripts.verify_mlx_spike_matrix_chain","--output",str(out/"chain")],"chain.log")
    run([python,"-m","scripts.verify_mlx_spike_matrix_chain","--asan","--output",str(out/"chain-asan")],"chain-asan.log")
    cases=[]
    for case in baseline_report["cases"]:
        name=case["case"];sparse=json.loads((out/"chain"/name/"device.json").read_text())
        if sparse!=json.loads((out/"chain-asan"/name/"device.json").read_text()):raise RuntimeError(f"sanitizer chain mismatch: {name}")
        backing=sparse.pop("host_memory_backing")
        if sparse!=json.loads((baseline/name/"device.json").read_text()):raise RuntimeError(f"dense/sparse events or counters differ: {name}")
        cases.append({"case":name,"host_memory_backing":backing})
    if sources!=identity() or any(sha(Path(p))!=digest for p,digest in baseline_hashes.items()):raise RuntimeError("sparse validation sources/baseline changed")
    report={"classification":"sparse_host_backing_and_high_offset_cpu_graph_validation_not_full_model_or_soc",
        "sources":sources,"dense_baseline":baseline_hashes,"graphs":graphs,"dense_equivalent_cases":cases,
        "low_high_graph_reports_equal":True,"only_dense_report_field_added":"host_memory_backing",
        "contract_sha256":sha(out/"contract.json"),"contract_asan_sha256":sha(out/"contract-asan.json"),
        "regression_sha256":sha(out/"regression.xml"),"chain_report_sha256":sha(out/"chain/report.json"),"chain_asan_report_sha256":sha(out/"chain-asan/report.json"),
        "full_model_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"SPARSE_MAPPED_MEMORY_VERIFIED {out/'report.json'}")


if __name__=="__main__":main()
