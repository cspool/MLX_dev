"""Verify CPU file-asset loading and graph equivalence; not full-model timing."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from scripts.run_mlx_spike_graph import ROOT,sha
from scripts.verify_mlx_spike_matrix_chain import source_identity as bridge_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh file-asset validation directory")
    out.mkdir(parents=True);python=str(ROOT/".venv/bin/python")
    def identity():
        result=bridge_sources()
        for path in (Path(__file__).resolve(),ROOT/"scripts/run_mlx_spike_graph.py",ROOT/"tests/asset_source_contract.cc",ROOT/"tests/test_asset_source.py",
                     ROOT/"tests/test_spike_graph_runtime.py",ROOT/"tests/test_model_memory_program.py",ROOT/"tests/test_model_tensor_semantics.py"):
            result[str(path.relative_to(ROOT))]=sha(path)
        return result
    sources=identity()
    def run(command,log,env=None):
        with (out/log).open("w") as output:subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=600,env=env)
    run([python,"-m","pytest","-q","tests/test_asset_source.py",f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],"pytest.log")
    graph=out/"pytest/dispatch-graph0";cases=[]
    for file in sorted((out/"pytest").rglob("files/report.json")):
        if any(p.is_symlink() for p in file.parents):continue
        case=file.parent.parent;label=case.name;destination=out/f"asan-{label}"
        run([python,"-m","scripts.run_mlx_spike_graph","--program",str(case/"mapped-program.json"),"--lifetimes",str(graph/"life.json"),
            "--reference",str(case/"reference.json"),"--asset-source","files","--source-offset",str(2**32+4096),"--asan","--output",str(destination)],f"{label}-asan.log")
        for name in ("device.json","asset-source.json"):
            if json.loads((file.parent/name).read_text())!=json.loads((destination/name).read_text()):raise RuntimeError(f"ASan {name} differs")
        cases.append({"case":label,"report_sha256":sha(file),"asan_report_sha256":sha(destination/"report.json")})
    if len(cases)!=2:raise RuntimeError("normal/perturbed file graphs missing")
    run([python,"-m","scripts.run_mlx_spike_graph","--program",str(graph/"program.json"),"--lifetimes",str(graph/"life.json"),
         "--asset-source","files","--load-only","--asan","--output",str(out/"asan-load-only")],"load-only-asan.log")
    ordinary=out/"pytest/test_actual_cpu_load_only_does0/load"
    for name in ("device.json","asset-source.json"):
        if json.loads((ordinary/name).read_text())!=json.loads((out/"asan-load-only"/name).read_text()):raise RuntimeError("ASan load-only differs")
    build=ROOT/"build/mlx-spike-matrix-asan"
    run(["cmake","--build",str(build),"--target","asset-source-contract","-j4"],"native-asan-build.log")
    environment=os.environ.copy();environment["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";environment["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    probes=[]
    for file in sorted((out/"pytest").rglob("job.json")):
        if any(p.is_symlink() for p in file.parents):continue
        job=json.loads(file.read_text())
        if not job["actions"] or any(a["kind"]=="truncate" for a in job["actions"]):continue
        label=file.parent.name
        run([str(ROOT/"build/mlx-spike-matrix/asset-source-contract"),str(file)],f"{label}-native.json")
        run([str(build/"asset-source-contract"),str(file)],f"{label}-asan.json",env=environment)
        if (out/f"{label}-native.json").read_bytes()!=(out/f"{label}-asan.json").read_bytes():raise RuntimeError("native ASan source result differs")
        probes.append({"case":label,"job_sha256":sha(file),"native_sha256":sha(out/f"{label}-native.json"),"asan_sha256":sha(out/f"{label}-asan.json")})
    if len(probes)!=2 or sources!=identity():raise RuntimeError("native probes missing or sources changed")
    report={"classification":"cpu_file_asset_loading_and_small_graph_equivalence_not_full_model_or_soc","sources":sources,"cases":cases,"native_probes":probes,
        "regression_sha256":sha(out/"regression.xml"),"load_only_report_sha256":sha(ordinary/"report.json"),"load_only_asan_report_sha256":sha(out/"asan-load-only/report.json"),
        "full_model_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"FILE_ASSET_SOURCE_VERIFIED {out/'report.json'}")


if __name__=="__main__":main()
