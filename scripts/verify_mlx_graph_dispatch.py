"""Validate generated RV64 graph dispatch and separately audit a full-model plan."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.run_mlx_spike_graph import ROOT
from scripts.verify_mlx_spike_matrix_chain import source_identity as bridge_sources


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--program",type=Path,required=True);parser.add_argument("--lifetimes",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh generic graph validation directory")
    out.mkdir(parents=True)
    def identity():
        sources=bridge_sources()
        for path in (Path(__file__).resolve(),ROOT/"scripts/run_mlx_spike_graph.py",ROOT/"tests/test_spike_graph_runtime.py",ROOT/"tests/test_model_memory_program.py",ROOT/"tests/test_model_tensor_semantics.py"):
            sources[str(path.relative_to(ROOT))]=sha(path)
        return sources
    sources=identity();inputs={str(p.resolve()):sha(p) for p in (args.program,args.lifetimes)}
    with (out/"pytest.log").open("w") as log:process=subprocess.run([str(ROOT/".venv/bin/python"),"-m","pytest","-q","tests/test_spike_graph_runtime.py",f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if process.returncode:raise RuntimeError("generic graph dispatch tests failed")
    with (out/"plan.log").open("w") as log:subprocess.run([str(ROOT/".venv/bin/python"),"-m","scripts.run_mlx_spike_graph","--plan-only","--program",str(args.program.resolve()),"--lifetimes",str(args.lifetimes.resolve()),"--device-bytes","17179934720","--output",str(out/"full-model-plan")],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    cases=[]
    for path in sorted((out/"pytest").rglob("run/report.json")):
        if any(p.is_symlink() for p in path.parents):continue
        report=json.loads(path.read_text())
        if not report["actual_cpu_dispatch"] or not report["output_bytes_checked_in_elf"] or report["source_calls"]!=45 or report["device_windows"]!=21:raise RuntimeError("generic graph run did not execute all planned work")
        cases.append({"case":str(path.parent.relative_to(out)),"report_sha256":sha(path),"elf_sha256":report["elf_sha256"],"payload_sha256":report["payload_sha256"],"source_calls":report["source_calls"],"device_windows":report["device_windows"]})
    if len(cases)!=2:raise RuntimeError("missing normal/perturbed generic graph executions")
    full=json.loads((out/"full-model-plan/report.json").read_text())
    if sources!=identity() or any(sha(Path(p))!=digest for p,digest in inputs.items()):raise RuntimeError("graph validation sources/inputs changed")
    result={"classification":"generic_cpu_graph_dispatch_plus_full_model_plan_not_full_model_execution","sources":sources,"inputs":inputs,"cases":cases,"regression_sha256":sha(out/"regression.xml"),"full_plan_report_sha256":sha(out/"full-model-plan/report.json"),
        "full_model_source_calls":full["source_calls"],"full_model_tasks":full["task_count"],"full_model_command_bytes":full["command_bytes"],"full_model_execution_verified":False,"rocket_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n");print(f"GENERIC_GRAPH_DISPATCH_VERIFIED {out/'report.json'}")


if __name__=="__main__":main()
