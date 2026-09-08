"""Bind vector cycle/port component regressions to implementation identity."""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.run_mlx_tensor_semantics import ROOT,sha,source_identity


def identity():
    result=source_identity()
    for name in ("tests/test_vector_window_scheduler.py","tests/vector_external_memory.cc","tests/test_model_memory_program.py","tests/test_model_tensor_semantics.py","scripts/verify_mlx_vector_window.py"):
        result[name]=sha(ROOT/name)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh vector-window verification directory")
    out.mkdir(parents=True);before=identity()
    with (out/"pytest.log").open("w") as stream:
        result=subprocess.run([str(ROOT/".venv/bin/python"),"-m","pytest","-q","tests/test_vector_window_scheduler.py","tests/test_model_memory_program.py",f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
    if result.returncode:raise RuntimeError(f"vector-window regressions failed: {out/'pytest.log'}")
    cases=[]
    for file in sorted((out/"pytest").rglob("out/result.json")):
        if any(parent.is_symlink() for parent in file.parents):continue
        report=json.loads(file.read_text())
        if report.get("classification")!="cpp_vector_window_cycle_component_not_full_system_validation":continue
        job=file.parent.parent/"job.json"
        if not report["done"] or report["dma_requests"]!=report["dma_responses"] or report["active_contexts"] or report["allocated_spm_vectors"]:raise RuntimeError(f"vector execution did not drain: {file}")
        cases.append({"case":str(file.parent.parent.relative_to(out)),"cycles":report["cycles"],"same_pe_context_overlap_cycles":report["same_pe_context_overlap_cycles"],"vector_sfu_overlap_pe_cycles":report["vector_sfu_overlap_pe_cycles"],"peak_contexts":report["peak_contexts"],"peak_spm_vectors":report["peak_spm_vectors"],"dma_requests":report["dma_requests"],"external_memory_port":report["external_memory_port"],"report_sha256":sha(file),"job_sha256":sha(job),"output_sha256":sha(file.parent/"output.bin")})
    if len(cases)<30 or not any(case["external_memory_port"] for case in cases) or before!=identity():raise RuntimeError("missing vector cases/external port proof or sources changed")
    binaries=[ROOT/"build/mlx-vector-window/mlx-vector-window",ROOT/"build/mlx-vector-window/vector-external-memory",ROOT/"build/mlx-model-tensor/mlx-tensor-semantics"]
    result={"classification":"vector_window_cycle_and_port_component_not_model_system_validation","sources":before,"binaries":{str(path):sha(path) for path in binaries},"cases":cases,"regression_xml_sha256":sha(out/"regression.xml"),"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(f"VECTOR_WINDOW_COMPONENT_VERIFIED {out/'report.json'}")


if __name__=="__main__":main()
