"""Validate shared matrix/vector memory contracts and physical binding adapters."""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.run_mlx_tensor_semantics import ROOT,sha,source_identity


def identity():
    result=source_identity()
    for name in ("tests/matrix_external_memory.cc","tests/model_io_contract.cc","tests/vector_external_memory.cc","tests/test_matrix_external_memory.py","tests/test_matrix_window_scheduler.py","tests/test_vector_window_scheduler.py","tests/test_model_tensor_semantics.py","scripts/verify_mlx_model_io.py"):
        result[name]=sha(ROOT/name)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh model-I/O validation directory")
    out.mkdir(parents=True);before=identity()
    with (out/"pytest.log").open("w") as stream:
        result=subprocess.run([str(ROOT/".venv/bin/python"),"-m","pytest","-q","tests/test_matrix_external_memory.py","tests/test_matrix_window_scheduler.py","tests/test_vector_window_scheduler.py",f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
    if result.returncode:raise RuntimeError(f"model-I/O validation failed: {out/'pytest.log'}")
    cases=[]
    for file in sorted((out/"pytest").rglob("out/result.json")):
        if any(p.is_symlink() for p in file.parents):continue
        report=json.loads(file.read_text())
        if not report.get("external_memory_port"):continue
        if not report["done"] or report["dma_requests"]!=report["dma_responses"] or not report["virtual_tensor_backing_used"]:raise RuntimeError("external memory execution did not drain or used local tensor data")
        job=file.parent.parent/"job.json"
        cases.append({"case":str(file.parent.parent.relative_to(out)),"cycles":report["cycles"],"dma_requests":report["dma_requests"],"virtual_tensor_backing_used":True,"physical_requests":len(report.get("physical_requests",[])),"physical_commits":report.get("physical_commits"),"transport":report.get("transport"),"output_sha256":sha(file.parent/"output.bin"),"report_sha256":sha(file),"job_sha256":sha(job)})
    if len(cases)<9 or not any(c["transport"] and c["transport"]["nacks"]>0 for c in cases) or before!=identity():raise RuntimeError("missing external/transport cases or source identity changed")
    binaries=[ROOT/"build/mlx-matrix-window/matrix-external-memory",ROOT/"build/mlx-matrix-window/model-io-contract",ROOT/"build/mlx-vector-window/vector-external-memory"]
    report={"classification":"shared_memory_contract_and_address_binding_not_chipyard_integration","sources":before,"cases":cases,"binaries":{str(p):sha(p) for p in binaries},"regression_xml_sha256":sha(out/"regression.xml"),"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"MODEL_IO_COMPONENT_VERIFIED {out/'report.json'}")


if __name__=="__main__":main()
