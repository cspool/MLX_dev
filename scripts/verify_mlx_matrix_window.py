"""Bind component tests and actual matrix-window outputs to source/build identity.

Does not certify full-model correctness, Chipyard integration or inference speed.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from mlxsim.model_matrix_reference import kasc_reference
from scripts.run_mlx_tensor_semantics import ROOT, sha, source_identity


def identity():
    result = source_identity()
    for name in ("src/mlxsim/model_matrix_reference.py", "src/mlxsim/model_execution_inventory.py", "tests/test_model_tensor_semantics.py", "tests/test_matrix_window_scheduler.py", "scripts/verify_mlx_matrix_window.py"):
        result[name] = sha(ROOT / name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("choose a fresh matrix-window verification directory")
    output.mkdir(parents=True)
    before = identity()
    with (output / "pytest.log").open("w") as stream:
        result = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "tests/test_matrix_window_scheduler.py", f"--basetemp={output / 'pytest'}", f"--junitxml={output / 'regression.xml'}"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(f"matrix window regression failed: {output / 'pytest.log'}")
    cases = []
    for file in sorted((output / "pytest").rglob("out/result.json")):
        if file.is_symlink() or any(p.is_symlink() for p in file.parents if p != output):
            continue
        job_path = file.parent.parent / "job.json"
        if not job_path.exists():
            continue
        job, native = json.loads(job_path.read_text()), json.loads(file.read_text())
        dtype = np.float16 if job["program"]["input_dtype"] == "f16" else np.float32
        output_dtype = np.float16 if job["program"]["output_dtype"] == "f16" else np.float32
        a, b = np.asarray(job["a"]["values"], dtype=dtype).reshape(job["a"]["shape"]), np.asarray(job["b"]["values"], dtype=dtype).reshape(job["b"]["shape"])
        bias = np.asarray(job["bias"]["values"], dtype=dtype) if "bias" in job else None
        expected = kasc_reference(a, b, transposed_b=job.get("transposed_b", True), bias=bias, output_dtype=output_dtype)
        actual = np.fromfile(file.parent / "output.bin", dtype=output_dtype).reshape(job["m"], job["n"])
        if actual.tobytes() != expected.tobytes() or not native["done"] or native["dma_requests"] != native["dma_responses"]:
            raise RuntimeError(f"matrix-window data/drain conformance failed: {file}")
        cases.append({"case": str(file.parent.parent.relative_to(output)), "shape_mnk": [job["m"], job["n"], job["k"]], "bitwise_equal": True, "cycles": native["cycles"], "same_pe_context_overlap_cycles": native["same_pe_context_overlap_cycles"], "peak_contexts": native["peak_contexts"], "peak_spm_vectors": native["peak_spm_vectors"], "dma_requests": native["dma_requests"], "dma_read_bytes": native["dma_read_bytes"], "dma_write_bytes": native["dma_write_bytes"], "report_sha256": sha(file), "job_sha256": sha(job_path), "output_sha256": sha(file.parent / "output.bin")})
    if len(cases) < 10 or before != identity():
        raise RuntimeError("missing matrix-window cases or sources changed during verification")
    binary = ROOT / "build/mlx-matrix-window/mlx-matrix-window"
    graph_binary = ROOT / "build/mlx-matrix-window/tensor-model/mlx-tensor-semantics"
    report = {"classification": "matrix_window_component_validated_not_full_model_or_chipyard", "sources": before, "binary_sha256": sha(binary), "tensor_graph_binary_sha256": sha(graph_binary), "cases": cases, "regression_xml_sha256": sha(output / "regression.xml"), "mlx_system_verified": False, "inference_performance_eligible": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"MATRIX_WINDOW_COMPONENT_VERIFIED {output / 'report.json'}")


if __name__ == "__main__":
    main()
