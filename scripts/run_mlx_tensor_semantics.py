#!/usr/bin/env python3
"""Compile and execute full-model tensor semantics in C++, without timing claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import scipy

from mlxsim.model_tensor_compiler import compile_inventory

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/mlx-model-tensor"
LOGIT_ATOL = 5e-3
LOGIT_RTOL = 5e-3


def source_identity():
    files = [Path(__file__).resolve(), ROOT / "src/mlxsim/model_gelu_program.py", ROOT / "src/mlxsim/model_composites.py", ROOT / "src/mlxsim/model_source_groups.py", ROOT / "src/mlxsim/model_value_outputs.py", ROOT / "src/mlxsim/model_tensor_compiler.py", ROOT / "src/mlxsim/model_matrix_program.py", ROOT / "src/mlxsim/model_vector_program.py", ROOT / "src/mlxsim/model_dtype_lowering.py", ROOT / "src/mlxsim/model_memory_program.py", ROOT / "src/mlxsim/model_control_program.py", ROOT / "src/mlxsim/model_block_pipeline.py"]
    for directory in (ROOT / "simulator_ext/tensor_model", ROOT / "simulator_ext/tagged", ROOT / "simulator_ext/matrix_schedule", ROOT / "simulator_ext/vector_model", ROOT / "simulator_ext/vector_schedule", ROOT / "simulator_ext/model_io", ROOT / "simulator_ext/memory_model", ROOT / "simulator_ext/control_model", ROOT / "simulator_ext/control_schedule", ROOT / "simulator_ext/shared_array", ROOT / "simulator_ext/model_events"):
        files.extend(path for path in directory.iterdir() if path.suffix in {".cc", ".h"} or path.name == "CMakeLists.txt")
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}


def compare_logits(actual, reference):
    shape = reference["logits_shape"]
    if actual["shape"] != shape or sha(Path(reference["logits_file"])) != reference["logits_sha256"]:
        raise RuntimeError("reference output identity/shape mismatch")
    ref = np.fromfile(reference["logits_file"], dtype=np.float16).astype(np.float32)
    got = np.fromfile(actual["logits_file"], dtype={"f16": np.float16, "f32": np.float32}[actual["dtype"]]).astype(np.float32)
    if ref.size != np.prod(shape) or got.shape != ref.shape or not ref.size:
        raise RuntimeError("native/reference logits byte count mismatch")
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise RuntimeError("nonfinite model logits cannot pass the numerical gate")
    error = np.abs(got - ref)
    failed = error > LOGIT_ATOL + LOGIT_RTOL * np.abs(ref)
    return {
        "forward_id": actual["forward_id"], "max_abs_error": float(error.max()),
        "rmse": float(np.sqrt(np.mean(error**2))), "rtol": LOGIT_RTOL, "atol": LOGIT_ATOL,
        "within_tolerance": not bool(failed.any()), "failed_elements": int(failed.sum()),
        "total_elements": int(ref.size), "tokens_equal": actual["tokens"] == [reference["token_id"]],
        "tokens": actual["tokens"], "actual_logits_sha256": sha(Path(actual["logits_file"])),
        "reference_logits_sha256": reference["logits_sha256"],
    }


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def cpu_blas():
    candidates = list(
        (Path(scipy.__file__).parent.parent / "scipy.libs").glob("libscipy_openblas-*.so")
    )
    if len(candidates) != 1:
        raise RuntimeError("expected exactly one explicit LP64 CPU OpenBLAS library")
    return candidates[0].resolve()


def execute(command, log, timeout=600):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False
        )
    if result.returncode:
        raise RuntimeError(f"tensor semantics command failed: {log}")


def build():
    execute(
        [
            "cmake",
            "-S",
            str(ROOT / "simulator_ext/tensor_model"),
            "-B",
            str(BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        BUILD / "configure.log",
    )
    execute(["cmake", "--build", str(BUILD), "-j4"], BUILD / "build.log")
    return BUILD / "mlx-tensor-semantics"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--observe-operators", type=Path)
    parser.add_argument("--matrix-backend", choices=("blas", "microcode", "scheduled"), default="blas")
    parser.add_argument("--vector-backend", choices=("functional", "microcode", "scheduled"), default="functional")
    parser.add_argument("--vector-schedule-options", type=Path)
    parser.add_argument("--memory-backend", choices=("functional", "planned", "scheduled"), default="functional")
    parser.add_argument("--memory-schedule-options", type=Path)
    parser.add_argument("--control-backend", choices=("functional", "rv64_leaf", "scheduled"), default="functional")
    parser.add_argument("--control-schedule-options", type=Path)
    parser.add_argument("--schedule-options", type=Path, help="explicit JSON timing/ready options for scheduled matrices only")
    args = parser.parse_args()
    if not 1 <= args.threads <= 256:
        raise RuntimeError("native BLAS threads must be in [1, 256]")
    sources = source_identity()
    inventory_hash = sha(args.inventory)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("choose a fresh tensor execution directory")
    output.mkdir(parents=True)
    inventory = json.loads(args.inventory.read_text())
    program, compiler = compile_inventory(inventory, matrix_backend=args.matrix_backend,
                                          schedule_options=json.loads(args.schedule_options.read_text()) if args.schedule_options else None,
                                          vector_backend=args.vector_backend,
                                          vector_schedule_options=json.loads(args.vector_schedule_options.read_text()) if args.vector_schedule_options else None,
                                          memory_backend=args.memory_backend,
                                          memory_schedule_options=json.loads(args.memory_schedule_options.read_text()) if args.memory_schedule_options else None,
                                          control_backend=args.control_backend,
                                          control_schedule_options=json.loads(args.control_schedule_options.read_text()) if args.control_schedule_options else None)
    (output / "program.json").write_text(json.dumps(program, indent=2) + "\n")
    (output / "compilation.json").write_text(json.dumps(compiler, indent=2) + "\n")
    if args.compile_only:
        print(f"NATIVE_SEMANTIC_COMPILATION_COMPLETE source_calls={len(program['nodes'])}")
        return
    built = build()
    # Execute an attempt-owned binary, so a later build cannot relabel this run.
    binary = output / "mlx-tensor-semantics"
    shutil.copy2(built, binary)
    binary_hash = sha(binary)
    print("Checking full weight identities before native execution", flush=True)
    for path, spec in inventory["model_identity"]["files"].items():
        if sha(Path(path)) != spec["sha256"]:
            raise RuntimeError(f"model asset changed: {path}")
    blas = cpu_blas() if args.matrix_backend == "blas" else None
    blas_hash = sha(blas) if blas else None
    program_hash = sha(output / "program.json")
    print("Running compiled full model in native tensor semantics (no MLX timing)", flush=True)
    execute(
        [
            str(binary),
            str(output / "program.json"),
            str(output / "native"),
            str(blas) if blas else "none",
            str(args.threads),
        ] + ([str(args.observe_operators.resolve())] if args.observe_operators else []),
        output / "native.log",
        timeout=7200,
    )
    native = json.loads((output / "native/result.json").read_text())
    if native["vector_microcode"]["calls"] != sum("vector_program" in node for node in program["nodes"]):
        raise RuntimeError("not every compiled vector microprogram executed")
    if native["memory_programs"]["calls"] != sum("memory_program" in node for node in program["nodes"]):
        raise RuntimeError("not every compiled memory plan executed")
    if native["control_programs"]["calls"] != sum("control_program" in node for node in program["nodes"]):
        raise RuntimeError("not every compiled controller program executed")
    if args.control_backend == "scheduled":
        windows = native.get("control_windows", [])
        control_ids = [node["source_operator_id"] for node in program["nodes"] if "control_program" in node]
        if [w["source_operator_id"] for w in windows] != control_ids or not all(w["done"] and w["dma_requests"] == w["dma_responses"] for w in windows):
            raise RuntimeError("controller cycle windows did not cover and drain every control operator")
    all_lowered = all(any(field in node for field in ("matrix_program", "vector_program", "memory_program", "control_program")) for node in program["nodes"])
    if all_lowered and native["functional_entry_calls"]:
        raise RuntimeError("fully lowered graph entered an unlowered functional helper")
    if args.matrix_backend != "blas" and (native["blas_calls"] or native["matrix_microcode"]["mul_active_lanes"] != native["matrix_macs"]):
        raise RuntimeError("microcode execution did not perform every matrix MAC without BLAS")
    if native["executed_source_calls"] != len(program["nodes"]) or [
        (event["source_operator_id"], event["entry"]) for event in native["events"]
    ] != [(node["source_operator_id"], f"tensor_model::{node['kind']}") for node in program["nodes"]]:
        raise RuntimeError("not every compiled operator executed")
    comparisons = []
    expected = {check["forward_id"]: check for check in inventory["reference_checks"]}
    if not expected or len(native["outputs"]) != len(expected) or {x["forward_id"] for x in native["outputs"]} != set(expected):
        raise RuntimeError("missing or duplicate model forward output")
    for actual in native["outputs"]:
        reference = expected[actual["forward_id"]]
        # Frozen before execution: allow legal FP32 reduction-order differences
        # under FP16 model I/O, but require the free-running greedy tokens to match.
        comparisons.append(compare_logits(actual, reference))
    if sources != source_identity() or inventory_hash != sha(args.inventory) or binary_hash != sha(binary) or (blas and blas_hash != sha(blas)) or program_hash != sha(output / "program.json"):
        raise RuntimeError("source/program runtime identity changed during execution")
    for path, spec in inventory["model_identity"]["files"].items():
        if sha(Path(path)) != spec["sha256"]:
            raise RuntimeError(f"model asset changed during execution: {path}")
    report = {
        "classification": "full_model_native_semantics_not_mlx_system_validation",
        "comparison": comparisons,
        "all_comparisons_passed": all(
            c["within_tolerance"] and c["tokens_equal"] for c in comparisons
        ),
        "compiled_source_calls": len(program["nodes"]),
        "executed_source_calls": native["executed_source_calls"],
        "inventory_sha256": inventory_hash,
        "sources": sources,
        "program_sha256": program_hash,
        "binary_sha256": binary_hash,
        "cpu_blas": {"path": str(blas), "sha256": blas_hash, "threads": args.threads} if blas else None,
        "matrix_backend": args.matrix_backend,
        "matrix_microcode": native["matrix_microcode"],
        "vector_backend": args.vector_backend,
        "vector_microcode": native["vector_microcode"],
        "memory_backend": args.memory_backend,
        "memory_programs": native["memory_programs"],
        "control_backend": args.control_backend,
        "control_programs": native["control_programs"],
        "all_source_calls_lowered": all_lowered,
        "functional_entry_calls": native["functional_entry_calls"],
        "blas_calls": native["blas_calls"],
        "mlx_system_verified": False,
        "inference_performance_eligible": False,
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_comparisons_passed"]:
        raise RuntimeError(f"full-model numerical comparison failed: {output / 'comparison.json'}")
    print(
        f"FULL_MODEL_NATIVE_SEMANTICS_PASS (not system timing) {output / 'comparison.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
