"""Full-shape isolated matrix checks against independent K-ascending arithmetic.

These comparisons are diagnostics, NOT a replacement for the model logits gate.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mlxsim.model_matrix_reference import kasc_reference
from scripts.run_mlx_tensor_semantics import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--operator-ids", nargs="+", type=int, default=[50, 53, 56])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("choose a fresh numerical conformance report path")
    program_file, native_file = args.run / "program.json", args.run / "native/result.json"
    program, native = json.loads(program_file.read_text()), json.loads(native_file.read_text())
    if program["matrix_backend"] != "microcode" or native["blas_calls"] != 0:
        raise RuntimeError("expected a completed matrix-microcode run with BLAS disabled")
    observations = {row["value_id"]: row for row in native["observations"]}
    rows = []
    def load_observation(identifier):
        spec = observations[identifier]
        return np.fromfile(spec["file"], dtype={"f16": np.float16, "f32": np.float32}[spec["dtype"]]).reshape(spec["shape"])
    for operator_id in args.operator_ids:
        node = program["nodes"][operator_id]
        if node["kind"] != "linear" or node["matrix_program"]["has_bias"]:
            raise RuntimeError("initial full-shape conformance entry requires bias-free linear")
        lhs, rhs = [arg["value"] for arg in node["args"]]
        asset = program["assets"][rhs]
        x = load_observation(lhs)
        weight = np.memmap(asset["path"], dtype={"f16": np.float16, "f32": np.float32}[asset["dtype"]], mode="r", offset=asset["byte_offset"], shape=tuple(asset["shape"]))
        expected = kasc_reference(x, weight, transposed_b=True, output_dtype={"f16": np.float16, "f32": np.float32}[node["output"]["dtype"]])
        actual = load_observation(node["id"])
        unsigned = np.uint16 if actual.dtype == np.float16 else np.uint32
        mismatch = actual.view(unsigned) != expected.view(unsigned)
        rows.append({"source_operator_id": operator_id, "module_path": node["module_path"], "input_shape": list(x.shape), "weight_shape": list(weight.shape), "output_shape": list(actual.shape), "bitwise_mismatches": int(mismatch.sum()), "elements": int(actual.size), "passed": not bool(mismatch.any()), "input_sha256": sha(Path(observations[lhs]["file"])), "actual_sha256": sha(Path(observations[node["id"]]["file"])), "reference_input_origin": "target_output_only_observation_not_injected_into_execution"})
    report = {"classification": "isolated_full_shape_matrix_numeric_conformance_not_model_acceptance", "profile": "mlx-matrix-f32-kasc-v1", "checks": rows, "all_passed": bool(rows) and all(row["passed"] for row in rows), "native_report_sha256": sha(native_file), "program_sha256": sha(program_file), "reference_source_sha256": sha(Path(__file__).resolve().parents[1] / "src/mlxsim/model_matrix_reference.py"), "mlx_system_verified": False, "inference_performance_eligible": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_passed"]:
        raise RuntimeError("matrix primitive numeric conformance failed")
    print(f"ISOLATED_MATRIX_NUMERIC_CONFORMANCE_PASS {args.output}")


if __name__ == "__main__":
    main()
