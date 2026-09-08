"""Compare output-only observations; never returns data to target execution."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_mlx_tensor_semantics import sha, compare_logits

DTYPES = {"f16": np.float16, "f32": np.float32, "i64": np.int64, "bool": np.bool_, "torch.float16": np.float16, "torch.float32": np.float32, "torch.int64": np.int64, "torch.bool": np.bool_}


def read_value(spec, *, check_hash=False):
    path = Path(spec["file"])
    if check_hash and sha(path) != spec["sha256"]:
        raise RuntimeError("diagnostic output hash mismatch")
    value = np.fromfile(path, dtype=DTYPES[spec["dtype"]])
    if value.size != np.prod(spec["shape"]):
        raise RuntimeError("diagnostic output extent mismatch")
    return value.reshape(spec["shape"])


def compare(reference_file, native_file, program_file, linear_ids=()):
    reference = json.loads(reference_file.read_text())
    native = json.loads(native_file.read_text())
    program = json.loads(program_file.read_text())
    refs = {row["value_id"]: row for row in reference["observations"]}
    actuals = {row["value_id"]: row for row in native["observations"]}
    if set(refs) != set(actuals):
        raise RuntimeError("reference/native observation ID mismatch")
    rows = []
    for identifier, source in refs.items():
        target = actuals[identifier]
        if source["shape"] != target["shape"] or DTYPES[source["dtype"]] != DTYPES[target["dtype"]]:
            raise RuntimeError("reference/native observation type mismatch")
        a, b = read_value(source, check_hash=True), read_value(target)
        changed = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        finite = np.isfinite(a) & np.isfinite(b)
        error = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
        rows.append({"value_id": identifier, "source_operator": source["source_operator"], "forward_id": source["forward_id"], "layer_idx": source["layer_idx"], "dtype": source["dtype"], "changed_elements": int(changed.sum()), "elements": int(a.size), "max_abs_finite_error": float(error.max()) if error.size else 0, "rmse_finite": float(np.sqrt(np.mean(error**2))) if error.size else 0, "nonfinite_mismatch": int((changed & ~finite).sum()), "native_sha256": sha(Path(target["file"]))})
    probes = []
    for operator_id in linear_ids:
        node = program["nodes"][operator_id]
        if node["kind"] != "linear" or len(node["args"]) != 2:
            raise RuntimeError("high-precision diagnostic probe supports only bias-free linear")
        input_id, weight_id = [arg["value"] for arg in node["args"]]
        asset = program["assets"][weight_id]
        if asset["kind"] != "mapped_file" or asset["dtype"] != "f16" or asset["bytes"] > 512 * 1024 * 1024:
            raise RuntimeError("high-precision probe requires bounded FP16 weight input")
        # Isolated diagnosis uses saved reference inputs, ONLY inside this
        # comparator. It is not native full-model execution or a new reference.
        x = read_value(refs[input_id], check_hash=True)
        target_x = read_value(actuals[input_id])
        if not np.array_equal(x, target_x):
            raise RuntimeError("linear probe inputs already differ; cannot isolate GEMM")
        weight = np.memmap(asset["path"], mode="r", dtype=np.float16, offset=asset["byte_offset"], shape=tuple(asset["shape"]))
        high = (x.astype(np.float64) @ weight.astype(np.float64).T).astype(np.float16)
        a, b = read_value(refs[node["id"]], check_hash=True), read_value(actuals[node["id"]])
        probes.append({"source_operator_id": operator_id, "reference_input_equals_native": True, "reference_vs_fp64_rounded_changed": int((a != high).sum()), "native_vs_fp64_rounded_changed": int((b != high).sum()), "elements": int(high.size), "weight_parameter": asset["parameter_name"]})
    return {"classification": "output_only_numerical_diagnosis_not_acceptance", "reference_sha256": sha(reference_file), "native_report_sha256": sha(native_file), "program_sha256": sha(program_file), "observations": rows, "isolated_linear_probes": probes, "mlx_system_verified": False, "inference_performance_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-linear-ids", type=int, nargs="*", default=[])
    parser.add_argument("--model-reference", type=Path, help="optional full-model reference inventory for the same diagnostic comparison")
    parser.add_argument("--other-reference", type=Path, help="optional independent reference inventory; compared to model-reference, never used for execution")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("choose a fresh diagnostic report path")
    result = compare(args.reference, args.native, args.program, args.probe_linear_ids)
    if args.model_reference:
        primary = json.loads(args.model_reference.read_text())
        checks = {row["forward_id"]: row for row in primary["reference_checks"]}
        actual = json.loads(args.native.read_text())["outputs"]
        result["full_model_comparison"] = [compare_logits(row, checks[row["forward_id"]]) for row in actual]
        result["model_reference_sha256"] = sha(args.model_reference)
        if args.other_reference:
            other = json.loads(args.other_reference.read_text())
            result["independent_reference_comparison"] = [compare_logits({"forward_id": row["forward_id"], "shape": row["logits_shape"], "dtype": "f16", "logits_file": row["logits_file"], "tokens": [row["token_id"]]}, checks[row["forward_id"]]) for row in other["reference_checks"]]
            result["other_reference_sha256"] = sha(args.other_reference)
    elif args.other_reference:
        raise RuntimeError("other-reference requires model-reference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"OUTPUT_ONLY_DIAGNOSIS_COMPLETE {args.output}")


if __name__ == "__main__":
    main()
