"""Audit complete GPU hardware measurements, never substitute a simulator result."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np

from scripts.benchmark_mlx_gpu_models import distribution, require
from scripts.mlx_system_attempt import record
from scripts.run_mlx_tensor_semantics import sha
from scripts.capture_mlx_bert_qa_reference import choose_span


def tensor_ids(value):
    if isinstance(value, dict):
        if "tensor_id" in value: yield value["tensor_id"]
        else:
            for child in value.values(): yield from tensor_ids(child)
    elif isinstance(value, list):
        for child in value: yield from tensor_ids(child)


def audit_samples(report, telemetry):
    samples = report["samples"]
    require(len(samples) == report["repetitions"] >= 20 and report["warmup"] >= 5, "insufficient measured repetitions or warmup")
    previous = 0
    for i, row in enumerate(samples):
        require(row["iteration"] == i and type(row["begin_monotonic_ns"]) is int and type(row["end_monotonic_ns"]) is int
                and previous <= row["begin_monotonic_ns"] < row["end_monotonic_ns"], "sample timestamps/order differ")
        previous = row["end_monotonic_ns"]
        for key in ("cuda_stream_interval_ms", "synchronized_graph_wall_ms", "task_wall_ms", "readback_postprocessing_wall_ms"):
            require(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] > 0, "invalid measured duration")
        require(len(row["forward_cuda_stream_ms"]) == 3 and all(math.isfinite(v) and v > 0 for v in row["forward_cuda_stream_ms"]), "GPU per-forward timing missing")
        require(abs(row["task_wall_ms"] - (row["end_monotonic_ns"] - row["begin_monotonic_ns"]) / 1e6) < 1e-6, "GPU task timing does not match actual timestamps")
        require(abs(row["task_wall_ms"] - row["synchronized_graph_wall_ms"] - row["readback_postprocessing_wall_ms"]) < 1e-6, "GPU task/model/postprocess scopes do not conserve")
        require(sum(row["forward_cuda_stream_ms"]) <= row["cuda_stream_interval_ms"] + .01, "per-forward stream ranges exceed their parent")
        require(row["cuda_stream_interval_ms"] <= row["synchronized_graph_wall_ms"] + .1, "GPU stream interval exceeds synchronized host interval")
    for key in ("cuda_stream_interval_ms", "synchronized_graph_wall_ms", "task_wall_ms"):
        require(distribution([r[key] for r in samples]) == report["statistics"][key], "GPU statistics differ from raw measurements")
    require(report["statistics"]["per_forward_cuda_stream_ms"] == [distribution([r["forward_cuda_stream_ms"][i] for r in samples]) for i in range(3)], "GPU forward statistics differ")
    require(not telemetry["errors"] and telemetry["locking"] is False, "telemetry failed or clocks were changed")
    overlapping = []
    for row in telemetry["samples"]:
        require(row["begin_monotonic_ns"] <= row["end_monotonic_ns"] and row["values"]["uuid"] == report["device"]["uuid"], "GPU telemetry time/identity differs")
        if row["phase"] == "measurement" and any(row["begin_monotonic_ns"] <= r["end_monotonic_ns"] and row["end_monotonic_ns"] >= r["begin_monotonic_ns"] for r in samples):
            require(float(row["values"]["clocks.current.sm"]) > 0 and float(row["values"]["clocks.current.memory"]) > 0, "invalid measured GPU clocks")
            overlapping.append(row)
    require(len(overlapping) >= 2, "insufficient telemetry during measured inference")
    return dict(measurement_samples=len(samples), overlapping_clock_samples=len(overlapping),
                sm_clock_mhz=sorted({float(r["values"]["clocks.current.sm"]) for r in overlapping}),
                memory_clock_mhz=sorted({float(r["values"]["clocks.current.memory"]) for r in overlapping}),
                throttle_masks=sorted({r["values"]["clocks_throttle_reasons.active"] for r in overlapping}))


def audit(directory, source_file):
    report = json.loads((directory / "result.json").read_text()); protocol = json.loads((directory / "protocol.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text()); correctness = json.loads((directory / "correctness.json").read_text())
    trace = json.loads((directory / "gpu-correctness-inventory.json").read_text()); source = json.loads(source_file.read_text())
    telemetry = json.loads((directory / "telemetry.json").read_text())
    for field, name in (("protocol_sha256", "protocol.json"), ("correctness_sha256", "correctness.json"), ("provenance_sha256", "provenance.json"), ("telemetry_sha256", "telemetry.json")):
        require(sha(directory / name) == report[field], "GPU measurement evidence changed")
    require(all(report.get(k) == v for k, v in protocol.items()), "GPU measured protocol differs from prebound protocol")
    require(protocol["input_inventory_sha256"] == sha(source_file), "GPU input inventory differs")
    for p, digest in {**provenance["sources"], **provenance["inputs"]}.items(): require(sha(Path(p)) == digest, "GPU source/input provenance changed")
    require(report["classification"] == "same_input_full_gpu_hardware_benchmark_not_mlx_simulation" and report["correctness_passed"]
            and not report["mlx_system_verified"] and not report["gpu_simulator_result"] and not report["cold_loading_measured"], "GPU hardware run relabelled as another scope")
    require(protocol["clock_locking"] is False and protocol["power_limit_changed"] is False and protocol["weights_and_inputs_resident"]
            and protocol["attention"] == "eager" and protocol["tf32"] is False and protocol["loaded_parameters_bitwise_checked"], "GPU execution conditions differ")
    require(correctness["passed"] and correctness["instrumentation_bitwise_equal"] and correctness["all_layers_verified"], "GPU reference/capture correctness missing")
    llama = report["model_family"] == "Llama2-7B"
    require(report["model_family"] == source["model_identity"]["family"] and report["model_variant"] == source["model_identity"]["variant"], "GPU model identity differs")
    require(source["model_identity"]["parameters"] == (6738415616 if llama else 108893186), "GPU model scope is not full")
    if llama:
        weights = json.loads((Path(source["model_identity"]["path"]) / "model.safetensors.index.json").read_text())["weight_map"]
        required = set(weights) - set(source["model_identity"]["ignored_nonpersistent_checkpoint_buffers"])
    else: required = set(source["model_identity"]["checkpoint_name_by_model_parameter"])
    require(len(required) == (291 if llama else 199), "GPU parameter count differs")
    require(len(trace["operations"]) == correctness["source_calls"], "GPU operator count differs")
    require(Counter((n["forward_id"], n["operator"]) for n in trace["operations"]) == Counter((n["forward_id"], n["operator"]) for n in source["operations"]), "GPU captured operator/forward coverage differs from full reference")
    for forward in range(3):
        used = {name for op in trace["operations"] if op["forward_id"] == forward for identifier in tensor_ids([op["inputs"], op["kwargs"]])
                for name in trace["tensors"][identifier]["parameter_or_buffer_names"]}
        require(required <= used, "GPU forward omitted a real model parameter")
        pattern = r"model\.layers\.\d+" if llama else r"bert\.encoder\.layer\.\d+"
        layers = [b["module_path"] for b in trace["boundaries"] if b["event"] == "enter" and b["forward_id"] == forward and re.fullmatch(pattern, b["module_path"])]
        require(len(layers) == len(set(layers)) == (32 if llama else 12), "GPU full layer coverage differs")
    roles = ("logits",) if llama else ("start_logits", "end_logits")
    expected_keys = {(f, role) for f in range(3) for role in roles}
    for rows in (correctness["comparison"], report["final_comparison"]):
        require(len(rows) == len(expected_keys) and {(r["forward_id"], r["role"]) for r in rows} == expected_keys, "GPU initial/final outputs missing")
        for row in rows:
            path = Path(row["file"]).resolve(); require(path.is_relative_to(directory.resolve()) and sha(path) == row["actual_sha256"], "GPU actual output changed or unowned")
            check = source["reference_checks"][row["forward_id"]]
            reference_path = Path(check["logits_file"] if llama else check["outputs"][row["role"]]["file"])
            require(sha(reference_path) == row["reference_sha256"], "GPU reference logits changed")
            expected_shape = check["logits_shape"] if llama else check["outputs"][row["role"]]["shape"]
            dtype = "<f2" if llama else "<f4"; width = 2 if llama else 4
            require(row["shape"] == expected_shape and path.stat().st_size == reference_path.stat().st_size == math.prod(expected_shape) * width, "GPU actual/reference output extent differs")
            got = np.fromfile(path, dtype=dtype).astype(np.float64); ref = np.fromfile(reference_path, dtype=dtype).astype(np.float64)
            require(np.isfinite(got).all() and np.isfinite(ref).all(), "GPU logits nonfinite")
            difference = np.abs(got-ref); failed = int((difference > protocol["atol"] + protocol["rtol"] * np.abs(ref)).sum())
            require(row["max_abs_error"] == float(difference.max()) and row["failed_elements"] == failed == 0, "GPU logit error report differs")
    initial = {(r["forward_id"], r["role"]): r for r in correctness["comparison"]}
    require(len(correctness["tasks"]) == 3, "GPU task results missing")
    for final in report["final_comparison"]:
        first = initial[(final["forward_id"], final["role"])]
        require(Path(final["file"]).read_bytes() == Path(first["file"]).read_bytes(), "GPU initial/final repeated logits differ")
    for i, (task, check) in enumerate(zip(correctness["tasks"], source["reference_checks"], strict=True)):
        if llama:
            logits = np.fromfile(initial[(i, "logits")]["file"], dtype="<f2")
            require(task["token"] == int(logits.argmax()) == check["token_id"] and task["kv_length"] == check["kv_len"], "GPU actual token/cache result differs")
        else:
            start = np.fromfile(initial[(i, "start_logits")]["file"], dtype="<f4").astype(np.float64).tolist()
            end = np.fromfile(initial[(i, "end_logits")]["file"], dtype="<f4").astype(np.float64).tolist()
            span = choose_span(start, end, check["context_mask"])
            span["text"] = check["context"][check["offsets"][span["start"]][0]:check["offsets"][span["end"]][1]]
            require(task == span and all(task[k] == check["span"][k] for k in ("start", "end", "text")), "GPU QA answer is not derived from actual logits")
    clocks = audit_samples(report, telemetry)
    return dict(classification="audited_full_gpu_measurement_not_mlx_or_gpu_simulator", model_family=report["model_family"],
                source_calls=len(trace["operations"]), actual_parameters_used_per_forward=len(required), full_layers_per_forward=32 if llama else 12,
                measurement=clocks, statistics=report["statistics"], result_sha256=sha(directory / "result.json"),
                sources_and_inputs_unchanged=True, measurements_recomputed=True, actual_task_outputs_recomputed=True,
                auditor_sha256=sha(Path(__file__)), mlx_system_verified=False, gpu_simulator_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "inventory", "output"): parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(); require(not args.output.exists(), "choose a fresh GPU audit report")
    result = audit(args.run.resolve(), args.inventory.resolve()); record(args.output, result)
    print(json.dumps(dict(output=str(args.output), model=result["model_family"], measurement=result["measurement"])))


if __name__ == "__main__": main()
