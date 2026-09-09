"""Full dense BERT QA in native C/C++; no reference activations enter execution."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from mlxsim.model_block_pipeline import compile_block_pipelines, references
from mlxsim.model_result_contract import verify_qa_result, validate_result_contract
from mlxsim.model_source_groups import verify_source_groups
from mlxsim.model_physical_evidence import verify_physical_execution
from mlxsim.model_ready_evidence import verify_ready_execution
from mlxsim.model_tensor_compiler import compile_inventory, safetensors_header
from scripts.mlx_system_attempt import record, run_process, snapshot_sources, linked_libraries
from scripts.run_mlx_tensor_semantics import ROOT, source_identity, sha

ATOL = 5e-3
RTOL = 5e-3


def require(condition, message):
    if not condition: raise RuntimeError(message)


def sources():
    result = source_identity()
    for directory in ("simulator_ext/model_system", "simulator_ext/model_storage"):
        for p in (ROOT / directory).iterdir():
            if p.suffix in {".cc", ".h"} or p.name == "CMakeLists.txt": result[str(p.relative_to(ROOT))] = sha(p)
    for name in ("scripts/run_mlx_qa_model.py", "scripts/mlx_system_attempt.py", "src/mlxsim/model_physical_evidence.py",
                 "src/mlxsim/model_ready_evidence.py", "src/mlxsim/model_shared_cycle_policy.py"):
        p = ROOT / name
        if p.exists(): result[name] = sha(p)
    return result


def full_bert_contract(inventory, program):
    model = inventory["model_identity"]
    require(model["family"] == "BERT-base-QA" and model["variant"] == "local_dense_baseline_not_paper_hybrid"
            and model["parameters"] == 108893186 and model["parameter_tensors"] == 199 and model["all_parameters_loaded"]
            and model["checkpoint_values_changed"] is False, "full BERT scope requires the unchanged complete dense checkpoint")
    for key, expected in dict(num_hidden_layers=12, hidden_size=768, intermediate_size=3072, num_attention_heads=12, vocab_size=30522).items():
        require(model["config"][key] == expected, "BERT topology changed or shrank")
    require(inventory["instrumentation_equivalence_passed"] and inventory["runtime"]["dtype"] == "torch.float32", "BERT reference precision/equivalence differs")
    weights = Path(model["path"]) / "model.safetensors"
    header, begin = safetensors_header(weights)
    required = set(header) - {"__metadata__"}
    mapped = [a for a in program["assets"].values() if a["kind"] == "mapped_file"]
    require(len(mapped) == len(required) == 199 and {a["parameter_name"] for a in mapped} == required, "BERT parameter bindings incomplete")
    elements = 0
    for asset in mapped:
        entry = header[asset["parameter_name"]]; first, end = entry["data_offsets"]
        require(Path(asset["path"]).resolve() == weights.resolve() and entry["dtype"] == "F32" and asset["dtype"] == "f32"
                and asset["shape"] == entry["shape"] and asset["byte_offset"] == begin + first
                and asset["bytes"] == end - first == math.prod(entry["shape"]) * 4 and begin + end <= weights.stat().st_size,
                "BERT asset differs from actual checkpoint range")
        elements += math.prod(entry["shape"])
    require(elements == 108893186, "BERT parameter extent count differs")
    checks = inventory["reference_checks"]
    require([c["forward_id"] for c in checks] == [0, 1, 2] and [len(c["context_mask"]) for c in checks] == [28, 64, 64], "full BERT validation needs all normal/padded/changed-context cases")
    require(checks[0]["context"] == checks[1]["context"] != checks[2]["context"], "BERT input perturbation is missing")
    for check in checks:
        forward = check["forward_id"]
        layers = [b["module_path"] for b in inventory["boundaries"] if b["event"] == "enter" and b["forward_id"] == forward and re.fullmatch(r"bert\.encoder\.layer\.\d+", b["module_path"])]
        require(len(layers) == 12 and set(layers) == {f"bert.encoder.layer.{i}" for i in range(12)}, "not all BERT layers executed")
        used = set()
        for node in program["nodes"]:
            if node["forward_id"] != forward: continue
            for value in references([node["args"], node.get("kwargs", {})]):
                asset = program["assets"].get(value, {})
                if asset.get("kind") == "mapped_file": used.add(asset["parameter_name"])
        require(used == required, "BERT forward does not consume every actual parameter")
        require(all(o["instrumentation_bitwise_equal"] for o in check["outputs"].values()), "QA reference outputs changed under capture")
    require(len(program["source_groups"]) == len(inventory["operations"]) == 1141, "BERT source coverage incomplete")
    require(all(sum(f + "_program" in n for f in ("matrix", "vector", "memory", "control")) == 1 for n in program["nodes"]), "BERT lowering contains a missing/ambiguous backend")
    validate_result_contract(program)
    return dict(parameters=elements, parameter_tensors=len(required), source_calls=1141, lowered_calls=len(program["nodes"]),
                forwards=3, full_layers_per_forward=12, model_variant=model["variant"], checkpoint_extents_verified=True)


def audit_outputs(program, inventory, native):
    verify_source_groups(program, native)
    require(native["executed_source_calls"] == 1141 and native["executed_lowered_calls"] == len(program["nodes"]), "full BERT execution incomplete")
    require(all(native[k] == 0 for k in ("functional_entry_calls", "blas_calls", "python_or_gpu_execution_fallbacks")), "BERT used an uncompiled numerical fallback")
    events = native["events"]
    require(len(events) == len(program["nodes"]), "BERT executed event count differs")
    observed = {e["source_operator_id"]: e for e in events}
    require(len(observed) == len(events), "BERT executed duplicate lowered source")
    for n in program["nodes"]:
        e = observed.get(n["source_operator_id"], {})
        route_matches = e.get("entry") == "tensor_model::" + n["kind"] if "entry" in e else e.get("kind") == n["kind"] and e.get("family", "invalid") + "_program" in n
        require(route_matches and e.get("forward_id") == n["forward_id"], "BERT event/backend identity differs")
    require([r["forward_id"] for r in native["outputs"]] == [0, 1, 2], "BERT outputs missing or reordered")
    comparisons = []
    for spec, check, actual in zip(program["outputs"], inventory["reference_checks"], native["outputs"], strict=True):
        verify_qa_result(program, spec, actual)
        row = dict(forward_id=spec["forward_id"], outputs={}, actual_span=actual["span"], reference_span=check["span"],
                   answer_equal=all(actual["span"][k] == check["span"][k] for k in ("start", "end", "text")))
        for role in ("start_logits", "end_logits"):
            a = actual["outputs"][role]; b = check["outputs"][role]
            require(sha(Path(b["file"])) == b["sha256"], "QA reference changed")
            got = np.fromfile(a["file"], dtype="<f4").astype(np.float64)
            ref = np.fromfile(b["file"], dtype="<f4").astype(np.float64)
            require(got.shape == ref.shape and np.isfinite(ref).all(), "QA reference shape/finite contract differs")
            error = np.abs(got - ref); failures = error > ATOL + RTOL * np.abs(ref)
            row["outputs"][role] = dict(atol=ATOL, rtol=RTOL, max_abs_error=float(error.max()), rmse=float(np.sqrt(np.mean(error**2))),
                                        failed_elements=int(failures.sum()), within_tolerance=not bool(failures.any()),
                                        actual_sha256=sha(Path(a["file"])), reference_sha256=b["sha256"])
        row["major_correctness_passed"] = row["answer_equal"] and all(r["within_tolerance"] for r in row["outputs"].values())
        comparisons.append(row)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("microcode", "physical", "paired"), default="microcode")
    parser.add_argument("--timeout", type=int, default=172800)
    parser.add_argument("--numeric-execution-reference", type=Path,
                        help="Completed full microcode attempt whose actual outputs must match during cycle simulation")
    args = parser.parse_args(); out = args.output.resolve()
    require(not out.exists(), "choose a fresh QA attempt directory")
    out.mkdir(parents=True)
    before = sources(); inventory = json.loads(args.inventory.read_text())
    scheduled = args.mode != "microcode"
    window_options = dict(max_cycles=10**12) if scheduled else None
    program, coverage = compile_inventory(inventory, matrix_backend="scheduled" if scheduled else "microcode",
        vector_backend="scheduled" if scheduled else "microcode", memory_backend="scheduled" if scheduled else "planned",
        control_backend="scheduled" if scheduled else "rv64_leaf", schedule_options=window_options,
        vector_schedule_options=window_options, memory_schedule_options=window_options, control_schedule_options=window_options)
    contract = full_bert_contract(inventory, program)
    files = {str(args.inventory.resolve()): sha(args.inventory)}
    reference_native = None
    if args.numeric_execution_reference:
        reference = args.numeric_execution_reference.resolve()
        state = json.loads((reference / "execution.json").read_text())
        require(state["status"] == "exited" and state["exit_code"] == 0 and state["mode"] == "microcode", "QA numeric execution reference is not complete")
        require(state["inputs"].get(str(args.inventory.resolve())) == files[str(args.inventory.resolve())], "QA cycle and numeric attempts use different inventories")
        require(sha(reference / "mlx-tensor-semantics") == state["binary_sha256"], "QA reference binary changed")
        require(all(sha(reference / "sources" / p) == h for p, h in state["sources"].items())
                and all(sha(Path(p)) == h for p, h in {**state["inputs"], **state["runtime_libraries"]}.items()), "QA reference provenance changed")
        ref_program = json.loads((reference / "program.json").read_text())
        def numerical(p):
            return {k: v for k, v in p.items() if not k.endswith("_backend") and not k.endswith("_schedule_options") and k != "block_pipeline_plan"}
        require(numerical(program) == numerical(ref_program), "QA cycle compilation changed the numerical graph")
        full_bert_contract(inventory, ref_program)
        reference_native = json.loads((reference / "native/result.json").read_text())
        require(all(r["major_correctness_passed"] for r in audit_outputs(ref_program, inventory, reference_native)), "QA reference major correctness failed")
        for name in ("execution.json", "program.json", "native/result.json"):
            files[str(reference / name)] = sha(reference / name)
        for row in reference_native["outputs"]:
            for output in row["outputs"].values(): files[output["file"]] = sha(Path(output["file"]))
    require(not scheduled or reference_native is not None, "QA timing runs require a completed full microcode reference")
    for name, info in inventory["files"].items():
        require(sha(Path(name)) == info["sha256"] and Path(name).stat().st_size == info["bytes"], "BERT bound input changed")
        files[str(Path(name).resolve())] = info["sha256"]
    for check in inventory["reference_checks"]:
        for row in check["outputs"].values():
            require(sha(Path(row["file"])) == row["sha256"], "BERT reference output changed")
            files[str(Path(row["file"]).resolve())] = row["sha256"]
    if args.mode == "paired": program = compile_block_pipelines(program, event_slots=32)
    options = dict(base=2**32, bytes=2**34, max_cycles=10**13, operator_progress=True,
                   memory=dict(latency=8, accept_period=1, nack_every=0, trace_limit=0))
    if args.mode == "paired": options.update(tile_pipeline=True, template_load_timing=True)
    record(out / "program.json", program); record(out / "coverage.json", coverage); record(out / "options.json", options)
    policy = dict(atol=ATOL, rtol=RTOL, require_same_answer=True, require_all_sources_parameters_and_layers=True,
                  allow_nonfinite=False, require_framework_bitwise_equal=False, performance_scope="native_only_after_major_correctness")
    record(out / "acceptance-policy.json", policy)
    for name in ("program.json", "options.json", "coverage.json", "acceptance-policy.json"): files[str(out / name)] = sha(out / name)
    build = ROOT / "build/qa-runtime"
    target = "mlx-tensor-semantics" if args.mode == "microcode" else "mlx-physical-model" if args.mode == "physical" else "mlx-ready-graph"
    with (out / "build.log").open("w") as log:
        for command in (["cmake", "-S", str(ROOT / "simulator_ext/model_system"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
                        ["cmake", "--build", str(build), "--target", target, "-j4"]):
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    require(sources() == before, "QA sources changed while building")
    snapshot_sources(ROOT, out / "sources", before)
    binary = out / target
    shutil.copy2(build / ("model-storage/tensor-model/" + target if args.mode == "microcode" else target), binary)
    binary_hash = sha(binary); libraries = linked_libraries(binary)
    command = [str(binary), str(out / "program.json")]
    command += [str(out / "native"), "none", "1"] if args.mode == "microcode" else [str(out / "options.json"), str(out / "native")]
    require(all(sha(Path(p)) == h for p, h in files.items()), "QA inputs changed before launch")
    run_process(command, out / "native.log", out / "execution.json", timeout=args.timeout, cwd=ROOT,
                metadata=dict(mode=args.mode, sources=before, inputs=files, binary_sha256=binary_hash, runtime_libraries=libraries, input_contract=contract, acceptance_policy=policy))
    require(all(sha(out / "sources" / p) == h for p, h in before.items()) and sha(binary) == binary_hash
            and all(sha(Path(p)) == h for p, h in {**files, **libraries}.items()), "owned QA execution provenance changed")
    native = json.loads((out / "native/result.json").read_text())
    comparison = audit_outputs(program, inventory, native)
    numeric_execution_equal = None
    if reference_native is not None:
        numeric_execution_equal = all(Path(a["outputs"][role]["file"]).read_bytes() == Path(b["outputs"][role]["file"]).read_bytes()
                                      for a, b in zip(native["outputs"], reference_native["outputs"], strict=True)
                                      for role in ("start_logits", "end_logits"))
    physical = None
    if args.mode == "physical": physical = verify_physical_execution(program, native, options)
    if args.mode == "paired": physical = verify_ready_execution(program, native, options)
    passed = all(r["major_correctness_passed"] for r in comparison) and numeric_execution_equal is not False
    record(out / "comparison.json", dict(classification="full_dense_bert_qa_native_not_rocket_or_author_hybrid", input_contract=contract,
        mode=args.mode, major_correctness_passed=passed, comparison=comparison, physical_coverage=physical,
        full_microcode_execution_bitwise_equal=numeric_execution_equal,
        source_and_program_identity_unchanged=True, native_report_sha256=sha(out / "native/result.json"),
        mlx_system_verified=False, system_inference_performance_eligible=False, native_cycle_experiment_eligible=passed and scheduled))
    print(f"BERT_QA_NATIVE_COMPLETE major_correctness={passed} mode={args.mode} {out}", flush=True)
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
