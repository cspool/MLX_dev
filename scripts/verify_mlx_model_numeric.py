"""Verify a full Llama2 run under declared numeric contracts, not system timing.

This is deliberately distinct from the historical framework/GPU error gate and
from a Chipyard/hardware certificate. Atomic libm sharing is made explicit.
"""

import argparse
import copy
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path

from mlxsim.model_tensor_compiler import compile_inventory
from mlxsim.model_physical_evidence import CLASSIFICATION as PHYSICAL_CLASSIFICATION, EXECUTION_CLASSIFICATION as PHYSICAL_EXECUTION_CLASSIFICATION, scheduled_compile_options, verify_physical_execution
from scripts.run_mlx_tensor_semantics import sha


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def normalize_reference_device(program, device):
    result = copy.deepcopy(program)
    normalized = Counter()
    for node in result["nodes"]:
        if node["kind"] == "arange" and "device" in node["kwargs"]:
            require(node["kwargs"]["device"] == device, "unexpected arange device placement")
            node["kwargs"]["device"] = "$REFERENCE_DEVICE"
            normalized["arange"] += 1
        if node["kind"] == "cast_device":
            require(node["args"][1] == device, "unexpected cast device placement")
            node["args"][1] = "$REFERENCE_DEVICE"
            normalized["cast_device"] += 1
    return result, dict(normalized)


def verify_generation_links(program, source):
    nodes = {node["id"]: node for node in program["nodes"]}
    outputs = {item["forward_id"]: item for item in program["outputs"]}
    embeddings = {node["forward_id"]: node for node in program["nodes"] if node["kind"] == "embedding"}
    require(len(embeddings) == len(outputs) == sum(node["kind"] == "embedding" for node in program["nodes"]), "missing or duplicate generation-step embedding")
    initial = embeddings[0]["args"][1]["value"]
    require(initial in program["assets"], "initial tokens are not explicitly bound")
    asset = program["assets"][initial]
    require(asset["kind"] == "literal" and asset["origin"] == "input" and asset["dtype"] == "i64", "initial tokens lack an input binding")
    require(asset["values"] == source["input"]["token_ids"], "initial token bytes differ from the source input")
    links = []
    for forward in range(1, len(outputs)):
        value = embeddings[forward]["args"][1]["value"]
        target = outputs[forward - 1]["token"]
        chain = []
        while value != target:
            require(value in nodes and value not in chain, "decode tokens come from a literal or cyclic binding")
            chain.append(value)
            node = nodes[value]
            require(node["kind"] in {"unsqueeze", "reshape", "alias", "contiguous", "cast", "cast_device"}, "decode token path contains a value-changing operation")
            require(node["output"]["dtype"] == "i64" and math.prod(node["output"]["shape"]) == 1, "decode token representation changed")
            value = node["args"][0]["value"]
        require(nodes[target]["kind"] == "argmax", "decode does not consume computed token selection")
        links.append({"forward_id": forward, "previous_token": target, "identity_view_chain": chain})
    return links


def verify_layers(inventory):
    checks = inventory["reference_checks"]
    require(len(checks) >= 2, "prefill/decode not exercised")
    require([row["forward_id"] for row in checks] == list(range(len(checks))), "nonsequential reference forwards")
    for row in checks:
        forward = row["forward_id"]
        require(row["past_len"] == (checks[forward - 1]["kv_len"] if forward else 0), "cache history does not connect adjacent forwards")
        require(row["q_len"] == (1 if forward else len(inventory["input"]["token_ids"][0])), "unexpected prefill/decode input length")
        entries = [event for event in inventory["boundaries"] if event["event"] == "enter" and event["forward_id"] == forward and re.fullmatch(r"model\.layers\.\d+", event["module_path"])]
        require(len(entries) == 32 and {event["layer_idx"] for event in entries} == set(range(32)), "not every layer executed")
        require(row["kv_len"] == row["past_len"] + row["q_len"], "cache length transition mismatch")
        require(len(row["cache_states"]) == 32, "missing layer cache bindings")
        require({item["layer_idx"] for item in row["cache_states"]} == set(range(32)), "cache bindings omit or duplicate a layer")
        require(all(item["key_shape"][-2] == row["kv_len"] and item["value_shape"][-2] == row["kv_len"] for item in row["cache_states"]), "cache shape does not follow generation state")
        require(row["logits_bitwise_equal"], "reference instrumentation changed logits")


def verify(run, source_file, reference_file):
    source, reference = json.loads(source_file.read_text()), json.loads(reference_file.read_text())
    program_file, native_file = run / "program.json", run / "native/result.json"
    program, native = json.loads(program_file.read_text()), json.loads(native_file.read_text())
    execution = json.loads((run / "comparison.json").read_text())
    physical=native["classification"]==PHYSICAL_CLASSIFICATION
    require(execution["classification"] == (PHYSICAL_EXECUTION_CLASSIFICATION if physical else "full_model_native_semantics_not_mlx_system_validation"), "unexpected execution evidence class")
    require(sha(source_file) == execution["inventory_sha256"] and sha(program_file) == execution["program_sha256"], "source/program identity mismatch")
    require(sha(run / ("mlx-physical-model" if physical else "mlx-tensor-semantics")) == execution["binary_sha256"], "attempt-owned binary identity mismatch")
    physical_coverage=None
    if physical:
        require(sha(native_file)==execution["native_report_sha256"] and sha(run/"system-options.json")==execution["system_options_sha256"],"physical report/system configuration identity mismatch")
        for name,digest in execution["sources"].items():
            require(sha(run/"sources"/name)==digest,"physical execution source snapshot mismatch")
        for name,digest in execution["runtime_libraries"].items():require(sha(Path(name))==digest,"physical runtime library changed")
        physical_coverage=verify_physical_execution(program,native,json.loads((run/"system-options.json").read_text()))
    for inventory in (source, reference):
        model = inventory["model_identity"]
        require(model["family"] == "Llama2-7B" and model["variant"] == "public_dense_not_paper_hybrid", "this verifier only registers full public dense Llama2-7B")
        require(model["parameters"] == 6738415616 and model["parameter_tensors"] == 291 and model["all_parameters_loaded"], "full checkpoint binding is incomplete")
        require(model["config"]["num_hidden_layers"] == 32 and model["config"]["hidden_size"] == 4096 and model["config"]["intermediate_size"] == 11008 and model["config"]["vocab_size"] == 32000, "model was reduced or changed")
        require(inventory["instrumentation_equivalence_passed"], "missing reference instrumentation equivalence")
        verify_layers(inventory)
    require(source["model_identity"]["files"] == reference["model_identity"]["files"], "reference uses a different model/tokenizer")
    require(source["input"] == reference["input"] and source["input"]["batch"] == 1, "reference input/generation contract differs")
    for name, info in source["model_identity"]["files"].items():
        require(sha(Path(name)) == info["sha256"], f"model asset identity mismatch: {name}")
    memory_backend = program.get("memory_backend", "functional")
    control_backend = program.get("control_backend", "functional")
    compile_options=scheduled_compile_options(program)
    require(compile_options["matrix_backend"] in {"microcode","scheduled"} and compile_options["vector_backend"] in {"microcode","scheduled"},"numeric verifier requires explicit matrix/vector instruction routes")
    current, _ = compile_inventory(source, **compile_options)
    require(current == program, "current compiler no longer produces the executed program")
    ref_program, _ = compile_inventory(reference, **compile_options)
    actual_normalized, changes = normalize_reference_device(program, source["runtime"]["device"])
    expected_normalized, ref_changes = normalize_reference_device(ref_program, reference["runtime"]["device"])
    require(actual_normalized == expected_normalized and changes == ref_changes, "numeric reference differs beyond explicit device placement metadata")
    linked_tokens = verify_generation_links(program, source)
    required_parameters = set(json.loads((Path(source["model_identity"]["path"]) / "model.safetensors.index.json").read_text())["weight_map"]) - set(source["model_identity"]["ignored_nonpersistent_checkpoint_buffers"])
    used = set()
    def visit(value):
        if isinstance(value, dict):
            if value.get("value") in program["assets"]:
                asset = program["assets"][value["value"]]
                if asset["kind"] == "mapped_file":
                    used.add(asset["parameter_name"])
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for node in program["nodes"]:
        visit(node["args"])
    require(len(required_parameters) == 291 and used == required_parameters, "not every full-model parameter is consumed")
    require(native["executed_source_calls"] == len(program["nodes"]) == execution["executed_source_calls"], "incomplete source-operator execution")
    require([(event["source_operator_id"], event["entry"]) for event in native["events"]] == [(node["source_operator_id"], f"tensor_model::{node['kind']}") for node in program["nodes"]], "executed operator routing differs from the compiled program")
    matrix_nodes = [node for node in program["nodes"] if "matrix_program" in node]
    vector_nodes = [node for node in program["nodes"] if "vector_program" in node]
    memory_nodes = [node for node in program["nodes"] if "memory_program" in node]
    control_nodes = [node for node in program["nodes"] if "control_program" in node]
    require(native["blas_calls"] == 0 and native["python_or_gpu_execution_fallbacks"] == 0, "tensor execution used a prohibited fallback")
    value_shapes = {name: asset["shape"] for name, asset in program["assets"].items()}
    value_shapes.update({node["id"]: node["output"]["shape"] for node in program["nodes"]})
    macs = sum(math.prod(node["output"]["shape"]) * value_shapes[node["args"][0]["value"]][-1] for node in matrix_nodes)
    require(macs == native["matrix_microcode"]["mul_active_lanes"] and (physical or macs == native["matrix_macs"]), "not every matrix MAC was executed by microcode")
    require(len(vector_nodes) == native["vector_microcode"]["calls"], "vector microcode coverage mismatch")
    if control_nodes:
        require(native["control_programs"]["calls"] == len(control_nodes), "controller leaf coverage mismatch")
        control_entry = "mlx::control_schedule::Simulator" if control_backend == "scheduled" else "mlx::control_model::execute"
        require(physical or sum(event.get("control_entry") == control_entry for event in native["events"]) == len(control_nodes), "controller instructions did not execute in the registered backend")
        if control_backend == "scheduled":
            windows = native["windows"]["control"] if physical else native.get("control_windows", [])
            require([w["source_operator_id"] for w in windows] == [n["source_operator_id"] for n in control_nodes], "controller cycle windows do not cover all control operators")
            require(all(w["done"] and w["dma_requests"] == w["dma_responses"] for w in windows), "controller cycle execution did not drain")
            for field, target in (("instructions", "instructions"), ("branches_taken", "branches_taken"), ("read_bytes", "read_bytes"), ("write_bytes", "write_bytes")):
                require(sum(w[field] for w in windows) == native["control_programs"][target], "controller cycle accounting mismatch")
    remaining = len(program["nodes"]) - len(matrix_nodes) - len(vector_nodes) - len(memory_nodes) - len(control_nodes)
    if remaining == 0:
        require(native.get("functional_entry_calls") == 0, "fully lowered model used unlowered functional entries")
    if memory_nodes:
        memory = native["memory_programs"]
        views = sum(node["memory_program"]["mode"] == "view" for node in memory_nodes)
        transfers = [node for node in memory_nodes if node["memory_program"]["mode"] == "transfer"]
        written = sum(math.prod(node["output"]["shape"]) * {"f16": 2, "f32": 4, "i64": 8, "bool": 1}[node["output"]["dtype"]] for node in transfers)
        require(memory["calls"] == len(memory_nodes) and memory["view_elisions"] == views and memory["allocations"] == len(transfers), "memory plan coverage/alias accounting mismatch")
        require(memory["write_bytes"] == written and (physical or memory["staging_bytes"] == 128 and memory["register_bytes_total"] == 32), "memory transfer work/capacity mismatch")
        memory_entry = "mlx::memory_model::Simulator" if memory_backend == "scheduled" else "mlx::memory_model::execute"
        require(physical or sum(event.get("memory_entry") == memory_entry for event in native["events"]) == len(memory_nodes), "memory plans were not consumed by the registered backend")
        if memory_backend == "scheduled":
            windows = native["windows"]["memory"] if physical else native.get("memory_windows", [])
            require([w["source_operator_id"] for w in windows] == [n["source_operator_id"] for n in memory_nodes], "memory cycle windows do not cover every planned operator")
            require(all(w["done"] and w["dma_requests"] == w["dma_responses"] and w["inflight_transactions"] == 0 for w in windows), "memory cycle execution did not drain")
            require(sum(w["dma_write_bytes"] for w in windows) == written, "memory cycle writes differ from the plan")
    require(reference["runtime"]["matrix_numeric_mode"] == "mlx-matrix-f32-kasc-v1" and reference["runtime"]["float_numeric_mode"] == "mlx-vector-fp32-v1", "reference numeric mode is not explicitly bound")
    require(reference["runtime"]["matrix_reference_calls"] == dict(Counter(node["kind"] for node in matrix_nodes)), "reference matrix coverage mismatch")
    require(reference["runtime"]["float_reference_calls"] == dict(Counter(node["kind"] for node in vector_nodes)), "reference vector coverage mismatch")
    primitive = reference["runtime"]["numeric_reference_provenance"]
    require(primitive["atomic_primitives_shared_with_cpp_fu"] is True, "reference primitive sharing must be explicitly disclosed")
    require(sha(Path(primitive["libm_path"])) == primitive["libm_sha256"], "reference primitive library identity changed")
    if physical:require(execution["runtime_libraries"].get(str(Path(primitive["libm_path"]).resolve()))==primitive["libm_sha256"],"physical libm differs from the declared numeric primitive")
    expected = {row["forward_id"]: row for row in reference["reference_checks"]}
    raw = {row["forward_id"]: row for row in execution["comparison"]}
    require(len(native["outputs"]) == len(expected) and {row["forward_id"] for row in native["outputs"]} == set(expected), "model outputs are missing or duplicated")
    comparisons = []
    for actual in native["outputs"]:
        ref = expected[actual["forward_id"]]
        a, b = Path(actual["logits_file"]), Path(ref["logits_file"])
        require(actual["dtype"] == "f16" and actual["shape"] == ref["logits_shape"] == [1, 32000], "unexpected model output type/shape")
        require(sha(a) == raw[actual["forward_id"]]["actual_logits_sha256"] and sha(b) == ref["logits_sha256"], "model output hash mismatch")
        require(a.stat().st_size == b.stat().st_size == 64000 and a.read_bytes() == b.read_bytes(), "model logits are not bitwise identical under the numeric contract")
        require(actual["tokens"] == [ref["token_id"]], "free-running token selection differs")
        logits = [value[0] for value in struct.iter_unpack("<e", a.read_bytes())]
        require(all(math.isfinite(value) for value in logits), "nonfinite logits cannot pass numeric conformance")
        require(actual["tokens"] == [max(range(len(logits)), key=logits.__getitem__)], "reported token is not the actual logits argmax")
        comparisons.append({"forward_id": actual["forward_id"], "elements": 32000, "bitwise_equal": True, "tokens": actual["tokens"], "logits_sha256": sha(a)})
    return {"classification": "full_model_numeric_contract_conformance_not_system_or_hardware_validation",
            "model_family": source["model_identity"]["family"], "model_variant": source["model_identity"]["variant"],
            "numeric_conformance_passed": True, "used_parameter_tensors": len(used), "executed_source_calls": len(program["nodes"]),
            "matrix_source_calls": len(matrix_nodes), "vector_source_calls": len(vector_nodes), "matrix_mac_lanes": macs,
            "memory_source_calls": len(memory_nodes),
            "control_source_calls": len(control_nodes),
            "remaining_functional_source_calls": remaining,
            "physical_execution_coverage": physical_coverage,
            "generation_links": linked_tokens, "comparisons": comparisons, "reference_primitive_provenance": primitive,
            "reference_device_metadata_normalization": changes, "source_inventory_sha256": sha(source_file),
            "numeric_reference_inventory_sha256": sha(reference_file), "program_sha256": sha(program_file),
            "native_report_sha256": sha(native_file), "execution_report_sha256": sha(run / "comparison.json"),
            "binary_sha256": execution["binary_sha256"], "execution_source_snapshot": execution["sources"],
            "historical_framework_comparison_passed": execution["all_comparisons_passed"],
            "mlx_system_verified": False, "mlx_hardware_mapping_complete": False, "inference_performance_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--numeric-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "choose a fresh numeric conformance report path")
    verifier_sha256 = sha(Path(__file__))
    result = verify(args.run.resolve(), args.source_inventory.resolve(), args.numeric_reference.resolve())
    require(sha(Path(__file__)) == verifier_sha256, "numeric verifier changed during validation")
    result["verifier_sha256"] = verifier_sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"FULL_MODEL_NUMERIC_CONTRACT_CONFORMANCE_PASS (not system validation) {args.output}")


if __name__ == "__main__":
    main()
