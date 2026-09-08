"""Bind a terminal Rocket run to complete compiler, input and output evidence.

Python only audits artifacts and rebuilds the CPU ELF; it executes no inference.
The full-model scope currently registers public dense Llama2, not paper hybrids.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

from mlxsim.model_physical_evidence import scheduled_compile_options
from mlxsim.model_system_evidence import require, terminal_execution, verify_system_execution
from mlxsim.model_tensor_compiler import compile_inventory
from scripts.mlx_system_attempt import digest, launch_metadata, record
from scripts.run_mlx_clocked_chipyard import ROOT, identity, prepare, source_identity
from scripts.run_mlx_tensor_semantics import compare_logits
from scripts.verify_mlx_model_numeric import normalize_reference_device, verify_generation_links, verify_layers
from system_sim.model_image.image import result_reference
from system_sim.physical_host.graph_lowering import literal_bytes


class Evidence:
    def __init__(self):
        self.files = {}

    def add(self, file, expected=None):
        file = Path(file).resolve()
        actual = digest(file)
        require(expected is None or actual == expected, f"evidence identity differs: {file}")
        require(str(file) not in self.files or self.files[str(file)] == actual,
                f"evidence changed during audit: {file}")
        self.files[str(file)] = actual
        return actual

    def read(self, file, expected=None):
        self.add(file, expected)
        return json.loads(Path(file).read_text())

    def finish(self):
        for file, expected in self.files.items():
            require(digest(file) == expected, f"evidence changed before audit completion: {file}")


def elf_load_segments(file):
    """Independently derive the initialized PT_LOAD rows from the bound ELF."""
    raw = file.read_bytes()
    require(len(raw) >= 64, "CPU ELF header is truncated")
    h = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
    require(h[0][:7] == b"\x7fELF\x02\x01\x01" and h[1:4] == (2, 243, 1)
            and h[4] % 2 == 0 and h[8] == 64 and h[9] == 56 and 0 < h[10] <= 128,
            "unsupported CPU ELF header")
    require(h[5] + h[9] * h[10] <= len(raw), "CPU ELF load table is truncated")
    rows, executable_entry = [], False
    for index in range(h[10]):
        kind, flags, offset, virtual, physical, size, memory_size, _ = struct.unpack_from(
            "<IIQQQQQQ", raw, h[5] + index * h[9])
        if kind != 1:
            continue
        require(size <= memory_size, "CPU ELF file segment exceeds memory size")
        if not memory_size:
            continue
        require(offset + size <= len(raw) and virtual == physical,
                "CPU ELF load extent is invalid")
        executable_entry |= bool(flags & 1) and physical <= h[4] < physical + memory_size
        rows.append({"name": f"elf:{index}", "path": str(file.resolve()), "address": physical,
                     "file_offset": offset, "file_bytes": size, "memory_bytes": memory_size,
                     "sha256": hashlib.sha256(raw[offset:offset + size]).hexdigest()})
    require(executable_entry, "CPU entry is outside executable load segments")
    return rows


def initialized_assets(program, plan, segments, memory, elf, replay_segments):
    require(plan["asset_initialization"] == "preloaded_resident_model_input_not_cpu_dma"
            and plan["cpu_asset_copy_bytes"] == 0, "audit requires the explicitly resident-input mode")
    require(memory["initialization_scope"] == "host_file_copy_before_clock_not_cpu_or_dma_execution",
            "initialization has an ambiguous execution scope")
    require(len(segments) == len(program["assets"]) and {r["name"] for r in segments} == set(program["assets"]),
            "initialization omits/duplicates assets or includes non-input data")
    actual = [r for r in memory["initialized_segments"] if not r["name"].startswith("elf:")]
    require(actual == segments, "actual RAM initialization differs from the compiled assets")
    elf_rows = [r for r in memory["initialized_segments"] if r["name"].startswith("elf:")]
    require(elf_rows == elf_load_segments(elf), "actual RAM did not initialize the exact CPU ELF load segments")
    expected = {r["name"]: r for r in replay_segments}
    require(set(expected) == set(program["assets"]), "replayed asset initialization scope changed")
    total, ranges = 0, []
    for row in segments:
        name = row["name"]
        binding, asset = plan["assets"][name], program["assets"][name]
        require(row["address"] == binding["base"] and row["file_bytes"] == row["memory_bytes"] == binding["bytes"],
                "resident asset address/extent differs from the graph binding")
        rebuilt = dict(expected[name])
        if asset["kind"] == "literal":
            # Only the freshly created literal container path may change.
            require(Path(row["path"]).resolve() == elf.parent / "literal_assets.bin",
                    "literal input was replaced with an external/golden file")
            rebuilt["path"] = row["path"]
            require(len(literal_bytes(asset)) == row["file_bytes"], "literal element count differs")
        require(rebuilt == row, "asset bytes, file offsets or source files changed in replay")
        total += row["file_bytes"]
    for row in [*elf_rows, *segments]:
        begin, end = row["address"], row["address"] + row["memory_bytes"]
        require(memory["base"] <= begin <= end <= memory["base"] + memory["bytes"],
                "initialized segment lies outside actual system RAM")
        if end > begin:
            ranges.append((begin, end))
    ranges.sort()
    require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), "ELF/input initialization overlaps")
    require(plan["initial_asset_count"] == len(segments) and plan["initial_asset_bytes"] == total,
            "initial asset count/byte accounting differs")
    return {"assets": len(segments), "bytes": total, "cpu_asset_copy_bytes": 0,
            "mode": "resident_input_not_cold_boot_or_dma_measurement"}


def audit_case(run, program_file, life_file, reference_file, out, evidence):
    state = evidence.read(run / "execution.json")
    evidence.add(run / "chipyard.log")
    terminal_execution(state, (run / "chipyard.log").read_text())
    require(state["sources"] == source_identity(), "current compiler/backend sources differ from the run")
    for name, expected in state["sources"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source snapshot path")
        evidence.add(ROOT / relative, expected)
        evidence.add(run.parent / "sources" / relative, expected)
    for entries in (state["inputs"], state["runtime_libraries"]):
        for name, expected in entries.items():
            evidence.add(name, expected)
    for file in (program_file, life_file, reference_file):
        require(str(file.resolve()) in state["inputs"], "audit input was not bound before the system run")
    program, reference = evidence.read(program_file), evidence.read(reference_file)
    plan, image = evidence.read(run / "plan.json"), evidence.read(run / "image.json")
    require(image["inputs"] == state["inputs"], "image/process input identities differ")
    for key, name in (("elf_sha256", "test.elf"), ("commands_sha256", "command_blob.bin"),
                      ("plan_sha256", "plan.json"), ("host_source_sha256", "test.c")):
        evidence.add(run / name, image[key])
    build = evidence.read(run.parent / "build.json")
    profile = evidence.read(run.parent / "profile.json", state["profile_identity"]["effective_sha256"])
    require(build["inputs"]["sources"] == state["sources"]
            and build["build_identity"] == state["build_identity"] == identity(build["inputs"]),
            "system build identity does not bind the executed sources")
    binary = Path(state["command"][0]).resolve()
    require(binary.parent == run.parent and binary.name == "simulator-chipyard-MLXClockedLargeRocketConfig",
            "execution is not the attempt-owned registered Rocket simulator")
    evidence.add(binary, build["simulator_sha256"])
    require(state["simulator_sha256"] == build["simulator_sha256"]
            and state["elf_sha256"] == image["elf_sha256"], "execution binary/ELF identity differs")
    require(state["command"][-1] == str(run / "test.elf")
            and f"+loadmem={run / 'test.elf'}" in state["command"]
            and f"+mlx_memory_segments={run / 'segments.json'}" in state["command"],
            "executed command did not select the bound ELF/resident inputs")
    require(state["source_calls"] == image["source_calls"] == plan["source_calls"]
            and state["task_count"] == image["task_count"] == plan["task_count"],
            "image/process source/task scopes differ")
    device, memory = evidence.read(run / "device.json"), evidence.read(run / "memory.json")
    require(device["build_identity"] == state["build_identity"], "device was not built from the bound inputs")
    coverage = verify_system_execution(program, plan, device, memory, profile)
    mapping = evidence.read(run / "launch-map.json")
    require(mapping == launch_metadata(program, plan), "observed launch identities differ from complete compilation")

    # No simulated cycle is run here. Rebuild all commands and the checker from
    # their original inputs; exact ELF equality connects the PASS marker to it.
    rebuilt_plan, _ = prepare({"program": str(program_file), "lifetimes": str(life_file),
                              "reference": str(reference_file)}, out / "elf-replay",
                             graph_base=plan["device_base"], graph_bytes=plan["device_bytes"],
                             memory_bytes=memory["bytes"], preload_assets=True,
                             block_pairs=plan.get("host_abi_version",1)==2,event_slots=plan.get("pair_event_slots",32))
    require(rebuilt_plan == plan, "complete compiler/address/lifetime replay changed the task graph")
    for name in ("command_blob.bin", "test.c", "test.elf", "launch-map.json"):
        require(evidence.add(out / "elf-replay" / name) == evidence.add(run / name),
                f"executed image differs from complete compiler/checker replay: {name}")
    segments = evidence.read(run / "segments.json")
    replay_segments = evidence.read(out / "elf-replay/segments.json")
    loading = initialized_assets(program, plan, segments, memory, run / "test.elf", replay_segments)
    if (run / "memory-init.json").exists():
        initial = evidence.read(run / "memory-init.json")
        require(initial["initialized_segments"] == memory["initialized_segments"]
                and initial["cycle"] == initial["ar_requests"] == initial["aw_requests"] == 0,
                "pre-clock asset initialization differs from the terminal memory record")
    outputs = result_reference(reference)["outputs"]
    require([r["forward_id"] for r in outputs] == [r["forward_id"] for r in program["outputs"]],
            "final reference checks do not cover exactly the compiled forwards")
    checks = []
    for row in outputs:
        path = Path(row["logits_file"])
        checksum = evidence.add(path)
        width = {"f16": "e", "f32": "f"}.get(row["dtype"])
        require(width is not None, "unregistered result reference precision")
        logits = [v[0] for v in struct.iter_unpack("<" + width, path.read_bytes())]
        require(len(logits) == math.prod(row["shape"]) and logits and all(map(math.isfinite, logits)),
                "nonfinite, empty or wrong-sized final logits")
        require(row["shape"][0] == 1 and row["tokens"] == [max(range(len(logits)), key=logits.__getitem__)],
                "reference tokens do not describe the free-running logits argmax")
        checks.append({"forward_id": row["forward_id"], "elements": len(logits), "tokens": row["tokens"],
                       "logits_sha256": checksum, "bitwise_checked_by_actual_cpu": True})
    return {"classification": "registered_rocket_graph_executable_and_output_audit_not_full_model_acceptance",
            "registered_graph_execution_verified": True, "complete_elf_rebuild_equal": True,
            "actual_output_dump_emitted": False,
            "output_evidence": "successful CPU byte comparisons in the exactly rebuilt executed ELF; not a new target dump",
            "coverage": coverage, "initialization": loading, "output_checks": checks,
            "full_model_execution_verified": False, "mlx_system_verified": False,
            "inference_performance_eligible": False}, program, state, profile


def llama2_contract(program, source, reference, state, profile, out, evidence):
    for inventory in (source, reference):
        model = inventory["model_identity"]
        require(model["family"] == "Llama2-7B" and model["variant"] == "public_dense_not_paper_hybrid",
                "full-model scope only registers public dense Llama2-7B")
        require(model["parameters"] == 6738415616 and model["parameter_tensors"] == 291
                and model["all_parameters_loaded"], "complete checkpoint is not bound")
        for key, expected in {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 11008,
                              "vocab_size": 32000, "num_attention_heads": 32, "num_key_value_heads": 32}.items():
            require(model["config"][key] == expected, "model topology was reduced or changed")
        require(inventory["instrumentation_equivalence_passed"], "reference instrumentation changed inference")
        verify_layers(inventory)
        evidence.add(model["model_source"], model["model_source_sha256"])
    require(source["model_identity"]["files"] == reference["model_identity"]["files"]
            and source["input"] == reference["input"] and source["input"]["batch"] == 1,
            "source/reference model or input contract differs")
    model = source["model_identity"]
    for name, info in model["files"].items():
        evidence.add(name, info["sha256"])
        require(Path(name).stat().st_size == info["bytes"], "checkpoint file extent differs")
    ignored = {f"model.layers.{i}.self_attn.rotary_emb.inv_freq" for i in range(32)}
    require(set(model["ignored_nonpersistent_checkpoint_buffers"]) == ignored,
            "unregistered checkpoint omissions")
    index = evidence.read(Path(model["path"]) / "model.safetensors.index.json")
    required = set(index["weight_map"]) - ignored
    used = set()

    def visit(value):
        if isinstance(value, dict):
            asset = program["assets"].get(value.get("value"))
            if asset and asset["kind"] == "mapped_file":
                used.add(asset["parameter_name"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for node in program["nodes"]:
        visit(node["args"])
        visit(node.get("kwargs", {}))
    require(len(required) == 291 and used == required, "not all full-model parameters are consumed")
    options = scheduled_compile_options(program)
    require(all(options[k + "_backend"] == "scheduled" for k in ("matrix", "vector", "memory", "control")),
            "full system input is not the completely scheduled program")
    compiled, _ = compile_inventory(source, **options)
    expected, _ = compile_inventory(reference, **options)
    require(compiled == program, "full source inventory does not compile to the executed program")
    actual_normalized, changes = normalize_reference_device(program, source["runtime"]["device"])
    expected_normalized, ref_changes = normalize_reference_device(expected, reference["runtime"]["device"])
    require(actual_normalized == expected_normalized and changes == ref_changes,
            "numeric reference changed computation beyond declared device placement")
    links = verify_generation_links(program, source)
    runtime = reference["runtime"]
    require(runtime["matrix_numeric_mode"] == "mlx-matrix-f32-kasc-v1"
            and runtime["float_numeric_mode"] == "mlx-vector-fp32-v1", "numeric contract is not explicit")
    for kind, field in (("matrix", "matrix_reference_calls"), ("vector", "float_reference_calls")):
        require(runtime[field] == dict(Counter(n["kind"] for n in program["nodes"] if kind + "_program" in n)),
                "numeric reference operator coverage differs")
    primitive = runtime["numeric_reference_provenance"]
    require(primitive["atomic_primitives_shared_with_cpp_fu"] is True
            and primitive["independent_matrix_and_reduction_control"] is True,
            "numeric reference independence/sharing is not disclosed")
    libm = Path(primitive["libm_path"]).resolve()
    evidence.add(libm, primitive["libm_sha256"])
    require(state["runtime_libraries"].get(str(libm)) == primitive["libm_sha256"],
            "system and explicit numeric reference used different atomic libm")
    native_profile = {"version": 1, "name": profile["name"], "max_busy_cycles": profile["max_busy_cycles"],
                      **{k + "_options": program[k + "_schedule_options"] for k in ("matrix", "vector", "memory")}}
    record(out / "native-profile.json", native_profile)
    parser = ROOT / "build/mlx-system-profile/system-profile-dump"
    evidence.add(parser, state["profile_identity"]["parser_sha256"])
    parsed = subprocess.run([str(parser), str(out / "native-profile.json")], capture_output=True,
                            text=True, check=True, timeout=30)
    require(json.loads(parsed.stdout) == profile, "system resources/timing differ from native scheduled input")
    gpu = {r["forward_id"]: r for r in source["reference_checks"]}
    comparison = []
    for row in result_reference(reference)["outputs"]:
        evidence.add(gpu[row["forward_id"]]["logits_file"], gpu[row["forward_id"]]["logits_sha256"])
        require(row["dtype"] == "f16" and row["shape"] == [1, 32000], "incomplete full-model final logits")
        # Equality to these reference bytes was proved by the actual CPU and
        # ELF rebuild, so this comparison is transitive, not a fresh target dump.
        comparison.append(compare_logits(row, gpu[row["forward_id"]]))
    return {"model_family": model["family"], "model_variant": model["variant"],
            "used_parameter_tensors": len(used), "generation_links": links,
            "reference_device_normalization": changes, "reference_primitive_provenance": primitive,
            "device_profile_matches_native": True, "framework_gpu_comparison": comparison,
            "framework_gpu_comparison_basis": "transitive through actual CPU bitwise equality to the numeric reference",
            "framework_gpu_comparison_passed": all(r["within_tolerance"] and r["tokens_equal"] for r in comparison)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "program", "lifetimes", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scope", choices=("public-dense-llama2", "registered-graph"), default="public-dense-llama2")
    parser.add_argument("--source-inventory", type=Path)
    args = parser.parse_args()
    if args.scope == "public-dense-llama2" and args.source_inventory is None:
        parser.error("full-model audit requires --source-inventory")
    run, out = args.run.resolve(), args.output.resolve()
    require(not out.exists() and out != run.parent and run.parent not in out.parents,
            "choose a fresh audit directory outside the immutable system attempt")
    out.mkdir(parents=True)
    evidence = Evidence()
    for file in (Path(__file__).resolve(), ROOT / "src/mlxsim/model_system_evidence.py",
                 ROOT / "scripts/verify_mlx_model_numeric.py", ROOT / "src/mlxsim/model_physical_evidence.py"):
        evidence.add(file)
    try:
        result, program, state, profile = audit_case(run, args.program.resolve(), args.lifetimes.resolve(),
                                                    args.reference.resolve(), out, evidence)
        result["scope"] = args.scope
        if args.scope == "public-dense-llama2":
            result["model_contract"] = llama2_contract(program, evidence.read(args.source_inventory),
                                                       evidence.read(args.reference), state, profile, out, evidence)
            result.update(classification="full_public_dense_model_system_numeric_contract_not_all_mlx_models_or_performance_acceptance",
                          full_model_execution_verified=True, full_model_system_numeric_contract_verified=True)
        evidence.finish()
        result.update(evidence=evidence.files, all_required_models_and_inputs_verified=False,
                      cross_operator_shared_resources_verified=False, full_rtl_verified=False)
        record(out / "report.json", result)
    except Exception as error:
        record(out / "failure.json", {"classification": "system_audit_failed_not_acceptance", "error": str(error),
                                      "full_model_execution_verified": False, "inference_performance_eligible": False})
        raise
    print(f"SYSTEM_AUDIT_COMPLETE scope={args.scope} (not overall MLX acceptance) {out / 'report.json'}")


if __name__ == "__main__":
    main()
