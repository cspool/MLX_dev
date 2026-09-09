"""Measured full-model native device-cycle baseline, never Rocket timing."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.run_mlx_tensor_semantics import compare_logits, sha
from scripts.verify_mlx_model_numeric import verify, require


def cycle_breakdown(program, native):
    require(native["classification"] == "shared_native_physical_model_execution_not_chipyard_system_validation", "baseline requires native serial physical execution")
    require(native["cross_operator_execution"] == "serial_with_shared_address_space", "cannot sum overlapped operator durations")
    nodes = program["nodes"]
    events = native["events"]
    require(len(nodes) == len(events) and [n["source_operator_id"] for n in nodes] == [e["source_operator_id"] for e in events], "cycle baseline lacks full source coverage")
    forwards = {}
    totals = Counter()
    operators = []
    for node, event in zip(nodes, events, strict=True):
        begin, end = event["shared_start_cycle"], event["shared_end_cycle"]
        require(type(begin) is int and type(end) is int and 0 <= begin <= end, "invalid measured source duration")
        families = [name for name in ("matrix", "vector", "memory", "control") if name + "_program" in node]
        require(len(families) == 1, "performance source lacks one executed route")
        family = families[0]
        forward = node["forward_id"]
        require(event["forward_id"] == forward, "cycle forward identity differs")
        row = forwards.setdefault(forward, dict(forward_id=forward, source_calls=0, measured_device_cycles=0,
                                                backend_cycles=Counter(), raw_begin_cycle=begin, raw_end_cycle=end))
        duration = end - begin
        row["source_calls"] += 1
        row["measured_device_cycles"] += duration
        row["backend_cycles"][family] += duration
        row["raw_end_cycle"] = end
        totals[family] += duration
        operators.append(dict(source_operator_id=node["source_operator_id"], forward_id=forward, layer_idx=node["layer_idx"],
                              operator=node["source_operator"], backend=family, measured_device_cycles=duration))
    total = sum(totals.values())
    require(total == native["device_component_cycles"], "measured source/device cycles do not conserve")
    require(native["shared_elapsed_cycles"] == total + native["host_readback_cycles"] + native["diagnostic_readback_cycles"], "readback/diagnostic cycles do not conserve")
    return dict(forwards=list(forwards.values()), measured_device_cycles=total, backend_cycles=dict(totals),
                host_readback_cycles=native["host_readback_cycles"], diagnostic_readback_cycles=native["diagnostic_readback_cycles"],
                shared_elapsed_cycles=native["shared_elapsed_cycles"], most_expensive_sources=sorted(operators, key=lambda r: r["measured_device_cycles"], reverse=True)[:20])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--numeric-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-task-and-numeric-contract", action="store_true", required=True,
                        help="Explicitly use the user-authorized 2026-09-09 native experiment gate; preserve GPU diagnostics")
    args = parser.parse_args()
    require(not args.output.exists(), "choose a fresh performance report")
    run, source_path, reference_path = args.run.resolve(), args.source_inventory.resolve(), args.numeric_reference.resolve()
    # Re-run full data, source, binary, physical conservation and numerical audit.
    # Do not trust a previous report's pass flag alone.
    correctness = verify(run, source_path, reference_path)
    require(correctness["numeric_conformance_passed"] and correctness["executed_source_calls"] == 6181
            and correctness["remaining_functional_source_calls"] == 0, "full native correctness is incomplete")
    program = json.loads((run / "program.json").read_text())
    native = json.loads((run / "native/result.json").read_text())
    inventory = json.loads(source_path.read_text())
    refs = {r["forward_id"]: r for r in inventory["reference_checks"]}
    comparisons = [compare_logits(row, refs[row["forward_id"]]) for row in native["outputs"]]
    require(all(row["tokens_equal"] for row in comparisons), "actual task tokens differ from framework")
    breakdown = cycle_breakdown(program, native)
    for row in breakdown["forwards"]:
        check = refs[row["forward_id"]]
        row.update(phase="prefill" if row["forward_id"] == 0 else "decode", input_tokens=check["q_len"], kv_length=check["kv_len"])
    report = dict(classification="complete_native_serial_device_cycle_baseline_not_rocket_system_performance",
                  acceptance_policy="user_2026_09_09_task_and_declared_numeric_contract_gpu_difference_diagnostic",
                  native_device_performance_experiment_eligible=True, mlx_system_verified=False,
                  system_inference_performance_eligible=False, author_hybrid_identity_verified=False,
                  model=program["model"], input_contract=inventory["input"], correctness=correctness,
                  framework_diagnostics=comparisons, original_gpu_gate_overridden=False,
                  measured=breakdown, system_options=json.loads((run / "system-options.json").read_text()),
                  backend_options={k: v for k, v in program.items() if k.endswith("_schedule_options")},
                  actual_cpu_execution=False, frequency_hz=None, system_ttft_seconds=None, system_tokens_per_second=None,
                  exclusions=["CPU execution/configuration", "CPU/DMA cold asset loading", "Rocket caches/real system interconnect", "general cross-operator CDC overlap"],
                  provenance=dict(run=str(run), source_inventory=str(source_path), numeric_reference=str(reference_path),
                                  native_report_sha256=sha(run / "native/result.json"), script_sha256=sha(Path(__file__))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), measured=breakdown["forwards"])))


if __name__ == "__main__": main()
