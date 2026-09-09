"""Compare completed full native serial execution with real ready-graph execution."""
import argparse
import json
from pathlib import Path

from mlxsim.model_result_contract import QA_CONTRACT
from mlxsim.model_physical_evidence import verify_physical_execution
from mlxsim.model_ready_evidence import verify_ready_execution
from scripts.report_mlx_native_performance import cycle_breakdown
from scripts.run_mlx_tensor_semantics import sha
from scripts.run_mlx_qa_model import audit_outputs, full_bert_contract
from scripts.run_mlx_ready_model import compare_outputs, full_input_contract
from scripts.verify_mlx_model_numeric import verify, require


def event_metrics(program, result):
    """An overlapping node's residence duration is NOT its isolated service time."""
    events = result["events"]
    ids = [n["source_operator_id"] for n in program["nodes"]]
    require(len(events) == len(ids) and {e["source_operator_id"] for e in events} == set(ids), "event comparison needs every executed source")
    by_id = {e["source_operator_id"]: e for e in events}
    intervals = []
    for e in events:
        a, b = e["start_cycle"], e["publish_cycle"]
        require(type(a) is int and type(b) is int and 0 <= a < b <= result["graph_cycles"], "invalid measured ready interval")
        intervals.append((a, b))
    union = 0; cursor = 0
    for a, b in sorted(intervals):
        union += max(0, b - max(a, cursor)); cursor = max(cursor, b)
    values = {}
    for n in program["nodes"]:
        names = [r["id"] for r in n["split_outputs"]] if n["kind"] == "split" else [n["id"]]
        for name in names: values[name] = by_id[n["source_operator_id"]]["publish_cycle"]
    roles = ("start_logits", "end_logits") if program.get("output_contract") == QA_CONTRACT else ("logits", "token")
    return dict(actual_graph_elapsed_cycles=result["graph_cycles"], sum_node_residence_cycles=sum(b-a for a,b in intervals),
                union_node_residence_cycles=union, cycles_with_no_live_source=result["graph_cycles"] - union,
                summed_overlap_residence_cycles=sum(b-a for a,b in intervals) - union,
                output_ready_cycles=[dict(forward_id=o["forward_id"], on_device_ready_cycle=max(values[o[r]] for r in roles)) for o in program["outputs"]],
                residence_sum_is_not_an_executed_serial_baseline=True)


def validate_owned_attempt(path, expected_mode=None):
    state = json.loads((path / "execution.json").read_text())
    require(state.get("status") == "exited" and state.get("exit_code") == 0, "performance comparison needs an actually completed attempt")
    if expected_mode is not None: require(state.get("mode") == expected_mode, "native execution mode differs")
    require(all(sha(path / "sources" / p) == h for p, h in state["sources"].items()), "owned execution sources changed")
    require(all(sha(Path(p)) == h for p, h in {**state["inputs"], **state["runtime_libraries"]}.items()), "owned execution inputs or libraries changed")
    name = "mlx-physical-model" if expected_mode == "physical" else "mlx-ready-graph"
    require(sha(path / name) == state["binary_sha256"], "owned simulator binary changed")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("serial-run", "paired-run", "source-inventory", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--numeric-reference", type=Path, help="Required for Llama2's explicit numerical contract")
    args = parser.parse_args(); serial = args.serial_run.resolve(); paired = args.paired_run.resolve()
    require(not args.output.exists(), "choose a fresh complete performance comparison")
    # Missing result files are a pending run, never a zero-cycle or estimated result.
    a = json.loads((serial / "program.json").read_text()); b = json.loads((paired / "program.json").read_text())
    require(a == {k: v for k, v in b.items() if k != "block_pipeline_plan"}, "serial/paired numerical programs, inputs or hardware profiles differ")
    left = json.loads((serial / "native/result.json").read_text()); right = json.loads((paired / "native/result.json").read_text())
    left_options_file = serial / ("options.json" if a.get("output_contract") == QA_CONTRACT else "system-options.json")
    left_options = json.loads(left_options_file.read_text()); right_options = json.loads((paired / "options.json").read_text())
    for key in ("base", "bytes", "max_cycles", "memory"):
        require(left_options.get(key) == right_options.get(key), "serial/paired memory/cycle profile differs")
    source = json.loads(args.source_inventory.read_text())
    if a.get("output_contract") == QA_CONTRACT:
        validate_owned_attempt(serial, "physical"); validate_owned_attempt(paired, "paired")
        for p, r in ((a, left), (b, right)):
            full_bert_contract(source, p)
            require(all(c["major_correctness_passed"] for c in audit_outputs(p, source, r)), "full BERT major correctness failed")
        roles = ("start_logits", "end_logits")
        outputs_equal = all(Path(x["outputs"][role]["file"]).read_bytes() == Path(y["outputs"][role]["file"]).read_bytes()
                            and x["span"] == y["span"] for x, y in zip(left["outputs"], right["outputs"], strict=True) for role in roles)
    else:
        require(args.numeric_reference is not None, "Llama2 comparison needs a full numeric reference")
        verify(serial, args.source_inventory.resolve(), args.numeric_reference.resolve())
        state = validate_owned_attempt(paired)
        require(state["scope"] == "public-dense-llama2", "paired run is not full Llama2")
        reference = json.loads(args.numeric_reference.read_text()); bound = {}
        full_input_contract(a, source, reference, bound)
        matches = compare_outputs(b, right, reference, paired / "native")
        require(all(c["bitwise_equal"] and c["tokens_equal"] for c in matches), "paired full-model numeric contract failed")
        outputs_equal = all(Path(x["logits_file"]).read_bytes() == Path(y["logits_file"]).read_bytes() and x["tokens"] == y["tokens"]
                            for x, y in zip(left["outputs"], right["outputs"], strict=True))
    require(outputs_equal, "scheduling changed final model values/task results")
    serial_coverage = verify_physical_execution(a, left, left_options)
    paired_coverage = verify_ready_execution(b, right, right_options)
    baseline = cycle_breakdown(a, left); end_to_end = event_metrics(b, right)
    report = dict(classification="full_native_end_to_end_vs_executed_serial_cycle_comparison_not_rocket",
                  model=a["model"], input_contract=a["input_contract"], same_numerical_program=True, all_actual_outputs_equal=True,
                  serial_measured=baseline, end_to_end_measured=end_to_end,
                  serial_over_end_to_end_device_cycle_ratio=baseline["measured_device_cycles"] / right["graph_cycles"],
                  device_cycles_saved=baseline["measured_device_cycles"] - right["graph_cycles"],
                  physical_requests=dict(serial=left["memory"]["submitted"], paired=right["memory"]["submitted"]),
                  scope_differences=dict(serial="independent per-operator array resources; no timed template load",
                                         paired="shared array and closed pair events; timed local template load",
                                         causal_overlap_speedup_claimed=False,
                                         reason="measured implementation comparison includes resource sharing and configuration, not only overlap"),
                  serial_coverage=serial_coverage, paired_coverage=paired_coverage,
                  mlx_system_verified=False, system_ttft_seconds=None, system_tokens_per_second=None,
                  files={str(p): sha(p) for p in (serial / "program.json", serial / "native/result.json", paired / "program.json", paired / "native/result.json", left_options_file, paired / "options.json")})
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"FULL_NATIVE_SERIAL_COMPARISON {args.output}")


if __name__ == "__main__": main()
