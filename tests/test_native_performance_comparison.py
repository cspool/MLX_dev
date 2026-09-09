"""Cycle accounting tests; small fixtures are not model performance evidence."""
import copy
import pytest

from scripts.report_mlx_native_performance import cycle_breakdown
from scripts.compare_mlx_execution_performance import event_metrics, validate_owned_attempt


def case():
    program = dict(nodes=[dict(id="v0", kind="add", vector_program={}, source_operator_id=0, source_operator="aten.add.Tensor", forward_id=0, layer_idx=0),
                          dict(id="v1", kind="argmax", control_program={}, source_operator_id=1, source_operator="aten.argmax.default", forward_id=0, layer_idx=None)],
                   outputs=[dict(forward_id=0, logits="v0", token="v1")])
    native = dict(classification="shared_native_physical_model_execution_not_chipyard_system_validation", cross_operator_execution="serial_with_shared_address_space",
                  events=[dict(source_operator_id=0, forward_id=0, shared_start_cycle=0, shared_end_cycle=10),
                          dict(source_operator_id=1, forward_id=0, shared_start_cycle=13, shared_end_cycle=18)],
                  device_component_cycles=15, host_readback_cycles=2, diagnostic_readback_cycles=3, shared_elapsed_cycles=20)
    return program, native


def test_serial_baseline_excludes_diagnostic_and_result_readback():
    p, r = case(); result = cycle_breakdown(p, r)
    assert result["measured_device_cycles"] == 15
    assert result["backend_cycles"] == {"vector": 10, "control": 5}
    assert result["forwards"][0]["measured_device_cycles"] == 15
    assert result["shared_elapsed_cycles"] == 20


@pytest.mark.parametrize("damage", ["missing_source", "negative", "cycle_sum", "readback_sum", "overlap_mode"])
def test_serial_accounting_rejects_missing_or_inconsistent_cycles(damage):
    p, r = case()
    if damage == "missing_source": r["events"].pop()
    if damage == "negative": r["events"][0]["shared_end_cycle"] = -1
    if damage == "cycle_sum": r["device_component_cycles"] += 1
    if damage == "readback_sum": r["shared_elapsed_cycles"] += 1
    if damage == "overlap_mode": r["cross_operator_execution"] = "parallel"
    with pytest.raises(RuntimeError): cycle_breakdown(p, r)


def test_parallel_residence_sum_is_separate_from_measured_end_to_end():
    p, _ = case()
    r = dict(events=[dict(source_operator_id=0, start_cycle=0, publish_cycle=10), dict(source_operator_id=1, start_cycle=5, publish_cycle=15)], graph_cycles=16)
    result = event_metrics(p, r)
    assert result["actual_graph_elapsed_cycles"] == 16
    assert result["sum_node_residence_cycles"] == 20
    assert result["union_node_residence_cycles"] == 15
    assert result["summed_overlap_residence_cycles"] == 5
    assert result["cycles_with_no_live_source"] == 1
    assert result["output_ready_cycles"] == [dict(forward_id=0, on_device_ready_cycle=15)]
    assert result["residence_sum_is_not_an_executed_serial_baseline"]
    bad = copy.deepcopy(r); bad["events"][0]["publish_cycle"] = 17
    with pytest.raises(RuntimeError): event_metrics(p, bad)


def test_active_run_cannot_generate_performance_comparison(tmp_path):
    (tmp_path / "execution.json").write_text('{"status":"running","pid":1234}')
    with pytest.raises(RuntimeError, match="actually completed"): validate_owned_attempt(tmp_path)
