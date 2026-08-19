import numpy as np
import yaml

from mlxsim.numerical_equivalence import MappingConfig, compare_execution, execute_graph
from mlxsim.workload_lowering import validate_suite
from scripts.audit_same_input_numerical_equivalence import DEFAULT_CONFIG, build_audit


def test_lowered_executor_matches_golden_for_one_graph() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    spec = yaml.safe_load(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["workload_spec"]["path"]).read_text()
    )
    graph_id = "figure23_complete_block"
    graph = spec["graphs"][graph_id]
    order = validate_suite(spec)[graph_id]
    contract = config["test_contract"]["graphs"][graph_id]
    golden = execute_graph(
        graph_id=graph_id,
        graph=graph,
        order=order,
        contract=contract,
        seed=189,
        dtype="float32",
        mapping=None,
    )
    lowered = execute_graph(
        graph_id=graph_id,
        graph=graph,
        order=order,
        contract=contract,
        seed=189,
        dtype="float32",
        mapping=MappingConfig("test", 8, (4, 4)),
    )
    comparison = compare_execution(lowered, golden)
    assert comparison["event_order_identity"]
    assert comparison["operation_count_identity"]
    assert comparison["tensor_element_identity"]
    assert comparison["final_maximum_absolute_error"] <= 1.0e-5
    assert np.isfinite(lowered["final"]).all()


def test_same_input_numerical_equivalence_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["graphs"] == 3
    assert report["summary"]["nodes"] == 14
    assert report["summary"]["runs"] == 72
    assert report["summary"]["boundary_comparisons"] == 336
    assert report["summary"]["boundary_passes"] == 336
    assert report["summary"]["final_comparisons"] == 72
    assert report["summary"]["final_passes"] == 72
    assert report["summary"]["mapping_invariance_passes"] == 54
    assert report["summary"]["same_input_numerical_equivalence_complete"]
