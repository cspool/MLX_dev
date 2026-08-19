from copy import deepcopy

import pytest
import yaml

from mlxsim.workload_lowering import WorkloadLoweringError, topological_order, validate_suite
from scripts.audit_unified_workload_lowering import DEFAULT_CONFIG, build_audit


def test_unified_workload_graph_schema_and_cycle_detection() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    spec_path = DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["workload_spec"]["path"]
    spec = yaml.safe_load(spec_path.read_text())
    orders = validate_suite(spec)
    assert len(orders) == 3
    assert sum(len(order) for order in orders.values()) == 14
    assert orders["figure23_complete_block"][0] == "rmsnorm"
    invalid = deepcopy(spec["graphs"]["figure19_fabnet_block"])
    invalid["operators"][0]["depends_on"] = ["global_ffn2"]
    invalid["operators"][1]["depends_on"] = ["fft2d_attention"]
    with pytest.raises(WorkloadLoweringError, match="cycle"):
        topological_order(invalid)


def test_unified_workload_lowering_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["graphs"] == 3
    assert report["summary"]["graph_nodes"] == 14
    assert report["summary"]["executable_units"] == 12
    assert report["summary"]["detailed_overlay_units"] == 4
    assert report["summary"]["memory_configs"] == 3
    assert report["summary"]["analytical_profiles"] == 8
    assert report["summary"]["lineage_nodes"] == 14
    assert report["summary"]["lowering_replays"] == 12
    assert report["summary"]["executions"] == 24
    assert report["summary"]["execution_replays"] == 12
    assert report["summary"]["numerically_complete_figures"] == 3
    assert report["summary"]["unified_toolchain_complete"]
    assert report["summary"]["author_toolchain_claimed"] is False
