from copy import deepcopy

import yaml

from mlxsim.active_window_coverage import compile_active_window_path
from scripts.audit_active_window_coverage import DEFAULT_CONFIG, build_audit


def test_active_window_capacity_partition() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    h120 = yaml.safe_load(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["h120_config"]["path"]).read_text()
    )
    h118 = yaml.safe_load(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["h118_config"]["path"]).read_text()
    )
    maxima = {}
    for window in config["window_sweep"]["compiled_windows"]:
        footprints = []
        for operator in config["workloads"]["operators"]:
            for size in config["workloads"]["sizes"]:
                overlay, memory, metadata = compile_active_window_path(
                    operator, int(size), int(window), config, h120, h118
                )
                assert overlay["active_window"] == int(window)
                assert memory["spad_ports"] == 4
                assert all(metadata["checks"].values())
                footprints.append(metadata["footprint"])
        maxima[int(window)] = max(footprints)
    assert maxima == {
        int(key): int(value)
        for key, value in config["window_sweep"][
            "expected_max_footprint_by_window"
        ].items()
    }


def test_active_window_only_changes_window_field() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    h120 = yaml.safe_load(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["h120_config"]["path"]).read_text()
    )
    h118 = yaml.safe_load(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["h118_config"]["path"]).read_text()
    )
    first, first_memory, _ = compile_active_window_path(
        "fft", 512, 1, config, h120, h118
    )
    fifth, fifth_memory, _ = compile_active_window_path(
        "fft", 512, 5, config, h120, h118
    )
    first_without_window = deepcopy(first)
    fifth_without_window = deepcopy(fifth)
    first_without_window.pop("active_window")
    fifth_without_window.pop("active_window")
    assert first_without_window == fifth_without_window
    assert first_memory == fifth_memory


def test_active_window_coverage_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["paper_performance_targets_consumed"] is False
    assert report["summary"]["compiled_paths"] == 128
    assert report["summary"]["executions"] == 192
    assert report["summary"]["workloads"] == 16
    assert report["summary"]["executed_windows"] == [1, 2, 3, 4, 5]
    assert report["summary"]["selected_window"] == 5
    assert all(report["work_checks"].values())
    assert all(report["window3_checks"].values())
