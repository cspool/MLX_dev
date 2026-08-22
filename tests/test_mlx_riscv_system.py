import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = {"load", "store", "fma", "add", "max", "exp", "div", "shuffle", "xfer", "mul"}


def test_system_workload_manifest_covers_completion_operators() -> None:
    manifest = json.loads(
        (ROOT / "artifacts/environment/h205/mlx-system-workload-manifest.json").read_text()
    )
    assert all(manifest["checks"].values())
    assert [item["name"] for item in manifest["workloads"]] == [
        "bsmm",
        "fft_cmp",
        "swa",
        "transformer_block",
    ]
    operations = {
        operation for workload in manifest["workloads"] for operation in workload["operation_coverage"]
    }
    assert operations == OPERATIONS
    assert all(item["active_pes"] >= 6 for item in manifest["workloads"])


def test_system_instruction_target_is_out_of_band_and_route_is_signed() -> None:
    manifest = json.loads(
        (ROOT / "artifacts/environment/h205/mlx-system-workload-manifest.json").read_text()
    )
    transfer = next(
        item
        for item in manifest["workloads"][0]["lineage"]
        if item["pe"] == 1 and item["normalized"]["op"] == "xfer"
    )
    instruction = transfer["normalized"]
    word = int(transfer["word"], 16)
    assert instruction["target_pe"] == 4
    assert instruction["dx"] == -1
    assert instruction["dy"] == 1
    assert word >> 60 == manifest["opcodes"]["xfer"]
    assert (word >> 33) & 0x1F == 0x1F
    assert (word >> 28) & 0x1F == 1


def test_cycle_model_and_physical_array_are_distinct_backends() -> None:
    cycle = (ROOT / "rtl/mlx/mlx_cycle_model.sv").read_text()
    array = (ROOT / "rtl/mlx/mlx_array_4x4.sv").read_text()
    assert "one shared SIMD functional service" in cycle
    assert "mlx_array_4x4" not in cycle
    assert "for (pe = 0; pe < PE_COUNT" in array
    assert "mlx_pe_top" in array
    assert "packet_route_grant" in array


def test_backend_runs_match_goldens_and_instruction_counts() -> None:
    result = json.loads((ROOT / "artifacts/results/mlx-system-backends-run210.json").read_text())
    assert result["status"] == "supported"
    assert all(result["checks"].values())
    assert len(result["records"]) == 4
    assert all(item["comparison"]["same_instruction_count"] for item in result["records"])
    assert all(
        item["comparison"]["event_sequence"]["same_instruction_multiset"]
        and item["comparison"]["event_sequence"]["same_per_pe_program_order"]
        for item in result["records"]
    )
    assert any(
        not item["comparison"]["event_sequence"]["same_global_issue_order"]
        for item in result["records"]
    )
    assert all(
        result["instruction_timing"][name]["observations"] > 0 for name in OPERATIONS
    )


def test_rocc_runtime_exposes_four_command_contract() -> None:
    runtime = (ROOT / "system_sim/software/mlx_runtime.h").read_text()
    scala = (ROOT / "system_sim/chipyard/MLXRoCC.scala").read_text()
    controller = (ROOT / "rtl/mlx/mlx_rocc_controller.sv").read_text()
    for name, funct in (("CONFIG", 0), ("LAUNCH", 1), ("WAIT", 2), ("STATUS", 3)):
        assert f"MLX_FUNCT_{name} {funct}" in runtime
        assert f"FUNCT_{name} = 7'd{funct}" in controller
    assert "OpcodeSet.custom0" in scala
    assert "MLXCycleRocketConfig" in scala
    assert "MLXRTLRocketConfig" in scala


def test_real_chipyard_bare_metal_closure_is_supported() -> None:
    result = json.loads(
        (ROOT / "artifacts/results/mlx-chipyard-system-run212.json").read_text()
    )
    assert result["status"] == "supported"
    assert all(result["checks"].values())
    assert len(result["records"]) == 8
    assert {item["backend"] for item in result["records"]} == {"cycle", "rtl"}
    assert all(item["returncode"] == 0 for item in result["records"])
    assert all(all(item["checks"].values()) for item in result["records"])
    assert all(
        item["summary"]["host_total"]
        == item["summary"]["host_config"] + item["summary"]["host_launch_wait"]
        for item in result["records"]
    )


def test_ppa_scope_is_real_array_and_unfitted() -> None:
    config = yaml.safe_load((ROOT / "configs/system/mlx_array_ppa_v1.yaml").read_text())
    assert config["top"] == "mlx_array_4x4"
    assert config["implementation_flow"] == (
        "recursive_lane_rf_hard_macros_direct_pe_then_integrated_4x4"
    )
    assert config["calibration"] == {"applied": False, "coefficients": None}
    assert "mlx_array_4x4.sv" in config["rtl_sources"][-1]
    assert config["activity"]["provenance"] == "measured_rtl_simulation"
    assert (
        config["activity"]["source_clock_period_ns"]
        * config["activity"]["timestamp_scale"]
        == config["activity"]["normalized_clock_period_ns"]
        == config["clock_period_ns"]
    )


def test_recursive_submacros_are_routed_and_vcd_powered() -> None:
    manifest = json.loads(
        (
            ROOT
            / "artifacts/environment/h206/pe_submacros/submacro-build-manifest.json"
        ).read_text()
    )
    assert set(manifest) == {
        "full_lane",
        "reduced_lane",
        "register_file",
        "functional_unit",
        "pe_top",
    }
    assert all(all(item["checks"].values()) for item in manifest.values())
    assert all(
        item["workload_power"]["annotated_pin_activities"] > 0
        and math.isfinite(item["workload_power"]["total_power_w"])
        and item["workload_power"]["total_power_w"] > 0
        for item in manifest.values()
    )


def test_hierarchical_integrated_ppa_is_supported() -> None:
    result = json.loads(
        (ROOT / "artifacts/results/mlx-array-ppa-run211.json").read_text()
    )
    assert result["status"] == "supported"
    assert all(result["checks"].values())
    assert result["hierarchical_top"]["macro_instances"] == 16
    assert result["physical"]["drc_violations"] == 0
    assert result["physical"]["die_area_um2"] > 0
    assert result["physical"]["fmax_ghz"] > 0
    assert result["physical"]["total_power_w"] > 0
    assert result["physical"]["power_aggregation"] == (
        "recursive_postroute_transformer_vcd_hierarchy"
    )
