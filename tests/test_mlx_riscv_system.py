import hashlib
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
    abstraction = config["abstract_lef_obstructions"]
    assert abstraction["integration_method"] == (
        "conservative_5um_raster_union_per_routing_layer"
    )
    assert abstraction["raster_pitch_um"] == 5.0
    assert abstraction["obstruction_cover"] == (
        "outward_quantized_any_overlap_after_access_halo_clip"
    )
    assert abstraction["preserve_pin_geometry"] is True
    assert len(abstraction["routing_layers"]) == 10
    macro_track = config["hierarchical_top_placement"][
        "macro_origin_track_alignment"
    ]
    assert macro_track["grid_dbu"] == math.lcm(
        *macro_track["routing_pitch_dbu"].values()
    )
    assert macro_track["grid_um"] == (
        macro_track["grid_dbu"] / macro_track["dbu_per_micron"]
    )
    assert macro_track["required_macro_instances"] == 16
    capacity = config["hierarchical_top_placement"][
        "detailed_placement_capacity_basis"
    ]
    assert capacity["reserved_site_capacity"] > 8 * capacity["estimated_required_sites"]
    route_plan = config["hierarchical_top_route_resource_plan"]
    assert route_plan["routing_layers"]["signal"] == "metal3-metal10"
    assert route_plan["routing_layers"]["clock"] == "metal5-metal10"
    assert route_plan["layer_capacity_adjustments"] == {}
    assert route_plan["grid_pitches_in_tile"] == 48
    assert route_plan["max_2d_edge_usage_multiplier"] == 101
    assert route_plan["verbose"] is True
    assert route_plan["stop_after_global_route"] is True
    assert route_plan["fast_route_edge_usage_contract"]["observed_usage"] == 2503
    patch = (ROOT / route_plan["global_route_patch"]).read_text()
    assert "pitches_in_tile_ = 15" in patch
    assert "pitches_in_tile_ = 48" in patch
    assert "max_usage_multiplier = 101" in patch
    route_tool_contract = config["toolchain"]["global_route_openroad"]
    route_archive = ROOT / route_tool_contract["archive"]
    assert route_archive.is_file()
    assert hashlib.sha256(route_archive.read_bytes()).hexdigest() == (
        route_tool_contract["archive_sha256"]
    )
    grid_basis = route_plan["grid_resource_basis"]
    assert grid_basis["selected_gcells"] * 4 == grid_basis["tile24_gcells"]
    assert grid_basis["selected_to_tile24_gcell_ratio"] == 0.25
    tile24_attempt = config["hierarchical_top_tile24_route_attempt"]
    assert tile24_attempt["elapsed_minutes_at_stop"] > 500
    assert tile24_attempt["observed_rss_gib"] > 200
    assert tile24_attempt["status"] == (
        "stopped_at_resource_safety_boundary_before_checkpoint"
    )
    assert (ROOT / tile24_attempt["evidence"]).is_file()
    full_interior_attempt = config["hierarchical_top_tile48_full_interior_attempt"]
    assert full_interior_attempt["total_overflow"] > 0
    assert full_interior_attempt["warning_message_limits_reached"] is True
    assert full_interior_attempt["result_consumed_as_final_ppa"] is False
    assert all(
        (ROOT / full_interior_attempt["evidence"][name]).is_file()
        for name in ("global_route_log", "detailed_route_log")
    )
    assert "mlx_array_4x4.sv" in config["rtl_sources"][-1]
    assert config["activity"]["provenance"] == "measured_rtl_simulation"
    assert (
        config["activity"]["source_clock_period_ns"]
        * config["activity"]["timestamp_scale"]
        == config["activity"]["normalized_clock_period_ns"]
        == config["clock_period_ns"]
    )


def test_paper_ppa_alignment_is_explicit_and_separated() -> None:
    targets = yaml.safe_load((ROOT / "artifacts/targets/paper_targets.yaml").read_text())[
        "table2_area_power"
    ]
    assert targets["provenance"] == "reported"
    assert targets["components"]["pe"] == {
        "area_mm2": 0.482,
        "power_mw": 365.4,
        "skip_hop_area_fraction": 0.062,
    }
    assert targets["components"]["pe_array"] == {
        "area_mm2": 7.712,
        "power_mw": 5846.4,
    }

    calibrated = json.loads(
        (ROOT / "artifacts/results/mlx-rtl-ppa-activity-calibrated-run208.json").read_text()
    )
    assert calibrated["hypothesis_status"] == "supported"
    assert calibrated["audit_integrity"] is True
    assert calibrated["paper_performance_targets_consumed"] is True
    assert calibrated["validation_eligible"] is False
    assert calibrated["summary"]["passing_area_values"] == 9
    assert calibrated["summary"]["passing_power_values"] == 9
    assert calibrated["summary"]["area_max_relative_error"] < 0.15
    assert calibrated["summary"]["power_max_relative_error"] < 0.15


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
    assert all(
        item["checks"]["pin_access"] is True
        and item["pin_access"]["all_pins_accessible"] is True
        and item["pin_access"]["stdcell_pins_without_access"] == 0
        and item["pin_access"]["macro_pins_without_access"] == 0
        and item["pin_access"]["no_access_errors"] == 0
        for item in result["submacro_chain"].values()
    )
    global_route = result["hierarchical_top"]["global_route_metrics"]
    assert result["checks"]["global_route_congestion"] is True
    assert global_route["overflow_resolved"] is True
    assert global_route["total_overflow"] == 0
    assert global_route["congestion_warning"] is False
    assert global_route["routed_nets"] > 0
    assert global_route["final_vias"] > 0
    assert global_route["total_wirelength_um"] > 0
    abstraction = result["hierarchical_top"]["integration_abstraction"]
    assert abstraction["pin_geometry_preserved"] is True
    assert abstraction["conservative_obstruction_cover"] is True
    assert abstraction["pin_count"] == abstraction["pin_rectangles"]
    assert abstraction["pin_rectangles"] == abstraction["accessible_pin_rectangles"]
    assert abstraction["source_obstruction_rectangles"] > 1_000_000
    assert abstraction["integration_obstruction_rectangles"] > 10
    assert abstraction["integration_obstruction_rectangles"] < (
        abstraction["source_obstruction_rectangles"]
    )
    assert abstraction["raster_pitch_um"] == 5.0
    assert abstraction["compression_ratio"] > 100
    legalization = result["hierarchical_top"]["channel_legalization"]
    assert legalization["cells"] == (
        result["hierarchical_top"]["synthesis"]["cell_count"]
        - result["hierarchical_top"]["macro_instances"]
    )
    assert legalization["rows"] > 0
    assert legalization["taps"] > 0
    assert legalization["minimum_capacity_ratio"] > 1.0
    macro_track = result["hierarchical_top"]["macro_track_contract"]
    assert result["checks"]["macro_track_alignment"] is True
    assert legalization["macro_instances_aligned"] == 16
    assert legalization["macro_origin_grid_dbu"] == macro_track["grid_dbu"]
    assert legalization["macro_max_displacement_dbu"] == 0
    assert macro_track["grid_dbu"] == 425600
    connectivity = result["hierarchical_top"]["route_connectivity"]
    assert result["checks"]["route_connectivity"] is True
    assert connectivity["all_pins_routed"] is True
    assert connectivity["global_route_completed"] is True
    assert connectivity["detailed_route_completed"] is True
    assert connectivity["detailed_pin_access_completed"] is True
    assert connectivity["global_missing_pin_routes"] == 0
    assert connectivity["global_missing_warning_limit_reached"] is False
    assert connectivity["detailed_stdcell_pins_without_access"] == 0
    assert connectivity["detailed_macro_pins_without_access"] == 0
    assert connectivity["detailed_no_access_errors"] == 0
    assert connectivity["detailed_off_grid_macro_terms"] >= 0
    assert connectivity["detailed_off_grid_block_terms"] >= 0
    route_contract = result["route_contract"]
    route_tool = result["global_route_tool"]
    assert result["checks"]["global_route_tool_provenance"] is True
    assert route_contract["grid_pitches_in_tile"] == 48
    assert route_contract["max_2d_edge_usage_multiplier"] == 101
    assert route_contract["stop_after_global_route"] is True
    assert route_contract["completion_markers"] == {
        "global_route": "MLX_ARRAY_STOP_AFTER_GRT",
        "detailed_route": "MLX_ARRAY_DROUTE_COMPLETE",
    }
    assert route_contract["required_zero_connectivity_failures"] == [
        "GRT-0026",
        "DRT-0073",
        "stdCellPinNoAp",
        "macroNoAp",
    ]
    assert route_contract["diagnostic_pin_access_warnings"] == [
        "DRT-0418",
        "DRT-0419",
        "DRT-0421",
    ]
    assert route_tool["grid_pitches_in_tile"] == 48
    assert route_tool["max_2d_edge_usage_multiplier"] == 101
    assert route_tool["binary"]["sha256"] == (
        "2fe0b0a5a576a4d940487b7ada0d62931ac0fc055e85653c498a08cef7f9a21f"
    )
