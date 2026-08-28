import hashlib
import json
import math
from pathlib import Path

import yaml

from scripts.run_mlx_hierarchical_ppa import (
    parse_channel_legalization,
    parse_cts_buffer_legalization,
    parse_global_route_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = {"load", "store", "fma", "add", "max", "exp", "div", "shuffle", "xfer", "mul"}


def test_channel_legalization_parser_tracks_resumable_hybrid_flow() -> None:
    metrics = parse_channel_legalization(
        """MLX_CHANNEL_ROWS_RESUME checkpoint=/tmp/rows.odb
MLX_MACRO_TRACK_ALIGNMENT macros=16 grid_dbu=425600 max_displacement_dbu=0
MLX_CHANNEL_ROW_SELECTION physical_rows=8192 row_segments=37504 removed_segments=0
MLX_CHANNEL_ROW_AUDIT nonoverlapping_macro_clear_segments=37504
MLX_CHANNEL_ROWS_CHECKPOINT checkpoint=/tmp/rows.odb
MLX_CHANNEL_ASSIGNMENT cells=775745 full_width_y_escapes=18361
MLX_CHANNEL_1D_LEGALIZATION backward_compactions=21
MLX_CHANNEL_CONSTRUCTIVE_AUDIT cells=775745 site_aligned=775745 segment_contained=775745 standard_nonoverlap=775745 row_segments=37504
MLX_CHANNEL_SEED_CHECKPOINT checkpoint=/tmp/seed.odb
MLX_CHANNEL_PRECHECK checkpoint=/tmp/precheck.odb
MLX_CHANNEL_LEGALIZER cells=775745 rows=8192 row_segments=37504 taps=1234 removed_rows=0 removed_tapcells=0 max_displacement_dbu=100 min_capacity_ratio=2.0 checkpoint=/tmp/legal.odb
MLX_CHANNEL_LOCALITY max_x_displacement_dbu=80 max_y_displacement_dbu=20 average_displacement_dbu=3.5
"""
    )
    assert metrics["selected_physical_rows"] == metrics["rows"] == 8192
    assert metrics["selected_row_segments"] == metrics["row_segments"] == 37504
    assert metrics["assigned_cells"] == metrics["cells"] == 775745
    assert metrics["full_width_y_escapes"] == 18361
    assert metrics["backward_compactions"] == 21
    assert metrics["audited_nonoverlapping_macro_clear_row_segments"] == 37504
    assert metrics["constructive_audit_cells"] == 775745
    assert metrics["site_aligned_cells"] == 775745
    assert metrics["segment_contained_cells"] == 775745
    assert metrics["standard_nonoverlap_cells"] == 775745
    assert metrics["constructive_audit_row_segments"] == 37504
    assert metrics["rows_checkpoint"] == "/tmp/rows.odb"
    assert metrics["resumed_from_rows_checkpoint"] is True
    assert metrics["seed_checkpoint"] == "/tmp/seed.odb"
    assert metrics["precheck_checkpoint"] == "/tmp/precheck.odb"


def test_cts_buffer_legalization_parser_tracks_constructive_audit() -> None:
    metrics = parse_cts_buffer_legalization(
        """MLX_ARRAY_CTS_SEED checkpoint=/tmp/cts-seed.odb
MLX_CTS_BUFFER_ASSIGNMENT buffers=6408 fixed_cells=880129 physical_rows=8192 row_segments=37504
MLX_CTS_BUFFER_LEGALIZATION buffers=6408 backward_compactions=3 site_aligned=6408 segment_contained=6408 fixed_clear=6408 standard_nonoverlap=6408 max_displacement_dbu=100 average_displacement_dbu=2.5
MLX_ARRAY_STOP_AFTER_CTS checkpoint=/tmp/post-cts.odb
"""
    )
    assert metrics["assigned_buffers"] == metrics["buffers"] == 6408
    assert metrics["fixed_cells"] == 880129
    assert metrics["physical_rows"] == 8192
    assert metrics["row_segments"] == 37504
    assert metrics["site_aligned_buffers"] == 6408
    assert metrics["segment_contained_buffers"] == 6408
    assert metrics["fixed_clear_buffers"] == 6408
    assert metrics["standard_nonoverlap_buffers"] == 6408
    assert metrics["seed_checkpoint"] == "/tmp/cts-seed.odb"
    assert metrics["checkpoint"] == "/tmp/post-cts.odb"


def test_global_route_metrics_use_64bit_layer_aggregation() -> None:
    metrics = parse_global_route_metrics(
        """MLX_GRT_ROUTE_ARGS -congestion_iterations 1 -critical_nets_percentage 0
[INFO GRT-0111] Final number of vias: 12
[INFO GRT-0112] Final usage 3D: 34
[INFO GRT-0096] Final congestion report:
Layer         Resource        Demand        Usage (%)    Max H / Max V / Total Overflow
metal3      1500000000             100            0.00%             1 /  2 /  3
metal4      1000000000             200            0.00%             4 /  5 /  6
Total       -1794967296             300           -0.00%             5 /  7 /  9
[INFO GRT-0018] Total wirelength: 56 um
[INFO GRT-0014] Routed nets: 78
[WARNING GRT-0115] Global routing finished with congestion.
"""
    )
    assert metrics["reported_total_resource"] == -1_794_967_296
    assert metrics["congestion_iterations"] == 1
    assert metrics["resource"] == 2_500_000_000
    assert metrics["demand"] == 300
    assert metrics["total_overflow"] == 9
    assert metrics["aggregate_overflow_consistent"] is True
    assert metrics["resource_total_uses_64bit_layer_sum"] is True
    assert metrics["overflow_resolved"] is False


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
    top_placement = config["hierarchical_top_placement"]
    assert top_placement["flow_tag"] == "compact-v7-u60-segment8192"
    assert top_placement["utilization_percent"] == 60
    floorplan_geometry = top_placement["floorplan_geometry"]
    assert floorplan_geometry["source"] == (
        "measured_compact_v7_u60_global_placement"
    )
    assert floorplan_geometry["horizontal_channel_um"] > 1800
    assert floorplan_geometry["vertical_channel_um"] > 1800
    assert floorplan_geometry["grid_growth_vs_u80"] == 1.333
    assert floorplan_geometry["status"] == "measured"
    assert top_placement["channel_legalizer"][
        "maximum_accepted_displacement_um"
    ] == 1803
    assert top_placement["channel_legalizer"][
        "maximum_accepted_displacement_basis"
    ] == "one_u60_pe_channel_span"
    macro_track = top_placement["macro_origin_track_alignment"]
    assert macro_track["grid_dbu"] == math.lcm(
        *macro_track["routing_pitch_dbu"].values()
    )
    assert macro_track["grid_um"] == (
        macro_track["grid_dbu"] / macro_track["dbu_per_micron"]
    )
    assert macro_track["required_macro_instances"] == 16
    capacity = top_placement["detailed_placement_capacity_basis"]
    assert capacity["reserved_site_capacity"] > 8 * capacity["estimated_required_sites"]
    route_plan = config["hierarchical_top_route_resource_plan"]
    assert route_plan["routing_layers"]["signal"] == "metal2-metal10"
    assert route_plan["routing_layers"]["clock"] == "metal5-metal10"
    assert route_plan["layer_capacity_adjustments"] == {}
    assert route_plan["grid_pitches_in_tile"] == 48
    assert route_plan["max_2d_edge_usage_multiplier"] == 101
    assert route_plan["verbose"] is True
    assert route_plan["congestion_report_iter_step"] == 1
    assert route_plan["stop_after_global_route"] is True
    post_legal_flow = (
        ROOT / "rtl/ppa/openroad_hierarchical_array_post_legal_flow.tcl"
    ).read_text()
    assert "-congestion_report_file" in post_legal_flow
    assert "-congestion_report_iter_step" in post_legal_flow
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
    assert global_route["congestion_iterations"] == result["route_contract"][
        "congestion_iterations"
    ]
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
    assert legalization["rows"] >= 8192
    assert legalization["row_segments"] >= 30000
    assert legalization["selected_physical_rows"] == legalization["rows"]
    assert legalization["selected_row_segments"] == legalization["row_segments"]
    assert legalization["assigned_cells"] == legalization["cells"]
    assert legalization["constructive_audit_cells"] == legalization["cells"]
    assert legalization["site_aligned_cells"] == legalization["cells"]
    assert legalization["segment_contained_cells"] == legalization["cells"]
    assert legalization["standard_nonoverlap_cells"] == legalization["cells"]
    assert legalization["audited_nonoverlapping_macro_clear_row_segments"] == (
        legalization["row_segments"]
    )
    assert legalization["constructive_audit_row_segments"] == legalization["row_segments"]
    assert legalization["full_width_y_escapes"] >= 0
    assert legalization["backward_compactions"] >= 0
    assert legalization["taps"] > 0
    assert legalization["minimum_capacity_ratio"] > 1.0
    assert legalization["maximum_accepted_displacement_dbu"] == 3_606_000
    assert legalization["maximum_accepted_displacement_basis"] == (
        "one_u60_pe_channel_span"
    )
    assert legalization["max_displacement_dbu"] <= legalization[
        "maximum_accepted_displacement_dbu"
    ]
    assert legalization["maximum_x_displacement_dbu"] <= legalization[
        "maximum_accepted_displacement_dbu"
    ]
    assert legalization["maximum_y_displacement_dbu"] <= legalization[
        "maximum_accepted_displacement_dbu"
    ]
    assert legalization["average_displacement_dbu"] >= 0
    macro_track = result["hierarchical_top"]["macro_track_contract"]
    assert result["checks"]["macro_track_alignment"] is True
    assert legalization["macro_instances_aligned"] == 16
    assert legalization["macro_origin_grid_dbu"] == macro_track["grid_dbu"]
    assert legalization["macro_max_displacement_dbu"] == 0
    cts_legalization = result["hierarchical_top"]["cts_buffer_legalization"]
    assert result["checks"]["cts_buffer_legalization"] is True
    assert cts_legalization["buffers"] > 0
    assert cts_legalization["assigned_buffers"] == cts_legalization["buffers"]
    assert cts_legalization["site_aligned_buffers"] == cts_legalization["buffers"]
    assert cts_legalization["segment_contained_buffers"] == cts_legalization["buffers"]
    assert cts_legalization["fixed_clear_buffers"] == cts_legalization["buffers"]
    assert cts_legalization["standard_nonoverlap_buffers"] == cts_legalization["buffers"]
    assert cts_legalization["max_displacement_dbu"] <= 7_731_275
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
    assert route_contract["congestion_iterations"] == 50
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
    assert route_contract["routing_layers"]["signal"] == "metal2-metal10"
    assert route_tool["grid_pitches_in_tile"] == 48
    assert route_tool["max_2d_edge_usage_multiplier"] == 101
    assert route_tool["binary"]["sha256"] == (
        "2fe0b0a5a576a4d940487b7ada0d62931ac0fc055e85653c498a08cef7f9a21f"
    )
