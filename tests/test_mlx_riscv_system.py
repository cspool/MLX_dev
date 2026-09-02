import hashlib
import json
import math
import os
from pathlib import Path

import yaml

from scripts.run_mlx_array_ppa import parse_openroad
from scripts.run_mlx_hierarchical_ppa import (
    aggregate_hierarchical_timing,
    all_congestion_iteration_reports,
    build_compact_macro_lef,
    congestion_iteration_reports,
    parse_channel_legalization,
    parse_congestion_marker_report,
    parse_cts_buffer_legalization,
    parse_global_route_metrics,
    parse_route_connectivity,
)

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = {"load", "store", "fma", "add", "max", "exp", "div", "shuffle", "xfer", "mul"}


def test_hierarchical_timing_uses_slowest_recursive_component() -> None:
    timing = aggregate_hierarchical_timing(
        1.0,
        {
            "hierarchical_top_shell": {"critical_path_delay_ns": 1.5},
            "pe_top": {"critical_path_delay_ns": 3.5},
            "functional_unit": {"critical_path_delay_ns": 27.5},
            "combinational_lane": {"critical_path_delay_ns": None},
        },
    )

    assert timing["critical_path_component"] == "functional_unit"
    assert timing["critical_path_delay_ns"] == 27.5
    assert math.isclose(timing["worst_slack_ns_at_target"], -26.5)
    assert math.isclose(timing["fmax_ghz"], 1.0 / 27.5)
    assert set(timing["candidates"]) == {
        "hierarchical_top_shell",
        "pe_top",
        "functional_unit",
    }


def test_compact_macro_lef_preserves_pins_and_outward_covers_obstructions(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("scripts.run_mlx_hierarchical_ppa.PROJECT_ROOT", tmp_path)
    source = tmp_path / "source.lef"
    destination = tmp_path / "compact.lef"
    prefix = """VERSION 5.8 ;
MACRO test_macro
  CLASS BLOCK ;
  ORIGIN 0 0 ;
  SIZE 10 BY 10 ;
  PIN A
    DIRECTION INPUT ;
    USE SIGNAL ;
    PORT
      LAYER metal2 ;
      RECT 0 4 1 5 ;
    END
  END A
"""
    source.write_text(
        prefix
        + """  OBS
    LAYER metal2 ;
      RECT 2.2 2.2 3.1 3.1 ;
      RECT 4.2 2.2 4.8 3.1 ;
    LAYER metal3 ;
      RECT 7.2 7.2 8.1 8.1 ;
  END
END test_macro
END LIBRARY
"""
    )

    result = build_compact_macro_lef(
        source,
        destination,
        {
            "integration_method": "test_conservative_raster_union",
            "source_method": "test_source",
            "integration_inset_um": 1.0,
            "raster_pitch_um": 2.0,
            "routing_layers": ["metal2", "metal3"],
            "preserve_pin_geometry": True,
        },
    )

    compact = destination.read_text()
    assert compact.startswith(prefix)
    assert "      RECT 1 1 5 5 ;" in compact
    assert "      RECT 7 7 9 9 ;" in compact
    assert result["pin_count"] == 1
    assert result["pin_rectangles"] == result["accessible_pin_rectangles"] == 1
    assert result["source_obstruction_rectangles"] == 3
    assert result["integration_obstruction_rectangles"] == 2
    assert result["occupied_raster_cells_by_layer"] == {"metal2": 4, "metal3": 1}
    assert result["conservative_obstruction_cover"] is True
    assert result["pin_geometry_preserved"] is True


def test_congestion_iteration_reports_reject_stale_suffixes(tmp_path: Path) -> None:
    report = tmp_path / "route-congestion.rpt"
    all_reports = [
        (2, tmp_path / "route-congestion-2.rpt"),
        (3, tmp_path / "route-congestion-3.rpt"),
        (10, tmp_path / "route-congestion-10.rpt"),
    ]
    for suffix, path in all_reports:
        path.write_text("iteration report\n")
        mtime_ns = {2: 300, 3: 200, 10: 100}[suffix]
        os.utime(path, ns=(mtime_ns, mtime_ns))
    (tmp_path / "route-congestion-not-an-iteration.rpt").write_text("ignored\n")
    (tmp_path / "other-congestion-3.rpt").write_text("ignored\n")

    assert all_congestion_iteration_reports(report) == all_reports
    assert congestion_iteration_reports(report) == all_reports[:1]


def test_congestion_marker_report_is_diagnostic_not_aggregate() -> None:
    metrics = parse_congestion_marker_report(
        """violation type: Horizontal congestion
\tsrcs: net:spm_rsp_rdata_i[1] net:tile_spm_wdata\\[0\\][2]
\tcomment: capacity:0 usage:2 overflow:2
violation type: Vertical congestion
\tsrcs: net:spm_rsp_rdata_i[2]
\tcomment: capacity:4 usage:5 overflow:1
"""
    )
    assert metrics == {
        "markers": 2,
        "horizontal_markers": 1,
        "vertical_markers": 1,
        "reported_marker_overflow_sum": 3,
        "max_marker_overflow": 2,
        "zero_capacity_markers": 1,
        "source_net_mentions": 3,
        "unique_source_nets": 3,
        "source_net_families": {
            "spm_rsp_rdata_i": 2,
            "tile_spm_wdata": 1,
        },
        "direction_marker_limit_reached": False,
        "aggregate_overflow_eligible": False,
    }


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


def test_route_connectivity_accepts_distributed_tile_completion_markers() -> None:
    connectivity = parse_route_connectivity(
        "MLX_TILE_STOP_AFTER_GRT checkpoint=/tmp/tile-grt.odb\n",
        """[INFO DRT-0166] Complete pin access.
#stdCellPinNoAp = 0
#macroNoAp = 0
MLX_TILE_DROUTE_COMPLETE odb=/tmp/tile.odb spef=/tmp/tile.spef drc=/tmp/tile.drc
""",
    )
    assert connectivity["global_route_completed"] is True
    assert connectivity["detailed_route_completed"] is True
    assert connectivity["all_pins_routed"] is True

    physical = parse_openroad(
        """[INFO DRT-0199]   Number of violations = 0.
MLX_TILE_DIE_UM 8603.000000 8603.000000
MLX_TILE_CORE_UM 8562.730000 8561.000000
""",
        1.0,
    )
    assert physical["drc_violations"] == 0
    assert physical["die_width_um"] == physical["die_height_um"] == 8603.0
    assert physical["core_width_um"] == 8562.73
    assert physical["core_height_um"] == 8561.0


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
    distributed = (ROOT / "rtl/mlx/mlx_array_4x4_distributed.sv").read_text()
    assert "one shared SIMD functional service" in cycle
    assert "mlx_array_4x4" not in cycle
    assert "for (pe = 0; pe < PE_COUNT" in array
    assert "mlx_pe_top" in array
    assert "packet_route_grant" in array
    assert "module mlx_array_4x4_centralized" in array
    assert "mlx_array_4x4_distributed" in array
    assert "module mlx_array_4x4_distributed" in distributed


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
        "conservative_2p5um_raster_union_per_routing_layer"
    )
    assert abstraction["raster_pitch_um"] == 2.5
    assert abstraction["obstruction_cover"] == (
        "outward_quantized_any_overlap_after_access_halo_clip"
    )
    assert abstraction["preserve_pin_geometry"] is True
    assert len(abstraction["routing_layers"]) == 10
    top_placement = config["hierarchical_top_placement"]
    assert top_placement["flow_tag"] == "compact-v8-u60-raster2p5-segment8192"
    assert top_placement["utilization_percent"] == 60
    floorplan_geometry = top_placement["floorplan_geometry"]
    assert floorplan_geometry["source"] == (
        "expected_same_u60_boundary_with_raster2p5_abstraction"
    )
    assert floorplan_geometry["horizontal_channel_um"] > 1800
    assert floorplan_geometry["vertical_channel_um"] > 1800
    assert floorplan_geometry["grid_growth_vs_u80"] == 1.333
    assert floorplan_geometry["status"] == "candidate"
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
    assert "rtl/mlx/mlx_array_pe_tile.sv" in config["rtl_sources"]
    assert "rtl/mlx/mlx_array_4x4_distributed.sv" in config["rtl_sources"]
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


def test_distributed_tile_candidate_is_promoted_but_not_final() -> None:
    config = yaml.safe_load((ROOT / "configs/system/mlx_array_ppa_v1.yaml").read_text())
    candidate = config["hierarchical_distributed_tile_candidate"]
    assert candidate["promoted_to_production_top"] is True
    assert candidate["result_consumed_as_final_ppa"] is False
    assert all(
        record["golden_match"]
        for record in candidate["functional_probe"]["workloads"].values()
    )
    structural = candidate["structural_probe"]
    assert structural["recursive_non_pe_cells"] == (
        16 * structural["tile_wrapper_nonmacro_cells_per_tile"]
        + structural["distributed_top_nonmacro_cells"]
    )
    assert structural["recursive_cell_reduction_vs_centralized"] > 0.70
    assert structural["recursive_area_reduction_vs_centralized"] > 0.65
    floorplan = candidate["tile_floorplan_v2_tight"]
    assert floorplan["pe_origin_grid_um"] == 212.8
    assert floorplan["tile48_grt_iter5"]["missing_route_warnings"] == 0
    assert floorplan["tile48_grt_iter5"]["aggregate_overflow"] > 0
    iter50 = floorplan["tile48_grt_iter50"]
    assert iter50["missing_route_warnings"] == 0
    assert iter50["final_2d_markers"] < 200
    assert iter50["aggregate_overflow"] > floorplan["tile48_grt_iter5"][
        "aggregate_overflow"
    ]
    assert all(value == 0 for value in floorplan["detailed_route_probe"]["pin_access_status"].values())
    loose = candidate["tile_floorplan_v1_rejected"]["tile48_grt_iter5"]
    assert loose["resource"] > floorplan["tile48_grt_iter5"]["resource"]
    assert loose["aggregate_overflow"] > floorplan["tile48_grt_iter5"][
        "aggregate_overflow"
    ]
    top_floorplan = candidate["distributed_top_floorplan_candidate"]
    assert top_floorplan["tile_macro_count"] == 16
    assert top_floorplan["utilization_percent"] == 70
    assert top_floorplan["expected_inter_tile_channel_um"] > 1300
    assert top_floorplan["status"] == (
        "tile48_grt_iter5_complete_droute_local_repair4_ready"
    )
    assert top_floorplan["legal_cells"] == 97260
    assert top_floorplan["cts_buffers"] == 1783
    top_grt = top_floorplan["tile48_grt_iter5_final"]
    assert top_grt["routed_nets"] == 117628
    assert top_grt["missing_route_warnings"] == 0
    assert top_grt["aggregate_overflow"] == 302129
    top_droute = top_floorplan["detailed_route_progress"]
    assert top_droute["pin_access"]["std_cell_pins_without_access"] == 0
    assert top_droute["pin_access"]["macro_pins_without_access"] == 0
    assert top_droute["pin_access"]["drt_0073"] == 0
    assert top_droute["pin_access"]["groups"] == 94679
    assert top_droute["initial_route"]["violations"] == 170675
    assert top_droute["initial_route"]["short_violations"] < 200000
    first_optimization = top_droute["optimization_iterations"][0]
    assert first_optimization["iteration"] == 1
    assert first_optimization["violations"] == 60216
    assert first_optimization["violations"] < top_droute["initial_route"]["violations"]
    assert first_optimization["metal2_short_violations"] == 38840
    full_route = top_droute["full_route_result"]
    assert full_route["completed_optimization_iterations"] == 20
    assert full_route["violation_curve"][0] == 170675
    assert full_route["violation_curve"][-1] == full_route["final_violations"] == 49
    assert full_route["final_short_violations"] == 39
    assert full_route["status"] == "complete_nonzero_drc_repair_required"
    assert top_droute["repair1"]["input_violations"] == 49
    assert top_droute["repair1"]["failure_code"] == "DRT-1010"
    assert top_droute["repair1"]["output_generated"] is False
    assert top_droute["clean_retry1"]["droute_end_iter"] == 50
    assert top_droute["clean_retry1"]["preserves_base_route_result"] is True
    clean_retry = top_droute["clean_retry1"]["full_route_result"]
    assert clean_retry["completed_optimization_iterations"] == 50
    assert len(clean_retry["violation_curve"]) == 51
    assert clean_retry["violation_curve"][-1] == clean_retry["final_violations"] == 22
    assert clean_retry["best_violations"] == 18
    assert clean_retry["best_iterations"] == [43, 44, 45, 46, 47, 48]
    assert clean_retry["final_short_violations"] == 21
    assert clean_retry["final_metal_spacing_violations"] == 1
    assert clean_retry["drt_completed"] is True
    assert clean_retry["rcx_sta_power_completed"] is True
    repair2 = top_droute["repair2"]
    assert repair2["input_violations"] == 22
    assert repair2["droute_end_iter"] == 64
    assert repair2["preserves_clean_retry_result"] is True
    assert repair2["first_iteration_violations"] == 55797939
    assert repair2["output_generated"] is False
    repair3 = top_droute["repair3"]
    assert repair3["import_probe"]["exact_layer_wirelength_match"] is True
    assert repair3["import_probe"]["wirelength_um"] == 682255030
    assert repair3["import_probe"]["vias"] == 2016530
    repair3_initial = repair3["optimization_iteration_0_result"]
    assert repair3_initial["final_violations"] == 55
    assert repair3_initial["exact_layer_wirelength_match"] is True
    assert repair3["output_generated"] is False
    repair4 = top_droute["repair4"]
    assert repair4["skip_redundant_incremental_iterations"] == [1, 2]
    assert repair4["first_drc_repair_iteration"] == 3
    assert top_droute["current_iteration"] == "local_repair4"
    droute = floorplan["detailed_route_probe"]
    assert droute["final_stubborn_iteration_violations"] == 0
    assert droute["status"] == "complete_zero_drc_zero_pin_access_failures"
    assert len(candidate["promotion_gates"]) == 9

    tile = (ROOT / candidate["rtl_sources"]["tile"]).read_text()
    distributed = (ROOT / candidate["rtl_sources"]["distributed_top"]).read_text()
    assert "module mlx_array_pe_tile" in tile
    assert ".rf_write_data_i(rf_write_data)" in tile
    assert "module mlx_array_4x4_distributed" in distributed
    assert "route_candidate" in distributed
    assert "mlx_array_pe_tile" in distributed
    hierarchical_flow = (
        ROOT / "rtl/ppa/openroad_hierarchical_array_flow.tcl"
    ).read_text()
    assert "PPA_TOP" in hierarchical_flow
    assert "PPA_MACRO_INSTANCE_KIND" in hierarchical_flow
    drt_patch = (ROOT / "patches/openroad/drt-point-ext-orthogonal.patch").read_text()
    assert "dbWireDecoder::POINT_EXT" in drt_patch
    assert "&& hasEndPoint" in drt_patch
    assert "prevLayer = decoder.getLayer()" in drt_patch
    assert "MLX_DRT_STOP_AFTER_IMPORT" in drt_patch
    assert "MLX_DRT_SKIP_REDUNDANT_INCREMENTAL" in drt_patch
    assert "GENERATE_TILES\\[%d\\].physical_tile" in hierarchical_flow
    runner = ROOT / candidate["physical_flows"]["runner"]
    assert runner.is_file() and os.access(runner, os.X_OK)
    top_runner = ROOT / candidate["physical_flows"]["top_signoff_runner"]
    assert top_runner.is_file() and os.access(top_runner, os.X_OK)
    signoff = candidate["distributed_top_signoff_contract"]
    assert signoff["macro_instances"] == 16
    assert signoff["congestion_iterations"] == 5
    assert signoff["require_zero_global_route_overflow"] is False
    assert signoff["require_zero_detailed_route_drc"] is True
    assert signoff["require_zero_detailed_pin_access_failures"] is True
    tile_result = json.loads(
        (ROOT / candidate["evidence"]["tile_candidate_summary"]).read_text()
    )
    assert tile_result["status"] == "supported"
    assert all(tile_result["required_checks"].values())
    assert tile_result["checks"]["global_route_overflow_is_zero"] is False
    assert tile_result["physical"]["drc_violations"] == 0


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
    assert result["schema_version"] == 2
    assert result["status"] == "supported"
    assert all(result["checks"].values())
    top = result["hierarchical_top"]
    assert top["implementation"] == "distributed_autonomous_pe_tiles"
    assert top["macro_master"] == "mlx_array_pe_tile"
    assert top["macro_instances"] == 16
    assert result["physical"]["drc_violations"] == 0
    assert result["physical"]["die_area_um2"] > 0
    assert result["physical"]["fmax_ghz"] > 0
    assert result["physical"]["total_power_w"] > 0
    timing_hierarchy = result["physical"]["timing_hierarchy"]
    assert timing_hierarchy["method"] == (
        "worst_postroute_delay_across_recursive_hierarchy"
    )
    assert timing_hierarchy["target_clock_period_ns"] == 1.0
    assert timing_hierarchy["critical_path_component"] == max(
        timing_hierarchy["candidates"],
        key=lambda name: timing_hierarchy["candidates"][name][
            "critical_path_delay_ns"
        ],
    )
    assert math.isclose(
        result["physical"]["critical_path_delay_ns"],
        timing_hierarchy["critical_path_delay_ns"],
    )
    assert math.isclose(
        result["physical"]["fmax_ghz"], timing_hierarchy["fmax_ghz"]
    )
    assert result["physical"]["power_aggregation"] == (
        "recursive_distributed_postroute_transformer_vcd_hierarchy"
    )
    assert "distributed_top_shell" in timing_hierarchy["candidates"]
    assert "autonomous_tile_shell" in timing_hierarchy["candidates"]
    assert all(
        item["checks"]["pin_access"] is True
        and item["pin_access"]["all_pins_accessible"] is True
        and item["pin_access"]["stdcell_pins_without_access"] == 0
        and item["pin_access"]["macro_pins_without_access"] == 0
        and item["pin_access"]["no_access_errors"] == 0
        for item in result["submacro_chain"].values()
    )
    assert result["tile_macro"]["status"] == "supported"
    assert all(result["tile_macro"]["required_checks"].values())
    route_contract = result["route_contract"]
    global_route = top["global_route_metrics"]
    assert result["checks"]["all_nets_globally_routed"] is True
    assert global_route["resource_total_uses_64bit_layer_sum"] is True
    assert global_route["aggregate_overflow_consistent"] is True
    assert global_route["resource"] > 0
    assert global_route["demand"] > 0
    assert global_route["total_overflow"] >= 0
    assert global_route["routed_nets"] > 0
    assert global_route["final_vias"] > 0
    assert global_route["total_wirelength_um"] > 0
    assert global_route["congestion_iterations"] == result["route_contract"][
        "congestion_iterations"
    ]
    assert result["diagnostics"]["global_route_overflow_is_zero"] == (
        global_route["overflow_resolved"] is True
    )
    iteration_reports = top["global_route_iteration_reports"]
    assert iteration_reports
    assert [item["file_suffix"] for item in iteration_reports] == sorted(
        item["file_suffix"] for item in iteration_reports
    )
    assert all(
        item["completed_iteration"] == item["file_suffix"] - 1
        for item in iteration_reports
    )
    assert iteration_reports[-1]["completed_iteration"] == route_contract[
        "congestion_iterations"
    ]
    assert all(
        item["marker_metrics"]["aggregate_overflow_eligible"] is False
        for item in iteration_reports
    )
    abstraction = top["integration_abstraction"]
    assert abstraction["pin_geometry_preserved"] is True
    assert abstraction["conservative_obstruction_cover"] is True
    assert abstraction["pin_count"] == abstraction["pin_rectangles"]
    assert abstraction["pin_rectangles"] == abstraction["accessible_pin_rectangles"]
    assert abstraction["source_obstruction_rectangles"] > 100_000
    assert abstraction["integration_obstruction_rectangles"] > 10
    assert abstraction["integration_obstruction_rectangles"] < (
        abstraction["source_obstruction_rectangles"]
    )
    assert abstraction["raster_pitch_um"] == 2.5
    assert abstraction["compression_ratio"] > 1
    legalization = top["channel_legalization"]
    assert legalization["cells"] == (
        top["synthesis"]["cell_count"] - top["macro_instances"]
    )
    assert legalization["rows"] >= route_contract["minimum_physical_rows"]
    assert legalization["row_segments"] >= route_contract["minimum_row_segments"]
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
    assert legalization["maximum_accepted_displacement_dbu"] == 8_603_000
    assert legalization["maximum_accepted_displacement_basis"] == (
        "half_of_8603um_tile_macro_span"
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
    macro_track = top["macro_track_contract"]
    assert result["checks"]["macro_track_alignment"] is True
    assert legalization["macro_instances_aligned"] == 16
    assert legalization["macro_origin_grid_dbu"] == macro_track["grid_dbu"]
    assert legalization["macro_max_displacement_dbu"] == 0
    cts_legalization = top["cts_buffer_legalization"]
    assert result["checks"]["cts_buffer_legalization"] is True
    assert cts_legalization["buffers"] > 0
    assert cts_legalization["assigned_buffers"] == cts_legalization["buffers"]
    assert cts_legalization["site_aligned_buffers"] == cts_legalization["buffers"]
    assert cts_legalization["segment_contained_buffers"] == cts_legalization["buffers"]
    assert cts_legalization["fixed_clear_buffers"] == cts_legalization["buffers"]
    assert cts_legalization["standard_nonoverlap_buffers"] == cts_legalization["buffers"]
    assert cts_legalization["max_displacement_dbu"] <= 8_603_000
    assert macro_track["grid_dbu"] == 425600
    connectivity = top["route_connectivity"]
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
    detailed_route = top["detailed_route_progress"]
    assert detailed_route["last_completed_optimization_iteration"] >= 20
    assert detailed_route["reported_violation_counts"] == len(
        detailed_route["violation_curve"]
    )
    assert detailed_route["final_violations"] == 0
    assert detailed_route["zero_drc_reached"] is True
    route_tool = result["global_route_tool"]
    assert result["checks"]["global_route_tool_provenance"] is True
    assert route_contract["grid_pitches_in_tile"] == 48
    assert route_contract["max_2d_edge_usage_multiplier"] == 101
    assert route_contract["congestion_iterations"] == 5
    assert route_contract["stop_after_global_route"] is True
    assert route_contract["require_zero_global_route_overflow"] is False
    assert route_contract["global_route_overflow_policy"] == (
        "diagnostic_after_all_nets_routed_actual_droute_is_authoritative"
    )
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
    local_repair_tool = route_tool["local_repair_openroad"]
    assert local_repair_tool["base_commit"] == route_tool["base_commit"]
    assert "drt-postroute-repair" in local_repair_tool["version"]
    assert local_repair_tool["binary"]["sha256"] == (
        "d43ecf4a09e1dbe25a38b6d4134d7d6ca059c305bae34cf3d9527a513cebcb67"
    )
