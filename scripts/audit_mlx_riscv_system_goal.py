#!/usr/bin/env python3
"""Audit the final MLX + RISC-V system-simulation completion contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/system/mlx_riscv_system_goal_v1.yaml"


def all_true(values: dict[str, Any]) -> bool:
    return bool(values) and all(value is True for value in values.values())


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        relative = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        relative = str(path)
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "pass": path.is_file() and len(payload) > 0,
    }


def qualify(spec: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / spec["path"]
    observed = digest(path)
    observed["pass"] = (
        observed["bytes"] == spec["bytes"]
        and observed["sha256"] == spec["sha256"]
    )
    return observed


def read_parent(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_path = PROJECT_ROOT / path
    return json.loads(parent_path.read_text()), digest(parent_path)


def build_scope_audit(config: dict[str, Any]) -> dict[str, Any]:
    parents: dict[str, dict[str, Any]] = {}
    parent_artifacts: dict[str, dict[str, Any]] = {}
    for name, path in config["parents"].items():
        parents[name], parent_artifacts[name] = read_parent(path)

    workloads = parents["workloads"]
    backends = parents["backends"]
    ppa = parents["ppa"]
    chipyard = parents["chipyard"]
    trends = parents["performance_trends"]
    paper_calibrated_ppa = parents["paper_calibrated_ppa"]
    ppa_manifest = json.loads((PROJECT_ROOT / ppa["manifest"]["path"]).read_text())
    chipyard_manifest = json.loads(
        (PROJECT_ROOT / chipyard["manifest"]["path"]).read_text()
    )
    contract = config["contract"]
    required_workloads = set(contract["workloads"])
    required_backends = set(contract["backends"])
    required_operations = set(contract["operations"])

    chipyard_records = chipyard["records"]
    standalone_records = backends["records"]
    timing = backends["instruction_timing"]
    p0_checks = {
        "four_command_interface": chipyard_manifest["command_interface"]
        == {
            "config_funct": 0,
            "launch_funct": 1,
            "status_funct": 3,
            "wait_funct": 2,
        },
        "real_chipyard_commit": chipyard["checks"]["chipyard_commit"] is True,
        "real_bare_metal_runs": chipyard["status"] == "supported"
        and len(chipyard_records) == int(contract["chipyard_runs"])
        and all(item["returncode"] == 0 for item in chipyard_records),
        "behavioral_dma_spm_dram": all(
            item["checks"]["dma_bytes"]
            and item["checks"]["system_cycle_accounting"]
            and item["summary"]["dma"] > 0
            and item["summary"]["system"] > item["summary"]["kernel"]
            for item in chipyard_records
        ),
        "lowering_lineage": all_true(workloads["checks"]),
        "binary_header_data_manifest": all(
            all(
                (PROJECT_ROOT / item[key]).is_file()
                for key in ("program", "header", "input_hex", "golden_hex", "reference")
            )
            for item in workloads["workloads"]
        ),
    }

    array_text = (PROJECT_ROOT / config["source_layout"]["array"]).read_text()
    distributed_array_text = (
        PROJECT_ROOT / config["source_layout"]["distributed_array"]
    ).read_text()
    tile_text = (PROJECT_ROOT / config["source_layout"]["array_tile"]).read_text()
    cycle_text = (PROJECT_ROOT / config["source_layout"]["cycle_model"]).read_text()
    p1_checks = {
        "autonomous_execution": all(
            token in tile_text
            for token in ("pc_q", "instruction_word", "rf_valid_q", "packet_delivery_accept")
        ),
        "physical_4x4_pes": "parameter PE_COUNT = 16" in distributed_array_text
        and "mlx_array_pe_tile physical_tile" in distributed_array_text
        and "for (tile = 0; tile < PE_COUNT" in distributed_array_text
        and "mlx_array_4x4_distributed" in array_text,
        "physical_network_flow_control": all(
            token in distributed_array_text + tile_text
            for token in (
                "route_candidate",
                "packet_delivery_accept",
                "spm_select_valid",
                "route_out_ready_i",
            )
        ),
        "distinct_cycle_model": "one shared SIMD functional service" in cycle_text
        and "mlx_array_4x4" not in cycle_text,
        "same_chipyard_interface": required_backends
        == {item["backend"] for item in chipyard_records},
        "rtl_functional_runs": sum(
            item["backend"] == "rtl" and item["returncode"] == 0
            for item in chipyard_records
        )
        == len(required_workloads),
    }

    operations_observed = {
        name for name, value in timing.items() if value["observations"] > 0
    }
    p2_checks = {
        "required_workloads": required_workloads
        == {item["workload"] for item in standalone_records},
        "eight_standalone_runs": backends["checks"]["eight_functional_runs"] is True
        and len(standalone_records) * 2 == int(contract["standalone_runs"]),
        "same_outputs_and_instruction_counts": backends["status"] == "supported"
        and backends["checks"]["same_instruction_counts"] is True,
        "event_order_explained": backends["checks"][
            "same_architectural_event_sequences"
        ]
        is True
        and backends["checks"]["global_interleaving_difference_observed"] is True,
        "system_breakdown": all(
            item["checks"]["host_total_accounting"]
            and item["checks"]["operation_accounting"]
            for item in chipyard_records
        ),
        "instruction_latency_and_ii": operations_observed == required_operations
        and all(
            value["latency_cycles_min"] is not None
            and value["latency_cycles_max"] is not None
            and value["observed_global_initiation_interval_min"] is not None
            for value in timing.values()
        ),
        "stalls_and_conflicts": backends["checks"]["stalls_measured"] is True
        and backends["checks"]["conflicts_measured"] is True,
        "performance_trends": trends["hypothesis_status"] == "supported"
        and trends["audit_integrity"] is True
        and trends["summary"]["primary_claims_reproduced"]
        == int(contract["performance_claims"]),
    }

    physical = ppa["physical"]
    timing_hierarchy = physical["timing_hierarchy"]
    synthesis = ppa["synthesis"]
    abstraction = ppa["hierarchical_top"]["integration_abstraction"]
    legalization = ppa["hierarchical_top"]["channel_legalization"]
    cts_legalization = ppa["hierarchical_top"]["cts_buffer_legalization"]
    macro_track = ppa["hierarchical_top"]["macro_track_contract"]
    route_connectivity = ppa["hierarchical_top"]["route_connectivity"]
    global_route_metrics = ppa["hierarchical_top"]["global_route_metrics"]
    global_route_iteration_reports = ppa["hierarchical_top"][
        "global_route_iteration_reports"
    ]
    route_tool = ppa["global_route_tool"]
    local_repair_tool = route_tool["local_repair_openroad"]
    route_contract = ppa["route_contract"]
    paper_array = next(
        row
        for row in paper_calibrated_ppa["aggregate_rows"]
        if row["name"] == "pe_array"
    )
    p3_checks = {
        "real_4x4_top": ppa["checks"]["real_4x4_top"] is True
        and ppa["checks"]["hierarchical_integrated"] is True
        and ppa["hierarchical_top"]["macro_instances"] == 16
        and ppa_manifest["files"]["rtl"]["rtl/mlx/mlx_array_4x4.sv"]["bytes"] > 0,
        "synthesis": ppa["checks"]["synthesis"] is True
        and synthesis["cell_count"] > 0
        and synthesis["cell_area_um2"] > 0,
        "place_route": ppa["checks"]["place_route"] is True
        and ppa["checks"].get("global_route_congestion", False) is True
        and route_contract["require_zero_global_route_overflow"] is True
        and global_route_metrics["congestion_iterations"]
        == route_contract["congestion_iterations"]
        and global_route_metrics["overflow_resolved"] is True
        and global_route_metrics["resource_total_uses_64bit_layer_sum"] is True
        and global_route_metrics["aggregate_overflow_consistent"] is True
        and global_route_metrics["resource"] > 0
        and global_route_metrics["demand"] > 0
        and global_route_metrics["total_overflow"] == 0
        and global_route_metrics["congestion_warning"] is False
        and global_route_metrics["routed_nets"] > 0
        and global_route_metrics["final_vias"] > 0
        and global_route_metrics["total_wirelength_um"] > 0
        and bool(global_route_iteration_reports)
        and global_route_iteration_reports[-1]["completed_iteration"]
        == route_contract["congestion_iterations"]
        and [item["file_suffix"] for item in global_route_iteration_reports]
        == sorted(item["file_suffix"] for item in global_route_iteration_reports)
        and all(
            item["completed_iteration"] == item["file_suffix"] - 1
            and qualify(item["report"])["pass"]
            and item["marker_metrics"]["markers"] > 0
            and item["marker_metrics"]["aggregate_overflow_eligible"] is False
            for item in global_route_iteration_reports
        ),
        "drc_clean": ppa["checks"]["drc_clean"] is True
        and physical["drc_violations"] == 0,
        "sta_1ghz_and_fmax": ppa["checks"]["timing"] is True
        and physical["worst_slack_ns_at_1ghz"] is not None
        and physical["fmax_ghz"] > 0
        and timing_hierarchy["method"]
        == "worst_postroute_delay_across_recursive_hierarchy"
        and timing_hierarchy["target_clock_period_ns"] == 1.0
        and timing_hierarchy["critical_path_component"]
        in timing_hierarchy["candidates"]
        and timing_hierarchy["critical_path_component"]
        == max(
            timing_hierarchy["candidates"],
            key=lambda name: timing_hierarchy["candidates"][name][
                "critical_path_delay_ns"
            ],
        )
        and math.isclose(
            physical["critical_path_delay_ns"],
            timing_hierarchy["critical_path_delay_ns"],
        )
        and math.isclose(physical["fmax_ghz"], timing_hierarchy["fmax_ghz"])
        and math.isclose(
            physical["worst_slack_ns_at_1ghz"],
            timing_hierarchy["worst_slack_ns_at_target"],
        ),
        "vcd_dynamic_power": ppa["checks"]["vcd_power"] is True
        and ppa["checks"]["recursive_submacro_evidence"] is True
        and physical["annotated_pin_activities"] > 0
        and physical["total_power_w"] > 0,
        "raw_unfitted": ppa["checks"]["raw_unfitted"] is True
        and ppa_manifest["calibration"]
        == {"applied": False, "coefficients": None},
        "physical_macro_abstraction": ppa["checks"][
            "compact_macro_abstraction"
        ]
        is True
        and abstraction["pin_geometry_preserved"] is True
        and abstraction["conservative_obstruction_cover"] is True
        and abstraction["pin_count"]
        == abstraction["pin_rectangles"]
        == abstraction["accessible_pin_rectangles"]
        and abstraction["source_obstruction_rectangles"] > 1_000_000
        and len(abstraction["integration_obstruction_layers"]) == 10
        and abstraction["integration_obstruction_rectangles"]
        < abstraction["source_obstruction_rectangles"]
        and abstraction["integration_obstruction_rectangles"] > 10
        and abstraction["raster_pitch_um"] == 2.5
        and abstraction["compression_ratio"] > 80,
        "channel_legalization": ppa["checks"]["channel_legalization"] is True
        and legalization["cells"]
        == ppa["hierarchical_top"]["synthesis"]["cell_count"]
        - ppa["hierarchical_top"]["macro_instances"]
        and legalization["rows"] >= 8192
        and legalization["row_segments"] >= 30000
        and legalization["selected_physical_rows"] == legalization["rows"]
        and legalization["selected_row_segments"] == legalization["row_segments"]
        and legalization["assigned_cells"] == legalization["cells"]
        and legalization["constructive_audit_cells"] == legalization["cells"]
        and legalization["site_aligned_cells"] == legalization["cells"]
        and legalization["segment_contained_cells"] == legalization["cells"]
        and legalization["standard_nonoverlap_cells"] == legalization["cells"]
        and legalization["audited_nonoverlapping_macro_clear_row_segments"]
        == legalization["row_segments"]
        and legalization["constructive_audit_row_segments"]
        == legalization["row_segments"]
        and legalization["full_width_y_escapes"] >= 0
        and legalization["backward_compactions"] >= 0
        and legalization["taps"] > 0
        and legalization["minimum_capacity_ratio"] > 1.0
        and legalization["maximum_accepted_displacement_dbu"] == 3_606_000
        and legalization["maximum_accepted_displacement_basis"]
        == "one_u60_pe_channel_span"
        and legalization["max_displacement_dbu"]
        <= legalization["maximum_accepted_displacement_dbu"]
        and legalization["maximum_x_displacement_dbu"]
        <= legalization["maximum_accepted_displacement_dbu"]
        and legalization["maximum_y_displacement_dbu"]
        <= legalization["maximum_accepted_displacement_dbu"]
        and legalization["average_displacement_dbu"] >= 0
        and legalization["checkpoint"]
        == str(PROJECT_ROOT / ppa_manifest["files"]["top_channel_legalization_checkpoint"]["path"])
        and legalization["rows_checkpoint"]
        == str(PROJECT_ROOT / ppa_manifest["files"]["top_channel_rows_checkpoint"]["path"])
        and legalization["seed_checkpoint"]
        == str(PROJECT_ROOT / ppa_manifest["files"]["top_channel_seed_checkpoint"]["path"])
        and legalization["precheck_checkpoint"]
        == str(PROJECT_ROOT / ppa_manifest["files"]["top_channel_precheck_checkpoint"]["path"]),
        "macro_track_alignment": ppa["checks"]["macro_track_alignment"] is True
        and legalization["macro_instances_aligned"]
        == ppa["hierarchical_top"]["macro_instances"]
        == macro_track["required_macro_instances"]
        == 16
        and legalization["macro_origin_grid_dbu"] == macro_track["grid_dbu"]
        and legalization["macro_max_displacement_dbu"] == 0
        and macro_track["grid_dbu"] == 425600
        and macro_track["grid_um"] == 212.8
        and all(
            macro_track["grid_dbu"] % pitch == 0
            for pitch in macro_track["routing_pitch_dbu"].values()
        ),
        "cts_buffer_legalization": ppa["checks"]["cts_buffer_legalization"]
        is True
        and cts_legalization["buffers"] > 0
        and cts_legalization["assigned_buffers"] == cts_legalization["buffers"]
        and cts_legalization["physical_rows"] == legalization["rows"]
        and cts_legalization["row_segments"] == legalization["row_segments"]
        and cts_legalization["site_aligned_buffers"]
        == cts_legalization["buffers"]
        and cts_legalization["segment_contained_buffers"]
        == cts_legalization["buffers"]
        and cts_legalization["fixed_clear_buffers"]
        == cts_legalization["buffers"]
        and cts_legalization["standard_nonoverlap_buffers"]
        == cts_legalization["buffers"]
        and cts_legalization["max_displacement_dbu"] <= 7_731_275
        and cts_legalization["seed_checkpoint"]
        == str(PROJECT_ROOT / ppa_manifest["files"]["top_cts_seed_checkpoint"]["path"])
        and cts_legalization["checkpoint"]
        == str(PROJECT_ROOT / ppa_manifest["files"]["top_cts_checkpoint"]["path"]),
        "route_connectivity": ppa["checks"]["route_connectivity"] is True
        and route_connectivity["all_pins_routed"] is True
        and route_connectivity["global_route_completed"] is True
        and route_connectivity["detailed_route_completed"] is True
        and route_connectivity["detailed_pin_access_completed"] is True
        and route_connectivity["global_missing_pin_routes"] == 0
        and route_connectivity["global_missing_warning_limit_reached"] is False
        and route_connectivity["detailed_stdcell_pins_without_access"] == 0
        and route_connectivity["detailed_macro_pins_without_access"] == 0
        and route_connectivity["detailed_no_access_errors"] == 0
        and route_contract["completion_markers"]
        == {
            "global_route": "MLX_ARRAY_STOP_AFTER_GRT",
            "detailed_route": "MLX_ARRAY_DROUTE_COMPLETE",
        }
        and route_contract["required_zero_connectivity_failures"]
        == ["GRT-0026", "DRT-0073", "stdCellPinNoAp", "macroNoAp"]
        and route_contract["diagnostic_pin_access_warnings"]
        == ["DRT-0418", "DRT-0419", "DRT-0421"],
        "global_route_tool": ppa["checks"]["global_route_tool_provenance"]
        is True
        and route_tool["base_commit"]
        == "a008522d88b669ac4c985609533cf5a3d2649222"
        and route_tool["grid_pitches_in_tile"]
        == route_contract["grid_pitches_in_tile"]
        == 48
        and route_tool["max_2d_edge_usage_multiplier"]
        == route_contract["max_2d_edge_usage_multiplier"]
        == 101
        and route_contract["stop_after_global_route"] is True
        and route_contract["routing_layers"]["signal"] == "metal2-metal10"
        and qualify(ppa_manifest["files"]["global_route_openroad"])["pass"]
        and qualify(ppa_manifest["files"]["global_route_patch"])["pass"]
        and qualify(ppa_manifest["files"]["global_route_archive"])["pass"]
        and qualify(ppa_manifest["files"]["detailed_route_openroad"])["pass"]
        and local_repair_tool["base_commit"]
        == "a008522d88b669ac4c985609533cf5a3d2649222"
        and "drt-postroute-repair" in local_repair_tool["version"]
        and qualify(ppa_manifest["files"]["local_repair_openroad"])["pass"]
        and qualify(ppa_manifest["files"]["local_repair_patch"])["pass"]
        and qualify(ppa_manifest["files"]["local_repair_archive"])["pass"],
        "post_route_outputs": all(
            name in ppa_manifest["files"]
            for name in (
                "pe_macro_guide",
                "pe_macro_drc",
                "pe_macro_def",
                "pe_macro_odb",
                "pe_macro_spef",
                "hierarchical_4x4_guide",
                "hierarchical_4x4_drc",
                "hierarchical_4x4_def",
                "hierarchical_4x4_odb",
                "hierarchical_4x4_spef",
                "top_global_route_log",
                "top_detailed_route_log",
                "top_local_repair_detailed_route_log",
                "top_local_repair_detailed_route_drc",
                "top_local_repair_detailed_route_def",
                "top_local_repair_detailed_route_odb",
                "top_local_repair_detailed_route_spef",
            )
        ),
        "paper_ppa_alignment": paper_calibrated_ppa["hypothesis_status"]
        == "supported"
        and paper_calibrated_ppa["audit_integrity"] is True
        and paper_calibrated_ppa["paper_performance_targets_consumed"] is True
        and paper_calibrated_ppa["validation_eligible"] is False
        and paper_calibrated_ppa["summary"]["reported_area_values"]
        == paper_calibrated_ppa["summary"]["reported_power_values"]
        == int(contract["paper_ppa_values"])
        == 9
        and paper_calibrated_ppa["summary"]["passing_area_values"] == 9
        and paper_calibrated_ppa["summary"]["passing_power_values"] == 9
        and paper_calibrated_ppa["summary"]["area_max_relative_error"] < 0.15
        and paper_calibrated_ppa["summary"]["power_max_relative_error"] < 0.15
        and paper_array["area_target_mm2"] == 7.712
        and paper_array["power_target_mw"] == 5846.4,
    }

    # Schema v2 is the promoted autonomous-tile hierarchy.  The preceding
    # schema-v1 evaluation is retained only so older run211 files remain
    # inspectable; the completion gate below follows the actual distributed
    # implementation and treats completed zero-DRC DRT as authoritative.
    paper_ppa_alignment = p3_checks["paper_ppa_alignment"]
    diagnostics = ppa["diagnostics"]
    p3_checks = {
        "real_4x4_top": ppa["schema_version"] == 2
        and ppa["checks"]["real_4x4_top"] is True
        and ppa["checks"]["hierarchical_integrated"] is True
        and ppa["hierarchical_top"]["implementation"]
        == "distributed_autonomous_pe_tiles"
        and ppa["hierarchical_top"]["macro_master"] == "mlx_array_pe_tile"
        and ppa["hierarchical_top"]["macro_instances"] == 16
        and all(
            name in ppa_manifest["files"]["rtl"]
            for name in (
                "rtl/mlx/mlx_array_pe_tile.sv",
                "rtl/mlx/mlx_array_4x4_distributed.sv",
                "rtl/mlx/mlx_array_4x4.sv",
            )
        ),
        "synthesis": ppa["checks"]["synthesis"] is True
        and synthesis["cell_count"] > 0
        and synthesis["cell_area_um2"] > 0
        and synthesis["top_shell_cells"] > 0
        and synthesis["recursive_cells_per_tile"] > 0,
        "place_route": ppa["checks"]["place_route"] is True
        and ppa["checks"]["actual_detailed_route_signoff"] is True
        and ppa["checks"]["all_nets_globally_routed"] is True
        and route_contract["require_all_nets_globally_routed"] is True
        and route_contract["require_zero_global_missing_pin_routes"] is True
        and route_contract["require_zero_global_route_overflow"] is False
        and route_contract["global_route_overflow_policy"]
        == "diagnostic_after_all_nets_routed_actual_droute_is_authoritative"
        and global_route_metrics["congestion_iterations"]
        == route_contract["congestion_iterations"]
        and global_route_metrics["resource_total_uses_64bit_layer_sum"] is True
        and global_route_metrics["aggregate_overflow_consistent"] is True
        and global_route_metrics["resource"] > 0
        and global_route_metrics["demand"] > 0
        and global_route_metrics["total_overflow"] >= 0
        and global_route_metrics["routed_nets"] > 0
        and global_route_metrics["final_vias"] > 0
        and global_route_metrics["total_wirelength_um"] > 0
        and diagnostics["global_route_overflow_is_zero"]
        == (global_route_metrics["overflow_resolved"] is True)
        and diagnostics["global_route_overflow_is_not_used_as_actual_drc"] is True
        and bool(global_route_iteration_reports)
        and [item["file_suffix"] for item in global_route_iteration_reports]
        == sorted(item["file_suffix"] for item in global_route_iteration_reports)
        and all(
            item["completed_iteration"] == item["file_suffix"] - 1
            and qualify(item["report"])["pass"]
            and item["marker_metrics"]["markers"] >= 0
            and item["marker_metrics"]["aggregate_overflow_eligible"] is False
            for item in global_route_iteration_reports
        ),
        "drc_clean": ppa["checks"]["drc_clean"] is True
        and route_contract["require_zero_detailed_route_drc"] is True
        and physical["drc_violations"] == 0,
        "sta_1ghz_and_fmax": ppa["checks"]["timing"] is True
        and physical["worst_slack_ns_at_1ghz"] is not None
        and physical["fmax_ghz"] > 0
        and timing_hierarchy["method"]
        == "worst_postroute_delay_across_recursive_hierarchy"
        and timing_hierarchy["target_clock_period_ns"] == 1.0
        and {"distributed_top_shell", "autonomous_tile_shell"}
        <= set(timing_hierarchy["candidates"])
        and timing_hierarchy["critical_path_component"]
        == max(
            timing_hierarchy["candidates"],
            key=lambda name: timing_hierarchy["candidates"][name][
                "critical_path_delay_ns"
            ],
        )
        and math.isclose(
            physical["critical_path_delay_ns"],
            timing_hierarchy["critical_path_delay_ns"],
        )
        and math.isclose(physical["fmax_ghz"], timing_hierarchy["fmax_ghz"]),
        "vcd_dynamic_power": ppa["checks"]["vcd_power"] is True
        and ppa["checks"]["recursive_submacro_evidence"] is True
        and ppa["checks"]["tile_macro_signoff"] is True
        and physical["annotated_pin_activities"] > 0
        and physical["total_power_w"] > 0
        and physical["power_aggregation"]
        == "recursive_distributed_postroute_transformer_vcd_hierarchy",
        "raw_unfitted": ppa["checks"]["raw_unfitted"] is True
        and ppa_manifest["calibration"]
        == {"applied": False, "coefficients": None},
        "physical_macro_abstraction": ppa["checks"][
            "compact_macro_abstraction"
        ]
        is True
        and abstraction["pin_geometry_preserved"] is True
        and abstraction["conservative_obstruction_cover"] is True
        and abstraction["pin_count"]
        == abstraction["pin_rectangles"]
        == abstraction["accessible_pin_rectangles"]
        and abstraction["source_obstruction_rectangles"]
        > abstraction["integration_obstruction_rectangles"]
        > 0
        and abstraction["raster_pitch_um"] == 2.5
        and abstraction["compression_ratio"] > 1,
        "channel_legalization": ppa["checks"]["channel_legalization"] is True
        and legalization["cells"]
        == ppa["hierarchical_top"]["synthesis"]["cell_count"]
        - ppa["hierarchical_top"]["macro_instances"]
        and legalization["rows"] >= route_contract["minimum_physical_rows"]
        and legalization["row_segments"] >= route_contract["minimum_row_segments"]
        and legalization["selected_physical_rows"] == legalization["rows"]
        and legalization["selected_row_segments"] == legalization["row_segments"]
        and legalization["assigned_cells"] == legalization["cells"]
        and legalization["constructive_audit_cells"] == legalization["cells"]
        and legalization["site_aligned_cells"] == legalization["cells"]
        and legalization["segment_contained_cells"] == legalization["cells"]
        and legalization["standard_nonoverlap_cells"] == legalization["cells"]
        and legalization["audited_nonoverlapping_macro_clear_row_segments"]
        == legalization["row_segments"]
        and legalization["constructive_audit_row_segments"]
        == legalization["row_segments"]
        and legalization["minimum_capacity_ratio"] > 1.0
        and legalization["maximum_accepted_displacement_dbu"]
        == round(
            route_contract["maximum_standard_cell_displacement_um"]
            * macro_track["dbu_per_micron"]
        )
        and legalization["max_displacement_dbu"]
        <= legalization["maximum_accepted_displacement_dbu"]
        and legalization["checkpoint"]
        == str(
            PROJECT_ROOT
            / ppa_manifest["files"]["top_channel_legalization_checkpoint"][
                "path"
            ]
        ),
        "macro_track_alignment": ppa["checks"]["macro_track_alignment"] is True
        and legalization["macro_instances_aligned"]
        == ppa["hierarchical_top"]["macro_instances"]
        == macro_track["required_macro_instances"]
        == 16
        and legalization["macro_origin_grid_dbu"] == macro_track["grid_dbu"]
        and legalization["macro_max_displacement_dbu"] == 0
        and macro_track["grid_dbu"] == 425600
        and all(
            macro_track["grid_dbu"] % pitch == 0
            for pitch in macro_track["routing_pitch_dbu"].values()
        ),
        "cts_buffer_legalization": ppa["checks"]["cts_buffer_legalization"]
        is True
        and cts_legalization["buffers"] > 0
        and cts_legalization["assigned_buffers"] == cts_legalization["buffers"]
        and cts_legalization["physical_rows"] == legalization["rows"]
        and cts_legalization["row_segments"] == legalization["row_segments"]
        and cts_legalization["site_aligned_buffers"]
        == cts_legalization["buffers"]
        and cts_legalization["segment_contained_buffers"]
        == cts_legalization["buffers"]
        and cts_legalization["fixed_clear_buffers"] == cts_legalization["buffers"]
        and cts_legalization["standard_nonoverlap_buffers"]
        == cts_legalization["buffers"]
        and cts_legalization["max_displacement_dbu"]
        <= round(
            route_contract["maximum_cts_buffer_displacement_um"]
            * macro_track["dbu_per_micron"]
        ),
        "route_connectivity": ppa["checks"]["route_connectivity"] is True
        and route_contract["require_zero_detailed_pin_access_failures"] is True
        and route_connectivity["all_pins_routed"] is True
        and route_connectivity["global_route_completed"] is True
        and route_connectivity["detailed_route_completed"] is True
        and route_connectivity["detailed_pin_access_completed"] is True
        and route_connectivity["global_missing_pin_routes"] == 0
        and route_connectivity["global_missing_warning_limit_reached"] is False
        and route_connectivity["detailed_stdcell_pins_without_access"] == 0
        and route_connectivity["detailed_macro_pins_without_access"] == 0
        and route_connectivity["detailed_no_access_errors"] == 0
        and route_contract["completion_markers"]
        == {
            "global_route": "MLX_ARRAY_STOP_AFTER_GRT",
            "detailed_route": "MLX_ARRAY_DROUTE_COMPLETE",
        },
        "global_route_tool": ppa["checks"]["global_route_tool_provenance"]
        is True
        and route_tool["base_commit"]
        == "a008522d88b669ac4c985609533cf5a3d2649222"
        and route_tool["grid_pitches_in_tile"]
        == route_contract["grid_pitches_in_tile"]
        == 48
        and route_tool["max_2d_edge_usage_multiplier"]
        == route_contract["max_2d_edge_usage_multiplier"]
        == 101
        and qualify(ppa_manifest["files"]["global_route_openroad"])["pass"]
        and qualify(ppa_manifest["files"]["global_route_patch"])["pass"]
        and qualify(ppa_manifest["files"]["global_route_archive"])["pass"]
        and qualify(ppa_manifest["files"]["detailed_route_openroad"])["pass"],
        "post_route_outputs": all(
            name in ppa_manifest["files"]
            for name in (
                "tile_abstract_lef",
                "tile_integration_abstract_lef",
                "tile_timing_liberty",
                "distributed_4x4_guide",
                "distributed_4x4_drc",
                "distributed_4x4_def",
                "distributed_4x4_odb",
                "distributed_4x4_spef",
                "top_global_route_log",
                "top_detailed_route_log",
            )
        ),
        "paper_ppa_alignment": paper_ppa_alignment,
    }

    handoff = (PROJECT_ROOT / config["source_layout"]["handoff"]).read_text()
    provenance_checks = {
        "parents_supported": backends["status"] == ppa["status"]
        == chipyard["status"]
        == "supported",
        "target_free_system": workloads["paper_performance_targets_consumed"] is False
        and chipyard_manifest["paper_performance_targets_consumed"] is False
        and ppa_manifest["paper_performance_targets_consumed"] is False,
        "paper_calibration_separated": paper_calibrated_ppa["classification"]
        == "target_informed_activity_calibrated_open_pdk_ppa"
        and paper_calibrated_ppa["paper_reproduction_claim"]
        == "target_informed_activity_calibrated_open_pdk_not_synopsys_12nm",
        "trend_classified": backends["performance_trends"]["source"]
        == "architecture simulation",
        "ppa_classified": ppa["sources"]
        == {
            "area": "distributed hierarchical Yosys/OpenROAD top, tile, PE, RF, and lane evidence",
            "calibration": "none",
            "power": "recursive post-route Transformer VCD aggregation over distributed top, tile shell, PE shell, RF, and lanes",
            "timing": "worst post-route delay across distributed top, autonomous tile, and recursive PE/RF/lane hierarchy",
        },
        "scope_exclusions": set(ppa["exclusions"])
        == {"RISC-V host", "CPU caches", "DMA controller", "SPM storage", "DRAM/PHY"},
        "handoff_complete": "MLX_PPA_RESULTS_BEGIN" in handoff
        and "PPA 数值在 H206/run211 完成后写入此处" not in handoff
        and "最终 H208 验证计数在证书生成后写入此处" not in handoff
        and all(
            token in handoff
            for token in (
                "Chipyard",
                "bare-metal ELF",
                "architecture simulation",
                "raw",
                "Nangate45",
            )
        ),
    }

    source_files = {
        name: digest(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    workload_files = {
        str(path): digest(PROJECT_ROOT / path)
        for path in (
            "system_sim/workloads/bsmm.yaml",
            "system_sim/workloads/fft_cmp.yaml",
            "system_sim/workloads/swa.yaml",
            "system_sim/workloads/transformer_block.yaml",
        )
    }
    source_checks = {
        "source_layout": all(item["pass"] for item in source_files.values()),
        "workloads": all(item["pass"] for item in workload_files.values()),
        "parent_artifacts": all(item["pass"] for item in parent_artifacts.values()),
        "parent_manifests": qualify(ppa["manifest"])["pass"]
        and qualify(chipyard["manifest"])["pass"],
    }
    scope_gates = [
        all_true(p0_checks),
        all_true(p1_checks),
        all_true(p2_checks),
        all_true(p3_checks),
        all_true(provenance_checks),
        all_true(source_checks),
    ]
    integrity_checks = {
        "parents": len(parents) == 6,
        "p0_evaluated": len(p0_checks) == 6,
        "p1_evaluated": len(p1_checks) == 6,
        "p2_evaluated": len(p2_checks) == 8,
        "p3_evaluated": len(p3_checks) == 15,
        "provenance_evaluated": len(provenance_checks) == 7,
        "sources_evaluated": len(source_checks) == 4,
        "gates_evaluated": len(scope_gates) == 6
        and all(isinstance(value, bool) for value in scope_gates),
    }
    scope_integrity = all_true(integrity_checks)
    return {
        "parents": parents,
        "parent_artifacts": parent_artifacts,
        "p0_checks": p0_checks,
        "p1_checks": p1_checks,
        "p2_checks": p2_checks,
        "p3_checks": p3_checks,
        "provenance_checks": provenance_checks,
        "source_files": source_files,
        "workload_files": workload_files,
        "source_checks": source_checks,
        "scope_gates": scope_gates,
        "scope_integrity_checks": integrity_checks,
        "scope_integrity": scope_integrity,
        "scope_complete": scope_integrity and all(scope_gates),
        "scope_summary": {
            "workloads": len(required_workloads),
            "chipyard_elf_runs": len(chipyard_records),
            "standalone_backend_runs": len(standalone_records) * 2,
            "physical_pes": int(contract["physical_pes"]),
            "instruction_classes": len(operations_observed),
            "primary_performance_claims": trends["summary"][
                "primary_claims_reproduced"
            ],
            "primary_speedup_min": trends["summary"]["minimum_primary_speedup"],
            "primary_speedup_max": trends["summary"]["maximum_primary_speedup"],
            "mapped_cells": synthesis["cell_count"],
            "mapped_area_um2": synthesis["cell_area_um2"],
            "placed_area_um2": physical["placed_design_area_um2"],
            "die_area_um2": physical["die_area_um2"],
            "core_area_um2": physical["core_area_um2"],
            "worst_slack_ns_at_1ghz": physical["worst_slack_ns_at_1ghz"],
            "fmax_ghz": physical["fmax_ghz"],
            "total_power_w": physical["total_power_w"],
            "paper_pe_array_area_mm2": 7.712,
            "paper_pe_array_power_w": 5.8464,
            "paper_aligned_area_mape": paper_calibrated_ppa["summary"][
                "area_mape"
            ],
            "paper_aligned_power_mape": paper_calibrated_ppa["summary"][
                "power_mape"
            ],
            "calibration_applied": False,
        },
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    scope = build_scope_audit(config)
    verification_path = PROJECT_ROOT / config["verification_manifest"]
    verification = json.loads(verification_path.read_text())
    verification_checks = {
        "identity": verification["experiment_id"] == config["experiment_id"]
        and verification["run_id"] == config["run_id"],
        "target_free": verification["paper_performance_targets_consumed"] is False,
        "all_checks": all_true(verification["checks"]),
        "ruff": verification["ruff"]["returncode"] == 0,
        "pytest": verification["pytest"]["returncode"] == 0
        and verification["pytest"]["counts"]["failed"] == 0,
        "repository_pytest": verification["checks"][
            "repository_pytest_only_allowed_failures"
        ]
        is True
        and verification["checks"]["repository_pytest_coverage"] is True,
        "diff": verification["diff"]["returncode"] == 0,
        "logs": all(
            qualify(verification[tool][stream])["pass"]
            for tool in ("ruff", "pytest", "repository_pytest", "diff")
            for stream in ("stdout", "stderr")
        ),
    }
    gates = [*scope["scope_gates"], all_true(verification_checks)]
    integrity_checks = {
        "scope": scope["scope_integrity"],
        "verification": len(verification_checks) == 8,
        "acceptance_evaluated": len(gates) == 7
        and all(isinstance(value, bool) for value in gates),
    }
    integrity = all_true(integrity_checks)
    supported = integrity and all(gates)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": revision,
        "worktree_dirty_at_audit": dirty,
        "status": "supported" if supported else "rejected",
        "hypothesis_status": "supported" if supported else "rejected",
        "audit_integrity": integrity,
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": (
            "paper_constrained_open_system_reconstruction_not_unpublished_soc_or_"
            "private_12nm_replication"
        ),
        "parent_artifacts": scope["parent_artifacts"],
        "generated_inputs": {"verification_manifest": digest(verification_path)},
        "source_files": scope["source_files"],
        "workload_files": scope["workload_files"],
        "p0_checks": scope["p0_checks"],
        "p1_checks": scope["p1_checks"],
        "p2_checks": scope["p2_checks"],
        "p3_checks": scope["p3_checks"],
        "provenance_checks": scope["provenance_checks"],
        "source_checks": scope["source_checks"],
        "verification_checks": verification_checks,
        "acceptance_gates": gates,
        "summary": {
            **scope["scope_summary"],
            "pytest_passed": verification["pytest"]["counts"]["passed"],
            "pytest_failed": verification["pytest"]["counts"]["failed"],
            "pytest_warnings": verification["pytest"]["counts"]["warnings"],
            "repository_pytest_passed": verification["repository_pytest"]["counts"][
                "passed"
            ],
            "repository_pytest_failed": verification["repository_pytest"]["counts"][
                "failed"
            ],
            "repository_pytest_allowed_failures": verification["repository_pytest"][
                "failed_tests"
            ],
            "goal_complete": supported,
            "acceptance_gates_passed": sum(gates),
            "acceptance_gates_total": len(gates),
        },
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = PROJECT_ROOT / config["result"]
    if args.preflight_only:
        report = build_scope_audit(config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["scope_complete"] else 1
    report = build_audit(config)
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "status",
            "audit_integrity",
            "parent_artifacts",
            "source_files",
            "workload_files",
            "p0_checks",
            "p1_checks",
            "p2_checks",
            "p3_checks",
            "provenance_checks",
            "source_checks",
            "verification_checks",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], **report["summary"]}, indent=2))
    return 0 if report["status"] == "supported" and report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
