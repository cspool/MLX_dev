#!/usr/bin/env python3
"""Run hierarchical, integrated P&R for one PE macro and the real 4x4 MLX top."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.extract_vcd_scope import extract_scope
from scripts.run_mlx_array_ppa import parse_openroad, parse_synthesis, run_to_log

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/system/mlx_array_ppa_v1.yaml"
PE_RTL = [
    "rtl/mlx/mlx_fp16.sv",
    "rtl/mlx/mlx_fu.sv",
    "rtl/mlx/mlx_register_file.sv",
    "rtl/mlx/mlx_tag_buffer.sv",
    "rtl/mlx/mlx_config_network.sv",
    "rtl/mlx/mlx_data_network.sv",
    "rtl/mlx/mlx_control_logic.sv",
    "rtl/mlx/mlx_pe_top.sv",
]


def artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    try:
        name = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        name = str(path)
    return {
        "path": name,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def yosys_run(commands: list[str], log: Path) -> int:
    return run_to_log(["yosys", "-Q", "-p", "; ".join(commands)], log)


def physical_paths(root: Path, stem: str) -> dict[str, Path]:
    return {
        "guide": root / f"{stem}.guide",
        "drc": root / f"{stem}.drc",
        "def": root / f"{stem}.def",
        "odb": root / f"{stem}.odb",
        "spef": root / f"{stem}.spef",
        "log": root / f"{stem}.log",
    }


def all_congestion_iteration_reports(report: Path) -> list[tuple[int, Path]]:
    """Return every matching FastRoute iteration report in numeric order."""

    pattern = re.compile(
        rf"^{re.escape(report.stem)}-(\d+){re.escape(report.suffix)}$"
    )
    reports: list[tuple[int, Path]] = []
    for candidate in report.parent.glob(f"{report.stem}-*{report.suffix}"):
        match = pattern.match(candidate.name)
        if match:
            reports.append((int(match.group(1)), candidate))
    return sorted(reports)


def congestion_iteration_reports(report: Path) -> list[tuple[int, Path]]:
    """Return the current run's contiguous, timestamp-ordered report prefix."""

    current: list[tuple[int, Path]] = []
    expected_suffix = 2
    prior_mtime_ns = -1
    for suffix, candidate in all_congestion_iteration_reports(report):
        mtime_ns = candidate.stat().st_mtime_ns
        if suffix != expected_suffix or mtime_ns < prior_mtime_ns:
            break
        current.append((suffix, candidate))
        expected_suffix += 1
        prior_mtime_ns = mtime_ns
    return current


def output_check(paths: dict[str, Path]) -> bool:
    return all(path.is_file() for key, path in paths.items() if key != "log") and all(
        path.stat().st_size > 0
        for key, path in paths.items()
        if key not in {"log", "drc"}
    )


def macro_metrics(text: str, clock_period: float) -> dict[str, Any]:
    metrics = parse_openroad(text, clock_period)
    die = re.findall(r"MLX_PE_DIE_UM\s+([0-9.]+)\s+([0-9.]+)", text)
    core = re.findall(r"MLX_PE_CORE_UM\s+([0-9.]+)\s+([0-9.]+)", text)
    if die:
        metrics["die_width_um"] = float(die[-1][0])
        metrics["die_height_um"] = float(die[-1][1])
    if core:
        metrics["core_width_um"] = float(core[-1][0])
        metrics["core_height_um"] = float(core[-1][1])
    return metrics


def parse_channel_legalization(text: str) -> dict[str, Any]:
    alignments = re.findall(
        r"MLX_MACRO_TRACK_ALIGNMENT macros=(\d+) grid_dbu=(\d+) "
        r"max_displacement_dbu=(\d+)",
        text,
    )
    records = re.findall(
        r"MLX_CHANNEL_LEGALIZER cells=(\d+) rows=(\d+) "
        r"(?:row_segments=(\d+) )?taps=(\d+) "
        r"removed_rows=(\d+) removed_tapcells=(\d+) "
        r"max_displacement_dbu=(\d+) min_capacity_ratio=([0-9.eE+-]+) "
        r"checkpoint=(\S+)",
        text,
    )
    locality_records = re.findall(
        r"MLX_CHANNEL_LOCALITY max_x_displacement_dbu=(\d+) "
        r"max_y_displacement_dbu=(\d+) average_displacement_dbu=([0-9.eE+-]+)",
        text,
    )
    row_selection_records = re.findall(
        r"MLX_CHANNEL_ROW_SELECTION physical_rows=(\d+) row_segments=(\d+) "
        r"removed_segments=(\d+)",
        text,
    )
    row_audit_records = re.findall(
        r"MLX_CHANNEL_ROW_AUDIT nonoverlapping_macro_clear_segments=(\d+)", text
    )
    assignment_records = re.findall(
        r"MLX_CHANNEL_ASSIGNMENT cells=(\d+) full_width_y_escapes=(\d+)",
        text,
    )
    legalizer_records = re.findall(
        r"MLX_CHANNEL_1D_LEGALIZATION backward_compactions=(\d+)", text
    )
    constructive_audit_records = re.findall(
        r"MLX_CHANNEL_CONSTRUCTIVE_AUDIT cells=(\d+) site_aligned=(\d+) "
        r"segment_contained=(\d+) standard_nonoverlap=(\d+) row_segments=(\d+)",
        text,
    )
    rows_checkpoints = re.findall(r"MLX_CHANNEL_ROWS_CHECKPOINT checkpoint=(\S+)", text)
    rows_resumes = re.findall(r"MLX_CHANNEL_ROWS_RESUME checkpoint=(\S+)", text)
    seed_checkpoints = re.findall(r"MLX_CHANNEL_SEED_CHECKPOINT checkpoint=(\S+)", text)
    precheck_checkpoints = re.findall(r"MLX_CHANNEL_PRECHECK checkpoint=(\S+)", text)
    if not records:
        return {
            "cells": None,
            "rows": None,
            "row_segments": None,
            "taps": None,
            "removed_rows": None,
            "removed_tapcells": None,
            "max_displacement_dbu": None,
            "minimum_capacity_ratio": None,
            "checkpoint": None,
            "macro_instances_aligned": None,
            "macro_origin_grid_dbu": None,
            "macro_max_displacement_dbu": None,
            "maximum_x_displacement_dbu": None,
            "maximum_y_displacement_dbu": None,
            "average_displacement_dbu": None,
            "selected_physical_rows": None,
            "selected_row_segments": None,
            "removed_row_segments": None,
            "audited_nonoverlapping_macro_clear_row_segments": None,
            "assigned_cells": None,
            "full_width_y_escapes": None,
            "backward_compactions": None,
            "constructive_audit_cells": None,
            "site_aligned_cells": None,
            "segment_contained_cells": None,
            "standard_nonoverlap_cells": None,
            "constructive_audit_row_segments": None,
            "rows_checkpoint": None,
            "resumed_from_rows_checkpoint": False,
            "seed_checkpoint": None,
            "precheck_checkpoint": None,
        }
    record = records[-1]
    alignment = alignments[-1] if alignments else (None, None, None)
    locality = locality_records[-1] if locality_records else (None, None, None)
    row_selection = (
        row_selection_records[-1] if row_selection_records else (None, None, None)
    )
    assignment = assignment_records[-1] if assignment_records else (None, None)
    constructive_audit = (
        constructive_audit_records[-1]
        if constructive_audit_records
        else (None, None, None, None, None)
    )
    return {
        "cells": int(record[0]),
        "rows": int(record[1]),
        "row_segments": int(record[2]) if record[2] else int(record[1]),
        "taps": int(record[3]),
        "removed_rows": int(record[4]),
        "removed_tapcells": int(record[5]),
        "max_displacement_dbu": int(record[6]),
        "minimum_capacity_ratio": float(record[7]),
        "checkpoint": record[8],
        "macro_instances_aligned": int(alignment[0]) if alignment[0] else None,
        "macro_origin_grid_dbu": int(alignment[1]) if alignment[1] else None,
        "macro_max_displacement_dbu": int(alignment[2]) if alignment[2] else None,
        "maximum_x_displacement_dbu": int(locality[0]) if locality[0] else None,
        "maximum_y_displacement_dbu": int(locality[1]) if locality[1] else None,
        "average_displacement_dbu": float(locality[2]) if locality[2] else None,
        "selected_physical_rows": (
            int(row_selection[0]) if row_selection[0] else None
        ),
        "selected_row_segments": int(row_selection[1]) if row_selection[1] else None,
        "removed_row_segments": int(row_selection[2]) if row_selection[2] else None,
        "audited_nonoverlapping_macro_clear_row_segments": (
            int(row_audit_records[-1]) if row_audit_records else None
        ),
        "assigned_cells": int(assignment[0]) if assignment[0] else None,
        "full_width_y_escapes": int(assignment[1]) if assignment[1] else None,
        "backward_compactions": (
            int(legalizer_records[-1]) if legalizer_records else None
        ),
        "constructive_audit_cells": (
            int(constructive_audit[0]) if constructive_audit[0] else None
        ),
        "site_aligned_cells": (
            int(constructive_audit[1]) if constructive_audit[1] else None
        ),
        "segment_contained_cells": (
            int(constructive_audit[2]) if constructive_audit[2] else None
        ),
        "standard_nonoverlap_cells": (
            int(constructive_audit[3]) if constructive_audit[3] else None
        ),
        "constructive_audit_row_segments": (
            int(constructive_audit[4]) if constructive_audit[4] else None
        ),
        "rows_checkpoint": rows_checkpoints[-1] if rows_checkpoints else None,
        "resumed_from_rows_checkpoint": bool(rows_resumes),
        "seed_checkpoint": seed_checkpoints[-1] if seed_checkpoints else None,
        "precheck_checkpoint": precheck_checkpoints[-1] if precheck_checkpoints else None,
    }


def parse_cts_buffer_legalization(text: str) -> dict[str, Any]:
    assignments = re.findall(
        r"MLX_CTS_BUFFER_ASSIGNMENT buffers=(\d+) fixed_cells=(\d+) "
        r"physical_rows=(\d+) row_segments=(\d+)",
        text,
    )
    records = re.findall(
        r"MLX_CTS_BUFFER_LEGALIZATION buffers=(\d+) backward_compactions=(\d+) "
        r"site_aligned=(\d+) segment_contained=(\d+) fixed_clear=(\d+) "
        r"standard_nonoverlap=(\d+) max_displacement_dbu=(\d+) "
        r"average_displacement_dbu=([0-9.eE+-]+)",
        text,
    )
    seed_checkpoints = re.findall(
        r"MLX_ARRAY_CTS_SEED(?:_RESUME)? checkpoint=(\S+)", text
    )
    final_checkpoints = re.findall(r"MLX_ARRAY_STOP_AFTER_CTS checkpoint=(\S+)", text)
    assignment = assignments[-1] if assignments else (None, None, None, None)
    record = records[-1] if records else (None, None, None, None, None, None, None, None)
    return {
        "buffers": int(record[0]) if record[0] else None,
        "assigned_buffers": int(assignment[0]) if assignment[0] else None,
        "fixed_cells": int(assignment[1]) if assignment[1] else None,
        "physical_rows": int(assignment[2]) if assignment[2] else None,
        "row_segments": int(assignment[3]) if assignment[3] else None,
        "backward_compactions": int(record[1]) if record[1] else None,
        "site_aligned_buffers": int(record[2]) if record[2] else None,
        "segment_contained_buffers": int(record[3]) if record[3] else None,
        "fixed_clear_buffers": int(record[4]) if record[4] else None,
        "standard_nonoverlap_buffers": int(record[5]) if record[5] else None,
        "max_displacement_dbu": int(record[6]) if record[6] else None,
        "average_displacement_dbu": float(record[7]) if record[7] else None,
        "seed_checkpoint": seed_checkpoints[-1] if seed_checkpoints else None,
        "checkpoint": final_checkpoints[-1] if final_checkpoints else None,
    }


def parse_route_connectivity(global_route_text: str, detailed_route_text: str) -> dict[str, Any]:
    """Require completed routing and zero unresolved pin-access failures."""

    global_route_completed = "MLX_ARRAY_STOP_AFTER_GRT checkpoint=" in global_route_text
    detailed_route_completed = "MLX_ARRAY_DROUTE_COMPLETE odb=" in detailed_route_text
    detailed_pin_access_completed = (
        "[INFO DRT-0166] Complete pin access." in detailed_route_text
    )
    global_missing_pin_routes = len(
        re.findall(r"\[WARNING GRT-0026\] Missing route to pin", global_route_text)
    )
    detailed_off_grid_macro_terms = len(
        re.findall(
            r"\[WARNING DRT-0418\] Term .* has no pins on routing grid",
            detailed_route_text,
        )
    )
    detailed_off_grid_block_terms = len(
        re.findall(
            r"\[WARNING DRT-0421\] Term .* has no pins on routing grid",
            detailed_route_text,
        )
    )
    stdcell_no_access = re.findall(
        r"#stdCellPinNoAp\s*=\s*(\d+)", detailed_route_text
    )
    macro_no_access = re.findall(r"#macroNoAp\s*=\s*(\d+)", detailed_route_text)
    detailed_no_access_errors = len(
        re.findall(r"\[(?:ERROR|WARNING) DRT-0073\]", detailed_route_text)
    )
    global_missing_warning_limit_reached = bool(
        re.search(r"\[WARNING GRT-0026\] message limit", global_route_text)
    )
    detailed_warning_limit_reached = bool(
        re.search(
            r"\[WARNING DRT-(?:0418|0419|0421)\] message limit",
            detailed_route_text,
        )
    )
    detailed_stdcell_pins_without_access = (
        int(stdcell_no_access[-1]) if stdcell_no_access else None
    )
    detailed_macro_pins_without_access = (
        int(macro_no_access[-1]) if macro_no_access else None
    )
    return {
        "global_route_completed": global_route_completed,
        "detailed_route_completed": detailed_route_completed,
        "detailed_pin_access_completed": detailed_pin_access_completed,
        "global_missing_pin_routes": global_missing_pin_routes,
        "detailed_off_grid_macro_terms": detailed_off_grid_macro_terms,
        "detailed_off_grid_block_terms": detailed_off_grid_block_terms,
        "detailed_stdcell_pins_without_access": (
            detailed_stdcell_pins_without_access
        ),
        "detailed_macro_pins_without_access": detailed_macro_pins_without_access,
        "detailed_no_access_errors": detailed_no_access_errors,
        "global_missing_warning_limit_reached": (
            global_missing_warning_limit_reached
        ),
        "detailed_warning_limit_reached": detailed_warning_limit_reached,
        "warning_limit_reached": (
            global_missing_warning_limit_reached or detailed_warning_limit_reached
        ),
        "all_pins_routed": global_route_completed
        and detailed_route_completed
        and detailed_pin_access_completed
        and global_missing_pin_routes == 0
        and not global_missing_warning_limit_reached
        and detailed_stdcell_pins_without_access == 0
        and detailed_macro_pins_without_access == 0
        and detailed_no_access_errors == 0,
    }


def parse_global_route_metrics(text: str) -> dict[str, Any]:
    congestion_iterations = re.findall(
        r"MLX_GRT_ROUTE_ARGS .*?-congestion_iterations (\d+)", text
    )
    final_report = (
        text.rsplit("[INFO GRT-0096] Final congestion report:", 1)[-1]
        if "[INFO GRT-0096] Final congestion report:" in text
        else ""
    )
    layer_rows = re.findall(
        r"^(metal\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?[0-9.]+)%\s+"
        r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*$",
        final_report,
        flags=re.MULTILINE,
    )
    totals = re.findall(
        r"^Total\s+(-?\d+)\s+(-?\d+)\s+(-?[0-9.]+)%\s+"
        r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*$",
        final_report,
        flags=re.MULTILINE,
    )
    final_vias = re.findall(r"\[INFO GRT-0111\] Final number of vias: (\d+)", text)
    final_usage_3d = re.findall(r"\[INFO GRT-0112\] Final usage 3D: (\d+)", text)
    wirelength = re.findall(r"\[INFO GRT-0018\] Total wirelength: (\d+) um", text)
    routed_nets = re.findall(r"\[INFO GRT-0014\] Routed nets: (\d+)", text)
    total = totals[-1] if totals else (None, None, None, None, None, None)
    layers = {
        row[0]: {
            "resource": int(row[1]),
            "demand": int(row[2]),
            "usage_percent": float(row[3]),
            "max_horizontal_overflow": int(row[4]),
            "max_vertical_overflow": int(row[5]),
            "total_overflow": int(row[6]),
        }
        for row in layer_rows
    }
    resource = sum(row["resource"] for row in layers.values()) if layers else None
    demand = sum(row["demand"] for row in layers.values()) if layers else None
    max_horizontal_overflow = (
        sum(row["max_horizontal_overflow"] for row in layers.values())
        if layers
        else None
    )
    max_vertical_overflow = (
        sum(row["max_vertical_overflow"] for row in layers.values())
        if layers
        else None
    )
    total_overflow = (
        sum(row["total_overflow"] for row in layers.values()) if layers else None
    )
    usage_percent = (
        100.0 * demand / resource
        if resource is not None and resource > 0 and demand is not None
        else None
    )
    congestion_warning = "[WARNING GRT-0115]" in text
    return {
        "congestion_iterations": (
            int(congestion_iterations[-1]) if congestion_iterations else None
        ),
        "resource": resource,
        "demand": demand,
        "usage_percent": usage_percent,
        "max_horizontal_overflow": max_horizontal_overflow,
        "max_vertical_overflow": max_vertical_overflow,
        "total_overflow": total_overflow,
        "layers": layers,
        "reported_total_resource": int(total[0]) if total[0] is not None else None,
        "reported_total_demand": int(total[1]) if total[1] is not None else None,
        "reported_total_usage_percent": (
            float(total[2]) if total[2] is not None else None
        ),
        "reported_total_overflow": int(total[5]) if total[5] is not None else None,
        "resource_total_uses_64bit_layer_sum": bool(layers),
        "aggregate_overflow_consistent": (
            total_overflow == int(total[5])
            if total_overflow is not None and total[5] is not None
            else False
        ),
        "final_vias": int(final_vias[-1]) if final_vias else None,
        "final_usage_3d": int(final_usage_3d[-1]) if final_usage_3d else None,
        "total_wirelength_um": int(wirelength[-1]) if wirelength else None,
        "routed_nets": int(routed_nets[-1]) if routed_nets else None,
        "congestion_warning": congestion_warning,
        "overflow_resolved": total_overflow == 0 and not congestion_warning,
    }


def build_compact_macro_lef(
    source: Path,
    destination: Path,
    obstruction_config: dict[str, Any],
) -> dict[str, Any]:
    """Preserve routed pins and conservatively raster-union internal OBS.

    OpenROAD's routed abstract contains every internal route rectangle.  That is
    useful as detailed evidence, but expanding those shapes sixteen times in a
    parent block exhausts memory.  The integration view retains the exact macro
    dimensions and pin PORT section, then outward-quantizes every internal OBS
    rectangle to a fixed raster and unions adjacent occupied bins.  Therefore it
    never opens space covered by the source abstraction, while avoiding the
    excessively pessimistic full-interior blockage used by the first compact
    attempt.  A narrow edge halo remains available for routed pin access.
    """

    source = source.resolve()
    destination = destination.resolve()
    size_pattern = re.compile(
        r"^\s*SIZE\s+([0-9.eE+-]+)\s+BY\s+([0-9.eE+-]+)\s*;"
    )
    layer_pattern = re.compile(r"^\s*LAYER\s+(\S+)\s*;")
    rect_pattern = re.compile(
        r"^\s*RECT\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
        r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*;"
    )
    layers = list(obstruction_config["routing_layers"])
    inset = float(obstruction_config["integration_inset_um"])
    raster_pitch = float(obstruction_config["raster_pitch_um"])
    if raster_pitch <= 0:
        raise ValueError("raster pitch must be positive")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    prefix_digest = hashlib.sha256()
    macro_name: str | None = None
    width: float | None = None
    height: float | None = None
    pin_count = 0
    pin_rectangles = 0
    accessible_pin_rectangles = 0
    pin_layers: set[str] = set()
    current_pin: str | None = None
    current_layer: str | None = None
    source_obstruction_rectangles = 0
    found_obstructions = False
    obstruction_layer: str | None = None
    raster_differences: dict[str, np.ndarray] = {}
    source_obstruction_rectangles_by_layer = {layer: 0 for layer in layers}

    with source.open() as input_stream, temporary.open("w") as output_stream:
        for line in input_stream:
            if line.startswith("  OBS"):
                found_obstructions = True
                break
            output_stream.write(line)
            prefix_digest.update(line.encode())
            if line.startswith("MACRO "):
                macro_name = line.split()[1]
            size_match = size_pattern.match(line)
            if size_match:
                width = float(size_match.group(1))
                height = float(size_match.group(2))
            if line.startswith("  PIN "):
                current_pin = line.split()[1]
                pin_count += 1
                current_layer = None
            elif current_pin and line.startswith("  END "):
                current_pin = None
                current_layer = None
            elif current_pin:
                layer_match = layer_pattern.match(line)
                if layer_match:
                    current_layer = layer_match.group(1)
                rect_match = rect_pattern.match(line)
                if rect_match:
                    if width is None or height is None or current_layer is None:
                        raise ValueError(f"malformed pin geometry in {source}")
                    x_min, y_min, x_max, y_max = map(float, rect_match.groups())
                    pin_rectangles += 1
                    pin_layers.add(current_layer)
                    if min(x_min, y_min, width - x_max, height - y_max) <= inset:
                        accessible_pin_rectangles += 1

        if not found_obstructions or macro_name is None or width is None or height is None:
            raise ValueError(f"missing macro boundary or OBS section in {source}")
        if width <= 2 * inset or height <= 2 * inset:
            raise ValueError(f"integration inset is too large for {source}")
        if pin_count == 0 or pin_rectangles == 0:
            raise ValueError(f"no routed pin geometry found in {source}")
        if accessible_pin_rectangles != pin_rectangles:
            raise ValueError(f"compact obstruction would cover a routed pin in {source}")

        raster_width = math.ceil((width - 2 * inset) / raster_pitch)
        raster_height = math.ceil((height - 2 * inset) / raster_pitch)
        if raster_width <= 0 or raster_height <= 0:
            raise ValueError(f"invalid obstruction raster for {source}")
        raster_differences = {
            layer: np.zeros((raster_height + 1, raster_width + 1), dtype=np.int32)
            for layer in layers
        }

        for line in input_stream:
            layer_match = layer_pattern.match(line)
            if layer_match:
                obstruction_layer = layer_match.group(1)
                continue
            rect_match = rect_pattern.match(line)
            if not rect_match:
                continue
            source_obstruction_rectangles += 1
            if obstruction_layer not in raster_differences:
                continue
            source_obstruction_rectangles_by_layer[obstruction_layer] += 1
            x_min, y_min, x_max, y_max = map(float, rect_match.groups())
            x_min = max(x_min, inset)
            y_min = max(y_min, inset)
            x_max = min(x_max, width - inset)
            y_max = min(y_max, height - inset)
            if x_min >= x_max or y_min >= y_max:
                continue
            x_start = max(
                0,
                min(raster_width, math.floor((x_min - inset) / raster_pitch)),
            )
            y_start = max(
                0,
                min(raster_height, math.floor((y_min - inset) / raster_pitch)),
            )
            x_stop = max(
                0,
                min(raster_width, math.ceil((x_max - inset) / raster_pitch)),
            )
            y_stop = max(
                0,
                min(raster_height, math.ceil((y_max - inset) / raster_pitch)),
            )
            if x_start >= x_stop or y_start >= y_stop:
                continue
            difference = raster_differences[obstruction_layer]
            difference[y_start, x_start] += 1
            difference[y_stop, x_start] -= 1
            difference[y_start, x_stop] -= 1
            difference[y_stop, x_stop] += 1

        output_stream.write("  OBS\n")
        integration_obstruction_rectangles_by_layer: dict[str, int] = {}
        occupied_raster_cells_by_layer: dict[str, int] = {}
        for layer in layers:
            output_stream.write(f"    LAYER {layer} ;\n")
            difference = raster_differences.pop(layer)
            occupied = np.cumsum(
                np.cumsum(difference[:-1, :-1], axis=0, dtype=np.int32),
                axis=1,
                dtype=np.int32,
            ) > 0
            occupied_raster_cells_by_layer[layer] = int(occupied.sum())
            active_runs: dict[tuple[int, int], int] = {}
            emitted = 0

            def emit(run: tuple[int, int], y_start: int, y_stop: int) -> None:
                nonlocal emitted
                x_start, x_stop = run
                rect_x_min = inset + x_start * raster_pitch
                rect_y_min = inset + y_start * raster_pitch
                rect_x_max = min(width - inset, inset + x_stop * raster_pitch)
                rect_y_max = min(height - inset, inset + y_stop * raster_pitch)
                output_stream.write(
                    f"      RECT {rect_x_min:g} {rect_y_min:g} "
                    f"{rect_x_max:g} {rect_y_max:g} ;\n"
                )
                emitted += 1

            for row_index, row in enumerate(occupied):
                padded_row = np.concatenate((np.array([False]), row, np.array([False])))
                transitions = np.flatnonzero(
                    padded_row[1:] != padded_row[:-1]
                )
                runs = list(zip(transitions[0::2].tolist(), transitions[1::2].tolist()))
                current_runs = set(runs)
                for run in list(active_runs):
                    if run not in current_runs:
                        emit(run, active_runs.pop(run), row_index)
                for run in runs:
                    active_runs.setdefault(run, row_index)
            for run, y_start in active_runs.items():
                emit(run, y_start, raster_height)
            integration_obstruction_rectangles_by_layer[layer] = emitted
        output_stream.write("  END\n")
        output_stream.write(f"END {macro_name}\n")
        output_stream.write("END LIBRARY\n")

    temporary.replace(destination)
    return {
        "method": obstruction_config["integration_method"],
        "source_method": obstruction_config["source_method"],
        "source_path": str(source.relative_to(PROJECT_ROOT)),
        "integration_path": str(destination.relative_to(PROJECT_ROOT)),
        "source_bytes": source.stat().st_size,
        "integration_bytes": destination.stat().st_size,
        "compression_ratio": source.stat().st_size / destination.stat().st_size,
        "macro_name": macro_name,
        "macro_width_um": width,
        "macro_height_um": height,
        "pin_count": pin_count,
        "pin_rectangles": pin_rectangles,
        "accessible_pin_rectangles": accessible_pin_rectangles,
        "pin_layers": sorted(pin_layers),
        "pin_prefix_sha256": prefix_digest.hexdigest(),
        "source_obstruction_rectangles": source_obstruction_rectangles,
        "source_obstruction_rectangles_by_layer": source_obstruction_rectangles_by_layer,
        "integration_obstruction_rectangles": sum(
            integration_obstruction_rectangles_by_layer.values()
        ),
        "integration_obstruction_rectangles_by_layer": (
            integration_obstruction_rectangles_by_layer
        ),
        "integration_obstruction_layers": layers,
        "integration_inset_um": inset,
        "raster_pitch_um": raster_pitch,
        "raster_width": raster_width,
        "raster_height": raster_height,
        "occupied_raster_cells_by_layer": occupied_raster_cells_by_layer,
        "conservative_obstruction_cover": True,
        "pin_geometry_preserved": bool(obstruction_config["preserve_pin_geometry"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-pe-synthesis", action="store_true")
    parser.add_argument("--reuse-pe-physical", action="store_true")
    parser.add_argument("--reuse-top-synthesis", action="store_true")
    parser.add_argument("--reuse-top-physical", action="store_true")
    parser.add_argument(
        "--top-grt-iterations",
        type=int,
        help=(
            "override only the executed top-level GRT iteration budget; "
            "the configured final signoff contract remains unchanged"
        ),
    )
    args = parser.parse_args()
    if args.top_grt_iterations is not None and args.top_grt_iterations < 0:
        parser.error("--top-grt-iterations must be non-negative")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    technology = config["technology"]
    route_plan = config["hierarchical_top_route_resource_plan"]
    global_route_openroad = Path(route_plan["global_route_openroad"])
    if not global_route_openroad.is_absolute():
        global_route_openroad = PROJECT_ROOT / global_route_openroad
    detailed_route_openroad = Path(route_plan["detailed_route_openroad"])
    if not detailed_route_openroad.is_absolute():
        detailed_route_openroad = PROJECT_ROOT / detailed_route_openroad
    global_route_patch = PROJECT_ROOT / route_plan["global_route_patch"]
    route_tool_contract = config["toolchain"]["global_route_openroad"]
    global_route_archive = PROJECT_ROOT / route_tool_contract["archive"]
    signoff_tool_contract = config["toolchain"][
        "detailed_route_and_signoff_openroad"
    ]
    global_route_tool = {
        "binary": artifact(global_route_openroad),
        "patch": artifact(global_route_patch),
        "archive": artifact(global_route_archive),
        "version": subprocess.run(
            [str(global_route_openroad), "-version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "base_commit": route_tool_contract["base_commit"],
        "grid_pitches_in_tile": int(route_plan["grid_pitches_in_tile"]),
        "max_2d_edge_usage_multiplier": int(
            route_plan["max_2d_edge_usage_multiplier"]
        ),
    }
    detailed_route_tool = artifact(detailed_route_openroad)
    route_tool_valid = (
        global_route_tool["binary"]["sha256"]
        == route_tool_contract["binary_sha256"]
        and global_route_tool["patch"]["sha256"]
        == route_tool_contract["patch_sha256"]
        and global_route_tool["archive"]["sha256"]
        == route_tool_contract["archive_sha256"]
        and route_tool_contract["base_commit"] in global_route_tool["version"]
        and global_route_tool["grid_pitches_in_tile"]
        == route_tool_contract["grid_pitches_in_tile"]
        == 48
        and global_route_tool["max_2d_edge_usage_multiplier"]
        == route_tool_contract["max_2d_edge_usage_multiplier"]
        == 101
        and detailed_route_tool["sha256"]
        == signoff_tool_contract["binary_sha256"]
    )
    if not route_tool_valid:
        print(
            json.dumps(
                {
                    "stage": "global_route_tool",
                    "global_route_tool": global_route_tool,
                    "detailed_route_tool": detailed_route_tool,
                },
                indent=2,
            )
        )
        return 1
    liberty = Path(technology["liberty"])
    output = PROJECT_ROOT / config["output_root"]
    pe_root = output / "pe_macro"
    top_root = output / "hierarchical_4x4"
    submacro_root = output / "pe_submacros"
    submacro_manifest_path = submacro_root / "submacro-build-manifest.json"
    pe_root.mkdir(parents=True, exist_ok=True)
    top_root.mkdir(parents=True, exist_ok=True)
    if not submacro_manifest_path.is_file():
        print(
            json.dumps(
                {
                    "stage": "submacro_chain",
                    "error": f"missing {submacro_manifest_path}",
                    "remedy": "run python -m scripts.build_mlx_pe_submacros --reuse",
                }
            )
        )
        return 1
    submacro_chain = json.loads(submacro_manifest_path.read_text())
    required_submacros = {
        "full_lane",
        "reduced_lane",
        "register_file",
        "functional_unit",
        "pe_top",
    }
    if set(submacro_chain) != required_submacros or not all(
        all(item["checks"].values()) for item in submacro_chain.values()
    ):
        print(json.dumps({"stage": "submacro_chain", "manifest": submacro_chain}, indent=2))
        return 1

    # Stage A: fixed-parameter, fully physical PE macro.
    pe_netlist = pe_root / "mlx-pe-top-mapped.v"
    pe_stats = pe_root / "mlx-pe-top-synthesis.stats"
    pe_synth_log = pe_root / "mlx-pe-top-synthesis.log"
    if not all(path.is_file() for path in (pe_netlist, pe_stats, pe_synth_log)):
        print(
            json.dumps(
                {
                    "stage": "pe_synthesis",
                    "error": "missing recursive submacro PE synthesis outputs",
                    "remedy": "run python -m scripts.build_mlx_pe_submacros --reuse",
                }
            )
        )
        return 1
    pe_synth_rc = 0
    pe_synthesis = parse_synthesis(pe_stats.read_text())

    source_vcd = PROJECT_ROOT / config["activity"]["source"]
    pe_vcd = pe_root / "transformer-block-pe0-ports.vcd"
    pe_vcd_extract = extract_scope(
        source_vcd,
        pe_vcd,
        "TOP.mlx_array_4x4.GENERATE_PES(0).physical_pe",
        ports_only=True,
        timestamp_scale=int(config["activity"]["timestamp_scale"]),
    )
    pe_phys = physical_paths(pe_root, "mlx-pe-top-routed")
    pe_lef = pe_root / "mlx-pe-top.abstract.lef"
    pe_lib = pe_root / "mlx-pe-top.lib"
    common_environment = os.environ.copy()
    common_environment.update(
        {
            "PPA_THREADS": str(config["threads"]),
            "PPA_TECH_LEF": technology["tech_lef"],
            "PPA_MACRO_LEF": technology["macro_lef"],
            "PPA_LIBERTY": technology["liberty"],
            "PPA_RCX_RULES": technology["rcx_rules"],
            "PPA_TAPCELL_TCL": technology["tapcell_tcl"],
            "TAP_CELL_NAME": "TAPCELL_X1",
            "PPA_CLOCK_PERIOD_NS": str(config["clock_period_ns"]),
            "PPA_UTILIZATION": str(config["placement_utilization_percent"]),
            "PPA_DENSITY": str(config["placement_density"]),
            "PPA_OVERFLOW_TARGET": str(config["placement_overflow_target"]),
            "PPA_INIT_DENSITY_PENALTY": str(
                config["placement_init_density_penalty"]
            ),
            "PPA_MIN_PHI_COEF": str(config["placement_min_phi_coef"]),
            "PPA_MAX_PHI_COEF": str(config["placement_max_phi_coef"]),
            "PPA_DROUTE_END_ITER": str(config["droute_end_iter"]),
            "PPA_GRT_CONGESTION_ITERATIONS": str(
                config["global_route_congestion_iterations"]
            ),
            "PPA_GRT_ALLOW_CONGESTION": (
                "1" if config["global_route_allow_congestion"] else "0"
            ),
        }
    )
    pe_physical_rc = (
        0
        if output_check(pe_phys) and pe_lef.is_file() and pe_lib.is_file()
        else 1
    )
    pe_physical = macro_metrics(pe_phys["log"].read_text(), float(config["clock_period_ns"]))
    pe_checks = {
        "synthesis": pe_synth_rc == 0
        and int(pe_synthesis["cell_count"] or 0) > 0
        and float(pe_synthesis["cell_area_um2"] or 0) > 0,
        "place_route": pe_physical_rc == 0
        and output_check(pe_phys)
        and pe_lef.is_file()
        and pe_lib.is_file(),
        "drc_clean": pe_physical["drc_violations"] == 0,
        "timing": pe_physical["fmax_ghz"] is not None,
        "vcd_power": int(pe_physical["annotated_pin_activities"] or 0) > 0
        and float(pe_physical["total_power_w"] or 0.0) > 0,
    }
    if not all(pe_checks.values()):
        print(json.dumps({"stage": "pe_physical", "checks": pe_checks, **pe_physical}, indent=2))
        return 1

    pe_integration_lef = pe_root / "mlx-pe-top.integration.lef"
    pe_abstraction = build_compact_macro_lef(
        pe_lef,
        pe_integration_lef,
        config["abstract_lef_obstructions"],
    )
    pe_abstraction_valid = (
        pe_abstraction["pin_geometry_preserved"]
        and pe_abstraction["conservative_obstruction_cover"]
        and pe_abstraction["pin_count"] > 0
        and pe_abstraction["pin_rectangles"]
        == pe_abstraction["accessible_pin_rectangles"]
        and pe_abstraction["source_obstruction_rectangles"]
        > pe_abstraction["integration_obstruction_rectangles"]
        and pe_abstraction["integration_bytes"] < pe_abstraction["source_bytes"]
    )
    if not pe_abstraction_valid:
        print(json.dumps({"stage": "pe_abstraction", **pe_abstraction}, indent=2))
        return 1

    # Stage B: synthesize the array controller/network around sixteen hard PE macros.
    top_netlist = top_root / "mlx-array-4x4-hierarchical-mapped.v"
    top_stats = top_root / "mlx-array-4x4-hierarchical-synthesis.stats"
    top_synth_log = top_root / "mlx-array-4x4-hierarchical-synthesis.log"
    blackbox = PROJECT_ROOT / "rtl/ppa/mlx_pe_top_blackbox.sv"
    array_source = PROJECT_ROOT / "rtl/mlx/mlx_array_4x4.sv"
    top_commands = [
        f"read_verilog -sv -lib {blackbox}",
        f"read_verilog -sv -DMLX_PPA_MACRO {array_source}",
        "hierarchy -check -top mlx_array_4x4",
        "synth -top mlx_array_4x4",
        f"dfflibmap -liberty {liberty}",
        f"abc -fast -liberty {liberty}",
        "hilomap -singleton -hicell LOGIC1_X1 Z -locell LOGIC0_X1 Z",
        "clean",
        f"tee -o {top_stats} stat -liberty {liberty}",
        f"write_verilog -noattr -noexpr -nodec {top_netlist}",
    ]
    if args.reuse_top_synthesis and top_netlist.is_file() and top_stats.is_file():
        top_synth_rc = 0
    else:
        top_synth_rc = yosys_run(top_commands, top_synth_log)
    if top_synth_rc != 0 or not top_netlist.is_file() or not top_stats.is_file():
        print(json.dumps({"stage": "top_synthesis", "returncode": top_synth_rc}))
        return 1
    top_synthesis = parse_synthesis(top_stats.read_text())
    macro_instances = len(
        re.findall(
            r"(?m)^\s*mlx_pe_top\s+(?:\\\S+|\S+)\s*\(",
            top_netlist.read_text(),
        )
    )

    top_vcd = output / "transformer-block-array-ports.vcd"
    top_vcd_extract = extract_scope(
        source_vcd,
        top_vcd,
        config["activity"]["source_scope"],
        ports_only=True,
        timestamp_scale=int(config["activity"]["timestamp_scale"]),
    )
    top_placement = config["hierarchical_top_placement"]
    macro_track_contract = top_placement["macro_origin_track_alignment"]
    routing_pitch_dbu = [
        int(value) for value in macro_track_contract["routing_pitch_dbu"].values()
    ]
    derived_macro_origin_grid_dbu = math.lcm(*routing_pitch_dbu)
    macro_track_contract_valid = (
        int(macro_track_contract["dbu_per_micron"]) == 2000
        and int(macro_track_contract["grid_dbu"]) == derived_macro_origin_grid_dbu
        and math.isclose(
            float(macro_track_contract["grid_um"]),
            derived_macro_origin_grid_dbu / int(macro_track_contract["dbu_per_micron"]),
        )
        and int(macro_track_contract["required_macro_instances"]) == macro_instances == 16
    )
    if not macro_track_contract_valid:
        print(
            json.dumps(
                {
                    "stage": "macro_track_contract",
                    "contract": macro_track_contract,
                    "derived_grid_dbu": derived_macro_origin_grid_dbu,
                },
                indent=2,
            )
        )
        return 1
    flow_tag = re.sub(r"[^A-Za-z0-9_.-]", "-", top_placement["flow_tag"])
    physical_stem = f"mlx-array-4x4-hierarchical-{flow_tag}-routed"
    checkpoint_stem = f"mlx-array-4x4-hierarchical-{flow_tag}"
    top_phys = physical_paths(top_root, physical_stem)
    top_gpl_checkpoint = top_root / f"{checkpoint_stem}-global-placement.odb"
    top_rows_checkpoint = top_root / f"{checkpoint_stem}-rows.odb"
    top_seed_checkpoint = top_root / f"{checkpoint_stem}-seed.odb"
    top_precheck_checkpoint = top_root / f"{checkpoint_stem}-precheck.odb"
    top_legal_checkpoint = top_root / f"{checkpoint_stem}-channel-legal.odb"
    top_cts_seed_checkpoint = top_root / f"{checkpoint_stem}-cts-seed.odb"
    top_cts_checkpoint = top_root / f"{checkpoint_stem}-post-cts.odb"
    top_grt_checkpoint = top_root / f"{checkpoint_stem}-global-route.odb"
    top_channel_log = top_root / f"{checkpoint_stem}-channel-legalize.log"
    top_cts_log = top_root / f"{checkpoint_stem}-cts.log"
    top_route_log = top_root / f"{checkpoint_stem}-route.log"
    top_congestion_report = top_root / f"{checkpoint_stem}-congestion.rpt"
    top_droute_resume_log = top_root / f"{checkpoint_stem}-droute-resume.log"
    top_grt_execution_iterations = (
        args.top_grt_iterations
        if args.top_grt_iterations is not None
        else int(
            top_placement.get(
                "global_route_congestion_iterations",
                config["global_route_congestion_iterations"],
            )
        )
    )
    top_environment = common_environment.copy()
    top_environment.update(
        {
            "PPA_PE_LEF": str(pe_integration_lef),
            "PPA_PE_LIBERTY": str(pe_lib),
            "PPA_NETLIST": str(top_netlist),
            "PPA_UTILIZATION": str(top_placement["utilization_percent"]),
            "PPA_DENSITY": str(top_placement["density"]),
            "PPA_ASPECT_RATIO": str(top_placement["aspect_ratio"]),
            "PPA_SKIP_INITIAL_PLACE": (
                "1" if top_placement["skip_initial_place"] else "0"
            ),
            "PPA_BIN_GRID_COUNT": str(top_placement["bin_grid_count"]),
            "PPA_OVERFLOW_TARGET": str(
                top_placement.get(
                    "overflow_target", config["placement_overflow_target"]
                )
            ),
            "PPA_GRT_CONGESTION_ITERATIONS": str(
                top_grt_execution_iterations
            ),
            "PPA_REPAIR_DESIGN": (
                "1" if top_placement.get("repair_design") else "0"
            ),
            "PPA_REPAIR_MAX_UTILIZATION": str(
                top_placement.get("repair_max_utilization_percent", 80)
            ),
            "PPA_TAPCELL_DISTANCE_UM": str(
                top_placement["tapcell_distance_um"]
            ),
            "PPA_DPL_ROW_LIMIT": str(
                top_placement.get("detailed_placement_full_width_rows", 0)
            ),
            "PPA_GUIDE": str(top_phys["guide"]),
            "PPA_DRC": str(top_phys["drc"]),
            "PPA_DEF": str(top_phys["def"]),
            "PPA_ODB": str(top_phys["odb"]),
            "PPA_SPEF": str(top_phys["spef"]),
            "PPA_VCD": str(top_vcd),
            "PPA_VCD_SCOPE": config["activity"]["promoted_scope"],
            "PPA_GPL_ODB": str(top_gpl_checkpoint),
            "PPA_ROWS_ODB": str(top_rows_checkpoint),
            "PPA_SEED_ODB": str(top_seed_checkpoint),
            "PPA_PRECHECK_ODB": str(top_precheck_checkpoint),
            "PPA_LEGAL_ODB": str(top_legal_checkpoint),
            "PPA_CTS_SEED_ODB": str(top_cts_seed_checkpoint),
            "PPA_CTS_ODB": str(top_cts_checkpoint),
            "PPA_GRT_ODB": str(top_grt_checkpoint),
            "PPA_POST_GPL_THREADS": str(top_placement.get("post_gpl_threads", 1)),
            "PPA_CHANNEL_TARGET_UTILIZATION": str(
                top_placement["channel_legalizer"]["target_row_utilization"]
            ),
            "PPA_PE_MACRO_MASTER": "mlx_pe_top",
            "PPA_MACRO_INSTANCE_COUNT": str(
                macro_track_contract["required_macro_instances"]
            ),
            "PPA_MACRO_ORIGIN_GRID_DBU": str(macro_track_contract["grid_dbu"]),
            "PPA_SIGNAL_ROUTING_LAYERS": route_plan["routing_layers"]["signal"],
            "PPA_CLOCK_ROUTING_LAYERS": route_plan["routing_layers"]["clock"],
            "PPA_MACRO_EXTENSION_GCELLS": str(
                route_plan["macro_extension_gcells"]
            ),
            "PPA_CRITICAL_NETS_PERCENTAGE": str(
                route_plan["critical_nets_percentage"]
            ),
            "PPA_GRT_VERBOSE": "1" if route_plan.get("verbose") else "0",
            "PPA_GRT_CONGESTION_REPORT_FILE": str(top_congestion_report),
            "PPA_GRT_CONGESTION_REPORT_ITER_STEP": str(
                route_plan.get("congestion_report_iter_step", 0)
            ),
            "PPA_LAYER_CAPACITY_ADJUSTMENTS": " ".join(
                f"{layer} {adjustment}"
                for layer, adjustment in route_plan[
                    "layer_capacity_adjustments"
                ].items()
            ),
            "PPA_STOP_AFTER_GRT": "0",
            "MALLOC_ARENA_MAX": "2",
        }
    )
    if args.reuse_top_physical and output_check(top_phys):
        top_physical_rc = 0
    else:
        prior_grt_metrics = parse_global_route_metrics(
            top_route_log.read_text() if top_route_log.is_file() else ""
        )
        prior_grt_connectivity = parse_route_connectivity(
            top_route_log.read_text() if top_route_log.is_file() else "", ""
        )
        resume_top_droute = (
            top_grt_checkpoint.is_file()
            and top_grt_checkpoint.stat().st_size > 0
            and prior_grt_metrics["overflow_resolved"] is True
            and prior_grt_metrics["congestion_iterations"]
            == int(route_plan["congestion_iterations"])
            == int(top_placement["global_route_congestion_iterations"])
            and prior_grt_connectivity["global_missing_pin_routes"] == 0
            and prior_grt_connectivity["global_missing_warning_limit_reached"]
            is False
        )
        resume_top_post_gpl = (
            not resume_top_droute
            and top_gpl_checkpoint.is_file()
            and top_gpl_checkpoint.stat().st_size > 0
        )
        top_physical_rc = 1
        if resume_top_droute:
            top_physical_rc = run_to_log(
                [
                    str(detailed_route_openroad),
                    "-no_init",
                    "-exit",
                    str(PROJECT_ROOT / "rtl/ppa/openroad_hierarchical_array_droute_resume.tcl"),
                ],
                top_droute_resume_log,
                top_environment,
            )
            prior_log = top_phys["log"].read_text() if top_phys["log"].is_file() else ""
            top_phys["log"].write_text(
                prior_log
                + "\nMLX_ARRAY_DROUTE_RESUME_LOG_BEGIN\n"
                + top_droute_resume_log.read_text()
            )
        else:
            if not resume_top_post_gpl:
                initial_environment = top_environment.copy()
                initial_environment["PPA_RESUME_GPL"] = "0"
                initial_environment["PPA_STOP_AFTER_GPL"] = "1"
                top_physical_rc = run_to_log(
                    [
                        "openroad",
                        "-no_init",
                        "-exit",
                        str(PROJECT_ROOT / "rtl/ppa/openroad_hierarchical_array_flow.tcl"),
                    ],
                    top_phys["log"],
                    initial_environment,
                )
                resume_top_post_gpl = (
                    top_physical_rc == 0
                    and top_gpl_checkpoint.is_file()
                    and top_gpl_checkpoint.stat().st_size > 0
                )
            if resume_top_post_gpl:
                legal_ready = (
                    top_legal_checkpoint.is_file()
                    and top_legal_checkpoint.stat().st_size > 0
                )
                if not legal_ready:
                    channel_environment = top_environment.copy()
                    channel_environment["PPA_RESUME_ROWS"] = (
                        "1"
                        if top_rows_checkpoint.is_file()
                        and top_rows_checkpoint.stat().st_size > 0
                        else "0"
                    )
                    channel_environment["PPA_THREADS"] = str(
                        top_placement.get("post_gpl_threads", 1)
                    )
                    channel_rc = run_to_log(
                        [
                            "openroad",
                            "-no_init",
                            "-exit",
                            str(
                                PROJECT_ROOT
                                / "rtl/ppa/openroad_hierarchical_array_channel_legalize.tcl"
                            ),
                        ],
                        top_channel_log,
                        channel_environment,
                    )
                    legal_ready = (
                        channel_rc == 0
                        and top_legal_checkpoint.is_file()
                        and top_legal_checkpoint.stat().st_size > 0
                    )
                    prior_log = (
                        top_phys["log"].read_text() if top_phys["log"].is_file() else ""
                    )
                    top_phys["log"].write_text(
                        prior_log
                        + "\nMLX_ARRAY_CHANNEL_LEGALIZE_LOG_BEGIN\n"
                        + top_channel_log.read_text()
                    )
                if legal_ready:
                    cts_ready = (
                        top_cts_checkpoint.is_file()
                        and top_cts_checkpoint.stat().st_size > 0
                    )
                    if not cts_ready:
                        cts_environment = top_environment.copy()
                        cts_environment["PPA_THREADS"] = str(
                            top_placement.get("post_gpl_threads", 1)
                        )
                        resume_cts_seed = (
                            top_cts_seed_checkpoint.is_file()
                            and top_cts_seed_checkpoint.stat().st_size > 0
                        )
                        cts_environment["PPA_RESUME_CTS"] = "0"
                        cts_environment["PPA_STOP_AFTER_CTS"] = "1"
                        cts_rc = run_to_log(
                            [
                                "openroad",
                                "-no_init",
                                "-exit",
                                str(
                                    PROJECT_ROOT
                                    / (
                                        "rtl/ppa/openroad_hierarchical_array_cts_legalize_resume.tcl"
                                        if resume_cts_seed
                                        else "rtl/ppa/openroad_hierarchical_array_post_legal_flow.tcl"
                                    )
                                ),
                            ],
                            top_cts_log,
                            cts_environment,
                        )
                        cts_ready = (
                            cts_rc == 0
                            and top_cts_checkpoint.is_file()
                            and top_cts_checkpoint.stat().st_size > 0
                        )
                        prior_log = (
                            top_phys["log"].read_text()
                            if top_phys["log"].is_file()
                            else ""
                        )
                        top_phys["log"].write_text(
                            prior_log
                            + "\nMLX_ARRAY_CTS_LOG_BEGIN\n"
                            + top_cts_log.read_text()
                        )
                    if cts_ready:
                        route_environment = top_environment.copy()
                        route_environment["PPA_THREADS"] = str(
                            top_placement.get("post_gpl_threads", 1)
                        )
                        route_environment["PPA_RESUME_CTS"] = "1"
                        route_environment["PPA_STOP_AFTER_CTS"] = "0"
                        route_environment["PPA_STOP_AFTER_GRT"] = (
                            "1" if route_plan["stop_after_global_route"] else "0"
                        )
                        for _, iteration_report in all_congestion_iteration_reports(
                            top_congestion_report
                        ):
                            iteration_report.unlink()
                        top_physical_rc = run_to_log(
                            [
                                str(global_route_openroad),
                                "-no_init",
                                "-exit",
                                str(
                                    PROJECT_ROOT
                                    / "rtl/ppa/openroad_hierarchical_array_post_legal_flow.tcl"
                                ),
                            ],
                            top_route_log,
                            route_environment,
                        )
                        prior_log = (
                            top_phys["log"].read_text()
                            if top_phys["log"].is_file()
                            else ""
                        )
                        top_phys["log"].write_text(
                            prior_log
                            + "\nMLX_ARRAY_ROUTE_LOG_BEGIN\n"
                            + top_route_log.read_text()
                        )
                        grt_ready = (
                            top_physical_rc == 0
                            and top_grt_checkpoint.is_file()
                            and top_grt_checkpoint.stat().st_size > 0
                        )
                        grt_metrics = parse_global_route_metrics(
                            top_route_log.read_text()
                            if top_route_log.is_file()
                            else ""
                        )
                        grt_connectivity = parse_route_connectivity(
                            top_route_log.read_text()
                            if top_route_log.is_file()
                            else "",
                            "",
                        )
                        grt_ready_for_droute = (
                            grt_ready
                            and grt_metrics["overflow_resolved"] is True
                            and grt_metrics["congestion_iterations"]
                            == int(route_plan["congestion_iterations"])
                            == int(
                                top_placement["global_route_congestion_iterations"]
                            )
                            and grt_connectivity["global_missing_pin_routes"] == 0
                            and grt_connectivity[
                                "global_missing_warning_limit_reached"
                            ]
                            is False
                        )
                        if (
                            route_plan["stop_after_global_route"]
                            and grt_ready_for_droute
                        ):
                            top_physical_rc = run_to_log(
                                [
                                    str(detailed_route_openroad),
                                    "-no_init",
                                    "-exit",
                                    str(
                                        PROJECT_ROOT
                                        / "rtl/ppa/openroad_hierarchical_array_droute_resume.tcl"
                                    ),
                                ],
                                top_droute_resume_log,
                                top_environment,
                            )
                            prior_log = top_phys["log"].read_text()
                            top_phys["log"].write_text(
                                prior_log
                                + "\nMLX_ARRAY_DROUTE_RESUME_LOG_BEGIN\n"
                                + top_droute_resume_log.read_text()
                            )
                        elif route_plan["stop_after_global_route"]:
                            top_physical_rc = 1
    channel_legalization = parse_channel_legalization(
        top_channel_log.read_text() if top_channel_log.is_file() else ""
    )
    channel_legalization["maximum_accepted_displacement_dbu"] = round(
        float(
            top_placement["channel_legalizer"][
                "maximum_accepted_displacement_um"
            ]
        )
        * int(macro_track_contract["dbu_per_micron"])
    )
    channel_legalization["maximum_accepted_displacement_basis"] = (
        top_placement["channel_legalizer"].get(
            "maximum_accepted_displacement_basis", "configured_absolute_limit"
        )
    )
    channel_legalization_valid = (
        channel_legalization["cells"]
        == int(top_synthesis["cell_count"] or 0) - macro_instances
        and (
            (
                int(top_placement["detailed_placement_full_width_rows"]) > 0
                and channel_legalization["rows"]
                == int(top_placement["detailed_placement_full_width_rows"])
            )
            or (
                int(top_placement["detailed_placement_full_width_rows"]) == 0
                and channel_legalization["rows"]
                >= int(
                    top_placement["channel_legalizer"]["minimum_physical_rows"]
                )
            )
        )
        and int(channel_legalization["row_segments"] or 0)
        >= int(top_placement["channel_legalizer"]["minimum_row_segments"])
        and channel_legalization["selected_physical_rows"]
        == channel_legalization["rows"]
        and channel_legalization["selected_row_segments"]
        == channel_legalization["row_segments"]
        and channel_legalization["assigned_cells"]
        == channel_legalization["cells"]
        and channel_legalization["constructive_audit_cells"]
        == channel_legalization["cells"]
        and channel_legalization["site_aligned_cells"]
        == channel_legalization["cells"]
        and channel_legalization["segment_contained_cells"]
        == channel_legalization["cells"]
        and channel_legalization["standard_nonoverlap_cells"]
        == channel_legalization["cells"]
        and channel_legalization[
            "audited_nonoverlapping_macro_clear_row_segments"
        ]
        == channel_legalization["row_segments"]
        and channel_legalization["constructive_audit_row_segments"]
        == channel_legalization["row_segments"]
        and channel_legalization["full_width_y_escapes"] is not None
        and channel_legalization["backward_compactions"] is not None
        and channel_legalization["rows_checkpoint"] == str(top_rows_checkpoint)
        and channel_legalization["seed_checkpoint"] == str(top_seed_checkpoint)
        and channel_legalization["precheck_checkpoint"] == str(top_precheck_checkpoint)
        and int(channel_legalization["taps"] or 0) > 0
        and float(channel_legalization["minimum_capacity_ratio"] or 0.0) > 1.0
        and channel_legalization["checkpoint"] == str(top_legal_checkpoint)
        and channel_legalization["max_displacement_dbu"] is not None
        and int(channel_legalization["max_displacement_dbu"])
        <= channel_legalization["maximum_accepted_displacement_dbu"]
        and channel_legalization["maximum_x_displacement_dbu"] is not None
        and channel_legalization["maximum_y_displacement_dbu"] is not None
        and channel_legalization["average_displacement_dbu"] is not None
    )
    cts_buffer_legalization = parse_cts_buffer_legalization(
        top_cts_log.read_text() if top_cts_log.is_file() else ""
    )
    cts_buffer_legalization_valid = (
        int(cts_buffer_legalization["buffers"] or 0) > 0
        and cts_buffer_legalization["assigned_buffers"]
        == cts_buffer_legalization["buffers"]
        and cts_buffer_legalization["fixed_cells"]
        == int(channel_legalization["cells"] or 0)
        + int(channel_legalization["taps"] or 0)
        and cts_buffer_legalization["physical_rows"]
        == channel_legalization["rows"]
        and cts_buffer_legalization["row_segments"]
        == channel_legalization["row_segments"]
        and cts_buffer_legalization["site_aligned_buffers"]
        == cts_buffer_legalization["buffers"]
        and cts_buffer_legalization["segment_contained_buffers"]
        == cts_buffer_legalization["buffers"]
        and cts_buffer_legalization["fixed_clear_buffers"]
        == cts_buffer_legalization["buffers"]
        and cts_buffer_legalization["standard_nonoverlap_buffers"]
        == cts_buffer_legalization["buffers"]
        and cts_buffer_legalization["max_displacement_dbu"] is not None
        and int(cts_buffer_legalization["max_displacement_dbu"])
        <= float(
            top_placement["channel_legalizer"][
                "maximum_cts_buffer_displacement_um"
            ]
        )
        * int(macro_track_contract["dbu_per_micron"])
        and cts_buffer_legalization["average_displacement_dbu"] is not None
        and cts_buffer_legalization["seed_checkpoint"]
        == str(top_cts_seed_checkpoint)
        and cts_buffer_legalization["checkpoint"] == str(top_cts_checkpoint)
    )
    macro_track_alignment_valid = (
        macro_track_contract_valid
        and channel_legalization["macro_instances_aligned"] == macro_instances
        and channel_legalization["macro_origin_grid_dbu"]
        == derived_macro_origin_grid_dbu
        and channel_legalization["macro_max_displacement_dbu"] is not None
        and int(channel_legalization["macro_max_displacement_dbu"]) == 0
    )
    route_connectivity = parse_route_connectivity(
        top_route_log.read_text() if top_route_log.is_file() else "",
        top_droute_resume_log.read_text() if top_droute_resume_log.is_file() else "",
    )
    global_route_metrics = parse_global_route_metrics(
        top_route_log.read_text() if top_route_log.is_file() else ""
    )
    global_route_congestion_valid = (
        global_route_metrics["overflow_resolved"] is True
        and global_route_metrics["congestion_iterations"]
        == int(route_plan["congestion_iterations"])
        == int(top_placement["global_route_congestion_iterations"])
    )
    top_physical = parse_openroad(
        top_phys["log"].read_text(), float(config["clock_period_ns"])
    )
    top_checks = {
        "synthesis": top_synth_rc == 0
        and int(top_synthesis["cell_count"] or 0) > 0,
        "sixteen_physical_macros": macro_instances == 16,
        "place_route": top_physical_rc == 0 and output_check(top_phys),
        "drc_clean": top_physical["drc_violations"] == 0,
        "timing": top_physical["fmax_ghz"] is not None,
        "vcd_power": int(top_physical["annotated_pin_activities"] or 0) > 0
        and float(top_physical["total_power_w"] or 0.0) > 0,
        "compact_macro_abstraction": pe_abstraction_valid,
        "channel_legalization": channel_legalization_valid,
        "cts_buffer_legalization": cts_buffer_legalization_valid,
        "macro_track_alignment": macro_track_alignment_valid,
        "route_connectivity": route_connectivity["all_pins_routed"],
        "global_route_congestion": global_route_congestion_valid,
        "global_route_tool_provenance": route_tool_valid,
    }
    if not all(top_checks.values()):
        print(
            json.dumps(
                {
                    "stage": "hierarchical_top_physical",
                    "checks": top_checks,
                    "cts_buffer_legalization": cts_buffer_legalization,
                    "global_route_metrics": global_route_metrics,
                    "route_connectivity": route_connectivity,
                },
                indent=2,
            )
        )
        return 1

    flat_stats = output / "mlx-array-4x4-synthesis.stats"
    flat_synthesis = parse_synthesis(flat_stats.read_text())
    component_synthesis = {
        name: submacro_chain[name]["synthesis"] for name in required_submacros
    }
    top_shell_cells = int(top_synthesis["cell_count"] or 0) - 16
    pe_shell_cells = int(component_synthesis["pe_top"]["cell_count"] or 0) - 33
    recursive_cell_count = top_shell_cells + 16 * (
        pe_shell_cells
        + int(component_synthesis["register_file"]["cell_count"] or 0)
        + 8 * int(component_synthesis["full_lane"]["cell_count"] or 0)
        + 24 * int(component_synthesis["reduced_lane"]["cell_count"] or 0)
    )
    recursive_cell_area = float(top_synthesis["cell_area_um2"] or 0.0) + 16 * (
        float(component_synthesis["pe_top"]["cell_area_um2"] or 0.0)
        + float(component_synthesis["register_file"]["cell_area_um2"] or 0.0)
        + 8 * float(component_synthesis["full_lane"]["cell_area_um2"] or 0.0)
        + 24 * float(component_synthesis["reduced_lane"]["cell_area_um2"] or 0.0)
    )
    synthesis = {
        "cell_count": recursive_cell_count,
        "cell_area_um2": recursive_cell_area,
        "method": "recursive hierarchical Yosys/ABC mapped standard-cell sum",
        "flat_crosscheck": flat_synthesis,
    }
    integrated_delay = max(
        float(pe_physical["critical_path_delay_ns"]),
        float(top_physical["critical_path_delay_ns"]),
    )
    power_names = (
        "internal_power_w",
        "switching_power_w",
        "leakage_power_w",
        "total_power_w",
    )
    component_power = {
        name: submacro_chain[name]["workload_power"] for name in required_submacros
    }
    per_pe_power = {
        metric: (
            float(component_power["pe_top"][metric] or 0.0)
            + float(component_power["register_file"][metric] or 0.0)
            + 8.0 * float(component_power["full_lane"][metric] or 0.0)
            + 24.0 * float(component_power["reduced_lane"][metric] or 0.0)
        )
        for metric in power_names
    }
    per_pe_annotated_pins = (
        int(component_power["pe_top"]["annotated_pin_activities"] or 0)
        + int(component_power["register_file"]["annotated_pin_activities"] or 0)
        + 8 * int(component_power["full_lane"]["annotated_pin_activities"] or 0)
        + 24 * int(component_power["reduced_lane"]["annotated_pin_activities"] or 0)
    )
    combined_power = {
        metric: float(top_physical[metric] or 0.0) + 16.0 * per_pe_power[metric]
        for metric in power_names
    }
    power_hierarchy = {
        "formula": "top_shell + 16 * (pe_fu_shell + rf + 8 * full_lane + 24 * reduced_lane)",
        "top_shell": {name: top_physical[name] for name in power_names},
        "per_pe": per_pe_power,
        "representative_component_reports": component_power,
    }
    die_area = float(top_physical["die_width_um"] or 0.0) * float(
        top_physical["die_height_um"] or 0.0
    )
    core_area = float(top_physical["core_width_um"] or 0.0) * float(
        top_physical["core_height_um"] or 0.0
    )
    physical = {
        **top_physical,
        **combined_power,
        "die_area_um2": die_area,
        "core_area_um2": core_area,
        "critical_path_delay_ns": integrated_delay,
        "fmax_ghz": 1.0 / integrated_delay,
        "worst_slack_ns_at_1ghz": 1.0 - integrated_delay,
        "drc_violations": int(pe_physical["drc_violations"])
        + int(top_physical["drc_violations"]),
        "annotated_pin_activities": int(top_physical["annotated_pin_activities"] or 0)
        + 16 * per_pe_annotated_pins,
        "pe_macro": pe_physical,
        "hierarchical_top": top_physical,
        "power_hierarchy": power_hierarchy,
        "power_aggregation": "recursive_postroute_transformer_vcd_hierarchy",
    }
    checks = {
        "real_4x4_top": macro_instances == 16,
        "synthesis": all((pe_checks["synthesis"], top_checks["synthesis"]))
        and recursive_cell_count > 0
        and recursive_cell_area > 0
        and int(flat_synthesis["cell_count"] or 0) > 0,
        "place_route": pe_checks["place_route"] and top_checks["place_route"],
        "drc_clean": pe_checks["drc_clean"] and top_checks["drc_clean"],
        "timing": pe_checks["timing"] and top_checks["timing"],
        "vcd_power": top_checks["vcd_power"]
        and all(
            int(item["annotated_pin_activities"] or 0) > 0
            and all(
                item[name] is not None and math.isfinite(float(item[name]))
                for name in power_names
            )
            for item in component_power.values()
        )
        and all(value > 0 and math.isfinite(value) for value in combined_power.values()),
        "raw_unfitted": config["calibration"]
        == {"applied": False, "coefficients": None},
        "hierarchical_integrated": macro_instances == 16
        and pe_physical_rc == 0
        and top_physical_rc == 0,
        "recursive_submacro_evidence": set(submacro_chain) == required_submacros
        and all(all(item["checks"].values()) for item in submacro_chain.values()),
        "compact_macro_abstraction": pe_abstraction_valid,
        "channel_legalization": channel_legalization_valid,
        "cts_buffer_legalization": cts_buffer_legalization_valid,
        "macro_track_alignment": macro_track_alignment_valid,
        "route_connectivity": route_connectivity["all_pins_routed"],
        "global_route_congestion": global_route_congestion_valid,
        "global_route_tool_provenance": route_tool_valid,
    }

    global_route_iteration_reports = [
        {
            "completed_iteration": file_suffix - 1,
            "file_suffix": file_suffix,
            "report": artifact(path),
        }
        for file_suffix, path in congestion_iteration_reports(top_congestion_report)
    ]

    files: dict[str, Any] = {
        "config": artifact(config_path),
        "liberty": artifact(liberty),
        "tech_lef": artifact(Path(technology["tech_lef"])),
        "macro_lef": artifact(Path(technology["macro_lef"])),
        "rcx_rules": artifact(Path(technology["rcx_rules"])),
        "source_vcd": artifact(source_vcd),
        "scoped_vcd": artifact(top_vcd),
        "pe_vcd": artifact(pe_vcd),
        "pe_netlist": artifact(pe_netlist),
        "pe_abstract_lef": artifact(pe_lef),
        "pe_integration_abstract_lef": artifact(pe_integration_lef),
        "top_channel_legalization_log": artifact(top_channel_log),
        "top_channel_rows_checkpoint": artifact(top_rows_checkpoint),
        "top_channel_seed_checkpoint": artifact(top_seed_checkpoint),
        "top_channel_precheck_checkpoint": artifact(top_precheck_checkpoint),
        "top_channel_legalization_checkpoint": artifact(top_legal_checkpoint),
        "top_cts_log": artifact(top_cts_log),
        "top_cts_seed_checkpoint": artifact(top_cts_seed_checkpoint),
        "top_cts_checkpoint": artifact(top_cts_checkpoint),
        "top_global_route_log": artifact(top_route_log),
        "top_detailed_route_log": artifact(top_droute_resume_log),
        "global_route_openroad": global_route_tool["binary"],
        "global_route_patch": global_route_tool["patch"],
        "global_route_archive": global_route_tool["archive"],
        "detailed_route_openroad": detailed_route_tool,
        "pe_timing_liberty": artifact(pe_lib),
        "top_netlist": artifact(top_netlist),
        "synthesis_stats": artifact(flat_stats),
        "pe_synthesis_stats": artifact(pe_stats),
        "top_synthesis_stats": artifact(top_stats),
        "openroad_log": artifact(top_phys["log"]),
        "pe_openroad_log": artifact(pe_phys["log"]),
        "submacro_manifest": artifact(submacro_manifest_path),
        "flat_physical_attempt": {
            name: artifact(PROJECT_ROOT / path)
            for name, path in config["flat_physical_attempt"]["evidence"].items()
        },
        "submacro_power_logs": {
            name: artifact(
                (pe_root if name == "pe_top" else submacro_root / name)
                / (
                    "mlx_pe_top-workload-power.log"
                    if name == "pe_top"
                    else f"{ {'full_lane': 'mlx_fp16_alu_lane', 'reduced_lane': 'mlx_fp16_reduced_lane', 'register_file': 'mlx_register_file', 'functional_unit': 'mlx_fu'}[name] }-workload-power.log"
                )
            )
            for name in required_submacros
        },
        "rtl": {
            item: artifact(PROJECT_ROOT / item)
            for item in [*PE_RTL, "rtl/mlx/mlx_array_4x4.sv"]
        },
    }
    if global_route_iteration_reports:
        files["top_global_route_congestion_iteration_reports"] = (
            global_route_iteration_reports
        )
    if top_congestion_report.is_file():
        files["top_global_route_congestion_report"] = artifact(
            top_congestion_report
        )
    for paths in (pe_phys, top_phys):
        for name, path in paths.items():
            if path.is_file():
                files[f"{path.parent.name}_{name}"] = artifact(path)

    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": "raw_unfitted_hierarchical_integrated_nangate45_4x4",
        "paper_performance_targets_consumed": False,
        "calibration": config["calibration"],
        "tools": {
            "yosys": subprocess.run(
                ["yosys", "-V"], capture_output=True, text=True, check=False
            ).stdout.strip(),
            "openroad": subprocess.run(
                ["openroad", "-version"], capture_output=True, text=True, check=False
            ).stdout.strip(),
            "openroad_global_route": global_route_tool,
        },
        "activity": {
            **config["activity"],
            "top_extraction": top_vcd_extract,
            "pe0_extraction": pe_vcd_extract,
        },
        "flat_synthesis": flat_synthesis,
        "recursive_synthesis": synthesis,
        "pe_macro": {
            "synthesis": pe_synthesis,
            "physical": pe_physical,
            "checks": pe_checks,
        },
        "submacro_chain": submacro_chain,
        "pe_integration_abstraction": pe_abstraction,
        "channel_legalization": channel_legalization,
        "cts_buffer_legalization": cts_buffer_legalization,
        "macro_track_contract": macro_track_contract,
        "route_connectivity": route_connectivity,
        "global_route_metrics": global_route_metrics,
        "global_route_iteration_reports": global_route_iteration_reports,
        "route_contract": route_plan,
        "hierarchical_top": {
            "synthesis": top_synthesis,
            "physical": top_physical,
            "macro_instances": macro_instances,
            "integration_abstraction": pe_abstraction,
            "channel_legalization": channel_legalization,
            "cts_buffer_legalization": cts_buffer_legalization,
            "macro_track_contract": macro_track_contract,
            "route_connectivity": route_connectivity,
            "global_route_metrics": global_route_metrics,
            "global_route_iteration_reports": global_route_iteration_reports,
            "route_contract": route_plan,
            "checks": top_checks,
        },
        "physical": physical,
        "checks": checks,
        "global_route_tool": global_route_tool,
        "files": files,
    }
    manifest_path = PROJECT_ROOT / config["manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "status": "supported" if all(checks.values()) else "rejected",
        "claim": "raw, unfitted hierarchical integrated PPA for the real 4x4 MLX array",
        "implementation": "one routed direct-lane/RF PE hard macro instantiated and routed sixteen times in the integrated 4x4 top",
        "exclusions": ["RISC-V host", "CPU caches", "DMA controller", "SPM storage", "DRAM/PHY"],
        "sources": {
            "area": "hierarchical OpenROAD integrated top database plus recursive and flat Yosys cross-checks",
            "timing": "worst of PE-macro and hierarchical-top post-route OpenROAD/OpenSTA",
            "power": "recursive representative-PE0 post-route Transformer VCD aggregation over top, combined PE/FU shell, RF, and lane macros",
            "calibration": "none",
        },
        "synthesis": synthesis,
        "pe_macro": {"synthesis": pe_synthesis, "physical": pe_physical},
        "submacro_chain": submacro_chain,
        "hierarchical_top": {
            "synthesis": top_synthesis,
            "physical": top_physical,
            "macro_instances": macro_instances,
            "integration_abstraction": pe_abstraction,
            "channel_legalization": channel_legalization,
            "cts_buffer_legalization": cts_buffer_legalization,
            "macro_track_contract": macro_track_contract,
            "route_connectivity": route_connectivity,
            "global_route_metrics": global_route_metrics,
            "global_route_iteration_reports": global_route_iteration_reports,
            "route_contract": route_plan,
        },
        "physical": physical,
        "checks": checks,
        "global_route_tool": global_route_tool,
        "route_contract": route_plan,
        "manifest": artifact(manifest_path),
    }
    result_path = PROJECT_ROOT / config["result"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "checks": checks, "physical": physical}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
