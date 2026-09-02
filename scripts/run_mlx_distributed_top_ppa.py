#!/usr/bin/env python3
"""Sign off the promoted distributed 4x4 MLX top and emit H206/run211."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.run_mlx_array_ppa import parse_openroad, parse_synthesis, run_to_log
from scripts.run_mlx_hierarchical_ppa import (
    aggregate_hierarchical_timing,
    artifact,
    build_compact_macro_lef,
    congestion_iteration_reports,
    parse_channel_legalization,
    parse_congestion_marker_report,
    parse_cts_buffer_legalization,
    parse_global_route_metrics,
    parse_route_connectivity,
    yosys_run,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/system/mlx_array_ppa_v1.yaml"
STAGES = (
    "inspect",
    "synthesis",
    "gpl",
    "legal",
    "cts",
    "grt",
    "droute",
    "repair",
    "retry",
    "drc-audit",
    "local-repair",
    "finalize",
    "all",
)
POWER_METRICS = (
    "internal_power_w",
    "switching_power_w",
    "leakage_power_w",
    "total_power_w",
)
RECURSIVE_COMPONENTS = {
    "full_lane",
    "reduced_lane",
    "register_file",
    "functional_unit",
    "pe_top",
}


def present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def distributed_paths(output: Path) -> dict[str, Path]:
    root = output / "distributed_4x4"
    stem = root / "mlx-array-4x4-distributed-u70"
    routed = root / "mlx-array-4x4-distributed-u70-iter5"
    clean_retry = root / "mlx-array-4x4-distributed-u70-iter5-clean-retry1"
    failed_local_repair2 = root / (
        "mlx-array-4x4-distributed-u70-iter5-clean-retry1-local-repair2"
    )
    failed_local_repair3 = root / (
        "mlx-array-4x4-distributed-u70-iter5-clean-retry1-local-repair3"
    )
    local_repair4 = root / (
        "mlx-array-4x4-distributed-u70-iter5-clean-retry1-local-repair4"
    )
    local_repair = root / (
        "mlx-array-4x4-distributed-u70-iter5-clean-retry1-local-repair5"
    )
    tile_root = output / "tile_macro"
    tile_droute = tile_root / "mlx-array-pe-tile-v2-tight-iter50-droute-probe"
    paths = {
        "root": root,
        "netlist": root / "mlx-array-4x4-distributed-mapped.v",
        "stats": root / "mlx-array-4x4-distributed-synthesis.stats",
        "synthesis_log": root / "mlx-array-4x4-distributed-synthesis.log",
        "gpl": stem.with_name(f"{stem.name}-global-placement.odb"),
        "gpl_log": stem.with_name(f"{stem.name}-gpl.log"),
        "rows": stem.with_name(f"{stem.name}-rows.odb"),
        "seed": stem.with_name(f"{stem.name}-seed.odb"),
        "precheck": stem.with_name(f"{stem.name}-precheck.odb"),
        "legal": stem.with_name(f"{stem.name}-channel-legal.odb"),
        "legal_log": stem.with_name(f"{stem.name}-channel-legalize.log"),
        "cts_seed": stem.with_name(f"{stem.name}-cts-seed.odb"),
        "cts": stem.with_name(f"{stem.name}-post-cts.odb"),
        "cts_log": stem.with_name(f"{stem.name}-cts.log"),
        "guide": routed.with_name(f"{routed.name}-routed.guide"),
        "grt": routed.with_name(f"{routed.name}-global-route.odb"),
        "grt_log": routed.with_name(f"{routed.name}-route.log"),
        "congestion": routed.with_name(f"{routed.name}-congestion.rpt"),
        "droute_log": routed.with_name(f"{routed.name}-droute.log"),
        "drc": routed.with_name(f"{routed.name}-routed.drc"),
        "def": routed.with_name(f"{routed.name}-routed.def"),
        "odb": routed.with_name(f"{routed.name}-routed.odb"),
        "spef": routed.with_name(f"{routed.name}-routed.spef"),
        "base_droute_log": routed.with_name(f"{routed.name}-droute.log"),
        "base_drc": routed.with_name(f"{routed.name}-routed.drc"),
        "base_def": routed.with_name(f"{routed.name}-routed.def"),
        "base_odb": routed.with_name(f"{routed.name}-routed.odb"),
        "base_spef": routed.with_name(f"{routed.name}-routed.spef"),
        "repair_log": routed.with_name(f"{routed.name}-repair1-droute.log"),
        "repair_drc": routed.with_name(f"{routed.name}-repair1-routed.drc"),
        "repair_def": routed.with_name(f"{routed.name}-repair1-routed.def"),
        "repair_odb": routed.with_name(f"{routed.name}-repair1-routed.odb"),
        "repair_spef": routed.with_name(f"{routed.name}-repair1-routed.spef"),
        "clean_retry_log": clean_retry.with_name(
            f"{clean_retry.name}-droute.log"
        ),
        "clean_retry_drc": clean_retry.with_name(
            f"{clean_retry.name}-routed.drc"
        ),
        "clean_retry_def": clean_retry.with_name(
            f"{clean_retry.name}-routed.def"
        ),
        "clean_retry_odb": clean_retry.with_name(
            f"{clean_retry.name}-routed.odb"
        ),
        "clean_retry_spef": clean_retry.with_name(
            f"{clean_retry.name}-routed.spef"
        ),
        "local_repair_log": local_repair.with_name(
            f"{local_repair.name}-droute.log"
        ),
        "failed_local_repair2_log": failed_local_repair2.with_name(
            f"{failed_local_repair2.name}-droute.log"
        ),
        "failed_local_repair3_log": failed_local_repair3.with_name(
            f"{failed_local_repair3.name}-droute.log"
        ),
        "local_repair4_log": local_repair4.with_name(
            f"{local_repair4.name}-droute.log"
        ),
        "local_repair4_drc": local_repair4.with_name(
            f"{local_repair4.name}-routed.drc"
        ),
        "local_repair4_def": local_repair4.with_name(
            f"{local_repair4.name}-routed.def"
        ),
        "local_repair4_odb": local_repair4.with_name(
            f"{local_repair4.name}-routed.odb"
        ),
        "local_repair4_spef": local_repair4.with_name(
            f"{local_repair4.name}-routed.spef"
        ),
        "residual_drc_audit_log": local_repair4.with_name(
            f"{local_repair4.name}-geometry-audit.log"
        ),
        "local_repair_drc": local_repair.with_name(
            f"{local_repair.name}-routed.drc"
        ),
        "local_repair_def": local_repair.with_name(
            f"{local_repair.name}-routed.def"
        ),
        "local_repair_odb": local_repair.with_name(
            f"{local_repair.name}-routed.odb"
        ),
        "local_repair_spef": local_repair.with_name(
            f"{local_repair.name}-routed.spef"
        ),
        "vcd": output / "transformer-block-distributed-top-ports.vcd",
        "tile_summary": tile_root / "mlx-array-pe-tile-candidate.json",
        "tile_lef": tile_droute.with_name(f"{tile_droute.name}.abstract.lef"),
        "tile_integration_lef": tile_droute.with_name(
            f"{tile_droute.name}.integration.lef"
        ),
        "tile_lib": tile_droute.with_name(f"{tile_droute.name}.lib"),
        "submacro_manifest": output
        / "pe_submacros"
        / "submacro-build-manifest.json",
        "preview": root / "mlx-array-4x4-distributed-candidate.json",
    }
    repair_complete = (
        paths["repair_log"].is_file()
        and "MLX_ARRAY_DROUTE_COMPLETE odb=" in paths["repair_log"].read_text()
        and all(
            present(paths[name])
            for name in ("repair_drc", "repair_def", "repair_odb", "repair_spef")
        )
    )
    clean_retry_complete = (
        paths["clean_retry_log"].is_file()
        and "MLX_ARRAY_DROUTE_COMPLETE odb="
        in paths["clean_retry_log"].read_text()
        and all(
            present(paths[name])
            for name in (
                "clean_retry_drc",
                "clean_retry_def",
                "clean_retry_odb",
                "clean_retry_spef",
            )
        )
    )
    local_repair_complete = (
        paths["local_repair_log"].is_file()
        and "MLX_ARRAY_DROUTE_COMPLETE odb="
        in paths["local_repair_log"].read_text()
        and all(
            present(paths[name])
            for name in (
                "local_repair_drc",
                "local_repair_def",
                "local_repair_odb",
                "local_repair_spef",
            )
        )
    )
    local_repair4_complete = (
        paths["local_repair4_log"].is_file()
        and "MLX_ARRAY_DROUTE_COMPLETE odb="
        in paths["local_repair4_log"].read_text()
        and all(
            present(paths[name])
            for name in (
                "local_repair4_drc",
                "local_repair4_def",
                "local_repair4_odb",
                "local_repair4_spef",
            )
        )
    )
    if local_repair_complete:
        for target, source in (
            ("droute_log", "local_repair_log"),
            ("drc", "local_repair_drc"),
            ("def", "local_repair_def"),
            ("odb", "local_repair_odb"),
            ("spef", "local_repair_spef"),
        ):
            paths[target] = paths[source]
    elif local_repair4_complete:
        for target, source in (
            ("droute_log", "local_repair4_log"),
            ("drc", "local_repair4_drc"),
            ("def", "local_repair4_def"),
            ("odb", "local_repair4_odb"),
            ("spef", "local_repair4_spef"),
        ):
            paths[target] = paths[source]
    elif clean_retry_complete:
        for target, source in (
            ("droute_log", "clean_retry_log"),
            ("drc", "clean_retry_drc"),
            ("def", "clean_retry_def"),
            ("odb", "clean_retry_odb"),
            ("spef", "clean_retry_spef"),
        ):
            paths[target] = paths[source]
    elif repair_complete:
        for target, source in (
            ("droute_log", "repair_log"),
            ("drc", "repair_drc"),
            ("def", "repair_def"),
            ("odb", "repair_odb"),
            ("spef", "repair_spef"),
        ):
            paths[target] = paths[source]
    return paths


def detailed_route_outputs_present(paths: dict[str, Path]) -> bool:
    return paths["drc"].is_file() and all(
        present(paths[name]) for name in ("def", "odb", "spef", "droute_log")
    )


def parse_detailed_route_progress(text: str) -> dict[str, Any]:
    started_iterations = [
        int(value)
        for value in re.findall(
            r"\[INFO DRT-0195\] Start (\d+)(?:th|st|nd|rd) optimization iteration\.",
            text,
        )
    ]
    violation_curve = [
        int(value)
        for value in re.findall(
            r"\[INFO DRT-0199\]\s+Number of violations = (\d+)\.", text
        )
    ]
    completed_iterations = started_iterations[: len(violation_curve)]
    return {
        "started_iteration_numbers": started_iterations,
        "completed_iteration_numbers": completed_iterations,
        "last_completed_optimization_iteration": (
            completed_iterations[-1] if completed_iterations else None
        ),
        "reported_violation_counts": len(violation_curve),
        "violation_curve": violation_curve,
        "final_violations": violation_curve[-1] if violation_curve else None,
        "zero_drc_reached": bool(violation_curve) and violation_curve[-1] == 0,
    }


def synthesize(config: dict[str, Any], paths: dict[str, Path]) -> int:
    liberty = Path(config["technology"]["liberty"])
    commands = [
        (
            "read_verilog -sv -lib "
            f"{PROJECT_ROOT / 'rtl/ppa/mlx_array_pe_tile_blackbox.sv'}"
        ),
        (
            "read_verilog -sv -DMLX_PPA_MACRO -DSYNTHESIS "
            f"{PROJECT_ROOT / 'rtl/mlx/mlx_array_4x4_distributed.sv'}"
        ),
        "hierarchy -check -top mlx_array_4x4_distributed",
        "synth -top mlx_array_4x4_distributed",
        f"dfflibmap -liberty {liberty}",
        f"abc -fast -liberty {liberty}",
        "hilomap -singleton -hicell LOGIC1_X1 Z -locell LOGIC0_X1 Z",
        "clean",
        f"tee -o {paths['stats']} stat -liberty {liberty}",
        f"write_verilog -noattr -noexpr -nodec {paths['netlist']}",
    ]
    return yosys_run(commands, paths["synthesis_log"])


def top_environment(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, str]:
    technology = config["technology"]
    candidate = config["hierarchical_distributed_tile_candidate"]
    floorplan = candidate["distributed_top_floorplan_candidate"]
    contract = candidate["distributed_top_signoff_contract"]
    macro_track = config["hierarchical_top_placement"][
        "macro_origin_track_alignment"
    ]
    env = os.environ.copy()
    env.update(
        {
            "MALLOC_ARENA_MAX": "2",
            "PPA_THREADS": str(config["threads"]),
            "PPA_POST_GPL_THREADS": "1",
            "PPA_TECH_LEF": technology["tech_lef"],
            "PPA_MACRO_LEF": technology["macro_lef"],
            "PPA_PE_LEF": str(paths["tile_integration_lef"]),
            "PPA_LIBERTY": technology["liberty"],
            "PPA_PE_LIBERTY": str(paths["tile_lib"]),
            "PPA_NETLIST": str(paths["netlist"]),
            "PPA_TOP": "mlx_array_4x4_distributed",
            "PPA_MACRO_INSTANCE_KIND": "tile",
            "PPA_CLOCK_PERIOD_NS": str(config["clock_period_ns"]),
            "PPA_UTILIZATION": str(floorplan["utilization_percent"]),
            "PPA_DENSITY": str(floorplan["placement_density"]),
            "PPA_ASPECT_RATIO": "1.0",
            "PPA_SKIP_INITIAL_PLACE": "0",
            "PPA_BIN_GRID_COUNT": str(floorplan["bin_grid_count"]),
            "PPA_OVERFLOW_TARGET": str(floorplan["overflow_target"]),
            "PPA_INIT_DENSITY_PENALTY": "0.001",
            "PPA_MIN_PHI_COEF": "0.95",
            "PPA_MAX_PHI_COEF": "1.20",
            "PPA_REPAIR_DESIGN": "0",
            "PPA_REPAIR_MAX_UTILIZATION": "80",
            "PPA_TAPCELL_DISTANCE_UM": "1000",
            "PPA_TAPCELL_TCL": technology["tapcell_tcl"],
            "TAP_CELL_NAME": "TAPCELL_X1",
            "PPA_DPL_ROW_LIMIT": str(contract["minimum_physical_rows"]),
            "PPA_CHANNEL_TARGET_UTILIZATION": "0.75",
            "PPA_PE_MACRO_MASTER": "mlx_array_pe_tile",
            "PPA_MACRO_INSTANCE_COUNT": str(contract["macro_instances"]),
            "PPA_MACRO_ORIGIN_GRID_DBU": str(macro_track["grid_dbu"]),
            "PPA_GPL_ODB": str(paths["gpl"]),
            "PPA_ROWS_ODB": str(paths["rows"]),
            "PPA_SEED_ODB": str(paths["seed"]),
            "PPA_PRECHECK_ODB": str(paths["precheck"]),
            "PPA_LEGAL_ODB": str(paths["legal"]),
            "PPA_CTS_SEED_ODB": str(paths["cts_seed"]),
            "PPA_CTS_ODB": str(paths["cts"]),
            "PPA_GUIDE": str(paths["guide"]),
            "PPA_GRT_ODB": str(paths["grt"]),
            "PPA_GRT_CONGESTION_ITERATIONS": str(
                contract["congestion_iterations"]
            ),
            "PPA_GRT_ALLOW_CONGESTION": "1",
            "PPA_GRT_VERBOSE": "1",
            "PPA_GRT_CONGESTION_REPORT_FILE": str(paths["congestion"]),
            "PPA_GRT_CONGESTION_REPORT_ITER_STEP": "1",
            "PPA_SIGNAL_ROUTING_LAYERS": contract["routing_layers"]["signal"],
            "PPA_CLOCK_ROUTING_LAYERS": contract["routing_layers"]["clock"],
            "PPA_LAYER_CAPACITY_ADJUSTMENTS": "",
            "PPA_MACRO_EXTENSION_GCELLS": "0",
            "PPA_CRITICAL_NETS_PERCENTAGE": "0",
            "PPA_DROUTE_END_ITER": str(contract["droute_end_iter"]),
            "PPA_DRC": str(paths["drc"]),
            "PPA_RCX_RULES": technology["rcx_rules"],
            "PPA_DEF": str(paths["def"]),
            "PPA_ODB": str(paths["odb"]),
            "PPA_SPEF": str(paths["spef"]),
            "PPA_VCD": str(paths["vcd"]),
            "PPA_VCD_SCOPE": config["activity"]["promoted_scope"],
            "PPA_RESUME_GPL": "0",
            "PPA_RESUME_ROWS": "0",
            "PPA_RESUME_CTS": "0",
            "PPA_STOP_AFTER_GPL": "1",
            "PPA_STOP_AFTER_CTS": "1",
            "PPA_STOP_AFTER_GRT": "1",
        }
    )
    return env


def run_physical_stage(
    stage: str,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> int:
    system_openroad = Path(
        config["toolchain"]["detailed_route_and_signoff_openroad"]["binary"]
    )
    global_openroad = PROJECT_ROOT / config["toolchain"]["global_route_openroad"][
        "binary"
    ]
    env = top_environment(config, paths)
    if stage == "gpl":
        return run_to_log(
            [
                str(system_openroad),
                "-no_init",
                "-exit",
                str(PROJECT_ROOT / "rtl/ppa/openroad_hierarchical_array_flow.tcl"),
            ],
            paths["gpl_log"],
            env,
        )
    if stage == "legal":
        return run_to_log(
            [
                str(system_openroad),
                "-no_init",
                "-exit",
                str(
                    PROJECT_ROOT
                    / "rtl/ppa/openroad_hierarchical_array_channel_legalize.tcl"
                ),
            ],
            paths["legal_log"],
            env,
        )
    if stage == "cts":
        env["PPA_STOP_AFTER_GRT"] = "1"
        return run_to_log(
            [
                str(system_openroad),
                "-no_init",
                "-exit",
                str(
                    PROJECT_ROOT
                    / "rtl/ppa/openroad_hierarchical_array_post_legal_flow.tcl"
                ),
            ],
            paths["cts_log"],
            env,
        )
    if stage == "grt":
        env["PPA_THREADS"] = str(
            config["hierarchical_distributed_tile_candidate"][
                "distributed_top_signoff_contract"
            ]["global_route_threads"]
        )
        env["PPA_RESUME_CTS"] = "1"
        env["PPA_STOP_AFTER_CTS"] = "0"
        return run_to_log(
            [
                str(global_openroad),
                "-no_init",
                "-exit",
                str(
                    PROJECT_ROOT
                    / "rtl/ppa/openroad_hierarchical_array_post_legal_flow.tcl"
                ),
            ],
            paths["grt_log"],
            env,
        )
    if stage == "droute":
        return run_droute(config, paths)
    if stage == "repair":
        return run_repair(config, paths)
    if stage == "retry":
        return run_clean_retry(config, paths)
    if stage == "drc-audit":
        return run_residual_drc_audit(config, paths)
    if stage == "local-repair":
        return run_local_repair(config, paths)
    raise ValueError(f"unsupported distributed-top stage {stage}")


def global_route_ready(paths: dict[str, Path]) -> tuple[bool, dict[str, Any]]:
    text = paths["grt_log"].read_text() if present(paths["grt_log"]) else ""
    connectivity = parse_route_connectivity(text, "")
    ready = (
        present(paths["grt"])
        and present(paths["guide"])
        and connectivity["global_route_completed"]
        and connectivity["global_missing_pin_routes"] == 0
        and not connectivity["global_missing_warning_limit_reached"]
    )
    return ready, connectivity


def droute_environment(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, str]:
    technology = config["technology"]
    contract = config["hierarchical_distributed_tile_candidate"][
        "distributed_top_signoff_contract"
    ]
    env = os.environ.copy()
    env.update(
        {
            "MALLOC_ARENA_MAX": "2",
            "PPA_THREADS": str(contract["detailed_route_threads"]),
            "PPA_GRT_ODB": str(paths["grt"]),
            "PPA_LIBERTY": technology["liberty"],
            "PPA_PE_LIBERTY": str(paths["tile_lib"]),
            "PPA_CLOCK_PERIOD_NS": str(config["clock_period_ns"]),
            "PPA_DROUTE_END_ITER": str(contract["droute_end_iter"]),
            "PPA_DRC": str(paths["drc"]),
            "PPA_RCX_RULES": technology["rcx_rules"],
            "PPA_DEF": str(paths["def"]),
            "PPA_ODB": str(paths["odb"]),
            "PPA_SPEF": str(paths["spef"]),
            "PPA_VCD": str(paths["vcd"]),
            "PPA_VCD_SCOPE": config["activity"]["promoted_scope"],
        }
    )
    return env


def run_droute(config: dict[str, Any], paths: dict[str, Path]) -> int:
    ready, connectivity = global_route_ready(paths)
    if not ready:
        raise RuntimeError(
            "refusing distributed-top DRT before GRT checkpoint/guide completion "
            f"and zero missing pin routes: {connectivity}"
        )
    for name in ("tile_lib", "vcd"):
        if not present(paths[name]):
            raise FileNotFoundError(paths[name])
    openroad = Path(
        config["toolchain"]["detailed_route_and_signoff_openroad"]["binary"]
    )
    return run_to_log(
        [
            str(openroad),
            "-no_init",
            "-exit",
            str(
                PROJECT_ROOT
                / "rtl/ppa/openroad_hierarchical_array_droute_resume.tcl"
            ),
        ],
        paths["droute_log"],
        droute_environment(config, paths),
    )


def run_repair(config: dict[str, Any], paths: dict[str, Path]) -> int:
    if not present(paths["base_odb"]):
        raise FileNotFoundError(paths["base_odb"])
    technology = config["technology"]
    contract = config["hierarchical_distributed_tile_candidate"][
        "distributed_top_signoff_contract"
    ]
    env = os.environ.copy()
    env.update(
        {
            "MALLOC_ARENA_MAX": "2",
            "PPA_THREADS": str(contract["detailed_route_threads"]),
            "PPA_GRT_ODB": str(paths["base_odb"]),
            "PPA_LIBERTY": technology["liberty"],
            "PPA_PE_LIBERTY": str(paths["tile_lib"]),
            "PPA_CLOCK_PERIOD_NS": str(config["clock_period_ns"]),
            "PPA_DROUTE_END_ITER": str(contract["repair_droute_end_iter"]),
            "PPA_DRC": str(paths["repair_drc"]),
            "PPA_RCX_RULES": technology["rcx_rules"],
            "PPA_DEF": str(paths["repair_def"]),
            "PPA_ODB": str(paths["repair_odb"]),
            "PPA_SPEF": str(paths["repair_spef"]),
            "PPA_VCD": str(paths["vcd"]),
            "PPA_VCD_SCOPE": config["activity"]["promoted_scope"],
        }
    )
    openroad = Path(
        config["toolchain"]["detailed_route_and_signoff_openroad"]["binary"]
    )
    return run_to_log(
        [
            str(openroad),
            "-no_init",
            "-exit",
            str(
                PROJECT_ROOT
                / "rtl/ppa/openroad_hierarchical_array_droute_resume.tcl"
            ),
        ],
        paths["repair_log"],
        env,
    )


def run_clean_retry(config: dict[str, Any], paths: dict[str, Path]) -> int:
    """Rerun DRT from the clean GRT checkpoint with an extended iteration budget."""
    ready, connectivity = global_route_ready(paths)
    if not ready:
        raise RuntimeError(
            "refusing clean distributed-top DRT retry before GRT checkpoint/guide "
            f"completion and zero missing pin routes: {connectivity}"
        )
    for name in ("tile_lib", "vcd"):
        if not present(paths[name]):
            raise FileNotFoundError(paths[name])
    contract = config["hierarchical_distributed_tile_candidate"][
        "distributed_top_signoff_contract"
    ]
    env = droute_environment(config, paths)
    env.update(
        {
            "PPA_DROUTE_END_ITER": str(contract["clean_retry_droute_end_iter"]),
            "PPA_DRC": str(paths["clean_retry_drc"]),
            "PPA_DEF": str(paths["clean_retry_def"]),
            "PPA_ODB": str(paths["clean_retry_odb"]),
            "PPA_SPEF": str(paths["clean_retry_spef"]),
        }
    )
    openroad = Path(
        config["toolchain"]["detailed_route_and_signoff_openroad"]["binary"]
    )
    return run_to_log(
        [
            str(openroad),
            "-no_init",
            "-exit",
            str(
                PROJECT_ROOT
                / "rtl/ppa/openroad_hierarchical_array_droute_resume.tcl"
            ),
        ],
        paths["clean_retry_log"],
        env,
    )


def run_local_repair(config: dict[str, Any], paths: dict[str, Path]) -> int:
    """Run repair5 from repair4 using the controlled stubborn-tile DRT build."""
    if not present(paths["local_repair4_odb"]):
        raise FileNotFoundError(paths["local_repair4_odb"])
    for name in ("tile_lib", "vcd"):
        if not present(paths[name]):
            raise FileNotFoundError(paths[name])
    technology = config["technology"]
    contract = config["hierarchical_distributed_tile_candidate"][
        "distributed_top_signoff_contract"
    ]
    tool = config["toolchain"]["stubborn_repair_openroad"]
    openroad = PROJECT_ROOT / tool["binary"]
    if not present(openroad):
        raise FileNotFoundError(openroad)
    if artifact(openroad)["sha256"] != tool["binary_sha256"]:
        raise RuntimeError(f"local-repair OpenROAD hash mismatch: {openroad}")
    env = os.environ.copy()
    env.update(
        {
            "MALLOC_ARENA_MAX": "2",
            "PPA_THREADS": str(contract["detailed_route_threads"]),
            "PPA_GRT_ODB": str(paths["local_repair4_odb"]),
            "PPA_LIBERTY": technology["liberty"],
            "PPA_PE_LIBERTY": str(paths["tile_lib"]),
            "PPA_CLOCK_PERIOD_NS": str(config["clock_period_ns"]),
            "PPA_DROUTE_END_ITER": str(
                contract["local_repair_droute_end_iter"]
            ),
            "MLX_DRT_SKIP_REDUNDANT_INCREMENTAL": "1",
            "MLX_DRT_STUBBORN_THRESHOLD": str(
                contract["local_repair_stubborn_threshold"]
            ),
            "PPA_DRC": str(paths["local_repair_drc"]),
            "PPA_RCX_RULES": technology["rcx_rules"],
            "PPA_DEF": str(paths["local_repair_def"]),
            "PPA_ODB": str(paths["local_repair_odb"]),
            "PPA_SPEF": str(paths["local_repair_spef"]),
            "PPA_VCD": str(paths["vcd"]),
            "PPA_VCD_SCOPE": config["activity"]["promoted_scope"],
        }
    )
    return run_to_log(
        [
            str(openroad),
            "-no_init",
            "-exit",
            str(
                PROJECT_ROOT
                / "rtl/ppa/openroad_hierarchical_array_droute_resume.tcl"
            ),
        ],
        paths["local_repair_log"],
        env,
    )


def run_residual_drc_audit(
    config: dict[str, Any], paths: dict[str, Path]
) -> int:
    """Prove whether repair4 macro markers overlap the integration OBS view."""
    for name in ("local_repair4_odb", "local_repair4_drc", "tile_integration_lef"):
        if not present(paths[name]):
            raise FileNotFoundError(paths[name])
    env = os.environ.copy()
    env.update(
        {
            "PPA_AUDIT_ODB": str(paths["local_repair4_odb"]),
            "PPA_AUDIT_DRC": str(paths["local_repair4_drc"]),
            "PPA_AUDIT_MACRO_LEF": str(paths["tile_integration_lef"]),
        }
    )
    openroad = Path(
        config["toolchain"]["detailed_route_and_signoff_openroad"]["binary"]
    )
    return run_to_log(
        [
            str(openroad),
            "-no_init",
            "-no_splash",
            "-exit",
            str(
                PROJECT_ROOT
                / "rtl/ppa/openroad_hierarchical_array_drc_geometry_audit.tcl"
            ),
        ],
        paths["residual_drc_audit_log"],
        env,
    )


def recursive_synthesis(
    top: dict[str, Any], tile: dict[str, Any], submacros: dict[str, Any]
) -> dict[str, Any]:
    top_shell_cells = int(top["cell_count"] or 0) - 16
    tile_shell_cells = int(tile["cell_count"] or 0) - 1
    pe_shell_cells = int(submacros["pe_top"]["synthesis"]["cell_count"] or 0) - 33
    per_pe_cells = (
        pe_shell_cells
        + int(submacros["register_file"]["synthesis"]["cell_count"] or 0)
        + 8 * int(submacros["full_lane"]["synthesis"]["cell_count"] or 0)
        + 24 * int(submacros["reduced_lane"]["synthesis"]["cell_count"] or 0)
    )
    per_tile_cells = tile_shell_cells + per_pe_cells
    cell_count = top_shell_cells + 16 * per_tile_cells

    per_pe_area = (
        float(submacros["pe_top"]["synthesis"]["cell_area_um2"] or 0.0)
        + float(
            submacros["register_file"]["synthesis"]["cell_area_um2"] or 0.0
        )
        + 8
        * float(submacros["full_lane"]["synthesis"]["cell_area_um2"] or 0.0)
        + 24
        * float(
            submacros["reduced_lane"]["synthesis"]["cell_area_um2"] or 0.0
        )
    )
    per_tile_area = float(tile["cell_area_um2"] or 0.0) + per_pe_area
    cell_area = float(top["cell_area_um2"] or 0.0) + 16 * per_tile_area
    return {
        "cell_count": cell_count,
        "cell_area_um2": cell_area,
        "top_shell_cells": top_shell_cells,
        "tile_shell_cells_per_tile": tile_shell_cells,
        "recursive_pe_cells_per_tile": per_pe_cells,
        "recursive_cells_per_tile": per_tile_cells,
        "top_shell_area_um2": float(top["cell_area_um2"] or 0.0),
        "tile_shell_area_um2_per_tile": float(tile["cell_area_um2"] or 0.0),
        "recursive_pe_area_um2_per_tile": per_pe_area,
        "recursive_area_um2_per_tile": per_tile_area,
        "method": (
            "distributed_top_shell + 16 * (autonomous_tile_shell + recursive "
            "PE/RF/lane standard-cell hierarchy)"
        ),
    }


def recursive_power(
    top_physical: dict[str, Any],
    tile_physical: dict[str, Any],
    submacros: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any], int]:
    component_power = {
        name: submacros[name]["workload_power"] for name in RECURSIVE_COMPONENTS
    }
    per_pe = {
        metric: (
            float(component_power["pe_top"][metric] or 0.0)
            + float(component_power["register_file"][metric] or 0.0)
            + 8 * float(component_power["full_lane"][metric] or 0.0)
            + 24 * float(component_power["reduced_lane"][metric] or 0.0)
        )
        for metric in POWER_METRICS
    }
    per_tile = {
        metric: float(tile_physical[metric] or 0.0) + per_pe[metric]
        for metric in POWER_METRICS
    }
    combined = {
        metric: float(top_physical[metric] or 0.0) + 16 * per_tile[metric]
        for metric in POWER_METRICS
    }
    per_pe_pins = (
        int(component_power["pe_top"]["annotated_pin_activities"] or 0)
        + int(component_power["register_file"]["annotated_pin_activities"] or 0)
        + 8
        * int(component_power["full_lane"]["annotated_pin_activities"] or 0)
        + 24
        * int(component_power["reduced_lane"]["annotated_pin_activities"] or 0)
    )
    annotated_pins = int(top_physical["annotated_pin_activities"] or 0) + 16 * (
        int(tile_physical["annotated_pin_activities"] or 0) + per_pe_pins
    )
    hierarchy = {
        "formula": (
            "top_shell + 16 * (autonomous_tile_shell + pe_fu_shell + rf + "
            "8 * full_lane + 24 * reduced_lane)"
        ),
        "top_shell": {name: top_physical[name] for name in POWER_METRICS},
        "tile_shell_per_tile": {
            name: tile_physical[name] for name in POWER_METRICS
        },
        "recursive_pe_per_tile": per_pe,
        "combined_per_tile": per_tile,
        "representative_component_reports": component_power,
    }
    return combined, hierarchy, annotated_pins


def tool_provenance(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    tool = config["toolchain"]["global_route_openroad"]
    signoff = config["toolchain"]["detailed_route_and_signoff_openroad"]
    prior_local_repair = config["toolchain"]["local_repair_openroad"]
    local_repair = config["toolchain"]["stubborn_repair_openroad"]
    binary = PROJECT_ROOT / tool["binary"]
    patch = PROJECT_ROOT / tool["patch"]
    archive = PROJECT_ROOT / tool["archive"]
    detailed = Path(signoff["binary"])
    local_repair_binary = PROJECT_ROOT / local_repair["binary"]
    local_repair_patch = PROJECT_ROOT / local_repair["patch"]
    local_repair_archive = PROJECT_ROOT / local_repair["archive"]
    prior_local_repair_binary = PROJECT_ROOT / prior_local_repair["binary"]
    prior_local_repair_patch = PROJECT_ROOT / prior_local_repair["patch"]
    prior_local_repair_archive = PROJECT_ROOT / prior_local_repair["archive"]
    prior_local_repair_record = {
        "base_commit": prior_local_repair["base_commit"],
        "binary": artifact(prior_local_repair_binary),
        "patch": artifact(prior_local_repair_patch),
        "archive": artifact(prior_local_repair_archive),
        "version": subprocess.run(
            [str(prior_local_repair_binary), "-version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
    }
    local_repair_record = {
        "base_commit": local_repair["base_commit"],
        "binary": artifact(local_repair_binary),
        "patch": artifact(local_repair_patch),
        "archive": artifact(local_repair_archive),
        "version": subprocess.run(
            [str(local_repair_binary), "-version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
    }
    record = {
        "base_commit": tool["base_commit"],
        "grid_pitches_in_tile": int(tool["grid_pitches_in_tile"]),
        "max_2d_edge_usage_multiplier": int(tool["max_2d_edge_usage_multiplier"]),
        "binary": artifact(binary),
        "patch": artifact(patch),
        "archive": artifact(archive),
        "detailed_route_binary": artifact(detailed),
        "prior_local_repair_openroad": prior_local_repair_record,
        "local_repair_openroad": local_repair_record,
        "version": subprocess.run(
            [str(binary), "-version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
    }
    valid = (
        record["binary"]["sha256"] == tool["binary_sha256"]
        and record["patch"]["sha256"] == tool["patch_sha256"]
        and record["archive"]["sha256"] == tool["archive_sha256"]
        and record["detailed_route_binary"]["sha256"]
        == signoff["binary_sha256"]
        and local_repair_record["binary"]["sha256"]
        == local_repair["binary_sha256"]
        and local_repair_record["patch"]["sha256"]
        == local_repair["patch_sha256"]
        and local_repair_record["archive"]["sha256"]
        == local_repair["archive_sha256"]
        and local_repair["version"] in local_repair_record["version"]
        and prior_local_repair_record["binary"]["sha256"]
        == prior_local_repair["binary_sha256"]
        and prior_local_repair_record["patch"]["sha256"]
        == prior_local_repair["patch_sha256"]
        and prior_local_repair_record["archive"]["sha256"]
        == prior_local_repair["archive_sha256"]
        and prior_local_repair["version"]
        in prior_local_repair_record["version"]
        and tool["base_commit"] in record["version"]
        and record["grid_pitches_in_tile"] == 48
        and record["max_2d_edge_usage_multiplier"] == 101
    )
    return record, valid


def build_result(
    config_path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    include_abstraction: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = config["hierarchical_distributed_tile_candidate"]
    contract = candidate["distributed_top_signoff_contract"]
    top_synthesis = (
        parse_synthesis(paths["stats"].read_text()) if present(paths["stats"]) else {}
    )
    tile_summary = (
        json.loads(paths["tile_summary"].read_text())
        if present(paths["tile_summary"])
        else {}
    )
    submacros = (
        json.loads(paths["submacro_manifest"].read_text())
        if present(paths["submacro_manifest"])
        else {}
    )
    top_netlist_text = paths["netlist"].read_text() if present(paths["netlist"]) else ""
    macro_instances = len(
        re.findall(
            r"(?m)^\s*mlx_array_pe_tile\s+(?:\\\S+|\S+)\s*\(",
            top_netlist_text,
        )
    )
    grt_text = paths["grt_log"].read_text() if present(paths["grt_log"]) else ""
    droute_text = (
        paths["droute_log"].read_text() if present(paths["droute_log"]) else ""
    )
    global_route = parse_global_route_metrics(grt_text)
    connectivity = parse_route_connectivity(grt_text, droute_text)
    top_physical = parse_openroad(droute_text, float(config["clock_period_ns"]))
    detailed_route_progress = parse_detailed_route_progress(droute_text)
    channel = parse_channel_legalization(
        paths["legal_log"].read_text() if present(paths["legal_log"]) else ""
    )
    cts = parse_cts_buffer_legalization(
        paths["cts_log"].read_text() if present(paths["cts_log"]) else ""
    )
    route_reports = [
        {
            "completed_iteration": suffix - 1,
            "file_suffix": suffix,
            "report": artifact(path),
            "marker_metrics": parse_congestion_marker_report(path.read_text()),
        }
        for suffix, path in congestion_iteration_reports(paths["congestion"])
    ]
    macro_track = {
        **config["hierarchical_top_placement"]["macro_origin_track_alignment"],
        "required_macro_instances": 16,
    }
    abstraction: dict[str, Any] = {}
    if include_abstraction and present(paths["tile_lef"]):
        abstraction = build_compact_macro_lef(
            paths["tile_lef"],
            paths["tile_integration_lef"],
            config["abstract_lef_obstructions"],
        )

    submacro_valid = set(submacros) == RECURSIVE_COMPONENTS and all(
        all(item["checks"].values()) for item in submacros.values()
    )
    synthesis = (
        recursive_synthesis(top_synthesis, tile_summary["synthesis"], submacros)
        if top_synthesis and tile_summary and submacro_valid
        else {}
    )
    tile_physical = tile_summary.get("physical", {})
    timing_hierarchy = aggregate_hierarchical_timing(
        float(config["clock_period_ns"]),
        {
            "distributed_top_shell": top_physical,
            "autonomous_tile_shell": tile_physical,
            **{
                name: submacros[name]["physical"]
                for name in sorted(RECURSIVE_COMPONENTS)
            },
        },
    )
    combined_power, power_hierarchy, annotated_pins = (
        recursive_power(top_physical, tile_physical, submacros)
        if submacro_valid
        else ({metric: 0.0 for metric in POWER_METRICS}, {}, 0)
    )
    physical = {
        **top_physical,
        **combined_power,
        "die_area_um2": float(top_physical.get("die_width_um") or 0.0)
        * float(top_physical.get("die_height_um") or 0.0),
        "core_area_um2": float(top_physical.get("core_width_um") or 0.0)
        * float(top_physical.get("core_height_um") or 0.0),
        "critical_path_delay_ns": timing_hierarchy["critical_path_delay_ns"],
        "fmax_ghz": timing_hierarchy["fmax_ghz"],
        "worst_slack_ns_at_1ghz": timing_hierarchy[
            "worst_slack_ns_at_target"
        ],
        "drc_violations": top_physical.get("drc_violations"),
        "annotated_pin_activities": annotated_pins,
        "tile_macro": tile_physical,
        "hierarchical_top": top_physical,
        "timing_hierarchy": timing_hierarchy,
        "power_hierarchy": power_hierarchy,
        "power_aggregation": "recursive_distributed_postroute_transformer_vcd_hierarchy",
    }

    top_nonmacro_cells = int(top_synthesis.get("cell_count") or 0) - macro_instances
    max_cell_displacement_dbu = round(
        float(contract["maximum_standard_cell_displacement_um"])
        * int(macro_track["dbu_per_micron"])
    )
    max_cts_displacement_dbu = round(
        float(contract["maximum_cts_buffer_displacement_um"])
        * int(macro_track["dbu_per_micron"])
    )
    channel.update(
        {
            "maximum_accepted_displacement_dbu": max_cell_displacement_dbu,
            "maximum_accepted_displacement_basis": contract[
                "standard_cell_displacement_basis"
            ],
        }
    )
    channel_valid = (
        channel["cells"] == top_nonmacro_cells > 0
        and channel["rows"] >= int(contract["minimum_physical_rows"])
        and channel["row_segments"] >= int(contract["minimum_row_segments"])
        and channel["selected_physical_rows"] == channel["rows"]
        and channel["selected_row_segments"] == channel["row_segments"]
        and channel["assigned_cells"] == channel["cells"]
        and channel["constructive_audit_cells"] == channel["cells"]
        and channel["site_aligned_cells"] == channel["cells"]
        and channel["segment_contained_cells"] == channel["cells"]
        and channel["standard_nonoverlap_cells"] == channel["cells"]
        and channel["constructive_audit_row_segments"] == channel["row_segments"]
        and channel["audited_nonoverlapping_macro_clear_row_segments"]
        == channel["row_segments"]
        and int(channel["max_displacement_dbu"] or max_cell_displacement_dbu + 1)
        <= max_cell_displacement_dbu
    )
    cts_valid = (
        int(cts["buffers"] or 0) > 0
        and cts["assigned_buffers"] == cts["buffers"]
        and cts["physical_rows"] == channel["rows"]
        and cts["row_segments"] == channel["row_segments"]
        and cts["site_aligned_buffers"] == cts["buffers"]
        and cts["segment_contained_buffers"] == cts["buffers"]
        and cts["fixed_clear_buffers"] == cts["buffers"]
        and cts["standard_nonoverlap_buffers"] == cts["buffers"]
        and int(cts["max_displacement_dbu"] or max_cts_displacement_dbu + 1)
        <= max_cts_displacement_dbu
    )
    macro_track_valid = (
        macro_instances == 16
        and channel["macro_instances_aligned"] == 16
        and channel["macro_origin_grid_dbu"] == int(macro_track["grid_dbu"])
        and channel["macro_max_displacement_dbu"] == 0
        and all(
            int(macro_track["grid_dbu"]) % int(pitch) == 0
            for pitch in macro_track["routing_pitch_dbu"].values()
        )
    )
    all_nets_globally_routed = (
        connectivity["global_route_completed"]
        and connectivity["global_missing_pin_routes"] == 0
        and not connectivity["global_missing_warning_limit_reached"]
        and int(global_route.get("routed_nets") or 0) > 0
    )
    global_route_iterations_complete = bool(route_reports) and route_reports[-1][
        "completed_iteration"
    ] == int(contract["congestion_iterations"])
    actual_droute_signoff = (
        detailed_route_outputs_present(paths)
        and connectivity["all_pins_routed"]
        and top_physical.get("drc_violations") == 0
    )
    abstraction_valid = (
        bool(abstraction)
        and abstraction["pin_geometry_preserved"]
        and abstraction["conservative_obstruction_cover"]
        and abstraction["pin_count"]
        == abstraction["pin_rectangles"]
        == abstraction["accessible_pin_rectangles"]
        and abstraction["source_obstruction_rectangles"]
        > abstraction["integration_obstruction_rectangles"]
        > 0
    )
    route_tool, route_tool_valid = tool_provenance(config)
    checks = {
        "real_4x4_top": macro_instances == 16,
        "synthesis": bool(synthesis)
        and int(synthesis.get("cell_count") or 0) > 0
        and float(synthesis.get("cell_area_um2") or 0.0) > 0.0,
        "gpl_checkpoint": present(paths["gpl"]),
        "channel_legalization": channel_valid,
        "cts_buffer_legalization": cts_valid,
        "macro_track_alignment": macro_track_valid,
        "global_route_checkpoint": present(paths["grt"]),
        "global_route_iterations_complete": global_route_iterations_complete,
        "all_nets_globally_routed": all_nets_globally_routed,
        "actual_detailed_route_signoff": actual_droute_signoff,
        "drc_clean": top_physical.get("drc_violations") == 0,
        "timing": top_physical.get("critical_path_delay_ns") is not None
        and timing_hierarchy["critical_path_component"] is not None
        and float(timing_hierarchy["critical_path_delay_ns"] or 0.0) > 0.0,
        "vcd_power": int(top_physical.get("annotated_pin_activities") or 0) > 0
        and float(top_physical.get("total_power_w") or 0.0) > 0.0
        and annotated_pins > 0
        and all(
            value > 0 and math.isfinite(value) for value in combined_power.values()
        ),
        "recursive_submacro_evidence": submacro_valid,
        "tile_macro_signoff": tile_summary.get("status") == "supported"
        and all(tile_summary.get("required_checks", {}).values()),
        "compact_macro_abstraction": abstraction_valid,
        "global_route_tool_provenance": route_tool_valid,
        "raw_unfitted": config["calibration"]
        == {"applied": False, "coefficients": None},
        "hierarchical_integrated": actual_droute_signoff and macro_instances == 16,
        "place_route": actual_droute_signoff and all_nets_globally_routed,
        "route_connectivity": connectivity["all_pins_routed"],
    }
    diagnostics = {
        "global_route_overflow_is_zero": global_route.get("overflow_resolved")
        is True,
        "global_route_overflow_policy": contract["global_route_overflow_policy"],
        "global_route_overflow_is_not_used_as_actual_drc": True,
    }

    file_paths = {
        "config": config_path,
        "tile_candidate_summary": paths["tile_summary"],
        "submacro_manifest": paths["submacro_manifest"],
        "top_netlist": paths["netlist"],
        "top_synthesis_stats": paths["stats"],
        "top_synthesis_log": paths["synthesis_log"],
        "top_gpl_checkpoint": paths["gpl"],
        "top_gpl_log": paths["gpl_log"],
        "top_channel_rows_checkpoint": paths["rows"],
        "top_channel_seed_checkpoint": paths["seed"],
        "top_channel_precheck_checkpoint": paths["precheck"],
        "top_channel_legalization_checkpoint": paths["legal"],
        "top_channel_legalization_log": paths["legal_log"],
        "top_cts_seed_checkpoint": paths["cts_seed"],
        "top_cts_checkpoint": paths["cts"],
        "top_cts_log": paths["cts_log"],
        "top_global_route_log": paths["grt_log"],
        "top_global_route_checkpoint": paths["grt"],
        "top_global_route_guide": paths["guide"],
        "top_detailed_route_log": paths["droute_log"],
        "top_base_detailed_route_log": paths["base_droute_log"],
        "top_base_detailed_route_drc": paths["base_drc"],
        "top_base_detailed_route_def": paths["base_def"],
        "top_base_detailed_route_odb": paths["base_odb"],
        "top_base_detailed_route_spef": paths["base_spef"],
        "top_repair_detailed_route_log": paths["repair_log"],
        "top_repair_detailed_route_drc": paths["repair_drc"],
        "top_repair_detailed_route_def": paths["repair_def"],
        "top_repair_detailed_route_odb": paths["repair_odb"],
        "top_repair_detailed_route_spef": paths["repair_spef"],
        "top_clean_retry_detailed_route_log": paths["clean_retry_log"],
        "top_clean_retry_detailed_route_drc": paths["clean_retry_drc"],
        "top_clean_retry_detailed_route_def": paths["clean_retry_def"],
        "top_clean_retry_detailed_route_odb": paths["clean_retry_odb"],
        "top_clean_retry_detailed_route_spef": paths["clean_retry_spef"],
        "top_local_repair_detailed_route_log": paths["local_repair_log"],
        "top_failed_local_repair2_detailed_route_log": paths[
            "failed_local_repair2_log"
        ],
        "top_failed_local_repair3_detailed_route_log": paths[
            "failed_local_repair3_log"
        ],
        "top_local_repair4_detailed_route_log": paths["local_repair4_log"],
        "top_local_repair4_detailed_route_drc": paths["local_repair4_drc"],
        "top_local_repair4_detailed_route_def": paths["local_repair4_def"],
        "top_local_repair4_detailed_route_odb": paths["local_repair4_odb"],
        "top_local_repair4_detailed_route_spef": paths["local_repair4_spef"],
        "top_local_repair4_geometry_audit_log": paths[
            "residual_drc_audit_log"
        ],
        "top_local_repair_detailed_route_drc": paths["local_repair_drc"],
        "top_local_repair_detailed_route_def": paths["local_repair_def"],
        "top_local_repair_detailed_route_odb": paths["local_repair_odb"],
        "top_local_repair_detailed_route_spef": paths["local_repair_spef"],
        "distributed_4x4_guide": paths["guide"],
        "distributed_4x4_drc": paths["drc"],
        "distributed_4x4_def": paths["def"],
        "distributed_4x4_odb": paths["odb"],
        "distributed_4x4_spef": paths["spef"],
        "source_vcd": paths["vcd"],
        "tile_abstract_lef": paths["tile_lef"],
        "tile_integration_abstract_lef": paths["tile_integration_lef"],
        "tile_timing_liberty": paths["tile_lib"],
    }
    files: dict[str, Any] = {
        name: artifact(path) for name, path in file_paths.items() if path.is_file()
    }
    files.update(
        {
            "global_route_openroad": route_tool["binary"],
            "global_route_patch": route_tool["patch"],
            "global_route_archive": route_tool["archive"],
            "detailed_route_openroad": route_tool["detailed_route_binary"],
            "prior_local_repair_openroad": route_tool[
                "prior_local_repair_openroad"
            ]["binary"],
            "prior_local_repair_patch": route_tool[
                "prior_local_repair_openroad"
            ]["patch"],
            "prior_local_repair_archive": route_tool[
                "prior_local_repair_openroad"
            ]["archive"],
            "local_repair_openroad": route_tool["local_repair_openroad"][
                "binary"
            ],
            "local_repair_patch": route_tool["local_repair_openroad"]["patch"],
            "local_repair_archive": route_tool["local_repair_openroad"][
                "archive"
            ],
            "rtl": {
                name: artifact(PROJECT_ROOT / name)
                for name in (
                    "rtl/mlx/mlx_array_pe_tile.sv",
                    "rtl/mlx/mlx_array_4x4_distributed.sv",
                    "rtl/mlx/mlx_array_4x4.sv",
                )
            },
        }
    )
    if route_reports:
        files["top_global_route_congestion_iteration_reports"] = route_reports
    if paths["congestion"].is_file():
        files["top_global_route_congestion_report"] = artifact(paths["congestion"])

    hierarchical_top = {
        "implementation": "distributed_autonomous_pe_tiles",
        "synthesis": top_synthesis,
        "physical": top_physical,
        "macro_instances": macro_instances,
        "macro_master": "mlx_array_pe_tile",
        "integration_abstraction": abstraction,
        "channel_legalization": channel,
        "cts_buffer_legalization": cts,
        "macro_track_contract": macro_track,
        "route_connectivity": connectivity,
        "global_route_metrics": global_route,
        "global_route_iteration_reports": route_reports,
        "detailed_route_progress": detailed_route_progress,
        "route_contract": contract,
        "checks": checks,
        "diagnostics": diagnostics,
    }
    status = "supported" if all(checks.values()) else "incomplete"
    common = {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": "raw_unfitted_distributed_hierarchical_nangate45_4x4",
        "paper_performance_targets_consumed": False,
        "calibration": config["calibration"],
        "status": status,
        "synthesis": synthesis,
        "tile_macro": tile_summary,
        "submacro_chain": submacros,
        "hierarchical_top": hierarchical_top,
        "physical": physical,
        "checks": checks,
        "diagnostics": diagnostics,
        "global_route_tool": route_tool,
        "route_contract": contract,
        "files": files,
    }
    manifest = {
        **common,
        "activity": {
            **config["activity"],
            "distributed_top_vcd": artifact(paths["vcd"]),
            "tile_vcd": tile_summary.get("files", {}).get("vcd"),
        },
        "tools": {
            "yosys": subprocess.run(
                ["yosys", "-V"], capture_output=True, text=True, check=False
            ).stdout.strip(),
            "openroad_global_route": route_tool,
        },
    }
    result = {
        key: value for key, value in common.items() if key not in {"files", "calibration"}
    }
    result.update(
        {
            "claim": (
                "raw, unfitted hierarchical PPA for the promoted distributed "
                "autonomous-tile 4x4 MLX array"
            ),
            "implementation": (
                "sixteen routed autonomous PE-tile hard macros in the integrated "
                "distributed 4x4 top"
            ),
            "exclusions": [
                "RISC-V host",
                "CPU caches",
                "DMA controller",
                "SPM storage",
                "DRAM/PHY",
            ],
            "sources": {
                "area": (
                    "distributed hierarchical Yosys/OpenROAD top, tile, PE, RF, "
                    "and lane evidence"
                ),
                "timing": (
                    "worst post-route delay across distributed top, autonomous "
                    "tile, and recursive PE/RF/lane hierarchy"
                ),
                "power": (
                    "recursive post-route Transformer VCD aggregation over "
                    "distributed top, tile shell, PE shell, RF, and lanes"
                ),
                "calibration": "none",
            },
        }
    )
    return manifest, result


def write_final(
    config: dict[str, Any], manifest: dict[str, Any], result: dict[str, Any]
) -> None:
    manifest_path = PROJECT_ROOT / config["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result["manifest"] = artifact(manifest_path)
    result_path = PROJECT_ROOT / config["result"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, default="inspect")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    paths = distributed_paths(PROJECT_ROOT / config["output_root"])
    paths["root"].mkdir(parents=True, exist_ok=True)

    requested = (
        ("synthesis", "gpl", "legal", "cts", "grt", "droute")
        if args.stage == "all"
        else (() if args.stage in {"inspect", "finalize"} else (args.stage,))
    )
    stage_outputs = {
        "synthesis": paths["netlist"],
        "gpl": paths["gpl"],
        "legal": paths["legal"],
        "cts": paths["cts"],
        "grt": paths["grt"],
        "droute": paths["odb"],
        "repair": paths["repair_odb"],
        "retry": paths["clean_retry_odb"],
        "drc-audit": paths["residual_drc_audit_log"],
        "local-repair": paths["local_repair_odb"],
    }
    for stage in requested:
        if present(stage_outputs[stage]) and not args.force:
            continue
        rc = (
            synthesize(config, paths)
            if stage == "synthesis"
            else run_physical_stage(stage, config, paths)
        )
        if rc != 0:
            raise RuntimeError(
                f"distributed-top stage {stage} failed with return code {rc}"
            )

    paths = distributed_paths(PROJECT_ROOT / config["output_root"])
    include_abstraction = args.stage in {
        "droute",
        "repair",
        "retry",
        "local-repair",
        "finalize",
        "all",
    } and present(
        paths["droute_log"]
    )
    manifest, result = build_result(
        config_path,
        config,
        paths,
        include_abstraction=include_abstraction,
    )
    paths["preview"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] == "supported":
        write_final(config, manifest, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": result["checks"],
                "diagnostics": result["diagnostics"],
                "global_route": result["hierarchical_top"][
                    "global_route_metrics"
                ],
                "connectivity": result["hierarchical_top"]["route_connectivity"],
                "physical": result["physical"],
            },
            indent=2,
        )
    )
    if args.stage in {
        "inspect",
        "synthesis",
        "gpl",
        "legal",
        "cts",
        "grt",
        "drc-audit",
    }:
        return 0
    return 0 if result["status"] == "supported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
