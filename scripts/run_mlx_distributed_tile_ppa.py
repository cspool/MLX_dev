#!/usr/bin/env python3
"""Build and sign off the autonomous MLX PE tile used by the distributed 4x4."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.run_mlx_array_ppa import artifact, parse_openroad, parse_synthesis, run_to_log
from scripts.run_mlx_hierarchical_ppa import (
    build_compact_macro_lef,
    parse_global_route_metrics,
    parse_route_connectivity,
    yosys_run,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/system/mlx_array_ppa_v1.yaml"
STAGES = ("inspect", "synthesis", "gpl", "legal", "cts", "grt", "droute", "all")


def tile_paths(output: Path, iterations: int) -> dict[str, Path]:
    root = output / "tile_macro"
    base = root / "mlx-array-pe-tile"
    tight = root / "mlx-array-pe-tile-v2-tight"
    routed = root / f"mlx-array-pe-tile-v2-tight-iter{iterations}"
    return {
        "root": root,
        "netlist": base.with_name(f"{base.name}-mapped.v"),
        "stats": base.with_name(f"{base.name}-synthesis.stats"),
        "synthesis_log": base.with_name(f"{base.name}-synthesis.log"),
        "gpl": tight.with_name(f"{tight.name}-global-placement.odb"),
        "gpl_log": tight.with_name(f"{tight.name}-gpl.log"),
        "legal": tight.with_name(f"{tight.name}-legal.odb"),
        "legal_log": tight.with_name(f"{tight.name}-legal.log"),
        "cts": tight.with_name(f"{tight.name}-cts.odb"),
        "cts_log": tight.with_name(f"{tight.name}-cts.log"),
        "guide": routed.with_name(f"{routed.name}-routed.guide"),
        "grt": routed.with_name(f"{routed.name}-global-route.odb"),
        "grt_log": routed.with_name(f"{routed.name}-route.log"),
        "congestion": routed.with_name(f"{routed.name}-congestion.rpt"),
        "droute_log": routed.with_name(f"{routed.name}-droute.log"),
        "drc": routed.with_name(f"{routed.name}-routed.drc"),
        "def": routed.with_name(f"{routed.name}-routed.def"),
        "odb": routed.with_name(f"{routed.name}-routed.odb"),
        "spef": routed.with_name(f"{routed.name}-routed.spef"),
        "lef": routed.with_name(f"{routed.name}.abstract.lef"),
        "integration_lef": routed.with_name(f"{routed.name}.integration.lef"),
        "lib": routed.with_name(f"{routed.name}.lib"),
        "vcd": root / "transformer-block-distributed-tile0-ports.vcd",
        "summary": root / "mlx-array-pe-tile-candidate.json",
    }


def synthesis(config: dict[str, Any], paths: dict[str, Path]) -> int:
    liberty = Path(config["technology"]["liberty"])
    commands = [
        f"read_verilog -sv -lib {PROJECT_ROOT / 'rtl/ppa/mlx_pe_top_blackbox.sv'}",
        (
            "read_verilog -sv -DMLX_PPA_MACRO -DSYNTHESIS "
            f"{PROJECT_ROOT / 'rtl/mlx/mlx_array_pe_tile.sv'}"
        ),
        "hierarchy -check -top mlx_array_pe_tile",
        "synth -top mlx_array_pe_tile",
        f"dfflibmap -liberty {liberty}",
        f"abc -fast -liberty {liberty}",
        "hilomap -singleton -hicell LOGIC1_X1 Z -locell LOGIC0_X1 Z",
        "clean",
        f"tee -o {paths['stats']} stat -liberty {liberty}",
        f"write_verilog -noattr -noexpr -nodec {paths['netlist']}",
    ]
    return yosys_run(commands, paths["synthesis_log"])


def environment(
    config: dict[str, Any], paths: dict[str, Path], iterations: int
) -> dict[str, str]:
    technology = config["technology"]
    candidate = config["hierarchical_distributed_tile_candidate"]
    floorplan = candidate["tile_floorplan_v2_tight"]
    die = floorplan["die_um"]
    origin = floorplan["pe_origin_um"]
    dbu_per_um = 2000
    env = os.environ.copy()
    env.update(
        {
            "MALLOC_ARENA_MAX": "2",
            "PPA_THREADS": "1",
            "PPA_TECH_LEF": technology["tech_lef"],
            "PPA_MACRO_LEF": technology["macro_lef"],
            "PPA_PE_LEF": str(
                PROJECT_ROOT / "artifacts/environment/h206/pe_macro/mlx-pe-top.integration.lef"
            ),
            "PPA_LIBERTY": technology["liberty"],
            "PPA_PE_LIBERTY": str(
                PROJECT_ROOT / "artifacts/environment/h206/pe_macro/mlx-pe-top.lib"
            ),
            "PPA_NETLIST": str(paths["netlist"]),
            "PPA_CLOCK_PERIOD_NS": str(config["clock_period_ns"]),
            "PPA_TILE_DIE_WIDTH_UM": str(die[0]),
            "PPA_TILE_DIE_HEIGHT_UM": str(die[1]),
            "PPA_TILE_CORE_MARGIN_UM": "20",
            "PPA_PE_ORIGIN_X_DBU": str(round(float(origin[0]) * dbu_per_um)),
            "PPA_PE_ORIGIN_Y_DBU": str(round(float(origin[1]) * dbu_per_um)),
            "PPA_TAPCELL_DISTANCE_UM": "1000",
            "TAP_CELL_NAME": "TAPCELL_X1",
            "PPA_DENSITY": "0.10",
            "PPA_BIN_GRID_COUNT": "128",
            "PPA_OVERFLOW_TARGET": "0.01",
            "PPA_INIT_DENSITY_PENALTY": "0.001",
            "PPA_MIN_PHI_COEF": "0.95",
            "PPA_MAX_PHI_COEF": "1.20",
            "PPA_GPL_ODB": str(paths["gpl"]),
            "PPA_LEGAL_ODB": str(paths["legal"]),
            "PPA_CTS_ODB": str(paths["cts"]),
            "PPA_SIGNAL_ROUTING_LAYERS": floorplan["signal_layers"],
            "PPA_CLOCK_ROUTING_LAYERS": floorplan["clock_layers"],
            "PPA_MACRO_EXTENSION_GCELLS": "0",
            "PPA_GRT_CONGESTION_ITERATIONS": str(iterations),
            "PPA_GRT_ALLOW_CONGESTION": "1",
            "PPA_GRT_VERBOSE": "1",
            "PPA_GRT_CONGESTION_REPORT_FILE": str(paths["congestion"]),
            "PPA_GRT_CONGESTION_REPORT_ITER_STEP": "5",
            "PPA_GUIDE": str(paths["guide"]),
            "PPA_GRT_ODB": str(paths["grt"]),
            "PPA_DROUTE_END_ITER": str(config["droute_end_iter"]),
            "PPA_DRC": str(paths["drc"]),
            "PPA_RCX_RULES": technology["rcx_rules"],
            "PPA_DEF": str(paths["def"]),
            "PPA_ODB": str(paths["odb"]),
            "PPA_SPEF": str(paths["spef"]),
            "PPA_ABSTRACT_LEF": str(paths["lef"]),
            "PPA_TIMING_LIB": str(paths["lib"]),
            "PPA_VCD": str(paths["vcd"]) if paths["vcd"].is_file() else "",
            "PPA_VCD_SCOPE": "component_activity",
            "PPA_STOP_AFTER_GPL": "1",
            "PPA_STOP_AFTER_LEGAL": "1",
            "PPA_STOP_AFTER_CTS": "1",
            "PPA_STOP_AFTER_GRT": "1",
        }
    )
    return env


def present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def run_stage(
    stage: str,
    config: dict[str, Any],
    paths: dict[str, Path],
    env: dict[str, str],
) -> int:
    system_openroad = Path(
        config["toolchain"]["detailed_route_and_signoff_openroad"]["binary"]
    )
    global_openroad = PROJECT_ROOT / config["toolchain"]["global_route_openroad"]["binary"]
    initial = PROJECT_ROOT / "rtl/ppa/openroad_array_pe_tile_flow.tcl"
    resume = PROJECT_ROOT / "rtl/ppa/openroad_array_pe_tile_resume.tcl"
    if stage == "gpl":
        return run_to_log(
            [str(system_openroad), "-no_init", "-exit", str(initial)],
            paths["gpl_log"],
            env,
        )
    stage_inputs = {"legal": ("gpl", "legal_log"), "cts": ("legal", "cts_log")}
    if stage in stage_inputs:
        input_key, log_key = stage_inputs[stage]
        local = env.copy()
        local["PPA_INPUT_ODB"] = str(paths[input_key])
        local["PPA_RESUME_STAGE"] = input_key
        local["PPA_STOP_AFTER_LEGAL"] = "1" if stage == "legal" else "0"
        local["PPA_STOP_AFTER_CTS"] = "1"
        return run_to_log(
            [str(system_openroad), "-no_init", "-exit", str(resume)],
            paths[log_key],
            local,
        )
    if stage == "grt":
        local = env.copy()
        local["PPA_INPUT_ODB"] = str(paths["cts"])
        local["PPA_RESUME_STAGE"] = "cts"
        local["PPA_STOP_AFTER_LEGAL"] = "0"
        local["PPA_STOP_AFTER_CTS"] = "0"
        local["PPA_STOP_AFTER_GRT"] = "1"
        return run_to_log(
            [str(global_openroad), "-no_init", "-exit", str(resume)],
            paths["grt_log"],
            local,
        )
    if stage == "droute":
        local = env.copy()
        local["PPA_INPUT_ODB"] = str(paths["grt"])
        local["PPA_RESUME_STAGE"] = "grt"
        local["PPA_STOP_AFTER_GRT"] = "0"
        return run_to_log(
            [str(system_openroad), "-no_init", "-exit", str(resume)],
            paths["droute_log"],
            local,
        )
    raise ValueError(f"unsupported execution stage {stage}")


def summarize(config: dict[str, Any], paths: dict[str, Path], iterations: int) -> dict[str, Any]:
    synthesis_metrics = parse_synthesis(paths["stats"].read_text()) if present(paths["stats"]) else {}
    grt_text = paths["grt_log"].read_text() if present(paths["grt_log"]) else ""
    droute_text = paths["droute_log"].read_text() if present(paths["droute_log"]) else ""
    grt = parse_global_route_metrics(grt_text)
    connectivity = parse_route_connectivity(grt_text, droute_text)
    physical = parse_openroad(droute_text, float(config["clock_period_ns"]))
    files = {
        name: artifact(path)
        for name, path in paths.items()
        if name not in {"root", "summary"} and present(path)
    }
    checks = {
        "synthesis": synthesis_metrics.get("cell_count", 0) > 0,
        "gpl_checkpoint": present(paths["gpl"]),
        "legal_checkpoint": present(paths["legal"]),
        "cts_checkpoint": present(paths["cts"]),
        "global_route_checkpoint": present(paths["grt"]),
        "zero_global_route_overflow": grt.get("overflow_resolved") is True,
        "global_connectivity": connectivity["global_missing_pin_routes"] == 0
        and not connectivity["global_missing_warning_limit_reached"],
        "detailed_route_outputs": all(
            present(paths[name]) for name in ("drc", "def", "odb", "spef", "lef", "lib")
        ),
        "zero_drc": physical.get("drc_violations") == 0,
        "all_pins_routed": connectivity["all_pins_routed"],
        "workload_vcd": present(paths["vcd"]),
    }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": "distributed_autonomous_pe_tile_candidate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "supported" if all(checks.values()) else "incomplete",
        "global_route_iterations": iterations,
        "synthesis": synthesis_metrics,
        "global_route": grt,
        "connectivity": connectivity,
        "physical": physical,
        "checks": checks,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--grt-iterations", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    paths = tile_paths(PROJECT_ROOT / config["output_root"], args.grt_iterations)
    paths["root"].mkdir(parents=True, exist_ok=True)
    env = environment(config, paths, args.grt_iterations)

    requested = (
        ("synthesis", "gpl", "legal", "cts", "grt", "droute")
        if args.stage == "all"
        else (() if args.stage == "inspect" else (args.stage,))
    )
    outputs = {
        "synthesis": paths["netlist"],
        "gpl": paths["gpl"],
        "legal": paths["legal"],
        "cts": paths["cts"],
        "grt": paths["grt"],
        "droute": paths["odb"],
    }
    for stage in requested:
        if present(outputs[stage]) and not args.force:
            continue
        if stage == "synthesis":
            rc = synthesis(config, paths)
        else:
            if stage == "droute":
                grt_text = paths["grt_log"].read_text() if present(paths["grt_log"]) else ""
                grt = parse_global_route_metrics(grt_text)
                if grt.get("overflow_resolved") is not True:
                    raise RuntimeError("refusing detailed route before zero global-route overflow")
            rc = run_stage(stage, config, paths, env)
        if rc != 0:
            raise RuntimeError(f"distributed tile stage {stage} failed with return code {rc}")

    if present(paths["lef"]):
        build_compact_macro_lef(
            paths["lef"],
            paths["integration_lef"],
            config["abstract_lef_obstructions"],
        )
    summary = summarize(config, paths, args.grt_iterations)
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
