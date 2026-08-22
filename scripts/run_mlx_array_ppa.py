#!/usr/bin/env python3
"""Synthesize, place/route, time, and VCD-power the real MLX 4x4 array."""

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

import yaml

from scripts.extract_vcd_scope import extract_scope

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/system/mlx_array_ppa_v1.yaml"


def artifact(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        relative = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        relative = str(path)
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def run_to_log(command: list[str], log: Path, environment: dict[str, str] | None = None) -> int:
    with log.open("w") as stream:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
            text=True,
        )
    return result.returncode


def parse_synthesis(text: str) -> dict[str, int | float | None]:
    cells = re.findall(r"Number of cells:\s+(\d+)", text)
    areas = re.findall(r"Chip area for (?:top )?module .*?:\s+([0-9.eE+-]+)", text)
    return {
        "cell_count": int(cells[-1]) if cells else None,
        "cell_area_um2": float(areas[-1]) if areas else None,
    }


def parse_openroad(text: str, clock_period: float) -> dict[str, Any]:
    totals = re.findall(
        r"^Total\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
        r"([0-9.eE+-]+)\s+([0-9.eE+-]+)",
        text,
        flags=re.MULTILINE,
    )
    annotations = re.findall(r"Annotated\s+(\d+)\s+pin activities", text)
    slacks = re.findall(r"^\s*([+-]?[0-9.]+)\s+slack", text, flags=re.MULTILINE)
    die = re.findall(r"MLX_PPA_DIE_UM\s+([0-9.]+)\s+([0-9.]+)", text)
    core = re.findall(r"MLX_PPA_CORE_UM\s+([0-9.]+)\s+([0-9.]+)", text)
    design_area = re.findall(r"Design area\s+([0-9.eE+-]+)", text)
    violations = re.findall(r"Number of violations\s*=\s*(\d+)", text)
    power = totals[-1] if totals else (None, None, None, None)
    slack = float(slacks[-1]) if slacks else None
    critical_delay = clock_period - slack if slack is not None else None
    return {
        "annotated_pin_activities": int(annotations[-1]) if annotations else None,
        "internal_power_w": float(power[0]) if power[0] is not None else None,
        "switching_power_w": float(power[1]) if power[1] is not None else None,
        "leakage_power_w": float(power[2]) if power[2] is not None else None,
        "total_power_w": float(power[3]) if power[3] is not None else None,
        "worst_slack_ns_at_1ghz": slack,
        "critical_path_delay_ns": critical_delay,
        "fmax_ghz": 1.0 / critical_delay if critical_delay and critical_delay > 0 else None,
        "die_width_um": float(die[-1][0]) if die else None,
        "die_height_um": float(die[-1][1]) if die else None,
        "core_width_um": float(core[-1][0]) if core else None,
        "core_height_um": float(core[-1][1]) if core else None,
        "placed_design_area_um2": float(design_area[-1]) if design_area else None,
        "drc_violations": int(violations[-1]) if violations else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--reuse-synthesis",
        action="store_true",
        help="reuse an existing mapped netlist/stats pair and rerun physical design",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    output = PROJECT_ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    netlist = output / "mlx-array-4x4-mapped.v"
    stats = output / "mlx-array-4x4-synthesis.stats"
    synthesis_log = output / "mlx-array-4x4-synthesis.log"
    scoped_vcd = output / "transformer-block-array-ports.vcd"
    openroad_log = output / "mlx-array-4x4-openroad.log"
    guide = output / "mlx-array-4x4.guide"
    drc = output / "mlx-array-4x4.drc"
    routed_def = output / "mlx-array-4x4-routed.def"
    odb = output / "mlx-array-4x4-routed.odb"
    spef = output / "mlx-array-4x4-routed.spef"

    technology = config["technology"]
    liberty = Path(technology["liberty"])
    sources = [PROJECT_ROOT / item for item in config["rtl_sources"]]
    source_command = " ".join(str(path) for path in sources)
    yosys_commands = [
        f"read_verilog -sv {source_command}",
        f"hierarchy -check -top {config['top']}",
        f"synth -top {config['top']}",
        f"dfflibmap -liberty {liberty}",
        f"abc {'-fast ' if config['abc_fast_mapping'] else ''}-liberty {liberty}",
        "hilomap -singleton -hicell LOGIC1_X1 Z -locell LOGIC0_X1 Z",
        "clean",
        f"tee -o {stats} stat -liberty {liberty}",
        f"write_verilog -noattr -noexpr -nodec {netlist}",
    ]
    if args.reuse_synthesis and netlist.is_file() and stats.is_file():
        synthesis_rc = 0
    else:
        synthesis_rc = run_to_log(
            ["yosys", "-Q", "-p", "; ".join(yosys_commands)], synthesis_log
        )
    if synthesis_rc != 0 or not netlist.is_file() or not stats.is_file():
        print(json.dumps({"stage": "synthesis", "returncode": synthesis_rc}, indent=2))
        return 1

    extraction = extract_scope(
        PROJECT_ROOT / config["activity"]["source"],
        scoped_vcd,
        config["activity"]["source_scope"],
        ports_only=True,
        timestamp_scale=int(config["activity"].get("timestamp_scale", 1)),
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PPA_TECH_LEF": technology["tech_lef"],
            "PPA_MACRO_LEF": technology["macro_lef"],
            "PPA_RCX_RULES": technology["rcx_rules"],
            "PPA_TAPCELL_TCL": technology["tapcell_tcl"],
            "TAP_CELL_NAME": "TAPCELL_X1",
            "PPA_LIBERTY": technology["liberty"],
            "PPA_NETLIST": str(netlist),
            "PPA_TOP": config["top"],
            "PPA_CLOCK_PERIOD_NS": str(config["clock_period_ns"]),
            "PPA_UTILIZATION": str(config["placement_utilization_percent"]),
            "PPA_DENSITY": str(config["placement_density"]),
            "PPA_BIN_GRID_COUNT": str(config["placement_bin_grid_count"]),
            "PPA_OVERFLOW_TARGET": str(config["placement_overflow_target"]),
            "PPA_INITIAL_PLACE_MAX_ITER": str(config["initial_place_max_iter"]),
            "PPA_INIT_DENSITY_PENALTY": str(
                config["placement_init_density_penalty"]
            ),
            "PPA_MIN_PHI_COEF": str(config["placement_min_phi_coef"]),
            "PPA_MAX_PHI_COEF": str(config["placement_max_phi_coef"]),
            "PPA_PHYSICAL_SEED": "1" if config["physical_seed"]["enabled"] else "0",
            "PPA_PE_CELL_COUNT": str(config["physical_seed"]["pe_cell_count"]),
            "PPA_TOP_CELL_COUNT": str(config["physical_seed"]["top_cell_count"]),
            "PPA_SEED_GRID_COLUMNS": str(config["physical_seed"]["grid_columns"]),
            "PPA_PRE_CTS_REPAIR": "1" if config["pre_cts_repair"] else "0",
            "PPA_POST_CTS_REPAIR": "1" if config["post_cts_repair"] else "0",
            "PPA_THREADS": str(config["threads"]),
            "PPA_DROUTE_END_ITER": str(config["droute_end_iter"]),
            "PPA_VCD_SCOPE": config["activity"]["promoted_scope"],
            "PPA_VCD": str(scoped_vcd),
            "PPA_GUIDE": str(guide),
            "PPA_DRC": str(drc),
            "PPA_DEF": str(routed_def),
            "PPA_ODB": str(odb),
            "PPA_SPEF": str(spef),
        }
    )
    openroad_rc = run_to_log(
        [
            "openroad",
            "-no_init",
            "-exit",
            str(PROJECT_ROOT / "rtl/ppa/openroad_array_flow.tcl"),
        ],
        openroad_log,
        environment,
    )

    synthesis_metrics = parse_synthesis(stats.read_text())
    openroad_metrics = parse_openroad(openroad_log.read_text(), float(config["clock_period_ns"]))
    expected_outputs = [guide, drc, routed_def, odb, spef]
    numeric_power = [
        openroad_metrics[name]
        for name in ("internal_power_w", "switching_power_w", "leakage_power_w", "total_power_w")
    ]
    checks = {
        "real_4x4_top": config["top"] == "mlx_array_4x4",
        "synthesis": synthesis_rc == 0
        and int(synthesis_metrics["cell_count"] or 0) > 0
        and float(synthesis_metrics["cell_area_um2"] or 0.0) > 0,
        "place_route": openroad_rc == 0
        and all(path.is_file() for path in expected_outputs)
        and all(
            path.stat().st_size > 0
            for path in expected_outputs
            if path != drc
        ),
        "drc_clean": openroad_metrics["drc_violations"] == 0,
        "timing": openroad_metrics["worst_slack_ns_at_1ghz"] is not None
        and openroad_metrics["fmax_ghz"] is not None,
        "vcd_power": int(openroad_metrics["annotated_pin_activities"] or 0) > 0
        and all(value is not None and math.isfinite(float(value)) for value in numeric_power),
        "raw_unfitted": not config["calibration"]["applied"]
        and config["calibration"]["coefficients"] is None,
    }
    files = {
        "config": artifact(config_path),
        "rtl": {str(path.relative_to(PROJECT_ROOT)): artifact(path) for path in sources},
        "liberty": artifact(liberty),
        "tech_lef": artifact(Path(technology["tech_lef"])),
        "macro_lef": artifact(Path(technology["macro_lef"])),
        "rcx_rules": artifact(Path(technology["rcx_rules"])),
        "tapcell_tcl": artifact(Path(technology["tapcell_tcl"])),
        "source_vcd": artifact(PROJECT_ROOT / config["activity"]["source"]),
        "scoped_vcd": artifact(scoped_vcd),
        "netlist": artifact(netlist),
        "synthesis_stats": artifact(stats),
        "synthesis_log": artifact(synthesis_log),
        "openroad_log": artifact(openroad_log),
    }
    for path in expected_outputs:
        if path.is_file():
            files[path.name] = artifact(path)
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": config["classification"],
        "paper_performance_targets_consumed": False,
        "calibration": config["calibration"],
        "tools": {
            "yosys": subprocess.run(
                ["yosys", "-V"], capture_output=True, text=True, check=False
            ).stdout.strip(),
            "openroad": subprocess.run(
                ["openroad", "-version"], capture_output=True, text=True, check=False
            ).stdout.strip(),
            "openroad_package_url": config["toolchain"]["openroad_package_url"],
            "openroad_package_sha256": config["toolchain"]["openroad_package_sha256"],
        },
        "activity": {**config["activity"], "extraction": extraction},
        "physical_seed": config["physical_seed"],
        "timing_repair": {
            "pre_cts": config["pre_cts_repair"],
            "post_cts": config["post_cts_repair"],
        },
        "synthesis": {
            "returncode": synthesis_rc,
            "reused": bool(args.reuse_synthesis),
            "abc_fast_mapping": bool(config["abc_fast_mapping"]),
            "abc_mapping_reason": config["abc_mapping_reason"],
            **synthesis_metrics,
        },
        "physical": {"returncode": openroad_rc, **openroad_metrics},
        "checks": checks,
        "files": files,
    }
    manifest_path = PROJECT_ROOT / config["manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "status": "supported" if all(checks.values()) else "rejected",
        "claim": "raw, unfitted Nangate45 PPA for the integrated executable 4x4 MLX array",
        "exclusions": ["RISC-V host", "CPU caches", "DMA controller", "SPM storage", "DRAM/PHY"],
        "sources": {
            "area": "Yosys/ABC mapped cell measurement",
            "timing": "OpenROAD/OpenSTA post-route measurement",
            "power": "OpenROAD post-route estimate from RTL-workload VCD port activity",
            "calibration": "none",
        },
        "synthesis": {
            "abc_fast_mapping": bool(config["abc_fast_mapping"]),
            "abc_mapping_reason": config["abc_mapping_reason"],
            **synthesis_metrics,
        },
        "physical": openroad_metrics,
        "checks": checks,
        "manifest": artifact(manifest_path),
    }
    result_path = PROJECT_ROOT / config["result"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], **synthesis_metrics, **openroad_metrics, "checks": checks}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
