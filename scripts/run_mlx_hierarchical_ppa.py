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
    payload = path.read_bytes()
    try:
        name = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        name = str(path)
    return {
        "path": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-pe-synthesis", action="store_true")
    parser.add_argument("--reuse-pe-physical", action="store_true")
    parser.add_argument("--reuse-top-synthesis", action="store_true")
    parser.add_argument("--reuse-top-physical", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    technology = config["technology"]
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
    top_phys = physical_paths(top_root, "mlx-array-4x4-hierarchical-routed")
    top_gpl_checkpoint = top_root / "mlx-array-4x4-global-placement.odb"
    top_grt_checkpoint = top_root / "mlx-array-4x4-global-route.odb"
    top_post_gpl_log = top_root / "mlx-array-4x4-post-gpl-resume.log"
    top_droute_resume_log = top_root / "mlx-array-4x4-droute-resume.log"
    top_placement = config["hierarchical_top_placement"]
    top_environment = common_environment.copy()
    top_environment.update(
        {
            "PPA_PE_LEF": str(pe_lef),
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
                top_placement.get(
                    "global_route_congestion_iterations",
                    config["global_route_congestion_iterations"],
                )
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
            "PPA_GRT_ODB": str(top_grt_checkpoint),
        }
    )
    if args.reuse_top_physical and output_check(top_phys):
        top_physical_rc = 0
    else:
        resume_top_droute = (
            top_grt_checkpoint.is_file()
            and top_grt_checkpoint.stat().st_size > 0
        )
        resume_top_post_gpl = (
            not resume_top_droute
            and top_gpl_checkpoint.is_file()
            and top_gpl_checkpoint.stat().st_size > 0
        )
        top_environment["PPA_RESUME_GPL"] = (
            "1" if resume_top_post_gpl else "0"
        )
        if resume_top_post_gpl:
            top_environment["PPA_THREADS"] = str(
                top_placement.get("post_gpl_threads", 1)
            )
            top_environment["MALLOC_ARENA_MAX"] = "2"
        physical_log = (
            top_droute_resume_log
            if resume_top_droute
            else top_post_gpl_log
            if resume_top_post_gpl
            else top_phys["log"]
        )
        top_physical_rc = run_to_log(
            [
                "openroad",
                "-no_init",
                "-exit",
                str(
                    PROJECT_ROOT
                    / "rtl/ppa"
                    / (
                        "openroad_hierarchical_array_droute_resume.tcl"
                        if resume_top_droute
                        else "openroad_hierarchical_array_flow.tcl"
                    )
                ),
            ],
            physical_log,
            top_environment,
        )
        if resume_top_droute or resume_top_post_gpl:
            prior_log = top_phys["log"].read_text() if top_phys["log"].is_file() else ""
            marker = (
                "MLX_ARRAY_DROUTE_RESUME_LOG_BEGIN"
                if resume_top_droute
                else "MLX_ARRAY_POST_GPL_RESUME_LOG_BEGIN"
            )
            top_phys["log"].write_text(
                prior_log
                + f"\n{marker}\n"
                + physical_log.read_text()
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
    }
    if not all(top_checks.values()):
        print(
            json.dumps(
                {"stage": "hierarchical_top_physical", "checks": top_checks},
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
    }

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
        "hierarchical_top": {
            "synthesis": top_synthesis,
            "physical": top_physical,
            "macro_instances": macro_instances,
            "checks": top_checks,
        },
        "physical": physical,
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
        },
        "physical": physical,
        "checks": checks,
        "manifest": artifact(manifest_path),
    }
    result_path = PROJECT_ROOT / config["result"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "checks": checks, "physical": physical}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
