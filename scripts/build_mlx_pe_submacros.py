#!/usr/bin/env python3
"""Build routed lane/RF/FU submacros and the hierarchical MLX PE hard macro."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from scripts.extract_vcd_scope import extract_scope
from scripts.run_mlx_array_ppa import parse_openroad, parse_synthesis, run_to_log

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/system/mlx_array_ppa_v1.yaml"


@dataclass(frozen=True)
class MacroSpec:
    name: str
    top: str
    read_commands: tuple[str, ...]
    chparam: str | None
    has_clock: bool
    dependencies: tuple[str, ...] = ()
    final_pe: bool = False


def run_yosys(commands: list[str], log: Path) -> int:
    return run_to_log(["yosys", "-Q", "-p", "; ".join(commands)], log)


def outputs(root: Path, top: str) -> dict[str, Path]:
    return {
        "netlist": root / f"{top}-mapped.v",
        "stats": root / f"{top}-synthesis.stats",
        "synth_log": root / f"{top}-synthesis.log",
        "guide": root / f"{top}.guide",
        "drc": root / f"{top}.drc",
        "def": root / f"{top}.def",
        "odb": root / f"{top}.odb",
        "spef": root / f"{top}.spef",
        "lef": root / f"{top}.abstract.lef",
        "lib": root / f"{top}.lib",
        "physical_log": root / f"{top}-physical.log",
    }


def ready(paths: dict[str, Path]) -> bool:
    return all(
        path.is_file() and (key == "drc" or path.stat().st_size > 0)
        for key, path in paths.items()
    )


def synthesis_ready(paths: dict[str, Path]) -> bool:
    return all(
        paths[key].is_file() and paths[key].stat().st_size > 0
        for key in ("netlist", "stats", "synth_log")
    )


def parse_pin_access(text: str) -> dict[str, object]:
    stdcell_no_access = re.findall(r"#stdCellPinNoAp\s*=\s*(\d+)", text)
    macro_no_access = re.findall(r"#macroNoAp\s*=\s*(\d+)", text)
    stdcell_pins_without_access = (
        int(stdcell_no_access[-1]) if stdcell_no_access else None
    )
    macro_pins_without_access = int(macro_no_access[-1]) if macro_no_access else None
    no_access_errors = len(re.findall(r"\[(?:ERROR|WARNING) DRT-0073\]", text))
    pin_access_completed = "[INFO DRT-0166] Complete pin access." in text
    return {
        "pin_access_completed": pin_access_completed,
        "stdcell_pins_without_access": stdcell_pins_without_access,
        "macro_pins_without_access": macro_pins_without_access,
        "no_access_errors": no_access_errors,
        "off_grid_macro_term_warnings": len(
            re.findall(r"\[WARNING DRT-0418\]", text)
        ),
        "non_center_macro_term_warnings": len(
            re.findall(r"\[WARNING DRT-0419\]", text)
        ),
        "off_grid_block_term_warnings": len(
            re.findall(r"\[WARNING DRT-0421\]", text)
        ),
        "diagnostic_warning_limit_reached": bool(
            re.search(r"\[WARNING DRT-(?:0418|0419|0421)\] message limit", text)
        ),
        "all_pins_accessible": pin_access_completed
        and stdcell_pins_without_access == 0
        and macro_pins_without_access == 0
        and no_access_errors == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--skip-power", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    technology = config["technology"]
    liberty = Path(technology["liberty"])
    root = PROJECT_ROOT / config["output_root"] / "pe_submacros"
    final_root = PROJECT_ROOT / config["output_root"] / "pe_macro"
    root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    fp16 = PROJECT_ROOT / "rtl/mlx/mlx_fp16.sv"
    rf = PROJECT_ROOT / "rtl/mlx/mlx_register_file.sv"
    fu = PROJECT_ROOT / "rtl/mlx/mlx_fu.sv"
    lane_blackbox = PROJECT_ROOT / "rtl/ppa/mlx_fp16_lanes_blackbox.sv"
    rf_blackbox = PROJECT_ROOT / "rtl/ppa/mlx_register_file_blackbox.sv"
    pe_sources = tuple(
        str(PROJECT_ROOT / item)
        for item in (
            "rtl/mlx/mlx_config_network.sv",
            "rtl/mlx/mlx_tag_buffer.sv",
            "rtl/mlx/mlx_data_network.sv",
            "rtl/mlx/mlx_control_logic.sv",
            "rtl/mlx/mlx_pe_top.sv",
        )
    )
    specs = (
        MacroSpec("full_lane", "mlx_fp16_alu_lane", (f"read_verilog -sv {fp16}",), None, False),
        MacroSpec(
            "reduced_lane",
            "mlx_fp16_reduced_lane",
            (f"read_verilog -sv {fp16}",),
            None,
            False,
        ),
        MacroSpec(
            "register_file",
            "mlx_register_file",
            (f"read_verilog -sv {rf}",),
            (
                "chparam -set SIMD_WIDTH 32 -set DATA_BITS 16 -set DEPTH 16 "
                "-set ADDR_BITS 4 -set GATED_CLOCK 0 mlx_register_file"
            ),
            True,
        ),
        MacroSpec(
            "functional_unit",
            "mlx_fu",
            (
                f"read_verilog -sv -lib {lane_blackbox}",
                f"read_verilog -sv {fu}",
            ),
            (
                "chparam -set SIMD_WIDTH 32 -set DATA_BITS 16 -set FULL_FEATURES 1 "
                "-set TRANS_LANES 8 -set HP_LANES 8 mlx_fu"
            ),
            True,
            ("full_lane", "reduced_lane"),
        ),
        MacroSpec(
            "pe_top",
            "mlx_pe_top",
            (
                f"read_verilog -sv -lib {lane_blackbox}",
                f"read_verilog -sv -lib {rf_blackbox}",
                f"read_verilog -sv {fu}",
                f"read_verilog -sv -DMLX_PPA_RF_SUBMACRO {' '.join(pe_sources)}",
            ),
            (
                "chparam -set SIMD_WIDTH 32 -set FULL_FEATURES 1 -set TRANS_LANES 8 "
                "-set RF_DEPTH 16 -set CONFIG_GATED_CLOCK 0 -set TAG_GATED_CLOCK 0 "
                "-set RF_GATED_CLOCK 0 -set NETWORK_GATED_CLOCK 0 mlx_pe_top"
            ),
            True,
            ("full_lane", "reduced_lane", "register_file"),
            True,
        ),
    )
    built: dict[str, dict[str, Path]] = {}
    summaries: dict[str, object] = {}
    source_vcd = PROJECT_ROOT / config["activity"]["source"]
    pe0_scope = "TOP.mlx_array_4x4.GENERATE_PES(0).physical_pe"
    activity_scopes = {
        "full_lane": (
            f"{pe0_scope}.functional_unit.GENERATE_LANES(0)"
            ".GENERATE_FULL_LANE.alu"
        ),
        "reduced_lane": (
            f"{pe0_scope}.functional_unit.GENERATE_LANES(10)"
            ".GENERATE_REDUCED_LANE.alu"
        ),
        "register_file": f"{pe0_scope}.register_file",
        "functional_unit": f"{pe0_scope}.functional_unit",
        "pe_top": pe0_scope,
    }

    for spec in specs:
        component_root = final_root if spec.final_pe else root / spec.name
        component_root.mkdir(parents=True, exist_ok=True)
        paths = outputs(component_root, spec.top)
        if spec.final_pe:
            paths.update(
                {
                    "netlist": final_root / "mlx-pe-top-mapped.v",
                    "stats": final_root / "mlx-pe-top-synthesis.stats",
                    "synth_log": final_root / "mlx-pe-top-synthesis.log",
                    "guide": final_root / "mlx-pe-top-routed.guide",
                    "drc": final_root / "mlx-pe-top-routed.drc",
                    "def": final_root / "mlx-pe-top-routed.def",
                    "odb": final_root / "mlx-pe-top-routed.odb",
                    "spef": final_root / "mlx-pe-top-routed.spef",
                    "lef": final_root / "mlx-pe-top.abstract.lef",
                    "lib": final_root / "mlx-pe-top.lib",
                    "physical_log": final_root / "mlx-pe-top-routed.log",
                }
            )
        component_vcd = component_root / (
            "transformer-block-pe0-ports.vcd"
            if spec.final_pe
            else f"transformer-block-{spec.name}-ports.vcd"
        )
        activity_extraction = extract_scope(
            source_vcd,
            component_vcd,
            activity_scopes[spec.name],
            ports_only=True,
            timestamp_scale=int(config["activity"]["timestamp_scale"]),
        )
        dependency_paths = [built[name] for name in spec.dependencies]
        grt_checkpoint = component_root / f"{spec.top}-global-route.odb"
        droute_resume_log = component_root / f"{spec.top}-droute-resume.log"
        if not (args.reuse and ready(paths)):
            # The PE path was also used by the earlier flat-PE experiment, so
            # never infer PE synthesis compatibility from filenames alone.
            if not (
                args.reuse and not spec.final_pe and synthesis_ready(paths)
            ):
                commands = [*spec.read_commands]
                if spec.chparam:
                    commands.append(spec.chparam)
                commands.extend(
                    (
                        f"hierarchy -check -top {spec.top}",
                        f"synth -top {spec.top}",
                        f"dfflibmap -liberty {liberty}",
                        f"abc -fast -liberty {liberty}",
                    )
                )
                if spec.final_pe:
                    commands.append(
                        "hilomap -singleton -hicell LOGIC1_X1 Z -locell LOGIC0_X1 Z"
                    )
                commands.extend(
                    (
                        "clean",
                        f"tee -o {paths['stats']} stat -liberty {liberty}",
                        f"write_verilog -noattr -noexpr -nodec {paths['netlist']}",
                    )
                )
                synthesis_rc = run_yosys(commands, paths["synth_log"])
                if synthesis_rc != 0:
                    print(
                        json.dumps(
                            {
                                "stage": f"{spec.name}_synthesis",
                                "returncode": synthesis_rc,
                            }
                        )
                    )
                    return 1

            environment = os.environ.copy()
            component_placement = config["component_placement"]
            needs_low_density = spec.name in component_placement[
                "low_density_components"
            ]
            macro_level = component_placement["macro_levels"].get(spec.name)
            if needs_low_density:
                utilization = component_placement["utilization_percent"]
                density = component_placement["density"]
                aspect_ratio = 1.0
                bin_grid_count = component_placement["bin_grid_count"]
                droute_iterations = component_placement["droute_end_iter"]
                skip_initial_place = True
            elif macro_level:
                utilization = macro_level["utilization_percent"]
                density = macro_level["density"]
                aspect_ratio = macro_level["aspect_ratio"]
                bin_grid_count = macro_level["bin_grid_count"]
                droute_iterations = macro_level["droute_end_iter"]
                skip_initial_place = macro_level["skip_initial_place"]
            else:
                utilization = config["placement_utilization_percent"]
                density = config["placement_density"]
                aspect_ratio = 1.0
                bin_grid_count = 128
                droute_iterations = config["droute_end_iter"]
                skip_initial_place = True
            grt_iterations = (
                macro_level.get(
                    "global_route_congestion_iterations",
                    config["global_route_congestion_iterations"],
                )
                if macro_level
                else config["global_route_congestion_iterations"]
            )
            overflow_target = (
                macro_level.get(
                    "overflow_target", config["placement_overflow_target"]
                )
                if macro_level
                else config["placement_overflow_target"]
            )
            environment.update(
                {
                    "PPA_THREADS": str(config["threads"]),
                    "PPA_TECH_LEF": technology["tech_lef"],
                    "PPA_MACRO_LEF": technology["macro_lef"],
                    "PPA_LIBERTY": technology["liberty"],
                    "PPA_RCX_RULES": technology["rcx_rules"],
                    "PPA_TAPCELL_TCL": technology["tapcell_tcl"],
                    "TAP_CELL_NAME": "TAPCELL_X1",
                    "PPA_TOP": spec.top,
                    "PPA_NETLIST": str(paths["netlist"]),
                    "PPA_HAS_CLOCK": "1" if spec.has_clock else "0",
                    "PPA_HAS_MACROS": "1" if dependency_paths else "0",
                    "PPA_MULTI_LAYER_PINS": (
                        "1"
                        if spec.name
                        in component_placement["multi_layer_pin_components"]
                        else "0"
                    ),
                    "PPA_EXTRA_LEFS": " ".join(str(item["lef"]) for item in dependency_paths),
                    "PPA_EXTRA_LIBS": " ".join(str(item["lib"]) for item in dependency_paths),
                    "PPA_CLOCK_PERIOD_NS": str(config["clock_period_ns"]),
                    "PPA_UTILIZATION": str(utilization),
                    "PPA_DENSITY": str(density),
                    "PPA_ASPECT_RATIO": str(aspect_ratio),
                    "PPA_SKIP_INITIAL_PLACE": "1" if skip_initial_place else "0",
                    "PPA_BIN_GRID_COUNT": str(bin_grid_count),
                    "PPA_OVERFLOW_TARGET": str(overflow_target),
                    "PPA_INIT_DENSITY_PENALTY": str(
                        config["placement_init_density_penalty"]
                    ),
                    "PPA_MIN_PHI_COEF": str(config["placement_min_phi_coef"]),
                    "PPA_MAX_PHI_COEF": str(config["placement_max_phi_coef"]),
                    "PPA_DROUTE_END_ITER": str(droute_iterations),
                    "PPA_GRT_CONGESTION_ITERATIONS": str(
                        grt_iterations
                    ),
                    "PPA_GRT_ALLOW_CONGESTION": (
                        "1" if config["global_route_allow_congestion"] else "0"
                    ),
                    "PPA_REPAIR_DESIGN": (
                        "1" if macro_level and macro_level.get("repair_design") else "0"
                    ),
                    "PPA_REPAIR_MAX_UTILIZATION": str(
                        macro_level.get("repair_max_utilization_percent", 80)
                        if macro_level
                        else 80
                    ),
                    "PPA_GUIDE": str(paths["guide"]),
                    "PPA_DRC": str(paths["drc"]),
                    "PPA_DEF": str(paths["def"]),
                    "PPA_ODB": str(paths["odb"]),
                    "PPA_SPEF": str(paths["spef"]),
                    "PPA_ABSTRACT_LEF": str(paths["lef"]),
                    "PPA_TIMING_LIB": str(paths["lib"]),
                    "PPA_LIBRARY_NAME": f"mlx_{spec.name}_macro_lib",
                    "PPA_VCD": str(component_vcd),
                    "PPA_VCD_SCOPE": "component_activity",
                    "PPA_GRT_ODB": str(grt_checkpoint),
                }
            )
            resume_droute = (
                args.reuse
                and grt_checkpoint.is_file()
                and grt_checkpoint.stat().st_size > 0
            )
            physical_rc = run_to_log(
                [
                    "openroad",
                    "-no_init",
                    "-exit",
                    str(
                        PROJECT_ROOT
                        / "rtl/ppa"
                        / (
                            "openroad_component_droute_resume.tcl"
                            if resume_droute
                            else "openroad_component_macro_flow.tcl"
                        )
                    ),
                ],
                droute_resume_log if resume_droute else paths["physical_log"],
                environment,
            )
            if resume_droute:
                paths["physical_log"].write_text(
                    paths["physical_log"].read_text()
                    + "\nMLX_COMPONENT_DROUTE_RESUME_LOG_BEGIN\n"
                    + droute_resume_log.read_text()
                )
            if physical_rc != 0:
                print(json.dumps({"stage": f"{spec.name}_physical", "returncode": physical_rc}))
                return 1
        metrics = parse_openroad(
            paths["physical_log"].read_text(), float(config["clock_period_ns"])
        )
        pin_access = parse_pin_access(paths["physical_log"].read_text())
        synthesis = parse_synthesis(paths["stats"].read_text())
        workload_power = None
        power_rc = None
        if not args.skip_power:
            power_log = component_root / f"{spec.top}-workload-power.log"
            power_environment = os.environ.copy()
            power_environment.update(
                {
                    "PPA_THREADS": str(config["threads"]),
                    "PPA_TECH_LEF": technology["tech_lef"],
                    "PPA_MACRO_LEF": technology["macro_lef"],
                    "PPA_LIBERTY": technology["liberty"],
                    "PPA_EXTRA_LEFS": " ".join(
                        str(item["lef"]) for item in dependency_paths
                    ),
                    "PPA_EXTRA_LIBS": " ".join(
                        str(item["lib"]) for item in dependency_paths
                    ),
                    "PPA_TOP": spec.top,
                    "PPA_NETLIST": str(paths["netlist"]),
                    "PPA_DEF": str(paths["def"]),
                    "PPA_ODB": str(paths["odb"]),
                    "PPA_SPEF": str(paths["spef"]),
                    "PPA_HAS_CLOCK": "1" if spec.has_clock else "0",
                    "PPA_CLOCK_PERIOD_NS": str(config["clock_period_ns"]),
                    "PPA_VCD": str(component_vcd),
                    "PPA_POWER_ONLY": "1",
                }
            )
            if args.reuse and power_log.is_file() and power_log.stat().st_size > 0:
                power_rc = 0
            else:
                power_rc = run_to_log(
                    [
                        "openroad",
                        "-no_init",
                        "-exit",
                        str(PROJECT_ROOT / "rtl/ppa/openroad_component_power.tcl"),
                    ],
                    power_log,
                    power_environment,
                )
            workload_power = parse_openroad(
                power_log.read_text(), float(config["clock_period_ns"])
            )
        checks = {
            "outputs": ready(paths),
            "cells": int(synthesis["cell_count"] or 0) > 0,
            "area": float(synthesis["cell_area_um2"] or 0.0) > 0,
            "drc": metrics["drc_violations"] == 0,
            "pin_access": pin_access["all_pins_accessible"] is True,
        }
        if not args.skip_power:
            assert workload_power is not None
            checks["workload_power"] = (
                power_rc == 0
                and int(workload_power["annotated_pin_activities"] or 0) > 0
                and float(workload_power["total_power_w"] or 0.0) > 0
            )
        if not all(checks.values()):
            print(json.dumps({"stage": spec.name, "checks": checks, "metrics": metrics}, indent=2))
            return 1
        built[spec.name] = paths
        summaries[spec.name] = {
            "synthesis": synthesis,
            "physical": metrics,
            "pin_access": pin_access,
            "activity": activity_extraction,
            "workload_power": workload_power,
            "checks": checks,
        }
        print(json.dumps({"completed": spec.name, **summaries[spec.name]}, indent=2))

    summary_path = root / "submacro-build-manifest.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
