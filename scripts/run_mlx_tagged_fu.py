#!/usr/bin/env python3
"""Build and verify functional FP16 RTL and bounded vector-FU timing."""

from __future__ import annotations

import argparse
import json
import re
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from functools import lru_cache
from pathlib import Path

from scripts.mlx_rtl_support import ROOT, digest, entry_gate, execute
from scripts.run_mlx_tagged import build_native

RTL = ROOT / "rtl/mlx/tagged"
BUILD = ROOT / "build/mlx-tagged-fu"
SOURCES = tuple(
    RTL / name
    for name in (
        "mlx_tagged_fu.sv",
        "mlx_tagged_fp16_lane.sv",
        "mlx_tagged_exp.sv",
        "mlx_fp16_functions.svh",
        "mlx_exp2_fraction.svh",
    )
)


def fingerprints():
    paths = (
        *SOURCES,
        ROOT / "tests/tagged_fp16_driver.cc",
        ROOT / "tests/tagged_fu_driver.cc",
        ROOT / "scripts/mlx_rtl_support.py",
        Path(__file__).resolve(),
    )
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def verify_constants():
    text = (RTL / "mlx_exp2_fraction.svh").read_text()
    values = {
        int(i): int(value, 16)
        for i, value in re.findall(r"6'd(\d+): exp2_fraction = 48'h([0-9a-f]+)", text)
    }
    with localcontext() as context:
        context.prec = 100
        expected = {
            i: int(
                (Decimal(2) ** (Decimal(i) / 64) * (Decimal(2) ** 40)).to_integral_value(
                    rounding=ROUND_HALF_EVEN
                )
            )
            for i in range(64)
        }
    if values != expected:
        raise RuntimeError("EXP coefficient table does not match its mathematical definition")


def build_fu(lanes=32):
    library = build_native().with_name("libmlx_tagged.a")
    key = tuple(fingerprints().items()) + (("native_library", digest(library)),)
    return _build("vector", lanes, key)


def build_scalar():
    library = build_native().with_name("libmlx_tagged.a")
    key = tuple(fingerprints().items()) + (("native_library", digest(library)),)
    return _build("scalar", 1, key)


@lru_cache(maxsize=6)
def _build(kind, lanes, key):
    entry_gate()
    verify_constants()
    library = build_native().with_name("libmlx_tagged.a")
    top = "mlx_tagged_fp16_lane" if kind == "scalar" else "mlx_tagged_fu"
    directory = BUILD / ("scalar" if kind == "scalar" else f"simd{lanes}")
    directory.mkdir(parents=True, exist_ok=True)
    sources = [
        str(p)
        for p in SOURCES
        if p.suffix == ".sv" and (kind != "scalar" or p.name != "mlx_tagged_fu.sv")
    ]
    parameters = [] if kind == "scalar" else [f"-GLANES={lanes}"]
    driver = (
        ROOT / "tests" / ("tagged_fp16_driver.cc" if kind == "scalar" else "tagged_fu_driver.cc")
    )
    common = ["-Wall", "--top-module", top, f"-I{RTL}", *parameters]
    execute(["verilator", "--lint-only", *common, *sources], directory / "lint.log")
    execute(
        [
            "verilator",
            "--cc",
            "--exe",
            "--build",
            "-j",
            "4",
            *common,
            "--Mdir",
            str(directory),
            "-CFLAGS",
            f"-std=c++17 -DMLX_FU_LANES={lanes} -I{ROOT / 'simulator_ext/tagged'}",
            "-LDFLAGS",
            f"{library} -ljsoncpp",
            *sources,
            str(driver),
        ],
        directory / "build.log",
        timeout=300,
    )
    if tuple(fingerprints().items()) + (("native_library", digest(library)),) != key:
        raise RuntimeError("FU sources changed during build")
    (directory / "sources.json").write_text(json.dumps(dict(key), indent=2) + "\n")
    return directory / f"V{top}"


def structural_check(output):
    output.mkdir(parents=True, exist_ok=True)
    netlist = output / "scalar-mapped.json"
    sources = " ".join(
        str(p) for p in SOURCES if p.suffix == ".sv" and p.name != "mlx_tagged_fu.sv"
    )
    execute(
        [
            "yosys",
            "-Q",
            "-T",
            "-q",
            "-l",
            str(output / "yosys-detail.log"),
            "-p",
            (
                f"read_verilog -sv -I{RTL} {sources}; hierarchy -check -top mlx_tagged_fp16_lane; "
                f"synth -top mlx_tagged_fp16_lane -noabc; check -assert; stat; write_json {netlist}"
            ),
        ],
        output / "yosys.log",
        timeout=600,
    )
    modules = json.loads(netlist.read_text())["modules"]
    types = {cell["type"] for module in modules.values() for cell in module["cells"].values()}
    if any("latch" in kind.lower() or kind in ("$div", "$mod", "$mul") for kind in types):
        raise RuntimeError("arithmetic/latches remain unmapped in scalar synthesis check")
    return {
        "classification": "bit_mapped_scalar_arithmetic_not_PPA",
        "modules": len(modules),
        "cells": sum(len(module["cells"]) for module in modules.values()),
        "cell_types": sorted(types),
        "netlist_sha256": digest(netlist),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/tagged/rtl-fu")
    parser.add_argument(
        "--phase", choices=("all", "numeric", "vector", "structural"), default="all"
    )
    parser.add_argument("--random-triples", type=int, default=200000)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entry_gate()
    before = fingerprints()
    verify_constants()
    report = {
        "classification": "functional_fu_rtl_not_M4_acceptance",
        "sources": before,
        "not_certified": ["full RTL SPM/NoC array", "Chipyard functional RTL", "PPA"],
    }
    if args.phase in ("all", "numeric"):
        print("FU: exhaustive EXP, anchor operands and random triples", flush=True)
        binary = build_scalar()
        log = execute(
            [str(binary), str(output / "numeric.json"), str(args.random_triples)],
            output / "numeric.log",
            timeout=300,
        )
        if "MLX_TAGGED_FP16_PASS" not in log:
            raise RuntimeError("missing scalar arithmetic pass marker")
        report["numeric"] = json.loads((output / "numeric.json").read_text())
        report["numeric"]["binary_sha256"] = digest(binary)
    if args.phase in ("all", "vector"):
        report["vector"] = {}
        for lanes in (4, 32):
            print(f"FU: SIMD{lanes} data, latency, II, backpressure and reset", flush=True)
            binary = build_fu(lanes)
            log = execute(
                [str(binary), str(output / f"simd{lanes}.json")],
                output / f"simd{lanes}.log",
                timeout=300,
            )
            if "MLX_TAGGED_FU_PASS" not in log:
                raise RuntimeError("missing vector FU pass marker")
            report["vector"][str(lanes)] = json.loads((output / f"simd{lanes}.json").read_text())
            report["vector"][str(lanes)]["binary_sha256"] = digest(binary)
    if args.phase in ("all", "structural"):
        print("FU: bit-level arithmetic synthesis", flush=True)
        report["structural"] = structural_check(output / "structural")
    if fingerprints() != before:
        raise RuntimeError("FU sources changed during verification")
    entry_gate()
    report["checks_passed"] = True
    (output / "fu-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"RTL_FU_CHECKS_PASS {output / 'fu-report.json'}", flush=True)


if __name__ == "__main__":
    main()
