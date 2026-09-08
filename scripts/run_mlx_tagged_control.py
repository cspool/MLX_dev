#!/usr/bin/env python3
"""Build/test synthesizable PE control; external C++ services are NOT full RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from mlxsim.tagged_program import Program
from scripts.mlx_rtl_support import ROOT, digest, entry_gate, execute
from scripts.run_mlx_tagged import build_native
from scripts.run_mlx_tagged_fu import build_fu
from scripts.run_mlx_tagged_fu import fingerprints as fu_fingerprints

RTL = ROOT / "rtl/mlx/tagged/mlx_tagged_pe_control.sv"
DRIVER = ROOT / "tests/tagged_pe_control_driver.cc"
BUILD = ROOT / "build/mlx-tagged-control"
PROTOCOL_CASES = (
    "arbiter",
    "rf-data",
    "missing-code",
    "reserved",
    "bad-pipeline",
    "bad-opcode",
    "register-range",
    "template-range",
    "zero-trips",
    "duplicate-id",
    "epoch-mismatch",
    "rf-overlap",
    "rf-capacity",
    "config-busy",
    "stale-block",
    "stale-epoch",
    "stale-iteration",
    "stale-pc",
    "missing-writeback",
    "unsolicited-complete",
    "stale-writeback",
    "overwrite-mailbox",
)


def control_sources():
    return {
        **fu_fingerprints(),
        **{
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                RTL,
                DRIVER,
                Path(__file__).resolve(),
                ROOT / "tests/test_tagged_rtl_control.py",
            )
        },
    }


def build_control(lanes=4, cpp_fu_service=False):
    library = build_native().with_name("libmlx_tagged.a")
    fu = None if cpp_fu_service else build_fu(lanes)
    fu_key = "cpp_service" if fu is None else digest(fu.parent / "Vmlx_tagged_fu__ALL.a")
    return _build_control(
        lanes, cpp_fu_service, tuple(control_sources().items()), digest(library), fu_key
    )


@lru_cache(maxsize=6)
def _build_control(lanes, cpp_fu_service, sources, library_hash, fu_key):
    entry_gate()
    library = build_native().with_name("libmlx_tagged.a")
    directory = BUILD / f"simd{lanes}-{'cpp-fu' if cpp_fu_service else 'rtl-fu'}"
    directory.mkdir(parents=True, exist_ok=True)
    build_id = hashlib.sha256(
        json.dumps([lanes, cpp_fu_service, sources, library_hash, fu_key], sort_keys=True).encode()
    ).hexdigest()
    identity_header = directory / "control-build-id.h"
    contents = f'#define MLX_CONTROL_BUILD_ID "{build_id}"\n'
    if not identity_header.exists() or identity_header.read_text() != contents:
        identity_header.write_text(contents)
    cflags = (
        f"-std=c++17 -DMLX_CONTROL_LANES={lanes} -I{ROOT / 'simulator_ext/tagged'} "
        f"-include {identity_header}"
    )
    libraries = f"{library} -ljsoncpp"
    if not cpp_fu_service:
        fu = build_fu(lanes)
        cflags += f" -DMLX_RTL_FU -I{fu.parent}"
        libraries += f" {fu.parent / 'Vmlx_tagged_fu__ALL.a'}"
    execute(
        [
            "verilator",
            "--lint-only",
            "--top-module",
            "mlx_tagged_pe_control",
            f"-GLANES={lanes}",
            "-Wall",
            str(RTL),
        ],
        directory / "lint.log",
    )
    execute(
        [
            "verilator",
            "--cc",
            "--exe",
            "--build",
            "-j",
            "4",
            "-Wall",
            "--top-module",
            "mlx_tagged_pe_control",
            f"-GLANES={lanes}",
            "--Mdir",
            str(directory),
            "-CFLAGS",
            cflags,
            "-LDFLAGS",
            libraries,
            str(RTL),
            str(DRIVER),
        ],
        directory / "build.log",
    )
    current_fu_key = (
        "cpp_service"
        if cpp_fu_service
        else digest(build_fu(lanes).parent / "Vmlx_tagged_fu__ALL.a")
    )
    if (
        sources != tuple(control_sources().items())
        or library_hash != digest(library)
        or fu_key != current_fu_key
    ):
        raise RuntimeError("RTL/test-service sources changed during build")
    (directory / "build-identity.json").write_text(
        json.dumps(
            {
                "build_identity": build_id,
                "binary_sha256": digest(directory / "Vmlx_tagged_pe_control"),
                "fu_library_sha256": fu_key,
                "native_library_sha256": library_hash,
                "sources": dict(sources),
            },
            indent=2,
        )
        + "\n"
    )
    return directory / "Vmlx_tagged_pe_control"


def run_control(
    program,
    output,
    *,
    overlap=True,
    load_latency=3,
    compute_ii=1,
    memory_period=1,
    receive_period=1,
    cpp_fu_service=False,
):
    program.validate()
    binary = build_control(program.hardware.lanes, cpp_fu_service)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "program.json").write_text(json.dumps(program.to_dict(), indent=2) + "\n")
    (output / "image.hex").write_text(
        "".join(f"{a:x} {w:016x}\n" for a, w in sorted(program.image().items()))
    )
    before = control_sources()
    log = execute(
        [
            str(binary),
            str(output / "program.json"),
            str(output / "image.hex"),
            str(output / "control.json"),
            str(int(overlap)),
            str(load_latency),
            str(compute_ii),
            str(memory_period),
            str(receive_period),
        ],
        output / "run.log",
    )
    if "MLX_TAGGED_CONTROL_PASS" not in log:
        raise RuntimeError("missing control component pass marker")
    if any(digest(ROOT / p) != sha for p, sha in before.items()):
        raise RuntimeError("RTL control sources changed during execution")
    data = json.loads((output / "control.json").read_text())
    build_record = json.loads((binary.parent / "build-identity.json").read_text())
    if (
        data.get("build_identity") != build_record["build_identity"]
        or digest(binary) != build_record["binary_sha256"]
    ):
        raise RuntimeError("executed front-end/FU build differs from its provenance")
    data.update(
        program_sha256=program.digest(),
        sources=before,
        binary_sha256=digest(binary),
        scope=(
            "single-PE RTL control and physical RF, "
            + ("C++ test FU" if cpp_fu_service else "functional RTL FU")
            + ", C++ test SPM/loopback; execution-domain comparison, not full RTL/system timing"
        ),
    )
    (output / "result.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def structural_check(output):
    output.mkdir(parents=True, exist_ok=True)
    netlist = output / "control.json"
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
                f"read_verilog -sv {RTL}; hierarchy -check -top mlx_tagged_pe_control; proc; opt; "
                "memory_dff; memory_collect; opt; "
                f"check -assert; stat; write_json {netlist}"
            ),
        ],
        output / "yosys.log",
        timeout=300,
    )
    module = json.loads(netlist.read_text())["modules"]["mlx_tagged_pe_control"]
    cells = module["cells"]
    if any("latch" in cell["type"].lower() for cell in cells.values()):
        raise RuntimeError("unexpected latch in control RTL")

    def parameter(value):
        return int(value, 2) if isinstance(value, str) else int(value)

    sequential_bits = sum(
        parameter(cell["parameters"]["WIDTH"])
        for cell in cells.values()
        if cell["type"] in ("$dff", "$adff", "$dffe", "$adffe")
    )
    memories = [
        {
            key: parameter(cell["parameters"][key])
            for key in ("WIDTH", "SIZE", "RD_PORTS", "WR_PORTS")
        }
        for cell in cells.values()
        if cell["type"] in ("$mem", "$mem_v2")
    ]
    if memories != [{"WIDTH": 512, "SIZE": 16, "RD_PORTS": 3, "WR_PORTS": 1}]:
        raise RuntimeError("generic RF shape/ports differ from the declared SIMD32 shared RF")
    memory_bits = sum(memory["WIDTH"] * memory["SIZE"] for memory in memories)
    result = {
        "classification": "generic_control_netlist_not_ppa",
        "rtl_sha256": digest(RTL),
        "generic_cells": len(cells),
        "sequential_bits": sequential_bits,
        "memory_bits": memory_bits,
        "total_state_bits": sequential_bits + memory_bits,
        "memories": memories,
        "netlist_sha256": digest(netlist),
        "latches": 0,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/tagged/rtl-control")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument(
        "--cpp-fu-service", action="store_true", help="explicit old component-test baseline"
    )
    parser.add_argument("--protocol", action="store_true")
    parser.add_argument("--structural", action="store_true")
    parser.add_argument("--regression", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = control_sources()
    report = {
        "classification": "rtl_pe_frontend_component_not_M4_acceptance",
        "sources": sources,
        "not_certified": [
            "RTL SPM/NoC array",
            "Chipyard functional RTL",
            "PPA",
        ],
    }
    binary = build_control(cpp_fu_service=args.cpp_fu_service)
    if args.program:
        program = Program.from_dict(json.loads(args.program.read_text()))
        run_control(
            program,
            output / "execution",
            overlap=not args.serial,
            cpp_fu_service=args.cpp_fu_service,
        )
    if args.protocol:
        for kind in PROTOCOL_CASES:
            log = execute([str(binary), "--protocol", kind], output / "protocol" / f"{kind}.log")
            if f"MLX_TAGGED_CONTROL_PROTOCOL_PASS {kind}" not in log:
                raise RuntimeError(f"missing protocol evidence: {kind}")
    if args.structural:
        report["structural"] = structural_check(output / "structural")
    if args.regression:
        attempt = Path(tempfile.mkdtemp(prefix="regression-", dir=output))
        junit = attempt / "pytest.xml"
        execute(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_tagged_rtl_control.py",
                f"--junitxml={junit}",
                f"--basetemp={attempt / 'work'}",
            ],
            attempt / "pytest.log",
            timeout=300,
        )
        cases = list(ET.parse(junit).getroot().iter("testcase"))
        if not cases or any(
            any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
            for case in cases
        ):
            raise RuntimeError("missing, failed or skipped RTL component test")
        report["regression"] = {
            "attempt": str(attempt),
            "cases": len(cases),
            "test_names": [case.attrib["name"] for case in cases],
        }
        report["runs"] = {
            str(p.relative_to(attempt)): {
                key: value
                for key, value in json.loads(p.read_text()).items()
                if key
                in (
                    "cycles",
                    "configuration_cycles",
                    "issue",
                    "retired",
                    "overlap_pe_cycles",
                    "state_comparisons",
                    "program_sha256",
                )
            }
            for p in sorted((attempt / "work").rglob("result.json"))
        }
        report["evidence_sha256"] = {
            str(p.relative_to(attempt)): digest(p)
            for p in sorted(attempt.rglob("*"))
            if p.is_file()
        }
    if any(digest(ROOT / p) != sha for p, sha in sources.items()):
        raise RuntimeError("RTL component sources changed during checks")
    entry_gate()
    report["component_checks_passed"] = True
    (output / "component-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"RTL_COMPONENT_CHECKS_PASS {output / 'component-report.json'}")


if __name__ == "__main__":
    main()
