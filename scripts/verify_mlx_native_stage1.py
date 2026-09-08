#!/usr/bin/env python3
"""Re-run the native-stage gates and bind their evidence to current sources.

This certificate permits system-integration work; it never certifies Chipyard,
RTL, PPA, arbitrary source languages or paper-scale workloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from mlxsim.tagged_program import Program
from mlxsim.tagged_workloads import workload
from scripts.run_mlx_tagged import BUILD, NATIVE, ROOT, build_native

COVERAGE = {
    "T1": "test_native_t1_t2_t4_t12_overlap_with_unchanged_resources",
    "T2": "test_native_t1_t2_t4_t12_overlap_with_unchanged_resources",
    "T3": "test_native_t3_t5_t8_t9_transport_and_iteration_ownership",
    "T4": "test_native_t1_t2_t4_t12_overlap_with_unchanged_resources",
    "T5": "test_native_t3_t5_t8_t9_transport_and_iteration_ownership",
    "T6": "test_native_t6_t11_reject_program_abi_and_capacity_without_python_validation",
    "T7": "test_native_t7_slot_reuse_across_40_logical_blocks",
    "T8": "test_native_t3_t5_t8_t9_transport_and_iteration_ownership",
    "T9": "test_native_destination_credit_prevents_future_iteration_router_blocking",
    "T10": "test_native_fused_regions_spill_between_resource_limited_waves",
    "T11_component_only": "test_c_api_tick_pause_poll_and_independent_instances",
    "T12": "test_native_folded_bsmm_overlaps_real_layers_with_identical_resources",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints():
    paths = [
        *NATIVE.iterdir(),
        *ROOT.glob("src/mlxsim/tagged_*.py"),
        ROOT / "tests/test_tagged_scheduler.py",
        ROOT / "scripts/run_mlx_tagged.py",
        Path(__file__).resolve(),
    ]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/tagged/stage1-acceptance")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = build_native()
    before = fingerprints()
    records = []

    def execute(name, command):
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=300, check=False
        )
        log = output / f"{name}.log"
        log.write_text(result.stdout + result.stderr)
        records.append(
            {
                "name": name,
                "command": command,
                "returncode": result.returncode,
                "log": str(log),
                "log_sha256": sha(log),
            }
        )
        if result.returncode:
            raise RuntimeError(f"native stage gate failed: {log}")

    junit = output / "pytest.xml"
    execute(
        "pytest",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_tagged_scheduler.py",
            f"--junitxml={junit}",
        ],
    )
    execute("ctest", ["ctest", "--test-dir", str(BUILD), "--output-on-failure"])
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    if not cases or any(
        any(case.find(tag) is not None for tag in ("failure", "error", "skipped")) for case in cases
    ):
        raise RuntimeError("test failures or skips prevent native-stage acceptance")
    covered = {
        requirement: [
            case.attrib["name"] for case in cases if case.attrib["name"].split("[")[0] == name
        ]
        for requirement, name in COVERAGE.items()
    }
    if any(not names for names in covered.values()):
        raise RuntimeError("missing named native-stage behavioral coverage")
    additional = (
        "test_source_regions_preserve_layer_identity_and_coarse_contexts",
        "test_typed_mlir_frontend_preserves_source_math_and_layer_regions",
        "test_mlir_cse_respects_context_boundaries_and_output_aliases",
    )
    if any(
        not any(case.attrib["name"].split("[")[0] == name for case in cases) for name in additional
    ):
        raise RuntimeError("missing compiler/context/MLIR coverage")

    workloads = {}
    for name in ("bsmm", "fft_cmp", "swa", "transformer_block"):
        directory = output / name
        execute(
            name,
            [
                sys.executable,
                "-m",
                "scripts.run_mlx_tagged",
                "--workload",
                name,
                "--no-trace",
                "--output",
                str(directory),
            ],
        )
        data = json.loads((directory / "result.json").read_text())
        program = Program.from_dict(json.loads((directory / "native/program.json").read_text()))
        _, golden = workload(name)
        aliases = program.lineage["source_output_aliases"]
        for backend in ("native", "serial"):
            result = data[backend]
            if result["program_sha256"] != program.digest() or result["binary_sha256"] != sha(
                binary
            ):
                raise RuntimeError("workload did not execute the current native program/binary")
            if any(sha(ROOT / path) != digest for path, digest in result["sources"].items()):
                raise RuntimeError("native source changed after workload execution")
            actual = {
                value: tuple(result["outputs"][str(program.lineage["spm_values"][aliases[value]])])
                for value in golden
            }
            if actual != golden:
                raise RuntimeError("source-level operator golden mismatch")
            counters = result["counters"]
            for left, right in (
                ("issue", "complete"),
                ("admitted", "retired"),
                ("xfer_sent", "xfer_delivered"),
                ("registers_reserved", "registers_released"),
            ):
                if counters[left] != counters[right]:
                    raise RuntimeError(f"resource conservation failure: {name}, {left}")
        if data["native"]["counters"]["issue"] != data["serial"]["counters"]["issue"]:
            raise RuntimeError("serial/overlap comparison changed dynamic work")
        workloads[name] = {
            "program_sha256": program.digest(),
            "blocks": len(program.blocks),
            "arithmetic_operations": sum(program.lineage["operation_counts"].values()),
            "mlir": program.lineage["mlir"],
            "native_cycles": data["native"]["cycles"],
            "serial_cycles": data["serial"]["cycles"],
            "result_sha256": sha(directory / "result.json"),
        }
    if before != fingerprints():
        raise RuntimeError("sources changed while native-stage gates were running")
    certificate = {
        "status": "native_stage_passed",
        "stage": "M0_M1_M2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "registered bounded FP16 vector/region workloads and named native scheduling regressions",
        "source_fingerprints": before,
        "binary_sha256": sha(binary),
        "test_count": len(cases),
        "junit_sha256": sha(junit),
        "coverage": covered,
        "commands": records,
        "workloads": workloads,
        "next_stage": "M3_system_simulation_integration",
        "system_integration_verified": False,
        "rtl_verified": False,
        "limitations": [
            "finite vector MLIR dialect, not arbitrary C/PyTorch or general Linalg lowering",
            "bounded wave admission; no runtime PE migration or cross-wave overlap",
            "SWA scalar-feature fixture uses the first SIMD/4 lanes",
            "no claim about full-model or paper-scale performance",
        ],
    }
    (output / "certificate.json").write_text(json.dumps(certificate, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": certificate["status"],
                "tests": len(cases),
                "certificate": str(output / "certificate.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
