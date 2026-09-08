#!/usr/bin/env python3
"""Certify actual Rocket/DMA execution of the native MLX model, never RTL/PPA."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

from mlxsim.tagged_program import Program
from mlxsim.tagged_simulator import execute_reference
from scripts.run_mlx_native_chipyard import ROOT, digest, validate_build


def fingerprints():
    paths = [
        *ROOT.glob("system_sim/native/*"),
        ROOT / "system_sim/chipyard/MLXNativeRoCC.scala",
        ROOT / "scripts/run_mlx_native_device.py",
        ROOT / "scripts/run_mlx_native_chipyard.py",
        ROOT / "tests/test_mlx_native_device.py",
        ROOT / "tests/native_device_lifecycle.cc",
        ROOT / "tests/test_mlx_native_system_gate.py",
        ROOT / "docs/mlx-native-isa.md",
        ROOT / "artifacts/tagged/stage1-acceptance/certificate.json",
        Path(__file__).resolve(),
    ]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths if path.is_file()}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit_dma(program, device):
    """Independently replay the one-outstanding bus trace and check actual data."""
    golden = execute_reference(program)
    inputs = [v for a in sorted(program.inputs) for v in program.inputs[a]]
    outputs = [v for a in sorted(program.outputs) for v in golden[a]]

    def pack(values):
        return [
            sum(values[i + lane] << (16 * lane) for lane in range(4))
            for i in range(0, len(values), 4)
        ]

    expected_reads, expected_writes = pack(inputs), pack(outputs)
    read_data, write_data = [], []
    addresses = {"memory_read_request": [], "memory_write_request": []}
    pending = None
    responses = 0
    for event in device["events"]:
        kind = event["event"]
        if kind in addresses:
            require(pending is None, "more than one outstanding DMA request")
            pending = event
            addresses[kind].append(event["address"])
            if kind == "memory_write_request":
                write_data.append(event["data"])
        elif kind == "memory_response":
            require(pending is not None, "unsolicited DMA response")
            require(event["device_cycle"] > pending["device_cycle"], "zero-latency DMA response")
            if pending["event"] == "memory_read_request":
                read_data.append(event["data"])
            responses += 1
            pending = None
    require(pending is None, "DMA request not drained at completion")
    require(read_data == expected_reads, "DMA read data differs from host input")
    require(write_data == expected_writes, "DMA write data differs from computed output")
    for values in addresses.values():
        require(all(b == a + 8 for a, b in pairwise(values)), "noncontiguous DMA layout")
    beats = len(expected_reads) + len(expected_writes)
    require(
        responses == beats == device["memory_requests"] == device["memory_responses"],
        "DMA request/response conservation failed",
    )
    require(device["dma_bytes"] == 8 * beats, "DMA byte conservation failed")


def audit_run(directory, build, *, rejection=None):
    data = json.loads((directory / "results.json").read_text())
    require(data["classification"] == "measured_chipyard_cpp_system", "wrong evidence backend")
    require(data["build"] == build, "run used a different simulator build")
    for name, record in data["records"].items():
        folder = directory / name
        device, host = record["device"], record["host"]
        require(digest(folder / "program.riscv") == record["elf_sha256"], "ELF changed after run")
        require(record["simulator_sha256"] == build["simulator_sha256"], "wrong simulator binary")
        require(device["build_identity"] == build["build_identity"], "wrong compiled identity")
        require(device["complete"] and (host["wait"] & 4), "wait returned before completion")
        require(host["mismatches"] == 0, "ELF observed incorrect output")
        log = (folder / "chipyard.log").read_text()
        marker = "MLX_NATIVE_ELF_REJECT_PASS" if rejection else "MLX_NATIVE_ELF_PASS"
        require(f"{marker} workload={name} " in log, "missing actual ELF pass marker")
        if rejection:
            require(record["expected_rejection"] == rejection, "wrong rejected image")
            require(device["error"] and host["error"] and (host["wait"] & 16), "error not visible")
            require(
                device["memory_requests"] == device["kernel_cycles"] == device["dma_bytes"] == 0,
                "invalid image caused execution",
            )
            continue
        require(device["error"] == host["error"] == 0, "execution error")
        program = Program.from_dict(json.loads((folder / "program.json").read_text()))
        require(program.digest() == record["program_sha256"], "program changed after execution")
        native = json.loads((folder / "standalone/native.json").read_text())
        kernel = device["kernel"]
        for key in ("cycles", "counters", "events", "outputs"):
            require(kernel[key] == native[key], f"integration changed kernel {key}")
        require(bool(kernel["events"]), "missing kernel event trace")
        counters = kernel["counters"]
        for a, b in (
            ("issue", "complete"),
            ("admitted", "retired"),
            ("xfer_sent", "xfer_delivered"),
            ("registers_reserved", "registers_released"),
        ):
            require(counters[a] == counters[b], f"resource conservation failed: {a}/{b}")
        for key, target in (
            ("system", "system_cycles"),
            ("dma", "dma_cycles"),
            ("kernel", "kernel_cycles"),
            ("bytes", "dma_bytes"),
        ):
            require(host[key] == device[target], f"host/device {key} counters differ")
        require(host["instructions"] == counters["issue"], "instruction status differs")
        require(host["overlap"] == counters.get("overlap_pe_cycles", 0), "overlap status differs")
        require(host["config"] >= device["config_commands"], "invalid host configuration interval")
        require(host["launch_wait"] >= device["system_cycles"], "invalid host launch/wait interval")
        require(
            device["system_cycles"] == device["dma_cycles"] + device["kernel_cycles"],
            "system busy cycle accounting failed",
        )
        audit_dma(program, device)
    return data["records"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/tagged/stage2-acceptance")
    parser.add_argument("--chipyard", type=Path, default=ROOT / "build/chipyard-native")
    parser.add_argument(
        "--skip-build", action="store_true", help="still require matching build provenance"
    )
    args = parser.parse_args()
    output, chipyard = args.output.resolve(), args.chipyard.resolve()
    output.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    certificate = output / "certificate.json"
    certificate.write_text(
        json.dumps({"status": "system_stage_running", "attempt": str(attempt)}) + "\n"
    )
    before = fingerprints()
    commands = []

    def execute(name, command, timeout=1200):
        print(f"stage2: {name}", flush=True)
        log = attempt / f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
        commands.append(
            {
                "name": name,
                "command": command,
                "returncode": result.returncode,
                "log": str(log),
                "log_sha256": digest(log),
            }
        )
        require(result.returncode == 0, f"system stage failed: {log}")

    runner = [sys.executable, "-m", "scripts.run_mlx_native_chipyard", "--chipyard", str(chipyard)]
    try:
        stage1 = json.loads(
            (ROOT / "artifacts/tagged/stage1-acceptance/certificate.json").read_text()
        )
        require(stage1["status"] == "native_stage_passed", "native-stage gate missing")
        require(
            all(digest(ROOT / path) == sha for path, sha in stage1["source_fingerprints"].items()),
            "native-stage gate is stale",
        )
        if not args.skip_build:
            execute(
                "build", [*runner, "--phase", "build", "--output", str(attempt / "build")], 3600
            )
        simulator = chipyard / "sims/verilator/simulator-chipyard-MLXNativeRocketConfig"
        build = validate_build(chipyard, simulator)
        junit = attempt / "pytest.xml"
        execute(
            "pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_mlx_native_device.py",
                "tests/test_mlx_native_system_gate.py",
                f"--junitxml={junit}",
            ],
        )
        cases = list(ET.parse(junit).getroot().iter("testcase"))
        require(
            bool(cases)
            and not any(
                any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
                for case in cases
            ),
            "missing, failed or skipped system regression",
        )
        named = {case.attrib["name"].split("[")[0] for case in cases}
        required = {
            "test_binary_loading_dma_and_same_native_kernel",
            "test_input_comes_from_dma_not_configuration_or_a_golden",
            "test_memory_backpressure_changes_dma_but_not_kernel_schedule",
            "test_invalid_image_is_rejected_before_dma_or_kernel",
            "test_dpi_signal_bridge_matches_direct_device_handshakes",
            "test_build_provenance_rejects_stale_inputs_and_changed_binary",
            "test_dma_audit_rejects_missing_or_corrupted_transactions",
            "test_native_device_relaunch_error_recovery_and_reset",
        }
        require(required <= named, "missing named system-stage coverage")
        scenarios = {
            "workloads": ["--workloads", "bsmm", "fft_cmp", "swa", "transformer_block"],
            "folded": ["--workloads", "bsmm_folded"],
            "serial": ["--workloads", "bsmm_folded", "--serial"],
            **{
                f"reject-{kind}": ["--workloads", "bsmm", "--reject", kind]
                for kind in ("magic", "route", "reserved")
            },
        }
        results = {}
        for name, options in scenarios.items():
            directory = attempt / name
            execute(name, [*runner, "--phase", "run", "--output", str(directory), *options])
            rejection = name.removeprefix("reject-") if name.startswith("reject-") else None
            results[name] = audit_run(directory, build, rejection=rejection)
        require(
            set(results["workloads"]) == {"bsmm", "fft_cmp", "swa", "transformer_block"},
            "missing registered system workload",
        )
        overlap, serial = (results[name]["bsmm_folded"] for name in ("folded", "serial"))
        require(
            overlap["program_sha256"] == serial["program_sha256"],
            "comparison changed program/resources",
        )
        for field in ("instructions", "bytes", "dma"):
            require(overlap["host"][field] == serial["host"][field], f"comparison changed {field}")
        require(
            overlap["device"]["kernel"]["outputs"] == serial["device"]["kernel"]["outputs"],
            "comparison changed output",
        )
        require(
            overlap["host"]["kernel"] < serial["host"]["kernel"]
            and overlap["host"]["overlap"] > 0
            and serial["host"]["overlap"] == 0,
            "folded case did not demonstrate controlled scheduling overlap",
        )
        require(fingerprints() == before, "system sources changed during acceptance")
        require(validate_build(chipyard, simulator) == build, "build changed during acceptance")
        evidence = {
            str(path.relative_to(attempt)): digest(path)
            for path in sorted(attempt.rglob("*"))
            if path.is_file()
        }
        certificate.write_text(
            json.dumps(
                {
                    "status": "system_stage_passed",
                    "stage": "M3",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "scope": "project_native_v2 bounded workloads, actual Rocket RoCC/DMA and native scheduling",
                    "classification": "measured_chipyard_cpp_system",
                    "attempt": str(attempt),
                    "source_fingerprints": before,
                    "build": build,
                    "pytest_cases": len(cases),
                    "commands": commands,
                    "evidence_sha256": evidence,
                    "coverage": {"T11_system": list(scenarios), "T12_system": ["folded", "serial"]},
                    "folded_comparison": {"overlap": overlap["host"], "serial": serial["host"]},
                    "not_certified": [
                        "functional scheduler/PE RTL",
                        "RTL PPA",
                        "complete unpublished paper ISA",
                        "arbitrary source languages",
                        "paper-scale performance",
                        "DMA/kernel overlap",
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"M3_SYSTEM_STAGE_PASS {certificate}", flush=True)
    except Exception as error:
        certificate.write_text(
            json.dumps(
                {
                    "status": "system_stage_failed",
                    "attempt": str(attempt),
                    "error": str(error),
                    "commands": commands,
                },
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
