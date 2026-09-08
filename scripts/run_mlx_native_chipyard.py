#!/usr/bin/env python3
"""Build and run real RISC-V ELFs against the Chipyard-hosted C++ MLX model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from mlxsim.tagged_mlir import compile_mlir, graph_to_mlir
from mlxsim.tagged_program import Hardware
from mlxsim.tagged_simulator import execute_reference
from mlxsim.tagged_workloads import workload
from scripts.run_mlx_native_device import ROOT, build_device, configuration
from scripts.run_mlx_tagged import build_native, run_native

CHIPYARD_COMMIT = "b5d013190d637e634113cb5179f8c8885df1945a"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def git_read(directory, *arguments):
    return subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(directory), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout


def build_inputs(chipyard):
    """Fingerprint compiled native sources, dependencies and actual checkout changes."""
    if git_read(chipyard, "rev-parse", "HEAD").strip() != CHIPYARD_COMMIT:
        raise RuntimeError("native integration requires the pinned Chipyard revision")
    modules = git_read(chipyard, "submodule", "status", "--recursive")
    checkouts = {".": CHIPYARD_COMMIT}
    for line in modules.splitlines():
        if line.startswith(("+", " ")):
            revision, name, *_ = line[1:].split()
            checkouts[name] = revision
    states = {
        name: {
            "commit": revision,
            "tracked_diff_sha256": hashlib.sha256(
                git_read(chipyard / name, "diff", "--binary", "HEAD").encode()
            ).hexdigest(),
        }
        for name, revision in sorted(checkouts.items())
    }
    sources = [
        *ROOT.glob("simulator_ext/tagged/*"),
        *(
            ROOT / "system_sim/native" / name
            for name in (
                "device.cc",
                "device.h",
                "dpi.cc",
                "MLXNativeRoCC.sv",
                "CMakeLists.txt",
            )
        ),
        ROOT / "system_sim/chipyard/MLXNativeRoCC.scala",
        Path(__file__).resolve(),
        ROOT / "scripts/run_mlx_native_device.py",
    ]
    dependencies = [
        ROOT / "build/mlx-native-device/libmlx_native_device.a",
        ROOT / "build/mlx-native-device/tagged-core/libmlx_tagged.a",
        ROOT / "build/riscv-native/lib/libfesvr.a",
    ]
    installed = [
        chipyard / "generators/chipyard/src/main/scala/MLXNativeRoCC.scala",
        chipyard / "generators/chipyard/src/main/resources/vsrc/MLXNativeRoCC.sv",
    ]
    versions = {}
    for name, command in {
        "cxx": ["c++", "--version"],
        "verilator": ["verilator", "--version"],
        "java": ["java", "-version"],
        "riscv_gcc": ["riscv64-unknown-elf-gcc", "--version"],
    }.items():
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        versions[name] = (result.stdout + result.stderr).strip()
    return {
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources if p.is_file()},
        "libraries": {str(p.relative_to(ROOT)): digest(p) for p in dependencies},
        "installed_bridge": {str(p.relative_to(chipyard)): digest(p) for p in installed},
        "checkouts": states,
        "submodule_status": modules,
        "tools": versions,
    }


def validate_build(chipyard, simulator):
    path = chipyard / "native-model-build.json"
    if not path.is_file():
        raise RuntimeError("native simulator lacks build provenance; run --phase build")
    manifest = json.loads(path.read_text())
    if (
        manifest["inputs"] != build_inputs(chipyard)
        or manifest["build_identity"] != identity(manifest["inputs"])
        or manifest["simulator_sha256"] != digest(simulator)
    ):
        raise RuntimeError("native simulator build is stale or changed; run --phase build")
    return manifest


def execute(command, log, *, cwd=ROOT, env=None, timeout=1800):
    print(f"Running {log.name}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    result.stdout = log.read_text()
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {log}")
    return result


def prepare(name, output, reject=None):
    output.mkdir(parents=True, exist_ok=True)
    folded = name == "bsmm_folded"
    graph, golden = workload("bsmm" if folded else name, iterations=4 if folded else 2)
    hardware = Hardware(rows=1, columns=2) if folded else Hardware()
    source = output / "source.mlir"
    source.write_text(graph_to_mlir(graph, hardware))
    program = compile_mlir(
        source, output / "mlir", build_native().with_name("mlx-mlir-front"), hardware
    )
    reference = execute_reference(program)
    aliases = program.lineage["source_output_aliases"]
    if any(
        reference[program.lineage["spm_values"][aliases[value]]] != vector
        for value, vector in golden.items()
    ):
        raise RuntimeError("source golden differs from compiled program")
    words = configuration(program)
    if reject == "magic":
        words[0] ^= 1
    elif reject == "route":
        address = next(a for a, w in words.items() if 0x1000 <= a < 0x1200 and w >> 60 == 8)
        words[address] ^= 1 << 33
    elif reject == "reserved":
        address = next(a for a in sorted(words) if 0x1000 <= a < 0x1200)
        words[address] |= 1 << 56

    def packed(vectors):
        return [
            sum(vector[base + lane] << (16 * lane) for lane in range(4))
            for vector in vectors
            for base in range(0, program.hardware.lanes, 4)
        ]

    inputs = packed([program.inputs[address] for address in sorted(program.inputs)])
    outputs = packed([reference[address] for address in sorted(program.outputs)])
    header = [
        "#include <stdint.h>",
        f"#define MLX_WORKLOAD_NAME {json.dumps(name)}",
        f"#define MLX_EXPECT_REJECT {int(reject is not None)}",
        f"#define MLX_INPUT_BEATS {len(inputs)}",
        f"#define MLX_OUTPUT_BEATS {len(outputs)}",
        f"#define MLX_CONFIG_WORDS {len(words)}",
        "static const struct { uint16_t address; uint64_t word; } mlx_configuration[] = {",
    ]
    header += [
        f"  {{0x{address:04x}, UINT64_C(0x{word:016x})}},"
        for address, word in sorted(words.items())
    ]
    header.append("};")
    for identifier, values in (("mlx_input", inputs), ("mlx_golden", outputs)):
        header.append(f"static const uint64_t {identifier}[] __attribute__((aligned(64))) = {{")
        header.extend(f"  UINT64_C(0x{word:016x})," for word in values)
        header.append("};")
    (output / "program.h").write_text("\n".join(header) + "\n")
    (output / "program.json").write_text(json.dumps(program.to_dict(), indent=2) + "\n")
    elf = output / "program.riscv"
    execute(
        [
            "riscv64-unknown-elf-gcc",
            "-march=rv64imac",
            "-mabi=lp64",
            "-mcmodel=medany",
            "-O2",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-common",
            "-nostdlib",
            "-nostartfiles",
            "-static",
            "-Wl,--no-relax",
            "-T",
            str(ROOT / "system_sim/native/link.ld"),
            "-I",
            str(output),
            '-DMLX_NATIVE_HEADER="program.h"',
            str(ROOT / "system_sim/native/start.S"),
            str(ROOT / "system_sim/native/host.c"),
            "-o",
            str(elf),
        ],
        output / "compile-elf.log",
        timeout=120,
    )
    return program, elf


def install(chipyard):
    result = subprocess.run(
        ["git", "-c", f"safe.directory={chipyard}", "-C", str(chipyard), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    )
    if result.stdout.strip() != CHIPYARD_COMMIT:
        raise RuntimeError("native integration requires the pinned Chipyard revision")
    resources = {
        ROOT / "system_sim/chipyard/MLXNativeRoCC.scala": chipyard
        / "generators/chipyard/src/main/scala/MLXNativeRoCC.scala",
        ROOT / "system_sim/native/MLXNativeRoCC.sv": chipyard
        / "generators/chipyard/src/main/resources/vsrc/MLXNativeRoCC.sv",
    }
    manifest = chipyard / "native-model-install.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    for source, target in resources.items():
        if (
            target.exists()
            and digest(target) != digest(source)
            and previous.get(str(target)) != digest(target)
        ):
            raise RuntimeError(f"installed bridge has unrecognized edits: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != digest(source):
            shutil.copyfile(source, target)
    # Existing compatibility patches replace retired snapshot dependencies.
    for directory, patch in (
        ("tools/chisel3", "chisel3-stable-deps.patch"),
        ("tools/treadle", "treadle-stable-firrtl.patch"),
        (".", "native-model-finalize.patch"),
    ):
        base = chipyard / directory
        command = ["git", "-c", f"safe.directory={base.resolve()}", "-C", str(base), "apply"]
        patch_path = str(ROOT / "patches/chipyard" / patch)
        forward = subprocess.run(
            [*command, "--check", patch_path], capture_output=True, check=False
        )
        if forward.returncode == 0:
            subprocess.run([*command, patch_path], check=True)
        elif subprocess.run(
            [*command, "--reverse", "--check", patch_path], capture_output=True, check=False
        ).returncode:
            raise RuntimeError(f"compatibility patch does not match: {patch}")
    manifest.write_text(
        json.dumps({str(target): digest(target) for target in resources.values()}, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chipyard", type=Path, default=ROOT / "build/chipyard-native")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/tagged/chipyard-native")
    parser.add_argument("--phase", choices=("prepare", "build", "run", "all"), default="all")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--reject", choices=("magic", "route", "reserved"))
    parser.add_argument(
        "--workloads", nargs="+", default=["bsmm", "fft_cmp", "swa", "transformer_block"]
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    chipyard = args.chipyard.resolve()
    gate = json.loads((ROOT / "artifacts/tagged/stage1-acceptance/certificate.json").read_text())
    if gate["status"] != "native_stage_passed" or any(
        digest(ROOT / name) != sha for name, sha in gate["source_fingerprints"].items()
    ):
        raise RuntimeError("native-stage gate is missing or stale")
    programs = {name: prepare(name, output / name, args.reject) for name in args.workloads}
    simulator = chipyard / "sims/verilator/simulator-chipyard-MLXNativeRocketConfig"
    if args.phase in ("build", "all"):
        install(chipyard)
        build_device()
        inputs = build_inputs(chipyard)
        build_id = identity(inputs)
        header = chipyard / "native-model-build-id.h"
        contents = f'#define MLX_NATIVE_BUILD_ID "{build_id}"\n'
        if not header.exists() or header.read_text() != contents:
            header.write_text(contents)
        sbt = (
            f"java -XX:ActiveProcessorCount=4 -Xmx6G -Dsbt.override.build.repos=true "
            f"-Dsbt.repository.config={ROOT / 'system_sim/native/sbt-repositories'} "
            f"-Dsbt.sourcemode=true -Dsbt.workspace={chipyard / 'tools'} "
            f"-jar {ROOT / 'build/toolchains/sbt/sbt-launch-1.4.9.jar'}"
        )
        native_libraries = (
            f"EXTRA_SIM_LDFLAGS={ROOT / 'build/mlx-native-device/libmlx_native_device.a'} "
            f"{ROOT / 'build/mlx-native-device/tagged-core/libmlx_tagged.a'} -ljsoncpp"
        )
        build_command = [
            "make",
            "-C",
            str(chipyard / "sims/verilator"),
            "CONFIG=MLXNativeRocketConfig",
            f"RISCV={ROOT / 'build/riscv-native'}",
            f"SBT={sbt}",
            f"EXTRA_SIM_SOURCES={ROOT / 'system_sim/native/dpi.cc'}",
            f"EXTRA_SIM_CXXFLAGS=-std=c++17 -I{ROOT / 'system_sim/native'} -I{ROOT / 'simulator_ext/tagged'} -include {header}",
            f"EXTRA_SIM_REQS={header} {ROOT / 'system_sim/native/dpi.cc'}",
            native_libraries,
            "-j4",
        ]
        execute(
            build_command,
            output / "build-chipyard.log",
            timeout=3600,
        )
        if inputs != build_inputs(chipyard):
            raise RuntimeError("native sources/dependencies changed during build")
        (chipyard / "native-model-build.json").write_text(
            json.dumps(
                {
                    "build_identity": build_id,
                    "inputs": inputs,
                    "simulator_sha256": digest(simulator),
                    "command": build_command,
                    "log_sha256": digest(output / "build-chipyard.log"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
    if args.phase in ("run", "all"):
        if not simulator.is_file():
            raise RuntimeError("native Chipyard simulator has not been built")
        build_manifest = validate_build(chipyard, simulator)
        records = {}
        for name, (program, elf) in programs.items():
            report = output / name / "device.json"
            result = execute(
                [str(simulator), "+max-cycles=20000000", str(elf)],
                output / name / "chipyard.log",
                env={
                    **os.environ,
                    "MLX_NATIVE_REPORT": str(report),
                    "MLX_NATIVE_TRACE": "1",
                    "MLX_NATIVE_SERIAL": "1" if args.serial else "0",
                },
                timeout=600,
            )
            marker = "MLX_NATIVE_ELF_REJECT_PASS" if args.reject else "MLX_NATIVE_ELF_PASS"
            match = re.search(rf"{marker} workload=\S+ (.+)", result.stdout)
            if not match or not report.is_file():
                raise RuntimeError(f"missing real ELF pass/device report for {name}")
            host = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", match[1])}
            device = json.loads(report.read_text())
            if device.get("build_identity") != build_manifest["build_identity"]:
                raise RuntimeError("running device does not match the recorded build")
            if args.reject:
                if (
                    not device["error"]
                    or not device["complete"]
                    or not host["error"]
                    or not (host["wait"] & 16)
                    or host["mismatches"]
                    or device["memory_requests"]
                    or device["kernel_cycles"]
                ):
                    raise RuntimeError("invalid ISA/image was not rejected before execution")
                records[name] = {
                    "expected_rejection": args.reject,
                    "host": host,
                    "device": device,
                    "elf_sha256": digest(elf),
                    "simulator_sha256": digest(simulator),
                }
                continue
            if device["error"] or not device["complete"] or host["error"] or host["mismatches"]:
                raise RuntimeError("system/native execution failed")
            if (
                host["system"] != device["system_cycles"]
                or host["kernel"] != device["kernel_cycles"]
            ):
                raise RuntimeError("ELF and native device counters disagree")
            reference = run_native(program, output / name / "standalone", overlap=not args.serial)
            if (
                device["kernel"]["cycles"] != reference["cycles"]
                or device["kernel"]["counters"] != reference["counters"]
            ):
                raise RuntimeError("system integration changed native scheduling/resource counts")
            if (
                device["kernel"]["events"] != reference["events"]
                or device["kernel"]["outputs"] != reference["outputs"]
            ):
                raise RuntimeError("system integration changed native events or outputs")
            expected_bytes = (
                (len(program.inputs) + len(program.outputs)) * program.hardware.lanes * 2
            )
            if (
                device["dma_bytes"] != expected_bytes
                or device["memory_requests"] != device["memory_responses"]
                or device["memory_requests"] * 8 != expected_bytes
            ):
                raise RuntimeError("system DMA byte/request conservation failed")
            if (
                device["system_cycles"] != device["dma_cycles"] + device["kernel_cycles"]
                or host["launch_wait"] < device["system_cycles"]
            ):
                raise RuntimeError("system/host cycle accounting failed")
            records[name] = {
                "host": host,
                "device": device,
                "program_sha256": program.digest(),
                "elf_sha256": digest(elf),
                "simulator_sha256": digest(simulator),
                "native_device_sources": {
                    str(path.relative_to(ROOT)): digest(path)
                    for path in (ROOT / "system_sim/native").iterdir()
                    if path.is_file()
                },
            }
        if validate_build(chipyard, simulator) != build_manifest:
            raise RuntimeError("simulator build changed during execution")
        (output / "results.json").write_text(
            json.dumps(
                {
                    "classification": "measured_chipyard_cpp_system",
                    "chipyard_commit": CHIPYARD_COMMIT,
                    "build": build_manifest,
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
