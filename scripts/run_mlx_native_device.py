#!/usr/bin/env python3
"""Prepare the binary device ABI and exercise C++ bus handshakes (not a CPU run)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from mlxsim.tagged_program import Program

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/mlx-native-device"


def configuration(program: Program) -> dict[int, int]:
    words = program.image()
    masks = [0] * 4
    for address in program.inputs:
        masks[address // 64] |= 1 << (address % 64)
    for address in program.outputs:
        masks[2 + address // 64] |= 1 << (address % 64)
    words.update({0x1E00 + index: value for index, value in enumerate(masks)})
    return words


def build_device() -> Path:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT / "system_sim/native"),
            "-B",
            str(BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run(
        ["cmake", "--build", str(BUILD), "-j4"], check=True, capture_output=True, timeout=120
    )
    return BUILD / "mlx-device-test"


def build_dpi() -> Path:
    build_device()
    directory = BUILD / "dpi"
    directory.mkdir(parents=True, exist_ok=True)
    cpp_flags = (
        f"-std=c++17 -DMLX_DPI_DRIVER -I{ROOT / 'system_sim/native'} "
        f"-I{ROOT / 'simulator_ext/tagged'}"
    )
    command = [
        "verilator",
        "--cc",
        "--exe",
        "--build",
        "-j",
        "4",
        "-Wno-fatal",
        "--top-module",
        "MLXNativeRoCCBlackBox",
        "--Mdir",
        str(directory),
        "-CFLAGS",
        cpp_flags,
        "-LDFLAGS",
        f"{BUILD / 'libmlx_native_device.a'} {BUILD / 'tagged-core/libmlx_tagged.a'} -ljsoncpp",
        str(ROOT / "system_sim/native/MLXNativeRoCC.sv"),
        str(ROOT / "system_sim/native/dpi.cc"),
        str(ROOT / "system_sim/native/protocol_driver.cc"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    (directory / "build.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"DPI build failed: {directory / 'build.log'}")
    return directory / "VMLXNativeRoCCBlackBox"


def run_device(program, output, *, delay=3, period=1, words=None, first_half=None, dpi=False):
    program.validate()
    binary = build_dpi() if dpi else build_device()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    words = configuration(program) if words is None else words
    (output / "image.hex").write_text(
        "".join(f"{a:x} {w:016x}\n" for a, w in sorted(words.items()))
    )
    (output / "program.json").write_text(json.dumps(program.to_dict(), indent=2) + "\n")
    command = [
        str(binary),
        str(output / "image.hex"),
        str(output / "program.json"),
        str(output / "result.json"),
        str(delay),
        str(period),
    ]
    if first_half is not None:
        command.append(str(first_half))
    environment = {
        **os.environ,
        "MLX_NATIVE_REPORT": str(output / "device.json"),
        "MLX_NATIVE_TRACE": "1",
        "MLX_NATIVE_SERIAL": "0",
    }
    process = subprocess.run(
        command, capture_output=True, text=True, timeout=120, check=False, env=environment
    )
    (output / "run.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f"device protocol driver failed: {process.stderr}")
    result = json.loads((output / "result.json").read_text())
    if dpi:
        device = json.loads((output / "device.json").read_text())
        result = {**device, **result, "host_kind": "dpi_signal_driver_not_riscv"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    program = Program.from_dict(json.loads(args.program.read_text()))
    result = run_device(program, args.output)
    print(
        json.dumps(
            {
                "error": result["error"],
                "system_cycles": result["system_cycles"],
                "classification": result["classification"],
            }
        )
    )


if __name__ == "__main__":
    main()
