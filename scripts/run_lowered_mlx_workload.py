#!/usr/bin/env python3
"""Execute every H187 lowered workload twice on its native simulator backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from mlxsim.schema import HardwareConfig, KernelProfile, StageSpec, Workload
from mlxsim.simulator import MLXSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/unified_workload_lowering_v1.yaml"
OVERLAY_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
BUILD_ROOT = PROJECT_ROOT / "build/unified-workload-lowering"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_drivers() -> dict[str, Path]:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    overlay = BUILD_ROOT / "mlx_overlay_json_driver"
    dpu = BUILD_ROOT / "historical_dpu_memory_driver"
    common = [
        "g++",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-O3",
        "-DNDEBUG",
        f"-I{OVERLAY_ROOT}",
        f"-I{EXT_ROOT}",
        "-I/usr/include/jsoncpp",
        str(OVERLAY_ROOT / "mlx_overlay.cc"),
        str(EXT_ROOT / "standalone_spad_adapter.cc"),
    ]
    subprocess.run(
        [*common, str(EXT_ROOT / "mlx_overlay_json_driver.cc"), "-ljsoncpp", "-o", str(overlay)],
        check=True,
    )
    subprocess.run(
        [
            *common,
            str(EXT_ROOT / "historical_dpu_memory.cc"),
            str(EXT_ROOT / "historical_dpu_memory_driver.cc"),
            "-ljsoncpp",
            "-o",
            str(dpu),
        ],
        check=True,
    )
    return {"overlay": overlay, "dpu": dpu}


def run_detailed(
    *,
    unit: dict[str, Any],
    replay: int,
    drivers: dict[str, Path],
    output_root: Path,
    max_cycles: int,
) -> dict[str, Any]:
    unit_root = output_root / "executions" / unit["unit_id"].replace(":", "_")
    unit_root.mkdir(parents=True, exist_ok=True)
    summary = unit_root / f"replay{replay}-summary.json"
    stdout = unit_root / f"replay{replay}-stdout.log"
    stderr = unit_root / f"replay{replay}-stderr.log"
    overlay_path = PROJECT_ROOT / unit["artifacts"]["overlay"]["primary"]["path"]
    if unit["execution_format"] == "mlx_overlay_json":
        command = [
            str(drivers["overlay"]),
            "--config",
            str(overlay_path),
            "--summary",
            str(summary),
            "--max-cycles",
            str(max_cycles),
        ]
    else:
        memory_path = PROJECT_ROOT / unit["artifacts"]["memory"]["primary"]["path"]
        command = [
            str(drivers["dpu"]),
            "--overlay-config",
            str(overlay_path),
            "--memory-config",
            str(memory_path),
            "--summary",
            str(summary),
            "--max-cycles",
            str(max_cycles),
        ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout.write_text(result.stdout)
    stderr.write_text(result.stderr)
    payload = json.loads(summary.read_text()) if summary.is_file() else None
    done = (
        payload.get("done") is True
        if unit["execution_format"] == "mlx_overlay_json" and payload
        else payload.get("overlay", {}).get("done") is True
        and payload.get("memory", {}).get("idle") is True
        if payload
        else False
    )
    return {
        "unit_id": unit["unit_id"],
        "execution_format": unit["execution_format"],
        "replay": replay,
        "returncode": result.returncode,
        "summary_path": str(summary.relative_to(PROJECT_ROOT)),
        "summary_sha256": sha256(summary) if summary.is_file() else None,
        "stdout_bytes": stdout.stat().st_size,
        "stderr_bytes": stderr.stat().st_size,
        "summary": payload,
        "pass": result.returncode == 0 and done and stderr.stat().st_size == 0,
    }


def run_analytical(
    *, unit: dict[str, Any], replay: int, hardware: HardwareConfig, output_root: Path
) -> dict[str, Any]:
    artifact_path = PROJECT_ROOT / unit["artifacts"]["profile"]["primary"]["path"]
    artifact = json.loads(artifact_path.read_text())
    workload = Workload(**artifact["workload"])
    profile_raw = artifact["profile"]
    profile = KernelProfile(
        operations=float(profile_raw["operations"]),
        offchip_bytes=float(profile_raw["offchip_bytes"]),
        output_elements=float(profile_raw["output_elements"]),
        stages=tuple(StageSpec(**stage) for stage in profile_raw["stages"]),
        metadata=profile_raw["metadata"],
    )
    simulator = MLXSimulator(hardware, trace_limit=64)
    result = simulator.simulate_profile(workload, profile).to_dict()
    payload = {
        "unit_id": unit["unit_id"],
        "execution_format": unit["execution_format"],
        "workload": workload.to_dict(),
        "profile_sha256": unit["artifacts"]["profile"]["primary"]["sha256"],
        "simulation": result,
    }
    unit_root = output_root / "executions" / unit["unit_id"].replace(":", "_")
    unit_root.mkdir(parents=True, exist_ok=True)
    summary = unit_root / f"replay{replay}-summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    finite = all(
        isinstance(result[field], (int, float)) and result[field] > 0
        for field in ("cycles", "latency_us", "operations", "offchip_bytes")
    )
    return {
        "unit_id": unit["unit_id"],
        "execution_format": unit["execution_format"],
        "replay": replay,
        "returncode": 0,
        "summary_path": str(summary.relative_to(PROJECT_ROOT)),
        "summary_sha256": sha256(summary),
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "summary": payload,
        "pass": finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    lowering = json.loads((PROJECT_ROOT / config["lowering_manifest"]).read_text())
    drivers = compile_drivers()
    hardware = HardwareConfig.from_yaml(
        PROJECT_ROOT / config["frozen_inputs"]["mlx_full_hardware"]["path"]
    )
    output_root = PROJECT_ROOT / config["output_root"]
    records: list[dict[str, Any]] = []
    for unit in lowering["units"]:
        for replay in range(1, int(config["toolchain_contract"]["required_replays_per_unit"]) + 1):
            if unit["execution_format"] == "analytical_kernel_profile_json":
                record = run_analytical(
                    unit=unit, replay=replay, hardware=hardware, output_root=output_root
                )
            else:
                record = run_detailed(
                    unit=unit,
                    replay=replay,
                    drivers=drivers,
                    output_root=output_root,
                    max_cycles=int(config["execution"]["detailed_max_cycles"]),
                )
            records.append(record)
            print(
                f"[H187] {record['unit_id']} replay={replay} pass={record['pass']}",
                flush=True,
            )
    replay_checks = {
        unit["unit_id"]: len(
            {
                record["summary_sha256"]
                for record in records
                if record["unit_id"] == unit["unit_id"]
            }
        )
        == 1
        for unit in lowering["units"]
    }
    checks = {
        "records": len(records) == int(config["toolchain_contract"]["required_executions"]),
        "passes": all(record["pass"] for record in records),
        "replays": all(replay_checks.values()),
        "units": len(replay_checks)
        == int(config["toolchain_contract"]["required_executable_units"]),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": True,
        "drivers": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in drivers.items()
        },
        "records": records,
        "replay_checks": replay_checks,
        "checks": checks,
    }
    path = PROJECT_ROOT / config["execution_manifest"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
