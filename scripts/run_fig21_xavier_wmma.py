#!/usr/bin/env python3
"""Compile and execute H144 WMMA repeat jobs on the Xavier model."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml
from run_xavier_matched_attention import parse_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_xavier_wmma_projection_v1.yaml"
GPGPUSIM_ROOT = PROJECT_ROOT / "third_party/accel-sim-framework/gpu-simulator/gpgpu-sim"
CUDA_SHIM = PROJECT_ROOT / "third_party/envs/cuda-11.8-cuobjdump"
NVCC = Path("/usr/local/cuda-11.8/bin/nvcc")
CUOBJDUMP = CUDA_SHIM / "bin/cuobjdump"
BUILD_ROOT = PROJECT_ROOT / "build/fig21-xavier-wmma"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path) -> tuple[Path, Path]:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    binary = BUILD_ROOT / "mlx_wmma_proxy"
    ptx = BUILD_ROOT / "mlx_wmma_proxy.compute_70.ptx"
    subprocess.run(
        [
            str(NVCC),
            "-ccbin=/usr/bin/g++-11",
            "-O3",
            "--cudart",
            "shared",
            "-gencode",
            "arch=compute_70,code=compute_70",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    result = subprocess.run(
        [str(CUOBJDUMP), "--dump-ptx", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    ptx.write_text(result.stdout)
    return binary, ptx


def execute(
    *,
    repeats: int,
    tiles: int,
    binary: Path,
    run_root: Path,
    xavier_config: Path,
    checksum_limit: float,
) -> dict[str, Any]:
    name = f"wmma-r{repeats}"
    run_dir = run_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("gpgpusim.config", "config_volta_islip.icnt"):
        (run_dir / filename).write_bytes((xavier_config / filename).read_bytes())
    setup_log = run_dir / "setup.log"
    run_log = run_dir / "run.log"
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_INSTALL_PATH": str(CUDA_SHIM),
            "PTXAS_CUDA_INSTALL_PATH": str(CUDA_SHIM),
            "OPENCL_REMOTE_GPU_HOST": environment.get("OPENCL_REMOTE_GPU_HOST", ""),
        }
    )
    command = (
        f'source "{GPGPUSIM_ROOT / "setup_environment"}" > "{setup_log}" && '
        f'"{binary}" wmma "{tiles}" "{repeats}" 0 > "{run_log}" 2>&1'
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=run_dir, env=environment, check=False)
    parsed = parse_run(run_log)
    summary = parsed["summary"] or {}
    expected_fma = tiles * repeats * 4096
    checks = {
        "operator": summary.get("operator") == "wmma",
        "tiles": summary.get("count") == tiles,
        "repeats": summary.get("parameter") == repeats,
        "parameter2": summary.get("parameter2") == 0,
        "work": summary.get("fma_equivalents") == expected_fma,
        "checksum": summary.get("relative_error", 1.0) <= checksum_limit,
        "cycles": int(parsed["cycles"] or 0) > 0,
        "instructions": int(parsed["instructions"] or 0) > 0,
        "ctas": int(parsed["ctas"] or 0) == tiles,
        "detailed": parsed["detailed_mode"],
        "exit": parsed["normal_exit"],
        "returncode": completed.returncode == 0,
    }
    measurement = {
        "schema_version": 1,
        "experiment_id": "H144",
        "name": name,
        "tiles": tiles,
        "repeats": repeats,
        "fma_equivalents": expected_fma,
        "run": parsed,
        "cycles": int(parsed["cycles"] or 0),
        "instructions": int(parsed["instructions"] or 0),
        "ctas": int(parsed["ctas"] or 0),
        "returncode": completed.returncode,
        "failure_stage": (
            "post_kernel_enqueue_crash"
            if completed.returncode != 0
            and "pushing kernel" in run_log.read_text(errors="replace")
            and parsed["cycles"] is None
            else None
        ),
        "checks": checks,
        "pass": all(checks.values()),
    }
    path = run_dir / "measurement.json"
    path.write_text(json.dumps(measurement, indent=2, sort_keys=True) + "\n")
    return {
        "name": name,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": digest(path),
        "cycles": measurement["cycles"],
        "fma_equivalents": expected_fma,
        "pass": measurement["pass"],
        "returncode": measurement["returncode"],
        "failure_stage": measurement["failure_stage"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only-repeat", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    source = PROJECT_ROOT / config["source_layout"]["cuda_source"]
    binary, ptx = build(source)
    repeats = [
        *config["wmma_workload"]["fit_repeats"],
        *config["wmma_workload"]["holdout_repeats"],
    ]
    if args.only_repeat is not None:
        repeats = [value for value in repeats if int(value) == args.only_repeat]
        if len(repeats) != 1:
            raise ValueError(f"unknown repeat: {args.only_repeat}")
    output_root = PROJECT_ROOT / config["output_root"]
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    records = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                execute,
                repeats=int(repeat),
                tiles=int(config["wmma_workload"]["tiles"]),
                binary=binary,
                run_root=run_root,
                xavier_config=PROJECT_ROOT / config["xavier"]["config_root"],
                checksum_limit=float(config["acceptance"]["checksum_relative_error_limit"]),
            ): int(repeat)
            for repeat in repeats
        }
        for completed, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            record = future.result()
            records[record["name"]] = record
            print(
                f"[H144] {completed}/{len(future_map)} {record['name']} "
                f"cycles={record['cycles']} pass={record['pass']}",
                flush=True,
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "binary": {
            "path": str(binary.relative_to(PROJECT_ROOT)),
            "bytes": binary.stat().st_size,
            "sha256": digest(binary),
        },
        "ptx": {
            "path": str(ptx.relative_to(PROJECT_ROOT)),
            "bytes": ptx.stat().st_size,
            "sha256": digest(ptx),
        },
        "records": records,
    }
    suffix = f"-r{args.only_repeat}" if args.only_repeat is not None else ""
    path = output_root / f"xavier-wmma-run-manifest{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    foundational_stop = (
        args.only_repeat == int(config["wmma_workload"]["fit_repeats"][0])
        and len(records) == 1
        and next(iter(records.values()))["failure_stage"] == "post_kernel_enqueue_crash"
    )
    if foundational_stop:
        manifest["stopping_rule_applied"] = config["acceptance"]["foundational_failure_stop"]
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return 0
    return (
        0
        if len(records) == len(repeats) and all(record["pass"] for record in records.values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
