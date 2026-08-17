#!/usr/bin/env python3
"""Build and run H84 jobs with execution-driven Xavier GPGPU-Sim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/xavier_matched_attention_v1.yaml"
GPGPUSIM_ROOT = PROJECT_ROOT / "third_party/accel-sim-framework/gpu-simulator/gpgpu-sim"
CUDA_SHIM = PROJECT_ROOT / "third_party/envs/cuda-11.8-cuobjdump"
BUILD_ROOT = PROJECT_ROOT / "build/xavier-matched-attention"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_int(text: str, key: str) -> int | None:
    matches = re.findall(rf"^{re.escape(key)} = (\d+)$", text, flags=re.MULTILINE)
    return int(matches[-1]) if matches else None


def parse_run(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8", errors="replace")
    summaries = re.findall(
        r"^MLX_GPU_PROXY_SUMMARY (\{.*\})$", content, flags=re.MULTILINE
    )
    return {
        "log_path": str(path.relative_to(PROJECT_ROOT)),
        "log_sha256": digest(path),
        "summary": json.loads(summaries[-1]) if summaries else None,
        "cycles": _last_int(content, "gpu_tot_sim_cycle"),
        "instructions": _last_int(content, "gpu_tot_sim_insn"),
        "ctas": _last_int(content, "gpu_tot_issued_cta"),
        "detailed_mode": "GPGPU-Sim PTX: simulation mode 0" in content,
        "normal_exit": "GPGPU-Sim: *** exit detected ***" in content,
    }


def build_binary(source: Path) -> Path:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    binary = BUILD_ROOT / "mlx_attention_proxy"
    subprocess.run(
        [
            "/usr/local/cuda-11.8/bin/nvcc",
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
    return binary


def run_job(
    job: dict[str, Any],
    *,
    binary: Path,
    run_root: Path,
    xavier_config: Path,
    checksum_limit: float,
) -> dict[str, Any]:
    run_dir = run_root / job["name"]
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
        f'"{binary}" "{job["family"]}" "{job["count"]}" '
        f'"{job["parameter"]}" "{job["parameter2"]}" > "{run_log}" 2>&1'
    )
    subprocess.run(["bash", "-lc", command], cwd=run_dir, env=environment, check=True)
    parsed = parse_run(run_log)
    summary = parsed["summary"] or {}
    checks = {
        "operator": summary.get("operator") == job["family"],
        "count": summary.get("count") == job["count"],
        "parameter": summary.get("parameter") == job["parameter"],
        "parameter2": summary.get("parameter2") == job["parameter2"],
        "checksum": summary.get("relative_error", 1.0) <= checksum_limit,
        "cycles": int(parsed["cycles"] or 0) > 0,
        "instructions": int(parsed["instructions"] or 0) > 0,
        "ctas": int(parsed["ctas"] or 0) > 0,
        "detailed": parsed["detailed_mode"],
        "exit": parsed["normal_exit"],
    }
    measurement = {
        "schema_version": 1,
        "experiment_id": "H84",
        "job": job,
        "run": parsed,
        "cycles": int(parsed["cycles"] or 0),
        "instructions": int(parsed["instructions"] or 0),
        "ctas": int(parsed["ctas"] or 0),
        "checks": checks,
        "pass": all(checks.values()),
    }
    path = run_dir / "measurement.json"
    path.write_text(json.dumps(measurement, indent=2, sort_keys=True) + "\n")
    return {
        "name": job["name"],
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": digest(path),
        "cycles": measurement["cycles"],
        "pass": measurement["pass"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", help="run one manifest job name")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    manifest_path = output_root / "xavier-attention-compile-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = PROJECT_ROOT / config["source_layout"]["cuda_source"]
    binary = build_binary(source)
    jobs = manifest["jobs"]
    if args.only:
        jobs = [job for job in jobs if job["name"] == args.only]
        if len(jobs) != 1:
            raise ValueError(f"unknown job: {args.only}")
    run_root = output_root / "runs"
    records = {}
    for index, job in enumerate(jobs, start=1):
        print(f"[xavier-attention] {index}/{len(jobs)} {job['name']}", flush=True)
        record = run_job(
            job,
            binary=binary,
            run_root=run_root,
            xavier_config=PROJECT_ROOT / "artifacts/environment/h56/config",
            checksum_limit=float(config["checksum_relative_error_limit"]),
        )
        records[record["name"]] = record
        print(f"  cycles={record['cycles']} pass={record['pass']}", flush=True)
    run_manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "binary": {
            "path": str(binary.relative_to(PROJECT_ROOT)),
            "bytes": binary.stat().st_size,
            "sha256": digest(binary),
        },
        "records": records,
    }
    suffix = f"-{args.only}" if args.only else ""
    path = output_root / f"xavier-attention-run-manifest{suffix}.json"
    path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    return 0 if len(records) == len(jobs) and all(item["pass"] for item in records.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
