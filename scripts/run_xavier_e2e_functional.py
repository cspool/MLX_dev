#!/usr/bin/env python3
"""Compile and execute H173's dense end-to-end Xavier proxy."""

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
from typing import Any

import yaml

from scripts.audit_gpgpusim_rtx3090_proxy import parse_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/xavier_e2e_functional_v1.yaml"
GPGPUSIM_ROOT = (
    PROJECT_ROOT / "third_party/accel-sim-framework/gpu-simulator/gpgpu-sim"
)
CUDA_SHIM = PROJECT_ROOT / "third_party/envs/cuda-11.8-cuobjdump"


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def parse_e2e_summary(text: str) -> dict[str, Any] | None:
    match = re.search(r"^XAVIER_E2E_SUMMARY (\{.*\})$", text, re.MULTILINE)
    return json.loads(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    build_root = PROJECT_ROOT / "build/xavier-e2e-functional"
    build_root.mkdir(parents=True, exist_ok=True)
    binary = build_root / "xavier_e2e_proxy"
    source = PROJECT_ROOT / config["source_layout"]["cuda_source"]
    compile_result = subprocess.run(
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
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    compile_stdout = output_root / "compile-stdout.log"
    compile_stderr = output_root / "compile-stderr.log"
    compile_stdout.write_text(compile_result.stdout)
    compile_stderr.write_text(compile_result.stderr)
    if compile_result.returncode != 0:
        raise SystemExit(compile_result.stderr)
    records: dict[str, Any] = {}
    for tokens_value in config["model"]["token_counts"]:
        tokens = int(tokens_value)
        key = f"N{tokens}"
        run_root = output_root / f"runs/{key}"
        run_root.mkdir(parents=True, exist_ok=True)
        for frozen_name, target_name in (
            ("xavier_config", "gpgpusim.config"),
            ("xavier_interconnect", "config_volta_islip.icnt"),
        ):
            shutil.copy2(
                PROJECT_ROOT / config["frozen_inputs"][frozen_name]["path"],
                run_root / target_name,
            )
        log_path = run_root / "run.log"
        environment = os.environ.copy()
        environment["CUDA_INSTALL_PATH"] = str(CUDA_SHIM)
        environment["PTXAS_CUDA_INSTALL_PATH"] = str(CUDA_SHIM)
        environment["OPENCL_REMOTE_GPU_HOST"] = environment.get(
            "OPENCL_REMOTE_GPU_HOST", ""
        )
        environment["PTX_SIM_MODE_FUNC"] = "0"
        command = (
            f"source '{GPGPUSIM_ROOT / 'setup_environment'}' >/dev/null && "
            f"'{binary}' {tokens} {int(config['model']['layers'])}"
        )
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=run_root,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        text = result.stdout + result.stderr
        log_path.write_text(text)
        parsed = parse_run(log_path)
        summary = parse_e2e_summary(text)
        records[key] = {
            "tokens": tokens,
            "returncode": result.returncode,
            "log": digest(log_path),
            "summary": summary,
            "cycles": parsed["cycles"],
            "instructions": parsed["instructions"],
            "ctas": parsed["ctas"],
            "detailed_mode": parsed["detailed_mode"],
            "normal_exit": parsed["normal_exit"],
            "kernel_launches_observed": len(
                re.findall(r"^kernel_name =", text, re.MULTILINE)
            ),
            "pass": result.returncode == 0
            and summary is not None
            and parsed["cycles"] is not None
            and parsed["normal_exit"],
        }
        print(
            f"[H173] {key} cycles={parsed['cycles']} "
            f"kernels={records[key]['kernel_launches_observed']} "
            f"error={summary['maximum_absolute_error'] if summary else None}",
            flush=True,
        )
    checks = {
        "compile": compile_result.returncode == 0 and binary.is_file(),
        "run_count": len(records) == int(config["execution"]["expected_runs"]),
        "runs": all(item["pass"] for item in records.values()),
        "target_free": config["execution"]["paper_performance_targets_consumed"]
        is False,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_performance_targets_consumed": False,
        "source": digest(source),
        "binary": digest(binary),
        "compile_stdout": digest(compile_stdout),
        "compile_stderr": digest(compile_stderr),
        "records": records,
        "checks": checks,
    }
    path = output_root / "xavier-e2e-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
