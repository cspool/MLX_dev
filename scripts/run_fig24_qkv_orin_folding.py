#!/usr/bin/env python3
"""Run H124 QKV q-folds in the frozen detailed GPGPU-Sim Orin proxy."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_qkv_orin_folding_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def compile_binary(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    toolchain = config["toolchain"]
    subprocess.run(
        [
            toolchain["nvcc"],
            f"-ccbin={toolchain['host_compiler']}",
            "-O3",
            "--cudart",
            "shared",
            "-gencode",
            f"arch={toolchain['architecture']},code={toolchain['architecture']}",
            str(PROJECT_ROOT / config["frozen_inputs"]["cuda_source"]["path"]),
            "-o",
            str(path),
        ],
        check=True,
    )


def run_one(task: tuple[Any, ...]) -> dict[str, Any]:
    key, count, stages, scale, block, binary, root, config = task
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        PROJECT_ROOT / config["frozen_inputs"]["orin_config"]["path"],
        root / "gpgpusim.config",
    )
    shutil.copy2(
        PROJECT_ROOT / config["frozen_inputs"]["orin_interconnect"]["path"],
        root / "config_ampere_islip.icnt",
    )
    setup = PROJECT_ROOT / config["toolchain"]["gpgpusim_root"] / "setup_environment"
    cuda_shim = PROJECT_ROOT / config["toolchain"]["cuda_shim"]
    command = (
        f"source {shlex.quote(str(setup))} > setup.log && "
        f"{shlex.quote(str(binary))} {count} {stages} {block}"
    )
    environment = os.environ.copy()
    environment["CUDA_INSTALL_PATH"] = str(cuda_shim)
    environment["PTXAS_CUDA_INSTALL_PATH"] = str(cuda_shim)
    environment.setdefault("OPENCL_REMOTE_GPU_HOST", "")
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    log = root / "run.log"
    log.write_text(result.stdout + result.stderr)
    text = log.read_text(errors="replace")
    passed = (
        result.returncode == 0
        and "MLX_FIG24_SCHEDULE_SUMMARY" in text
        and re.search(r"^gpu_tot_sim_cycle = [1-9][0-9]*$", text, re.MULTILINE)
        and "GPGPU-Sim: *** exit detected ***" in text
    )
    return {
        "key": key,
        "stages": stages,
        "scale": scale,
        "count": count,
        "block_threads": block,
        "returncode": result.returncode,
        "artifact": digest(log),
        "pass": bool(passed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    if output_root.exists() and not args.resume:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    binary = PROJECT_ROOT / "build/fig24-qkv-orin-folding/mlx_fig24_schedule_witness"
    compile_binary(config, binary)
    binary_record = digest(binary)
    scales = [*config["folding"]["fit_scales"], *config["folding"]["holdout_scales"]]
    tasks = []
    records = []
    for template, specification in config["templates"].items():
        stages = int(specification["stages"])
        for scale_value in scales:
            scale = int(scale_value)
            key = f"{template}-q{scale}"
            root = output_root / "runs" / key
            log = root / "run.log"
            if args.resume and log.is_file():
                text = log.read_text(errors="replace")
                if "GPGPU-Sim: *** exit detected ***" in text:
                    records.append(
                        {
                            "key": key,
                            "stages": stages,
                            "scale": scale,
                            "count": int(config["folding"]["base_element_count"])
                            * scale,
                            "block_threads": int(config["folding"]["block_threads"]),
                            "returncode": 0,
                            "artifact": digest(log),
                            "pass": True,
                        }
                    )
                    continue
            tasks.append(
                (
                    key,
                    int(config["folding"]["base_element_count"]) * scale,
                    stages,
                    scale,
                    int(config["folding"]["block_threads"]),
                    binary,
                    root,
                    config,
                )
            )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records.append(record)
            print(
                f"[H124] {completed}/{len(tasks)} {record['key']} "
                f"pass={record['pass']}",
                flush=True,
            )
    records.sort(key=lambda item: item["key"])
    checks = {
        "records": len(records) == int(config["folding"]["required_runs"]),
        "runs": all(item["pass"] for item in records),
        "binary": binary_record["bytes"] > 0,
        "target_free": config["acceptance"]["targets_consumed"] is False,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "binary": binary_record,
        "records": records,
        "checks": checks,
    }
    path = output_root / "fig24-qkv-orin-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
