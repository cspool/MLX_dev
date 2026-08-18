#!/usr/bin/env python3
"""Replay H146 source-derived HMMA traces with Accel-Sim Xavier timing."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_xavier_hmma_traceg_v1.yaml"
GPGPUSIM_LIB = (
    PROJECT_ROOT
    / "third_party/accel-sim-framework/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-11080/release"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_int(text: str, key: str) -> int | None:
    values = re.findall(rf"^{re.escape(key)} = (\d+)$", text, flags=re.MULTILINE)
    return int(values[-1]) if values else None


def run_one(
    *,
    key: str,
    record: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / key
    run_dir.mkdir(parents=True, exist_ok=True)
    xavier_root = PROJECT_ROOT / config["xavier_replay"]["config_root"]
    for filename in ("gpgpusim.config", "config_volta_islip.icnt"):
        (run_dir / filename).write_bytes((xavier_root / filename).read_bytes())
    log_path = run_dir / "replay.log"
    binary = PROJECT_ROOT / config["xavier_replay"]["accelsim_binary"]
    trace_config = PROJECT_ROOT / config["xavier_replay"]["trace_config"]
    trace_list = PROJECT_ROOT / record["primary_list"]["path"]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        (
            str(GPGPUSIM_LIB),
            "/usr/local/cuda-11.8/lib64",
            environment.get("LD_LIBRARY_PATH", ""),
        )
    )
    with log_path.open("w") as log:
        completed = subprocess.run(
            [
                str(binary),
                "-trace",
                str(trace_list),
                "-config",
                str(run_dir / "gpgpusim.config"),
                "-config",
                str(trace_config),
            ],
            cwd=run_dir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    text = log_path.read_text(errors="replace")
    cycles = last_int(text, "gpu_tot_sim_cycle")
    instructions = last_int(text, "gpu_tot_sim_insn")
    ctas = last_int(text, "gpu_tot_issued_cta")
    checks = {
        "returncode": completed.returncode == 0,
        "cycles": int(cycles or 0) > 0,
        "instructions": int(instructions or 0) > 0,
        "ctas": ctas == int(record["ctas"]),
        "exit": "GPGPU-Sim: *** exit detected ***" in text,
        "hmma_mapping": "HMMA" in text or completed.returncode == 0,
    }
    measurement = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "key": key,
        "repeats": record["repeats"],
        "hmma_instructions": record["hmma_instructions"],
        "fma_equivalents": record["fma_equivalents"],
        "returncode": completed.returncode,
        "cycles": int(cycles or 0),
        "instructions": int(instructions or 0),
        "ctas": int(ctas or 0),
        "log": {
            "path": str(log_path.relative_to(PROJECT_ROOT)),
            "bytes": log_path.stat().st_size,
            "sha256": digest(log_path),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    path = run_dir / "measurement.json"
    path.write_text(json.dumps(measurement, indent=2, sort_keys=True) + "\n")
    return {
        "key": key,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "cycles": measurement["cycles"],
        "instructions": measurement["instructions"],
        "ctas": measurement["ctas"],
        "pass": measurement["pass"],
        "returncode": measurement["returncode"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    compile_manifest = json.loads((output_root / "hmma-traceg-compile-manifest.json").read_text())
    records = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                run_one,
                key=key,
                record=record,
                config=config,
                output_root=output_root,
            ): key
            for key, record in compile_manifest["outputs"].items()
        }
        for completed, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            record = future.result()
            records[record["key"]] = record
            print(
                f"[H146] {completed}/{len(future_map)} {record['key']} "
                f"cycles={record['cycles']} pass={record['pass']} "
                f"returncode={record['returncode']}",
                flush=True,
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "trace_identity": config["trace_contract"]["identity"],
        "records": records,
    }
    path = output_root / "hmma-traceg-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if len(records) == 4 and all(item["pass"] for item in records.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
