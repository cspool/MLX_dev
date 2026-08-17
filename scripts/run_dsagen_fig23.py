#!/usr/bin/env python3
"""Execute every Figure 23 config twice with the optimized MLX JSON driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "artifacts/environment/h46"
BUILD_ROOT = PROJECT_ROOT / "build/mlx-fig23"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
DRIVER_SOURCE = PROJECT_ROOT / "simulator_ext/dsagen/mlx_overlay_json_driver.cc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "runs")
    parser.add_argument("--experiment-id", default="H46")
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_driver() -> Path:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    output = BUILD_ROOT / "mlx_overlay_json_driver_opt"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DNDEBUG",
            "-O3",
            f"-I{SOURCE_ROOT}",
            "-I/usr/include/jsoncpp",
            str(SOURCE_ROOT / "mlx_overlay.cc"),
            str(DRIVER_SOURCE),
            "-ljsoncpp",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


def run(binary: Path, config: Path, summary: Path) -> dict:
    subprocess.run(
        [
            str(binary),
            "--config",
            str(config),
            "--summary",
            str(summary),
            "--max-cycles",
            "10000000",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return json.loads(summary.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    config_root = args.config_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compiler = json.loads(
        (config_root / "fig23-compile-manifest.json").read_text(encoding="utf-8")
    )
    binary = build_driver()
    runs: dict[str, dict] = {}
    checks: dict[str, bool] = {}
    for key, item in compiler["outputs"].items():
        print(f"[fig23] {key}", flush=True)
        config = config_root / item["path"]
        first_path = output_dir / f"{key}-first.json"
        second_path = output_dir / f"{key}-second.json"
        first = run(binary, config, first_path)
        second = run(binary, config, second_path)
        checks[f"{key}_replay"] = first == second
        checks[f"{key}_done"] = first.get("done") is True
        checks[f"{key}_instruction_count"] = first.get("instructions_completed") == item[
            "metadata"
        ]["instruction_count"]
        runs[key] = {
            "summary": first,
            "summary_sha256": digest(first_path),
            "replay_sha256": digest(second_path),
        }
    cycles: dict[str, dict[str, int]] = {}
    speedups: dict[str, list[float]] = {
        "simd32_4x4": [],
        "simd8_8x8": [],
        "simd32_8x8": [],
    }
    for sequence in (512, 1024, 2048, 4096, 8192):
        values = {
            name: int(runs[f"{sequence}-{name}"]["summary"]["cycles"])
            for name in ("baseline", *speedups)
        }
        cycles[str(sequence)] = values
        for name, series in speedups.items():
            series.append(values["baseline"] / values[name])
    checks["targets_consumed_by_runner"] = False
    manifest = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "driver": {"bytes": binary.stat().st_size, "sha256": digest(binary)},
        "runs": runs,
        "cycles": cycles,
        "speedups": speedups,
        "checks": checks,
    }
    path = output_dir.parent / "fig23-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cycles": cycles, "speedups": speedups}, indent=2))
    return 0 if all(value for key, value in checks.items() if key != "targets_consumed_by_runner") else 1


if __name__ == "__main__":
    raise SystemExit(main())
