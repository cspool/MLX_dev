#!/usr/bin/env python3
"""Build and run H105 DPU semantics under four compiler modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/simict_dpu_contract_v1.yaml"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
DRIVER = PROJECT_ROOT / "simulator_ext/dsagen/dpu_contract_driver.cc"
LEGACY_DRIVER = PROJECT_ROOT / "simulator_ext/dsagen/mlx_overlay_driver.cc"

MODE_FLAGS = {
    "debug": ["-O0", "-g"],
    "optimized": ["-O3", "-DNDEBUG"],
    "asan": ["-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer"],
    "ubsan": ["-O1", "-g", "-fsanitize=undefined", "-fno-omit-frame-pointer"],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_driver(output: Path, mode: str, source: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            *MODE_FLAGS[mode],
            f"-I{SOURCE_ROOT}",
            "-I/usr/include/jsoncpp",
            str(SOURCE_ROOT / "mlx_overlay.cc"),
            str(source),
            "-ljsoncpp",
            "-o",
            str(output),
        ],
        check=True,
    )


def run_one(
    *,
    driver: Path,
    mode: str,
    name: str,
    replay: int,
    config_path: Path,
    expected_failure: str | None,
    run_root: Path,
) -> dict[str, Any]:
    stem = f"{name}-r{replay}"
    summary_path = run_root / mode / f"{stem}-summary.json"
    trace_path = run_root / mode / f"{stem}-trace.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if mode == "asan":
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
    if mode == "ubsan":
        environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    result = subprocess.run(
        [
            str(driver),
            "--config",
            str(config_path),
            "--summary",
            str(summary_path),
            "--trace",
            str(trace_path),
            "--max-cycles",
            "100000",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if expected_failure:
        passed = result.returncode == 1 and expected_failure in result.stderr
        return {
            "mode": mode,
            "scenario": name,
            "replay": replay,
            "expected_failure": expected_failure,
            "returncode": result.returncode,
            "stderr": result.stderr.strip(),
            "pass": passed,
        }
    passed = result.returncode == 0 and summary_path.is_file() and trace_path.is_file()
    return {
        "mode": mode,
        "scenario": name,
        "replay": replay,
        "expected_failure": None,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": sha(summary_path) if summary_path.is_file() else None,
        "summary": json.loads(summary_path.read_text()) if summary_path.is_file() else None,
        "trace_path": str(trace_path.relative_to(PROJECT_ROOT)),
        "trace_sha256": sha(trace_path) if trace_path.is_file() else None,
        "pass": passed,
    }


def run_legacy(build_root: Path, run_root: Path) -> dict[str, Any]:
    reports = {}
    for mode in ("debug", "optimized"):
        driver = build_root / f"legacy-{mode}"
        compile_driver(driver, mode, LEGACY_DRIVER)
        report_path = run_root / "legacy" / f"{mode}-report.json"
        trace_path = run_root / "legacy" / f"{mode}-trace.jsonl"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [str(driver), "--report", str(report_path), "--trace", str(trace_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        reports[mode] = {
            "returncode": result.returncode,
            "report_path": str(report_path.relative_to(PROJECT_ROOT)),
            "report_sha256": sha(report_path),
            "report": json.loads(report_path.read_text()),
            "trace_path": str(trace_path.relative_to(PROJECT_ROOT)),
            "trace_sha256": sha(trace_path),
            "pass": result.returncode == 0,
        }
    reports["debug_optimized_equivalent"] = (
        reports["debug"]["report"] == reports["optimized"]["report"]
        and reports["debug"]["trace_sha256"] == reports["optimized"]["trace_sha256"]
    )
    return reports


def run_h52_regression(
    config: dict[str, Any], drivers: dict[str, Path], run_root: Path
) -> dict[str, Any]:
    specification = config["regressions"]["h52"]
    reference_summary = json.loads(
        (PROJECT_ROOT / specification["reference_summary"]).read_text(
            encoding="utf-8"
        )
    )
    reference_trace = [
        json.loads(line)
        for line in (PROJECT_ROOT / specification["reference_trace"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    reports: dict[str, Any] = {}
    for mode in ("debug", "optimized"):
        summary_path = run_root / "h52" / f"{mode}-summary.json"
        trace_path = run_root / "h52" / f"{mode}-trace.jsonl"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                str(drivers[mode]),
                "--config",
                str(PROJECT_ROOT / specification["config"]),
                "--summary",
                str(summary_path),
                "--trace",
                str(trace_path),
                "--max-cycles",
                "1000000",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trace = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        normalized_trace = []
        for event in trace:
            normalized = dict(event)
            normalized["scenario"] = reference_trace[0]["scenario"]
            normalized_trace.append(normalized)
        summary_common_exact = all(
            summary.get(key) == value
            for key, value in reference_summary.items()
            if key != "scenario"
        )
        trace_semantics_exact = normalized_trace == reference_trace
        dpu_fields_absent = not {
            "dpu_frfo_issues",
            "max_active_blocks_per_pe",
            "route_hops_by_plane",
        }.intersection(summary)
        reports[mode] = {
            "returncode": result.returncode,
            "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
            "summary_sha256": sha(summary_path),
            "trace_path": str(trace_path.relative_to(PROJECT_ROOT)),
            "trace_sha256": sha(trace_path),
            "summary_common_exact": summary_common_exact,
            "trace_semantics_exact": trace_semantics_exact,
            "dpu_fields_absent": dpu_fields_absent,
            "pass": result.returncode == 0
            and summary_common_exact
            and trace_semantics_exact
            and dpu_fields_absent,
        }
    reports["debug_optimized_byte_exact"] = (
        reports["debug"]["summary_sha256"]
        == reports["optimized"]["summary_sha256"]
        and reports["debug"]["trace_sha256"]
        == reports["optimized"]["trace_sha256"]
    )
    reports["pass"] = (
        reports["debug"]["pass"]
        and reports["optimized"]["pass"]
        and reports["debug_optimized_byte_exact"]
    )
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    compiler = json.loads(
        (output_root / "simict-dpu-compile-manifest.json").read_text(encoding="utf-8")
    )
    build_root = PROJECT_ROOT / "build/simict-dpu-contract"
    run_root = output_root / "runs"
    drivers = {}
    for mode in MODE_FLAGS:
        driver = build_root / f"dpu-contract-{mode}"
        compile_driver(driver, mode, DRIVER)
        drivers[mode] = driver
    records = []
    for mode in MODE_FLAGS:
        replay_count = 2 if mode in {"debug", "optimized"} else 1
        for name, item in compiler["outputs"].items():
            for replay in range(1, replay_count + 1):
                records.append(
                    run_one(
                        driver=drivers[mode],
                        mode=mode,
                        name=name,
                        replay=replay,
                        config_path=PROJECT_ROOT / item["artifact"]["path"],
                        expected_failure=item["expected_failure"],
                        run_root=run_root,
                    )
                )
    by_key = {(item["mode"], item["scenario"], item["replay"]): item for item in records}
    replay_checks = {}
    cross_build_checks = {}
    for name, item in compiler["outputs"].items():
        if item["expected_failure"]:
            replay_checks[name] = all(
                by_key[(mode, name, replay)]["pass"]
                for mode in ("debug", "optimized")
                for replay in (1, 2)
            )
            cross_build_checks[name] = all(
                by_key[(mode, name, 1)]["pass"] for mode in MODE_FLAGS
            )
        else:
            replay_checks[name] = all(
                by_key[(mode, name, 1)]["summary_sha256"]
                == by_key[(mode, name, 2)]["summary_sha256"]
                and by_key[(mode, name, 1)]["trace_sha256"]
                == by_key[(mode, name, 2)]["trace_sha256"]
                for mode in ("debug", "optimized")
            )
            cross_build_checks[name] = (
                by_key[("debug", name, 1)]["summary"]
                == by_key[("optimized", name, 1)]["summary"]
                and by_key[("debug", name, 1)]["trace_sha256"]
                == by_key[("optimized", name, 1)]["trace_sha256"]
                and by_key[("asan", name, 1)]["summary"]
                == by_key[("debug", name, 1)]["summary"]
                and by_key[("ubsan", name, 1)]["summary"]
                == by_key[("debug", name, 1)]["summary"]
            )
    legacy = run_legacy(build_root, run_root)
    h52 = run_h52_regression(config, drivers, run_root)
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "records": records,
        "replay_checks": replay_checks,
        "cross_build_checks": cross_build_checks,
        "legacy": legacy,
        "h52": h52,
        "checks": {
            "all_runs": all(item["pass"] for item in records),
            "all_replays": all(replay_checks.values()),
            "all_builds": all(cross_build_checks.values()),
            "legacy": all(legacy[mode]["pass"] for mode in ("debug", "optimized"))
            and legacy["debug_optimized_equivalent"],
            "h52": h52["pass"],
        },
    }
    path = output_root / "simict-dpu-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["checks"], indent=2))
    return 0 if all(manifest["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
