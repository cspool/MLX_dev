#!/usr/bin/env python3
"""Build and execute H106 under debug, optimized, ASan and UBSan."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/historical_dpu_memory_v1.yaml"
OVERLAY_ROOT = (
    PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
)
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
DRIVER = EXT_ROOT / "historical_dpu_memory_driver.cc"
H105_DRIVER = EXT_ROOT / "dpu_contract_driver.cc"
LEGACY_DRIVER = EXT_ROOT / "mlx_overlay_driver.cc"

MODE_FLAGS = {
    "debug": ["-O0", "-g"],
    "optimized": ["-O3", "-DNDEBUG"],
    "asan": ["-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer"],
    "ubsan": ["-O1", "-g", "-fsanitize=undefined", "-fno-omit-frame-pointer"],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_h106(output: Path, mode: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            *MODE_FLAGS[mode],
            f"-I{OVERLAY_ROOT}",
            f"-I{EXT_ROOT}",
            "-I/usr/include/jsoncpp",
            str(OVERLAY_ROOT / "mlx_overlay.cc"),
            str(EXT_ROOT / "standalone_spad_adapter.cc"),
            str(EXT_ROOT / "historical_dpu_memory.cc"),
            str(DRIVER),
            "-ljsoncpp",
            "-o",
            str(output),
        ],
        check=True,
    )


def compile_overlay_driver(output: Path, mode: str, source: Path) -> None:
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            *MODE_FLAGS[mode],
            f"-I{OVERLAY_ROOT}",
            "-I/usr/include/jsoncpp",
            str(OVERLAY_ROOT / "mlx_overlay.cc"),
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
    overlay_config: Path,
    memory_config: Path,
    expected_failure: str | None,
    run_root: Path,
) -> dict[str, Any]:
    stem = f"{name}-r{replay}"
    summary_path = run_root / mode / f"{stem}-summary.json"
    overlay_trace_path = run_root / mode / f"{stem}-overlay.jsonl"
    memory_trace_path = run_root / mode / f"{stem}-memory.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if mode == "asan":
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
    if mode == "ubsan":
        environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    result = subprocess.run(
        [
            str(driver),
            "--overlay-config",
            str(overlay_config),
            "--memory-config",
            str(memory_config),
            "--summary",
            str(summary_path),
            "--overlay-trace",
            str(overlay_trace_path),
            "--memory-trace",
            str(memory_trace_path),
            "--max-cycles",
            "100000",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if expected_failure:
        return {
            "mode": mode,
            "scenario": name,
            "replay": replay,
            "returncode": result.returncode,
            "expected_failure": expected_failure,
            "stderr": result.stderr.strip(),
            "pass": result.returncode == 1 and expected_failure in result.stderr,
        }
    passed = (
        result.returncode == 0
        and summary_path.is_file()
        and overlay_trace_path.is_file()
        and memory_trace_path.is_file()
    )
    return {
        "mode": mode,
        "scenario": name,
        "replay": replay,
        "returncode": result.returncode,
        "expected_failure": None,
        "stderr": result.stderr.strip(),
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": sha(summary_path) if summary_path.is_file() else None,
        "summary": json.loads(summary_path.read_text()) if summary_path.is_file() else None,
        "overlay_trace_path": str(overlay_trace_path.relative_to(PROJECT_ROOT)),
        "overlay_trace_sha256": (
            sha(overlay_trace_path) if overlay_trace_path.is_file() else None
        ),
        "memory_trace_path": str(memory_trace_path.relative_to(PROJECT_ROOT)),
        "memory_trace_sha256": (
            sha(memory_trace_path) if memory_trace_path.is_file() else None
        ),
        "pass": passed,
    }


def run_legacy(build_root: Path, run_root: Path) -> dict[str, Any]:
    frozen = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/environment/h105/simict-dpu-run-manifest.json"
        ).read_text()
    )["legacy"]
    reports: dict[str, Any] = {}
    for mode in ("debug", "optimized"):
        binary = build_root / f"legacy-{mode}"
        compile_overlay_driver(binary, mode, LEGACY_DRIVER)
        report_path = run_root / "regressions/legacy" / f"{mode}-report.json"
        trace_path = run_root / "regressions/legacy" / f"{mode}-trace.jsonl"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [str(binary), "--report", str(report_path), "--trace", str(trace_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        reports[mode] = {
            "returncode": result.returncode,
            "report_path": str(report_path.relative_to(PROJECT_ROOT)),
            "report_sha256": sha(report_path),
            "trace_path": str(trace_path.relative_to(PROJECT_ROOT)),
            "trace_sha256": sha(trace_path),
            "frozen_report_exact": sha(report_path) == frozen[mode]["report_sha256"],
            "frozen_trace_exact": sha(trace_path) == frozen[mode]["trace_sha256"],
        }
        reports[mode]["pass"] = (
            result.returncode == 0
            and reports[mode]["frozen_report_exact"]
            and reports[mode]["frozen_trace_exact"]
        )
    reports["pass"] = all(reports[mode]["pass"] for mode in ("debug", "optimized"))
    return reports


def run_h105_semantics(build_root: Path, run_root: Path) -> dict[str, Any]:
    compile_manifest = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/environment/h105/simict-dpu-compile-manifest.json"
        ).read_text()
    )
    frozen = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/environment/h105/simict-dpu-run-manifest.json"
        ).read_text()
    )
    frozen_records = {
        item["scenario"]: item
        for item in frozen["records"]
        if item["mode"] == "debug" and item["replay"] == 1
    }
    reports: dict[str, Any] = {}
    binaries = {}
    for mode in ("debug", "optimized"):
        binary = build_root / f"h105-{mode}"
        compile_overlay_driver(binary, mode, H105_DRIVER)
        binaries[mode] = binary
        reports[mode] = {}
        for name, item in compile_manifest["outputs"].items():
            root = run_root / "regressions/h105" / mode
            root.mkdir(parents=True, exist_ok=True)
            summary_path = root / f"{name}-summary.json"
            trace_path = root / f"{name}-trace.jsonl"
            result = subprocess.run(
                [
                    str(binary),
                    "--config",
                    str(PROJECT_ROOT / item["artifact"]["path"]),
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
            )
            if item["expected_failure"]:
                passed = (
                    result.returncode == 1
                    and item["expected_failure"] in result.stderr
                )
                reports[mode][name] = {
                    "returncode": result.returncode,
                    "expected_failure": item["expected_failure"],
                    "pass": passed,
                }
            else:
                reference = frozen_records[name]
                passed = (
                    result.returncode == 0
                    and json.loads(summary_path.read_text()) == reference["summary"]
                    and sha(trace_path) == reference["trace_sha256"]
                )
                reports[mode][name] = {
                    "returncode": result.returncode,
                    "summary_exact": json.loads(summary_path.read_text())
                    == reference["summary"],
                    "trace_exact": sha(trace_path) == reference["trace_sha256"],
                    "pass": passed,
                }

    h52_reference_summary = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/environment/h52/runs/standalone/debug-summary.json"
        ).read_text()
    )
    h52_reference_trace = [
        json.loads(line)
        for line in (
            PROJECT_ROOT
            / "artifacts/environment/h52/runs/standalone/debug-trace.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    reports["h52"] = {}
    for mode, binary in binaries.items():
        root = run_root / "regressions/h52"
        root.mkdir(parents=True, exist_ok=True)
        summary_path = root / f"{mode}-summary.json"
        trace_path = root / f"{mode}-trace.jsonl"
        result = subprocess.run(
            [
                str(binary),
                "--config",
                str(PROJECT_ROOT / "artifacts/environment/h52/mlx-full-block-fixed.json"),
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
        summary = json.loads(summary_path.read_text())
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        for event in trace:
            event["scenario"] = "json_driver"
        common_exact = all(
            summary.get(key) == value
            for key, value in h52_reference_summary.items()
            if key != "scenario"
        )
        trace_exact = trace == h52_reference_trace
        reports["h52"][mode] = {
            "returncode": result.returncode,
            "summary_common_exact": common_exact,
            "trace_semantics_exact": trace_exact,
            "pass": result.returncode == 0 and common_exact and trace_exact,
        }
    reports["h105_pass"] = all(
        item["pass"]
        for mode in ("debug", "optimized")
        for item in reports[mode].values()
    )
    reports["h52_pass"] = all(
        reports["h52"][mode]["pass"] for mode in ("debug", "optimized")
    )
    reports["pass"] = reports["h105_pass"] and reports["h52_pass"]
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    compile_manifest = json.loads(
        (output_root / "historical-dpu-memory-compile-manifest.json").read_text()
    )
    build_root = PROJECT_ROOT / "build/historical-dpu-memory"
    run_root = output_root / "runs"
    drivers = {}
    for mode in MODE_FLAGS:
        driver = build_root / f"historical-dpu-memory-{mode}"
        compile_h106(driver, mode)
        drivers[mode] = driver
    records = []
    for mode in MODE_FLAGS:
        replays = 2 if mode in {"debug", "optimized"} else 1
        for name, item in compile_manifest["outputs"].items():
            for replay in range(1, replays + 1):
                records.append(
                    run_one(
                        driver=drivers[mode],
                        mode=mode,
                        name=name,
                        replay=replay,
                        overlay_config=PROJECT_ROOT / item["overlay"]["path"],
                        memory_config=PROJECT_ROOT / item["memory"]["path"],
                        expected_failure=item["expected_failure"],
                        run_root=run_root,
                    )
                )
    auxiliary_records = []
    for name, item in compile_manifest["auxiliary_outputs"].items():
        for mode in MODE_FLAGS:
            auxiliary_records.append(
                run_one(
                    driver=drivers[mode],
                    mode=mode,
                    name=name,
                    replay=1,
                    overlay_config=PROJECT_ROOT / item["overlay"]["path"],
                    memory_config=PROJECT_ROOT / item["memory"]["path"],
                    expected_failure=item["expected_failure"],
                    run_root=run_root / "auxiliary",
                )
            )
    by_key = {(r["mode"], r["scenario"], r["replay"]): r for r in records}
    replay_checks = {}
    cross_build_checks = {}
    for name, item in compile_manifest["outputs"].items():
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
                and by_key[(mode, name, 1)]["overlay_trace_sha256"]
                == by_key[(mode, name, 2)]["overlay_trace_sha256"]
                and by_key[(mode, name, 1)]["memory_trace_sha256"]
                == by_key[(mode, name, 2)]["memory_trace_sha256"]
                for mode in ("debug", "optimized")
            )
            cross_build_checks[name] = all(
                by_key[(mode, name, 1)]["summary"]
                == by_key[("debug", name, 1)]["summary"]
                and by_key[(mode, name, 1)]["overlay_trace_sha256"]
                == by_key[("debug", name, 1)]["overlay_trace_sha256"]
                and by_key[(mode, name, 1)]["memory_trace_sha256"]
                == by_key[("debug", name, 1)]["memory_trace_sha256"]
                for mode in MODE_FLAGS
            )
    legacy = run_legacy(build_root, run_root)
    h105 = run_h105_semantics(build_root, run_root)
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "records": records,
        "auxiliary_records": auxiliary_records,
        "replay_checks": replay_checks,
        "cross_build_checks": cross_build_checks,
        "legacy": legacy,
        "h105": h105,
        "checks": {
            "all_runs": all(item["pass"] for item in records),
            "all_replays": all(replay_checks.values()),
            "all_builds": all(cross_build_checks.values()),
            "invalid_relative_address": all(
                item["pass"] for item in auxiliary_records
            )
            and len(auxiliary_records) == len(MODE_FLAGS),
            "legacy": legacy["pass"],
            "h105_h52": h105["pass"],
        },
    }
    path = output_root / "historical-dpu-memory-run-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["checks"], indent=2))
    return 0 if all(manifest["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
