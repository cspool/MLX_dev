#!/usr/bin/env python3
"""Execute every H118 workload twice optimized and under ASan/UBSan."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig22_coupled_workloads_v1.yaml"
OVERLAY_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
DRIVER = EXT_ROOT / "historical_dpu_memory_driver.cc"

MODE_FLAGS = {
    "optimized": ["-O3", "-DNDEBUG"],
    "asan": ["-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer"],
    "ubsan": ["-O1", "-g", "-fsanitize=undefined", "-fno-omit-frame-pointer"],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_driver(path: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            str(path),
        ],
        check=True,
    )


def compile_overlay_driver(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O3",
            "-DNDEBUG",
            f"-I{OVERLAY_ROOT}",
            "-I/usr/include/jsoncpp",
            str(OVERLAY_ROOT / "mlx_overlay.cc"),
            str(EXT_ROOT / "dpu_contract_driver.cc"),
            "-ljsoncpp",
            "-o",
            str(path),
        ],
        check=True,
    )


def frozen_record(
    manifest: dict[str, Any], *, scenario: str | None = None, run_key: str | None = None
) -> dict[str, Any]:
    matches = [
        item
        for item in manifest["records"]
        if item["mode"] == "optimized"
        and int(item["replay"]) == 1
        and (scenario is None or item.get("scenario") == scenario)
        and (run_key is None or item.get("run_key") == run_key)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen regression record, got {len(matches)}")
    return matches[0]


def overlay_regression(
    *,
    driver: Path,
    name: str,
    config_path: Path,
    reference: dict[str, Any],
    output_root: Path,
    expected_failure: str | None = None,
) -> dict[str, Any]:
    root = output_root / "regressions" / name
    root.mkdir(parents=True, exist_ok=True)
    summary = root / "summary.json"
    trace = root / "trace.jsonl"
    result = subprocess.run(
        [
            str(driver),
            "--config",
            str(config_path),
            "--summary",
            str(summary),
            "--trace",
            str(trace),
            "--max-cycles",
            "1000000",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if expected_failure is not None:
        checks = {
            "returncode": result.returncode != 0,
            "expected_failure": expected_failure in result.stderr,
            "no_summary": not summary.exists(),
        }
    else:
        checks = {
            "returncode": result.returncode == 0 and not result.stderr,
            "summary": summary.is_file()
            and sha(summary) == reference["summary_sha256"],
            "trace": trace.is_file() and sha(trace) == reference["trace_sha256"],
        }
    return {
        "summary_path": str(summary.relative_to(PROJECT_ROOT)),
        "trace_path": str(trace.relative_to(PROJECT_ROOT)),
        "stderr": result.stderr.strip(),
        "checks": checks,
        "pass": all(checks.values()),
    }


def memory_regression(
    *,
    driver: Path,
    name: str,
    overlay_path: Path,
    memory_path: Path,
    reference: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    root = output_root / "regressions" / name
    root.mkdir(parents=True, exist_ok=True)
    summary = root / "summary.json"
    result = subprocess.run(
        [
            str(driver),
            "--overlay-config",
            str(overlay_path),
            "--memory-config",
            str(memory_path),
            "--summary",
            str(summary),
            "--max-cycles",
            "500000000",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    checks = {
        "returncode": result.returncode == 0 and not result.stderr,
        "summary": summary.is_file()
        and sha(summary) == reference["summary_sha256"],
    }
    return {
        "summary_path": str(summary.relative_to(PROJECT_ROOT)),
        "stderr": result.stderr.strip(),
        "checks": checks,
        "pass": all(checks.values()),
    }


def run_one(task: tuple[Any, ...]) -> dict[str, Any]:
    key, mode, replay, driver, overlay, memory, summary, max_cycles = task
    summary.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if mode == "asan":
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
    if mode == "ubsan":
        environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    result = subprocess.run(
        [
            str(driver),
            "--overlay-config",
            str(overlay),
            "--memory-config",
            str(memory),
            "--summary",
            str(summary),
            "--max-cycles",
            str(max_cycles),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    passed = result.returncode == 0 and not result.stderr and summary.is_file()
    return {
        "key": key,
        "mode": mode,
        "replay": replay,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "summary_path": str(summary.relative_to(PROJECT_ROOT)),
        "summary_sha256": sha(summary) if summary.is_file() else None,
        "summary": json.loads(summary.read_text()) if summary.is_file() else None,
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    compiled = json.loads(
        (output_root / "fig22-coupled-compile-manifest.json").read_text()
    )
    build_root = PROJECT_ROOT / "build/fig22-coupled-workloads"
    drivers: dict[str, Path] = {}
    for mode in MODE_FLAGS:
        driver = build_root / f"fig22-coupled-{mode}"
        compile_driver(driver, mode)
        drivers[mode] = driver
    overlay_driver = build_root / "fig22-coupled-overlay-optimized"
    compile_overlay_driver(overlay_driver)

    records: list[dict[str, Any]] = []
    tasks: list[tuple[Any, ...]] = []
    specifications = [
        ("optimized", 1),
        ("optimized", 2),
        ("asan", 1),
        ("ubsan", 1),
    ]
    for key, item in compiled["outputs"].items():
        for mode, replay in specifications:
            summary = output_root / f"runs/{mode}/{key}-r{replay}.json"
            if args.resume and summary.is_file():
                payload = json.loads(summary.read_text())
                if payload["overlay"]["done"] and payload["memory"]["idle"]:
                    records.append(
                        {
                            "key": key,
                            "mode": mode,
                            "replay": replay,
                            "returncode": 0,
                            "stderr": "",
                            "summary_path": str(summary.relative_to(PROJECT_ROOT)),
                            "summary_sha256": sha(summary),
                            "summary": payload,
                            "pass": True,
                        }
                    )
                    continue
            tasks.append(
                (
                    key,
                    mode,
                    replay,
                    drivers[mode],
                    PROJECT_ROOT / item["overlay"]["path"],
                    PROJECT_ROOT / item["memory"]["path"],
                    summary,
                    int(config["execution"]["max_cycles"]),
                )
            )
    workers = args.workers or int(config["execution"]["workers"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records.append(record)
            if completed % 16 == 0 or completed == len(tasks):
                cycles = (
                    record["summary"]["end_to_end_cycles"]
                    if record["summary"]
                    else None
                )
                print(
                    f"[H118] {completed}/{len(tasks)} {record['key']} "
                    f"{record['mode']} cycles={cycles}",
                    flush=True,
                )
    records.sort(key=lambda item: (item["key"], item["mode"], item["replay"]))
    by_key = {
        (item["key"], item["mode"], int(item["replay"])): item
        for item in records
    }
    replay_checks = {
        key: by_key[(key, "optimized", 1)]["summary_sha256"]
        == by_key[(key, "optimized", 2)]["summary_sha256"]
        for key in compiled["outputs"]
    }
    sanitizer_checks = {
        key: all(
            by_key[(key, mode, 1)]["summary_sha256"]
            == by_key[(key, "optimized", 1)]["summary_sha256"]
            for mode in ("asan", "ubsan")
        )
        for key in compiled["outputs"]
    }
    regression_config = config["regressions"]
    h105_compile = json.loads(
        (PROJECT_ROOT / regression_config["h105"]["compile_manifest"]).read_text()
    )
    h105_run = json.loads(
        (PROJECT_ROOT / regression_config["h105"]["run_manifest"]).read_text()
    )
    h109_compile = json.loads(
        (PROJECT_ROOT / regression_config["h109"]["compile_manifest"]).read_text()
    )
    h109_run = json.loads(
        (PROJECT_ROOT / regression_config["h109"]["run_manifest"]).read_text()
    )
    regressions: dict[str, Any] = {}
    for name, scenario in (
        ("h105_valid", regression_config["h105"]["valid_scenario"]),
        ("h105_invalid", regression_config["h105"]["invalid_scenario"]),
    ):
        output = h105_compile["outputs"][scenario]
        regressions[name] = overlay_regression(
            driver=overlay_driver,
            name=name,
            config_path=PROJECT_ROOT / output["artifact"]["path"],
            reference=frozen_record(h105_run, scenario=scenario),
            output_root=output_root,
            expected_failure=output["expected_failure"],
        )
    h109_scenario = regression_config["h109"]["scenario"]
    regressions["h109"] = overlay_regression(
        driver=overlay_driver,
        name="h109",
        config_path=PROJECT_ROOT
        / h109_compile["outputs"][h109_scenario]["artifact"]["path"],
        reference=frozen_record(h109_run, scenario=h109_scenario),
        output_root=output_root,
    )
    for parent_name in ("h106", "h113"):
        specification = regression_config[parent_name]
        parent_compile = json.loads(
            (PROJECT_ROOT / specification["compile_manifest"]).read_text()
        )
        parent_run = json.loads(
            (PROJECT_ROOT / specification["run_manifest"]).read_text()
        )
        output = parent_compile["outputs"][specification["scenario"]]
        regressions[parent_name] = memory_regression(
            driver=drivers["optimized"],
            name=parent_name,
            overlay_path=PROJECT_ROOT / output["overlay"]["path"],
            memory_path=PROJECT_ROOT / output["memory"]["path"],
            reference=frozen_record(parent_run, scenario=specification["scenario"]),
            output_root=output_root,
        )
    h114_spec = regression_config["h114"]
    h114_compile = json.loads(
        (PROJECT_ROOT / h114_spec["compile_manifest"]).read_text()
    )
    h114_run = json.loads((PROJECT_ROOT / h114_spec["run_manifest"]).read_text())
    h114_output = h114_compile["outputs"][h114_spec["run_key"]]
    regressions["h114"] = memory_regression(
        driver=drivers["optimized"],
        name="h114",
        overlay_path=PROJECT_ROOT / h114_output["overlay"]["path"],
        memory_path=PROJECT_ROOT / h114_output["memory"]["path"],
        reference=frozen_record(h114_run, run_key=h114_spec["run_key"]),
        output_root=output_root,
    )
    checks = {
        "execution_count": len(records)
        == int(config["execution"]["required_executions"]),
        "all_runs": all(item["pass"] for item in records),
        "replays": all(replay_checks.values()),
        "sanitizers": all(sanitizer_checks.values()),
        "regressions": all(item["pass"] for item in regressions.values()),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "records": records,
        "replay_checks": replay_checks,
        "sanitizer_checks": sanitizer_checks,
        "regressions": regressions,
        "checks": checks,
    }
    path = output_root / "fig22-coupled-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
