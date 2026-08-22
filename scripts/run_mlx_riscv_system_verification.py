#!/usr/bin/env python3
"""Run fresh repository verification for the MLX RISC-V system certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/system/mlx_riscv_system_goal_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def execute(
    command: list[str], *, environment: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess[str], float]:
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    return result, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = PROJECT_ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    project_paths = os.pathsep.join((str(PROJECT_ROOT), str(PROJECT_ROOT / "src")))
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        project_paths
        if not prior_pythonpath
        else f"{project_paths}{os.pathsep}{prior_pythonpath}"
    )

    records: dict[str, Any] = {}
    for name, command_key in (
        ("ruff", "ruff_command"),
        ("pytest", "pytest_command"),
        ("repository_pytest", "repository_pytest_command"),
        ("diff", "diff_command"),
    ):
        command = [str(item) for item in config["verification"][command_key]]
        result, seconds = execute(command, environment=environment)
        stdout = output / f"{name}-stdout.log"
        stderr = output / f"{name}-stderr.log"
        stdout.write_text(result.stdout)
        stderr.write_text(result.stderr)
        records[name] = {
            "command": command,
            "returncode": result.returncode,
            "seconds": seconds,
            "stdout": digest(stdout),
            "stderr": digest(stderr),
        }

    for name in ("pytest", "repository_pytest"):
        pytest_text = (
            (output / f"{name}-stdout.log").read_text()
            + "\n"
            + (output / f"{name}-stderr.log").read_text()
        )
        passed = re.findall(r"(\d+) passed", pytest_text)
        failed = re.findall(r"(\d+) failed", pytest_text)
        warnings = re.findall(r"(\d+) warnings", pytest_text)
        records[name]["counts"] = {
            "passed": int(passed[-1]) if passed else None,
            "failed": int(failed[-1]) if failed else 0,
            "warnings": int(warnings[-1]) if warnings else 0,
        }
        records[name]["failed_tests"] = sorted(
            set(re.findall(r"^FAILED\s+(\S+)", pytest_text, flags=re.MULTILINE))
        )

    counts = records["pytest"]["counts"]
    repository_counts = records["repository_pytest"]["counts"]
    allowed_failures = sorted(config["verification"]["allowed_repository_failures"])
    checks = {
        "ruff": records["ruff"]["returncode"] == 0,
        "pytest": records["pytest"]["returncode"] == 0,
        "pytest_summary": counts["passed"] is not None,
        "pytest_coverage": int(counts["passed"] or 0)
        >= int(config["verification"]["minimum_pytest_passed"]),
        "pytest_no_failures": counts["failed"] == 0,
        "repository_pytest_summary": repository_counts["passed"] is not None,
        "repository_pytest_coverage": int(repository_counts["passed"] or 0)
        >= int(config["verification"]["minimum_repository_pytest_passed"]),
        "repository_pytest_only_allowed_failures": records["repository_pytest"][
            "failed_tests"
        ]
        == allowed_failures
        and repository_counts["failed"] == len(allowed_failures),
        "repository_pytest_expected_returncode": records["repository_pytest"][
            "returncode"
        ]
        == (1 if allowed_failures else 0),
        "diff_check": records["diff"]["returncode"] == 0,
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": revision,
        "paper_performance_targets_consumed": False,
        **records,
        "checks": checks,
    }
    manifest_path = PROJECT_ROOT / config["verification_manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "checks": checks,
                "pytest": counts,
                "repository_pytest": repository_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
