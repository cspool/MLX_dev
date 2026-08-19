#!/usr/bin/env python3
"""Run fresh repository verification for the H188 final certificate."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/numerical_convergence_goal_certificate_v1.yaml"


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
        env=environment,
        check=False,
    )
    return result, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    ruff_command = [str(PROJECT_ROOT / config["verification"]["ruff_command"][0])]
    ruff_command.extend(config["verification"]["ruff_command"][1:])
    ruff, ruff_seconds = execute(ruff_command)
    ruff_stdout = output_root / "ruff-stdout.log"
    ruff_stderr = output_root / "ruff-stderr.log"
    ruff_stdout.write_text(ruff.stdout)
    ruff_stderr.write_text(ruff.stderr)
    pytest_command = [str(PROJECT_ROOT / config["verification"]["pytest_command"][0])]
    pytest_command.extend(config["verification"]["pytest_command"][1:])
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH", "")
    project_paths = os.pathsep.join((str(PROJECT_ROOT), str(PROJECT_ROOT / "src")))
    environment["PYTHONPATH"] = (
        project_paths if not old_pythonpath else f"{project_paths}{os.pathsep}{old_pythonpath}"
    )
    pytest, pytest_seconds = execute(pytest_command, environment=environment)
    pytest_stdout = output_root / "pytest-stdout.log"
    pytest_stderr = output_root / "pytest-stderr.log"
    pytest_stdout.write_text(pytest.stdout)
    pytest_stderr.write_text(pytest.stderr)
    combined = f"{pytest.stdout}\n{pytest.stderr}"
    match = re.search(
        r"(?P<passed>\d+) passed"
        r"(?:, (?P<failed>\d+) failed)?"
        r"(?:, (?P<warnings>\d+) warnings)? in ",
        combined,
    )
    counts = {
        "passed": int(match.group("passed")) if match else None,
        "failed": int(match.group("failed") or 0) if match else None,
        "warnings": int(match.group("warnings") or 0) if match else None,
    }
    expected = config["verification"]
    checks = {
        "ruff_returncode": ruff.returncode == 0,
        "ruff_message": "All checks passed" in ruff.stdout,
        "pytest_returncode": pytest.returncode == 0,
        "pytest_summary_parsed": match is not None,
        "pytest_passed": counts["passed"] == int(expected["expected_pytest_passed"]),
        "pytest_failed": counts["failed"] == int(expected["expected_pytest_failed"]),
        "pytest_warnings": counts["warnings"]
        == int(expected["expected_pytest_warnings"]),
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
        "ruff": {
            "command": config["verification"]["ruff_command"],
            "returncode": ruff.returncode,
            "seconds": ruff_seconds,
            "stdout": digest(ruff_stdout),
            "stderr": digest(ruff_stderr),
        },
        "pytest": {
            "command": config["verification"]["pytest_command"],
            "returncode": pytest.returncode,
            "seconds": pytest_seconds,
            "counts": counts,
            "stdout": digest(pytest_stdout),
            "stderr": digest(pytest_stderr),
        },
        "checks": checks,
    }
    path = PROJECT_ROOT / config["verification_manifest"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": checks, "pytest": counts}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
