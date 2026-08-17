"""Fig. 19 digitization and external FABNet simulator audit helpers."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIXEL_MANIFEST = PROJECT_ROOT / "artifacts/targets/fig19_digitization_pixels.yaml"
DEFAULT_FABNET_ROOT = PROJECT_ROOT / "third_party/butterfly-acc"
FABNET_REVISION = "d5e313605fed593c8765c70acbf78231cfab3e00"
LATENCY_PATTERN = re.compile(r"The overall latecy is:\s*([0-9.eE+-]+)")


def load_fig19_manifest(path: str | Path = PIXEL_MANIFEST) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _latency_from_y(y: float, axis: dict[str, float]) -> float:
    y_zero = axis["y_at_zero_ms"]
    y_twenty = axis["y_at_twenty_ms"]
    return (y_zero - y) * 20.0 / (y_zero - y_twenty)


def derive_fig19_targets(manifest: dict[str, Any]) -> dict[str, Any]:
    axis = manifest["axis"]
    bars = manifest["bars"]
    fabnet = [_latency_from_y(y, axis) for y in bars["fabnet_total_endpoint_y"]]
    mlx = [_latency_from_y(y, axis) for y in bars["mlx_total_endpoint_y"]]
    uncertainty_ms = axis["uncertainty_pixels"] * 20.0 / (
        axis["y_at_zero_ms"] - axis["y_at_twenty_ms"]
    )
    return {
        "sequence_lengths": bars["sequence_lengths"],
        "fabnet_total_latency_ms": fabnet,
        "mlx_total_latency_ms": mlx,
        "latency_uncertainty_ms": uncertainty_ms,
        "reported_speedup": bars["reported_speedup_annotations"],
    }


def audit_fig19_digitization(
    manifest: dict[str, Any], *, verify_source: bool = False
) -> dict[str, Any]:
    metadata = manifest["metadata"]
    source_check: dict[str, Any] = {}
    if verify_source:
        path = PROJECT_ROOT / metadata["source"]
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        source_check = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "expected_sha256": metadata["sha256"],
            "actual_sha256": actual_hash,
            "pass": actual_hash == metadata["sha256"],
        }

    targets = derive_fig19_targets(manifest)
    checks: list[dict[str, Any]] = []
    for length, fabnet, mlx, reported in zip(
        targets["sequence_lengths"],
        targets["fabnet_total_latency_ms"],
        targets["mlx_total_latency_ms"],
        targets["reported_speedup"],
        strict=True,
    ):
        derived = fabnet / mlx
        relative_error = abs(derived - reported) / reported
        checks.append(
            {
                "sequence_length": length,
                "derived_speedup": derived,
                "reported_speedup": reported,
                "relative_error": relative_error,
                "tolerance": 0.03,
                "pass": relative_error <= 0.03,
            }
        )

    source_pass = source_check.get("pass", True)
    return {
        "source_check": source_check,
        "targets": targets,
        "speedup_cross_checks": checks,
        "summary": {
            "source_hash_pass": source_pass,
            "max_speedup_relative_error": max(check["relative_error"] for check in checks),
            "all_speedup_cross_checks_pass": all(check["pass"] for check in checks),
            "pass": source_pass and all(check["pass"] for check in checks),
        },
    }


def parse_fabnet_latency(stdout: str) -> float:
    matches = LATENCY_PATTERN.findall(stdout)
    if len(matches) != 1:
        raise ValueError(f"expected one FABNet latency line, found {len(matches)}")
    return float(matches[0])


def fabnet_command(
    simulator_path: Path, sequence_length: int, *, python_executable: str = sys.executable
) -> list[str]:
    return [
        python_executable,
        str(simulator_path),
        "--num_len",
        str(sequence_length),
        "--version",
        "large",
        "--frequency",
        "200",
        "--efficiency",
        "0.85",
        "--fpga_board",
        "zcu128",
        "--offchip_mem",
        "hbm",
        "--parallesm_be",
        "40",
        "--head_dim",
        "32",
    ]


def inspect_fabnet_checkout(repo_root: str | Path = DEFAULT_FABNET_ROOT) -> dict[str, Any]:
    root = Path(repo_root)
    if not (root / ".git").is_dir():
        return {
            "path": str(root),
            "expected_revision": FABNET_REVISION,
            "actual_revision": None,
            "tracked_files_clean": None,
            "pass": False,
        }
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    worktree_clean = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet"], check=False
    ).returncode == 0
    index_clean = subprocess.run(
        ["git", "-C", str(root), "diff", "--cached", "--quiet"], check=False
    ).returncode == 0
    tracked_clean = worktree_clean and index_clean
    return {
        "path": str(root.relative_to(PROJECT_ROOT) if root.is_relative_to(PROJECT_ROOT) else root),
        "expected_revision": FABNET_REVISION,
        "actual_revision": revision,
        "tracked_files_clean": tracked_clean,
        "pass": revision == FABNET_REVISION and tracked_clean,
    }


def run_fabnet_simulator(
    sequence_lengths: Sequence[int],
    *,
    repo_root: str | Path = DEFAULT_FABNET_ROOT,
    python_executable: str = sys.executable,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    simulator_dir = root / "hardware/npu_design/simulator"
    simulator_path = simulator_dir / "simulator_bfly.py"
    results: list[dict[str, Any]] = []
    for length in sequence_lengths:
        command = fabnet_command(
            simulator_path, int(length), python_executable=python_executable
        )
        completed = subprocess.run(
            command,
            cwd=simulator_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "sequence_length": int(length),
                "latency_ms": parse_fabnet_latency(completed.stdout),
                "command": command,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            }
        )
    return results


def compare_fabnet_results(
    targets: dict[str, Any], simulator_results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    expected_by_length = dict(
        zip(
            targets["sequence_lengths"],
            targets["fabnet_total_latency_ms"],
            strict=True,
        )
    )
    points: list[dict[str, Any]] = []
    for result in simulator_results:
        length = result["sequence_length"]
        expected = expected_by_length[length]
        actual = result["latency_ms"]
        relative_error = abs(actual - expected) / expected
        points.append(
            {
                **result,
                "target_latency_ms": expected,
                "absolute_relative_error": relative_error,
                "tolerance": 0.10,
                "pass": relative_error <= 0.10,
            }
        )
    errors = [point["absolute_relative_error"] for point in points]
    return {
        "points": points,
        "summary": {
            "point_count": len(points),
            "mape": sum(errors) / len(errors),
            "max_absolute_relative_error": max(errors),
            "all_points_pass": all(point["pass"] for point in points),
        },
    }
