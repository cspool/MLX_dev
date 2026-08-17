"""Qualification and metric helpers for the H27 WinoGrande baseline."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mlxsim.llama_perplexity import sha256_file


def canonical_row_json(row: Mapping[str, Any]) -> str:
    """Serialize one dataset row using the frozen H27 canonical encoding."""

    return json.dumps(
        dict(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash newline-terminated canonical JSON rows in their evaluation order."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_row_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_row_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_row_json(row).encode("utf-8")).hexdigest()


def qualify_parquet_dataset(
    path: Path, dataset_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Qualify the independently downloaded official validation parquet."""

    import pyarrow.parquet as pq

    exists = path.is_file()
    if not exists:
        return {
            "path": str(path),
            "exists": False,
            "pass": False,
        }

    table = pq.read_table(path)
    rows = table.to_pylist()
    labels: dict[str, int] = {}
    for row in rows:
        label = str(row["answer"])
        labels[label] = labels.get(label, 0) + 1

    checks = {
        "bytes": path.stat().st_size == int(dataset_config["parquet_bytes"]),
        "parquet_sha256": sha256_file(path) == dataset_config["parquet_sha256"],
        "rows": len(rows) == int(dataset_config["expected_rows"]),
        "columns": table.column_names == list(dataset_config["expected_columns"]),
        "label_counts": labels
        == {str(key): int(value) for key, value in dataset_config["expected_label_counts"].items()},
        "content_sha256": canonical_rows_sha256(rows)
        == dataset_config["canonical_content_sha256"],
        "first_row_sha256": canonical_row_sha256(rows[0])
        == dataset_config["first_row_sha256"],
        "last_row_sha256": canonical_row_sha256(rows[-1])
        == dataset_config["last_row_sha256"],
    }
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "parquet_sha256": sha256_file(path),
        "rows": len(rows),
        "columns": table.column_names,
        "label_counts": labels,
        "canonical_content_sha256": canonical_rows_sha256(rows),
        "first_row_sha256": canonical_row_sha256(rows[0]),
        "last_row_sha256": canonical_row_sha256(rows[-1]),
        "checks": checks,
        "pass": all(checks.values()),
    }


def audit_accuracy(
    *,
    sample_values: Sequence[float],
    aggregate_accuracy: float,
    paper_target_pct: float,
    relative_error_gate: float,
    mean_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Cross-check lm-eval's aggregate against samples and the paper target."""

    if not sample_values:
        raise ValueError("sample_values cannot be empty")
    values = [float(value) for value in sample_values]
    if any(not math.isfinite(value) or value not in (0.0, 1.0) for value in values):
        raise ValueError("sample accuracy values must be finite zeros or ones")
    if not math.isfinite(aggregate_accuracy) or not 0.0 <= aggregate_accuracy <= 1.0:
        raise ValueError("aggregate_accuracy must be finite and in [0, 1]")
    if paper_target_pct <= 0.0:
        raise ValueError("paper_target_pct must be positive")
    if relative_error_gate < 0.0:
        raise ValueError("relative_error_gate must be non-negative")

    sample_mean = sum(values) / len(values)
    aggregate_matches_samples = math.isclose(
        aggregate_accuracy,
        sample_mean,
        rel_tol=mean_tolerance,
        abs_tol=mean_tolerance,
    )
    accuracy_pct = aggregate_accuracy * 100.0
    relative_error = abs(accuracy_pct - paper_target_pct) / paper_target_pct
    return {
        "sample_count": len(values),
        "correct_count": int(sum(values)),
        "sample_mean_accuracy": sample_mean,
        "aggregate_accuracy": aggregate_accuracy,
        "aggregate_matches_samples": aggregate_matches_samples,
        "accuracy_pct": accuracy_pct,
        "paper_target_accuracy_pct": paper_target_pct,
        "relative_error": relative_error,
        "relative_error_gate": relative_error_gate,
        "pass": aggregate_matches_samples and relative_error <= relative_error_gate,
    }
