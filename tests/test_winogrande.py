import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml

from mlxsim.winogrande import (
    audit_accuracy,
    canonical_row_json,
    canonical_rows_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/analysis/llama2_winogrande_v1.yaml"


def test_canonical_rows_hash_is_ordered_and_newline_terminated() -> None:
    rows = [{"b": 2, "a": "é"}, {"a": "x", "b": 1}]
    expected = ''.join(canonical_row_json(row) + "\n" for row in rows).encode("utf-8")
    assert canonical_rows_sha256(rows) == hashlib.sha256(expected).hexdigest()
    assert canonical_rows_sha256(reversed(rows)) != canonical_rows_sha256(rows)


def test_accuracy_audit_cross_checks_aggregate_and_target() -> None:
    report = audit_accuracy(
        sample_values=[1.0, 0.0, 1.0, 1.0],
        aggregate_accuracy=0.75,
        paper_target_pct=75.0,
        relative_error_gate=0.10,
    )
    assert report["correct_count"] == 3
    assert report["aggregate_matches_samples"] is True
    assert report["relative_error"] == pytest.approx(0.0)
    assert report["pass"] is True

    mismatch = audit_accuracy(
        sample_values=[1.0, 0.0],
        aggregate_accuracy=0.75,
        paper_target_pct=75.0,
        relative_error_gate=0.10,
    )
    assert mismatch["aggregate_matches_samples"] is False
    assert mismatch["pass"] is False


@pytest.mark.parametrize("values", [[], [0.5], [float("nan")]])
def test_accuracy_audit_rejects_invalid_samples(values: list[float]) -> None:
    with pytest.raises(ValueError):
        audit_accuracy(
            sample_values=values,
            aggregate_accuracy=0.5,
            paper_target_pct=90.1,
            relative_error_gate=0.10,
        )


def test_h27_task_files_match_frozen_hashes_and_partial_scoring() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    harness = config["harness"]
    for key in ("task_yaml", "preprocessor"):
        path = PROJECT_ROOT / harness[key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == harness[f"{key}_sha256"]

    module_path = PROJECT_ROOT / harness["preprocessor"]
    spec = importlib.util.spec_from_file_location("h27_winogrande_preprocess", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    doc = {
        "sentence": "Alice thanked Bob because _ had helped.",
        "option1": "Alice",
        "option2": "Bob",
        "answer": "2",
    }
    assert module.doc_to_text(doc) == 1
    assert module.doc_to_choice(doc) == [
        "Alice thanked Bob because Alice",
        "Alice thanked Bob because Bob",
    ]
    assert module.doc_to_target(doc) == "had helped."
