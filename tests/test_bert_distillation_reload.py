from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts import audit_bert_structured_distillation_reload as audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs/analysis/bert_structured_distillation_reload_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def load_corrected_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs/analysis/bert_structured_distillation_reload_v2.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_h39_metric_differences_are_absolute_percentage_points() -> None:
    assert audit.metric_differences(
        {"f1": 80.01, "exact_match": 70.02}, {"f1": 80.0, "exact_match": 70.0}
    ) == pytest.approx({"f1": 0.01, "exact_match": 0.02})


def test_h39_preflight_binds_all_h38_sources_and_checkpoints() -> None:
    config = load_config()
    output = PROJECT_ROOT / config["run"]["output"]
    report = audit.preflight(
        config,
        require_output_absent=not output.exists(),
        require_clean=False,
    )
    assert report["checks"]["source_files"] is True
    assert report["checks"]["checkpoints"] is True
    assert report["checks"]["result_bindings"] is True
    assert report["checks"]["h38_integrity"] is True
    assert report["actual_output_absent"] is (not output.exists())
    assert report["pass"] is True


def test_h39_corrected_preflight_binds_inconclusive_run_and_canonical_keys() -> None:
    config = load_corrected_config()
    output = PROJECT_ROOT / config["run"]["output"]
    report = audit.preflight(
        config,
        require_output_absent=not output.exists(),
        require_clean=False,
    )
    assert config["structured"]["expected_layernorm_alias_count"] == 0
    assert report["checks"]["prior_inconclusive"] is True
    assert report["pass"] is True
