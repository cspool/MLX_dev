import math

import yaml

from mlxsim.performance_service import CrossFittedLogNContrastService
from scripts.audit_fig20_attention_holdout_repair import DEFAULT_CONFIG, build_audit


def test_cross_fitted_log_n_service_excludes_requested_shape() -> None:
    values = {256: 0.0, 512: 0.1, 1024: 0.2, 2048: 0.3, 4096: 2.0}
    service = CrossFittedLogNContrastService(
        values_by_sequence=values,
        reference_sequence=256,
        model_name="unit-cross-fit",
        target_informed=False,
        provenance="unit-test",
    )
    fit = service.predict_excluding(4096)
    assert fit["training_sequences"] == [256, 512, 1024, 2048]
    assert 4096 not in fit["training_sequences"]
    assert math.isclose(float(fit["slope"]), 0.1)
    assert math.isclose(float(fit["prediction"]), 0.4)


def test_fig20_attention_holdout_repair_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert all(report["acceptance_gates"])
    assert report["summary"]["n4096_passing_points"] == 2
    assert report["summary"]["n4096_max_relative_error"] <= 0.15
    assert report["summary"]["passing_points"] == 48
    assert report["summary"]["direction_matches"] == 36
    assert report["summary"]["changed_points"] == 6
    assert report["summary"]["unchanged_points"] == 42
    assert report["summary"]["parameters_refit"] is False
    assert report["summary"]["independent_validation_claimed"] is False
