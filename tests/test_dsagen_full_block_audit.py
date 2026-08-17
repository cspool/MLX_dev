from __future__ import annotations

from scripts.audit_dsagen_mlx_dma_memory import load_yaml
from scripts.audit_dsagen_mlx_full_block import DEFAULT_CONFIG, compiler_audit


def test_h48_compiler_audit_has_expected_primitive_and_pipeline_counts() -> None:
    result = compiler_audit(load_yaml(DEFAULT_CONFIG))
    assert result["pass"] is True
    assert result["pipeline_counts"] == {
        "load": 24,
        "store": 16,
        "compute": 840,
        "xfer": 472,
    }
    assert result["operation_counts"]["fma"] == 344
    assert result["operation_counts"]["fexp"] == 16
