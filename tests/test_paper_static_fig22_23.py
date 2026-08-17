from __future__ import annotations

from scripts.audit_paper_static_fig22_23 import paper_static_transform, relative_error
from scripts.compile_paper_static_fig22_23 import transform


def test_transform_preserves_payload_and_adds_model_only() -> None:
    source = {"blocks": [{"tag": 1}], "metadata": {"x": 1}}
    output = transform(source)
    assert output["pe_dependency_model"] == "paper_static"
    assert output["metadata"] == {
        "x": 1,
        "pe_dependency_model": "paper_static",
        "scoreboard_is_paper_semantics": False,
    }
    assert output["blocks"] == source["blocks"]
    assert paper_static_transform(source) == output


def test_relative_error_gate_boundary() -> None:
    assert relative_error(0.9, 1.0) <= 0.10
    assert relative_error(0.899, 1.0) > 0.10
