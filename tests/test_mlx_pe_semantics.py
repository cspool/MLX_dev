from __future__ import annotations

from scripts.compile_mlx_pe_semantics import paper_static_document


def test_paper_static_transform_changes_only_registered_fields() -> None:
    parent = {
        "active_window": 4,
        "memory_backend": "fixed",
        "metadata": {"paper_performance_targets_consumed": False},
        "blocks": [{"tag": 1}],
    }
    transformed = paper_static_document(parent)
    assert "pe_dependency_model" not in parent
    assert transformed["pe_dependency_model"] == "paper_static"
    assert transformed["metadata"]["pe_dependency_model"] == "paper_static"
    assert transformed["metadata"]["scoreboard_is_paper_semantics"] is False
    assert transformed["blocks"] == parent["blocks"]
