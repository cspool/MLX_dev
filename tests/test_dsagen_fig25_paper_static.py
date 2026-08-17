from __future__ import annotations

from scripts.compile_dsagen_fig25_paper_static import transform


def test_transform_only_adds_registered_dependency_fields() -> None:
    parent = {
        "active_window": 4,
        "memory_backend": "dsagen_dma",
        "metadata": {"arithmetic_expanded": True},
        "blocks": [{"tag": 1, "instructions": []}],
    }
    output = transform(parent)
    assert output["pe_dependency_model"] == "paper_static"
    assert output["metadata"] == {
        "arithmetic_expanded": True,
        "pe_dependency_model": "paper_static",
        "scoreboard_is_paper_semantics": False,
    }
    assert output["blocks"] == parent["blocks"]
    assert "pe_dependency_model" not in parent
