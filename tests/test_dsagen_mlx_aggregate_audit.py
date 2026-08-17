from __future__ import annotations

from scripts import audit_dsagen_mlx_aggregate as audit


def test_dsagen_parser_extracts_overlay_and_adapter() -> None:
    text = """MLX_OVERLAY_SUMMARY {"done":true,"cycles":10}
MLX_SPAD_ADAPTER_SUMMARY {"requests":3,"responses":3}
Cycles: 569
CGRA Instances: 256 -- Activity Ratio: 1
CGRA Insts / Cycle: 1024 / 569 = 1
[single-core] sanity check passed successfully!
Exiting @ tick 1 because exiting with last active thread context
"""
    parsed = audit.parse_dsagen(text)
    assert parsed["overlay"] == {"done": True, "cycles": 10}
    assert parsed["adapter"] == {"requests": 3, "responses": 3}
    assert parsed["roi_cycles"] == 569
    assert parsed["normal_exit"] is True


def test_aggregate_document_rejects_padding() -> None:
    document = {
        "routing": {"mesh_width": 1, "mesh_height": 1},
        "metadata": {
            "operator": "bsmm",
            "width": 2,
            "stages": 1,
            "pairs_per_stage": 1,
            "total_pairs": 1,
            "instruction_count": 6,
            "memory_requests": 3,
            "transfers": 1,
            "compilation_mode": "aggregate",
            "paper_performance_targets_consumed": False,
        },
        "blocks": [],
    }
    report = audit.audit_aggregate_document(document)
    assert report["pass"] is False
    assert report["checks"]["pair_coverage_no_padding"] is False
