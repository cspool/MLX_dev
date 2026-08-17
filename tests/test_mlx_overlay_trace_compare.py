from __future__ import annotations

from scripts.compare_mlx_overlay_traces import normalize_event, normalize_name


def test_aggregate_ids_normalize_to_pairwise_ids() -> None:
    assert normalize_name("bsmm_aggregate_stage2_slot7") == "bsmm_stage2_pair7"
    assert normalize_name("bsmm_s1_slot3_ready") == "bsmm_s1_p3_ready"


def test_event_normalization_preserves_timing_and_changes_only_ids() -> None:
    event = {
        "cycle": 7,
        "block": "bsmm_aggregate_stage1_slot2",
        "detail": {"event": "bsmm_s1_slot2_ready", "count": "1"},
    }
    assert normalize_event(event) == {
        "cycle": 7,
        "block": "bsmm_stage1_pair2",
        "detail": {"event": "bsmm_s1_p2_ready", "count": "1"},
    }
