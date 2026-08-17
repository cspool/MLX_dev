from __future__ import annotations

import math
from pathlib import Path

from scripts.audit_dsagen_mlx_dma_memory import parse_prefixed_json, parse_stats


def test_parse_prefixed_json_uses_last_summary() -> None:
    text = (
        'MLX_DMA_ADAPTER_SUMMARY {"requests":1}\n'
        'MLX_DMA_ADAPTER_SUMMARY {"requests":128,"responses":128}'
    )
    assert parse_prefixed_json(text, "MLX_DMA_ADAPTER_SUMMARY") == {
        "requests": 128,
        "responses": 128,
    }


def test_parse_stats_handles_integer_float_and_text(tmp_path: Path) -> None:
    path = tmp_path / "stats.txt"
    path.write_text(
        "metric.integer 64 # comment\nmetric.float 1.25 # comment\nmetric.nan nan\n",
        encoding="utf-8",
    )
    parsed = parse_stats(path)
    assert parsed["metric.integer"] == 64
    assert parsed["metric.float"] == 1.25
    assert math.isnan(parsed["metric.nan"])
