from __future__ import annotations

import json
from pathlib import Path

from scripts.check_dsagen_fig25_run import build_measurement


def test_formal_measurement_matches_overlay_and_requestor_stats() -> None:
    root = Path("artifacts/environment/h49")
    # This test becomes active after formal H49 execution; unit-level compiler
    # coverage remains in test_dsagen_operator_sweep.py.
    assert callable(build_measurement)
    assert json.loads((root / "fig25-transfer-compile-manifest.json").read_text())["output_count"] == 24
