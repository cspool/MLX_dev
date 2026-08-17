from __future__ import annotations

from scripts.check_fig24_gpu_run import build_measurement


def test_fig24_gpu_checker_is_callable() -> None:
    assert callable(build_measurement)
