from __future__ import annotations

from scripts.audit_mlx_scaling_mechanism import NAMES


def test_scaling_names_are_frozen() -> None:
    assert NAMES == ("baseline", "simd32_4x4", "simd8_8x8", "simd32_8x8")
