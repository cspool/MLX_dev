from __future__ import annotations

from scripts.compile_dsagen_fig22 import flatten_parameters


def test_parameter_flattening_is_stable() -> None:
    value = {"a": 1, "b": {"c": 2, "d": [1, 2]}}
    assert flatten_parameters(value) == ["a", "b.c", "b.d"]
