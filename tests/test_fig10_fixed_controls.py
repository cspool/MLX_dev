from __future__ import annotations

from mlxsim.fig10_mapping import compile_fig10_mapping
from scripts.compile_fig10_fixed_controls import fixed_control


def test_fixed_control_changes_only_backend() -> None:
    parent, _ = compile_fig10_mapping("bsmm", 64)
    control = fixed_control(parent)
    assert control["memory_backend"] == "fixed"
    control["memory_backend"] = parent["memory_backend"]
    assert control == parent
