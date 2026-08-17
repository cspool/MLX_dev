from scripts.audit_fma_fig25_transfer import nested, relative_error


def test_nested_and_relative_error() -> None:
    assert nested({"a": {"b": [1]}}, "a.b") == [1]
    assert relative_error(0.9, 1.0) < 0.11
