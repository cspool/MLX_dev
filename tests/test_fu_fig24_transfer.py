from scripts.audit_fu_fig24_transfer import nested


def test_nested_target_lookup() -> None:
    assert nested({"a": {"b": 1}}, "a.b") == 1
