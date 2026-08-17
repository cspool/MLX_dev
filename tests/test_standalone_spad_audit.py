from scripts.audit_standalone_dsagen_spad import relative_error


def test_relative_error_handles_zero_reference() -> None:
    assert relative_error(0.0, 0.0) == 0.0
    assert relative_error(1.0, 0.0) == 1.0
