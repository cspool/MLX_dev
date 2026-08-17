from scripts.audit_fig10_scalability import qualify


def test_qualify_binds_expected_digest(tmp_path) -> None:
    path = tmp_path / "value.json"
    path.write_text("{}\n", encoding="utf-8")
    actual = qualify(path)
    checked = qualify(path, {"bytes": 3, "sha256": actual["sha256"]})
    assert checked["pass"]
