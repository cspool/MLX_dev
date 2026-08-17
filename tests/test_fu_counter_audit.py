from scripts.audit_fu_counters import qualify


def test_qualify_handles_tmp_path(tmp_path) -> None:
    path = tmp_path / "value.json"
    path.write_text("{}\n", encoding="utf-8")
    assert qualify(path)["pass"]
