from scripts.audit_spad_scalability import qualify


def test_qualify_checks_digest(tmp_path) -> None:
    path = tmp_path / "value.json"
    path.write_text("{}\n", encoding="utf-8")
    first = qualify(path)
    assert qualify(path, {"sha256": first["sha256"]})["pass"]
