from scripts.audit_fig24_fu_rerun import qualify


def test_qualify_accepts_file(tmp_path) -> None:
    path = tmp_path / "x"
    path.write_text("x", encoding="utf-8")
    assert qualify(path)["pass"]
