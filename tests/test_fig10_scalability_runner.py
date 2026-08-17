from scripts.run_fig10_scalability import digest


def test_runner_digest_is_callable(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text("{}\n", encoding="utf-8")
    assert len(digest(path)) == 64
