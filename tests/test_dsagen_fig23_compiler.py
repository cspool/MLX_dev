from __future__ import annotations

from scripts.run_dsagen_fig23 import digest


def test_digest(tmp_path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"x")
    assert len(digest(path)) == 64
