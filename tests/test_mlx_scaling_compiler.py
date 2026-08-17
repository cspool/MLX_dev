from __future__ import annotations

from scripts.compile_mlx_scaling import digest


def test_digest_is_sha256(tmp_path) -> None:
    path = tmp_path / "value"
    path.write_bytes(b"abc")
    assert digest(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
