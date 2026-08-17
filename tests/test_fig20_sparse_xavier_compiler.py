from __future__ import annotations

from scripts.compile_fig20_sparse_xavier import digest


def test_compiler_digest_is_callable() -> None:
    assert callable(digest)
