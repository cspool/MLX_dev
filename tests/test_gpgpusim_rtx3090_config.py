from __future__ import annotations

from scripts.build_gpgpusim_rtx3090_config import derive_config


def test_registered_config_substitutions_are_exact() -> None:
    source = (
        "-gpgpu_n_clusters 46\n"
        "-gpgpu_n_mem 16\n"
        "-gpgpu_clock_domains 1132:1132:1132:3500.5\n"
            "unchanged"
    )
    derived, counts = derive_config(source)
    assert counts == {"clusters": 1, "memory_partitions": 1, "clocks": 1}
    assert "-gpgpu_n_clusters 82" in derived
    assert derived.count("-gpgpu_n_mem 24") == 1
    assert "-gpgpu_clock_domains 1695:1695:1695:5250" in derived
    assert derived.endswith("unchanged")
