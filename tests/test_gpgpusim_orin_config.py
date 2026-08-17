from __future__ import annotations

from scripts.build_gpgpusim_orin_config import derive_config


def test_orin_derivation_changes_only_clusters_and_clocks() -> None:
    source = (
        "-gpgpu_n_clusters 46\n"
        "-gpgpu_n_mem 16\n"
        "-gpgpu_clock_domains 1132:1132:1132:3500.5\n"
        "unchanged"
    )
    derived, counts = derive_config(source)
    assert counts == {"clusters": 1, "clocks": 1}
    assert "-gpgpu_n_clusters 16" in derived
    assert "-gpgpu_n_mem 16" in derived
    assert "-gpgpu_clock_domains 1300:1300:1300:1600" in derived
    assert derived.endswith("unchanged")
