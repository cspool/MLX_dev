from __future__ import annotations

from scripts.build_gpgpusim_xavier_config import derive_config


def test_xavier_derivation_changes_only_registered_fields() -> None:
    source = (
        "-gpgpu_n_clusters 40\n"
        "-gpgpu_n_cores_per_cluster 2\n"
        "-gpgpu_n_mem 24\n"
        "-gpgpu_clock_domains 1200.0:1200.0:1200.0:850.0\n"
        "unchanged"
    )
    derived, counts = derive_config(source)
    assert counts == {
        "clusters": 1,
        "cores_per_cluster": 1,
        "memory_partitions": 1,
        "clocks": 1,
    }
    assert "-gpgpu_n_clusters 8" in derived
    assert "-gpgpu_n_cores_per_cluster 1" in derived
    assert "-gpgpu_n_mem 16" in derived
    assert "-gpgpu_clock_domains 1377:1377:1377:2133" in derived
