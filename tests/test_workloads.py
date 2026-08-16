import pytest

from mlxsim.schema import Workload
from mlxsim.workloads import compile_workload


def test_hierarchical_bsmm_density_matches_equation_two() -> None:
    workload = Workload(kernel="bsmm", n=128, d=512, block_size=32)
    profile = compile_workload(workload)
    expected_ratio = 2 * 5 / 32
    dense_ops = 2 * 128 * 512 * 512
    assert profile.metadata["butterfly_density"] == pytest.approx(expected_ratio)
    assert profile.operations == pytest.approx(dense_ops * expected_ratio)
    assert len(profile.stages) == 5


def test_fft_compression_shortens_output() -> None:
    workload = Workload(kernel="fft_cmp", n=1024, d=512, chunk_length=64, compression_ratio=0.5)
    profile = compile_workload(workload)
    assert profile.output_elements == pytest.approx(1024 * 512 * 0.5)
    assert profile.metadata["retained_length"] == 32
    assert len(profile.stages) == 11


def test_cdc_stage_tags_are_forward_only() -> None:
    profile = compile_workload(Workload(kernel="transformer", n=512, d=512, batch=8, block_size=32))
    tags = [stage.tag for stage in profile.stages]
    assert tags == list(range(len(tags)))


def test_rectangular_ffn_profiles_preserve_dense_shape() -> None:
    dense = compile_workload(Workload(kernel="gemm", n=128, d=4096, output_dim=11008))
    sparse = compile_workload(
        Workload(kernel="bsmm", n=128, d=4096, output_dim=11008, block_size=32)
    )
    assert dense.operations == pytest.approx(2 * 128 * 4096 * 11008)
    assert sparse.operations / dense.operations == pytest.approx(2 * 5 / 32)
    assert dense.output_elements == sparse.output_elements == 128 * 11008


def test_full_attention_has_quadratic_score_work() -> None:
    short = compile_workload(Workload(kernel="attention", n=128, d=512))
    long = compile_workload(Workload(kernel="attention", n=256, d=512))
    assert len(short.stages) == 4
    assert long.operations / short.operations == pytest.approx(4.0)
