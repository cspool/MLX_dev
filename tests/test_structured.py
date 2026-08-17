import torch

from mlxsim.structured import (
    HierarchicalButterflyLinear,
    chunked_fft_compress,
    chunked_fft_decompress,
    fourier_resample_real,
    hierarchical_butterfly_weight_count,
)


def test_fourier_resample_preserves_constant_amplitude() -> None:
    value = torch.ones(3, 64)
    for output_length in (16, 32, 48, 96):
        output = fourier_resample_real(value, output_length)
        assert torch.allclose(output, torch.ones_like(output), atol=1e-6)


def test_fourier_resample_promotes_unsupported_low_precision_fft() -> None:
    value = torch.randn(2, 32, dtype=torch.bfloat16)
    output = fourier_resample_real(value, 16)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_chunked_s1_roundtrip_and_padding_shape() -> None:
    torch.manual_seed(1)
    value = torch.randn(2, 70, 5)
    compressed, context = chunked_fft_compress(
        value, chunk_length=32, compression_ratio=1.0, dim=1
    )
    assert compressed.shape == (2, 96, 5)
    restored = chunked_fft_decompress(compressed, context, dim=1)
    assert restored.shape == value.shape
    assert torch.allclose(restored, value, atol=1e-5)


def test_chunks_are_independent() -> None:
    first = torch.zeros(1, 64, 2)
    second = first.clone()
    second[:, :32] = 1.0
    compressed_first, _ = chunked_fft_compress(
        first, chunk_length=32, compression_ratio=0.5, dim=1
    )
    compressed_second, _ = chunked_fft_compress(
        second, chunk_length=32, compression_ratio=0.5, dim=1
    )
    assert torch.equal(compressed_first[:, 16:], compressed_second[:, 16:])


def test_hierarchical_parameter_formula() -> None:
    for block_size in (16, 32, 64):
        layer = HierarchicalButterflyLinear(128, 256, block_size=block_size, bias=False)
        assert layer.structured_weight_count == hierarchical_butterfly_weight_count(
            128, 256, block_size
        )
        assert layer.analytical_density == 2 * block_size.bit_length() / block_size - 2 / block_size


def test_dense_and_factorized_forwards_match_with_gradients() -> None:
    torch.manual_seed(2)
    layer = HierarchicalButterflyLinear(32, 48, block_size=16, initialization="fit_start")
    value = torch.randn(4, 32, requires_grad=True)
    dense = layer(value)
    factorized = layer.factorized_forward(value)
    assert torch.allclose(dense, factorized, atol=1e-5)
    dense.sum().backward()
    assert torch.isfinite(value.grad).all()
    assert torch.isfinite(layer.factors.grad).all()


def test_identity_initialization() -> None:
    layer = HierarchicalButterflyLinear(64, 64, block_size=16, bias=True)
    value = torch.randn(3, 64)
    assert torch.equal(layer(value), value)
