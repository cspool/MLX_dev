"""Explicitly inferred functional forms of MLX's structured operators.

These implementations prioritize auditable tensor semantics and portable
autograd. They are not performance kernels and are not presented as author
source recovery.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FFTCompressionContext:
    original_length: int
    padded_length: int
    chunk_length: int
    compressed_chunk_length: int
    num_chunks: int
    compression_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _replace_last(value: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    """Return ``value`` with its last scalar replaced without in-place mutation."""
    return torch.cat((value[..., :-1], replacement.unsqueeze(-1)), dim=-1)


def fourier_resample_real(value: torch.Tensor, output_length: int, *, dim: int = -1) -> torch.Tensor:
    """Fourier-resample a real tensor with amplitude and Nyquist correction.

    This follows the standard FFT resampling convention: retain the common low
    modes, split/combine an even-length Nyquist term as needed, then scale by
    ``output_length / input_length``.
    """

    if output_length <= 0:
        raise ValueError("output_length must be positive")
    dim = dim % value.ndim
    moved = value.movedim(dim, -1)
    input_length = moved.shape[-1]
    output_dtype = moved.dtype
    fft_input = (
        moved.float() if moved.dtype in {torch.float16, torch.bfloat16} else moved
    )
    if input_length <= 0:
        raise ValueError("input length must be positive")
    if output_length == input_length:
        result = torch.fft.irfft(
            torch.fft.rfft(fft_input, dim=-1), n=input_length, dim=-1
        )
        return result.to(output_dtype).movedim(-1, dim)

    spectrum = torch.fft.rfft(fft_input, dim=-1)
    if output_length < input_length:
        output_bins = output_length // 2 + 1
        resized = spectrum[..., :output_bins]
        if output_length % 2 == 0:
            resized = _replace_last(resized, resized[..., -1] * 2.0)
    else:
        copied = spectrum
        if input_length % 2 == 0:
            copied = _replace_last(copied, copied[..., -1] * 0.5)
        output_bins = output_length // 2 + 1
        copied_bins = copied.shape[-1]
        resized = F.pad(copied, (0, output_bins - copied_bins))

    result = torch.fft.irfft(resized, n=output_length, dim=-1)
    result = result * (output_length / input_length)
    return result.to(output_dtype).movedim(-1, dim)


def chunked_fft_compress(
    value: torch.Tensor,
    *,
    chunk_length: int,
    compression_ratio: float,
    dim: int = -2,
) -> tuple[torch.Tensor, FFTCompressionContext]:
    """Compress independent token chunks to ``compression_ratio * chunk_length``."""

    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if not 0.0 < compression_ratio <= 1.0:
        raise ValueError("compression_ratio must be in (0, 1]")
    compressed_length_float = chunk_length * compression_ratio
    compressed_chunk_length = round(compressed_length_float)
    if not math.isclose(compressed_length_float, compressed_chunk_length, abs_tol=1e-9):
        raise ValueError("chunk_length * compression_ratio must be an integer")

    dim = dim % value.ndim
    moved = value.movedim(dim, -1)
    original_length = moved.shape[-1]
    num_chunks = math.ceil(original_length / chunk_length)
    padded_length = num_chunks * chunk_length
    padded = F.pad(moved, (0, padded_length - original_length))
    chunks = padded.reshape(*padded.shape[:-1], num_chunks, chunk_length)
    compressed_chunks = fourier_resample_real(chunks, compressed_chunk_length, dim=-1)
    compressed = compressed_chunks.flatten(-2)
    context = FFTCompressionContext(
        original_length=original_length,
        padded_length=padded_length,
        chunk_length=chunk_length,
        compressed_chunk_length=compressed_chunk_length,
        num_chunks=num_chunks,
        compression_ratio=compression_ratio,
    )
    return compressed.movedim(-1, dim), context


def chunked_fft_decompress(
    value: torch.Tensor, context: FFTCompressionContext, *, dim: int = -2
) -> torch.Tensor:
    """Symmetrically restore the original token count from compressed chunks."""

    dim = dim % value.ndim
    moved = value.movedim(dim, -1)
    expected = context.num_chunks * context.compressed_chunk_length
    if moved.shape[-1] != expected:
        raise ValueError(
            f"compressed token length mismatch: expected {expected}, got {moved.shape[-1]}"
        )
    chunks = moved.reshape(
        *moved.shape[:-1], context.num_chunks, context.compressed_chunk_length
    )
    restored_chunks = fourier_resample_real(chunks, context.chunk_length, dim=-1)
    restored = restored_chunks.flatten(-2)[..., : context.original_length]
    return restored.movedim(-1, dim)


def hierarchical_butterfly_weight_count(
    in_features: int, out_features: int, block_size: int
) -> int:
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if in_features % block_size or out_features % block_size:
        raise ValueError("block_size must divide both feature widths")
    return (
        (in_features // block_size)
        * (out_features // block_size)
        * 2
        * block_size
        * int(math.log2(block_size))
    )


class HierarchicalButterflyLinear(nn.Module):
    """Blocked linear map with an independent log-depth butterfly per tile."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        block_size: int,
        bias: bool = True,
        initialization: str = "identity",
    ) -> None:
        super().__init__()
        hierarchical_butterfly_weight_count(in_features, out_features, block_size)
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.in_blocks = in_features // block_size
        self.out_blocks = out_features // block_size
        self.num_stages = int(math.log2(block_size))
        self.factors = nn.Parameter(
            torch.empty(
                self.out_blocks,
                self.in_blocks,
                self.num_stages,
                block_size // 2,
                2,
                2,
            )
        )
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters(initialization)

    @property
    def structured_weight_count(self) -> int:
        return self.factors.numel()

    @property
    def analytical_density(self) -> float:
        return self.structured_weight_count / (self.in_features * self.out_features)

    @torch.no_grad()
    def reset_parameters(self, initialization: str = "identity") -> None:
        if initialization not in {"identity", "fit_start"}:
            raise ValueError(f"unknown initialization: {initialization}")
        eye = torch.eye(2, dtype=self.factors.dtype, device=self.factors.device)
        self.factors.copy_(eye.expand_as(self.factors))
        if initialization == "identity":
            for out_block in range(self.out_blocks):
                for in_block in range(self.in_blocks):
                    if out_block != in_block:
                        self.factors[out_block, in_block, 0].zero_()
        else:
            self.factors.add_(torch.randn_like(self.factors) * 0.01)
            for out_block in range(self.out_blocks):
                for in_block in range(self.in_blocks):
                    if out_block != in_block:
                        self.factors[out_block, in_block, 0].normal_(mean=0.0, std=0.01)
        if self.bias is not None:
            self.bias.zero_()

    def _apply_stage(
        self, values: torch.Tensor, stage_factors: torch.Tensor, stage: int
    ) -> torch.Tensor:
        prefix = values.shape[:-3]
        stride = 1 << stage
        groups = self.block_size // (2 * stride)
        paired = values.reshape(
            *prefix, self.out_blocks, self.in_blocks, groups, 2, stride
        ).movedim(-1, -2)
        factors = stage_factors.reshape(
            self.out_blocks, self.in_blocks, groups, stride, 2, 2
        )
        mixed = torch.einsum("...abgsi,abgsji->...abgsj", paired, factors)
        return mixed.movedim(-1, -2).reshape(
            *prefix, self.out_blocks, self.in_blocks, self.block_size
        )

    def factorized_forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input width {self.in_features}, got {value.shape[-1]}"
            )
        prefix = value.shape[:-1]
        blocks = value.reshape(*prefix, self.in_blocks, self.block_size)
        values = blocks.unsqueeze(-3).expand(
            *prefix, self.out_blocks, self.in_blocks, self.block_size
        )
        for stage in range(self.num_stages):
            values = self._apply_stage(values, self.factors[:, :, stage], stage)
        output = values.sum(dim=-2).reshape(*prefix, self.out_features)
        return output + self.bias if self.bias is not None else output

    def dense_weight(self) -> torch.Tensor:
        eye = torch.eye(
            self.block_size, dtype=self.factors.dtype, device=self.factors.device
        )
        values = eye[:, None, None, :].expand(
            self.block_size, self.out_blocks, self.in_blocks, self.block_size
        )
        for stage in range(self.num_stages):
            values = self._apply_stage(values, self.factors[:, :, stage], stage)
        tiles = values.permute(1, 2, 3, 0)
        return tiles.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.dense_weight(), self.bias)

    def fit_to_dense_(
        self,
        dense: nn.Linear,
        *,
        steps: int,
        learning_rate: float,
        seed: int,
    ) -> dict[str, float | int]:
        """Fit factors to a dense layer with an explicitly inferred MSE objective."""

        if dense.in_features != self.in_features or dense.out_features != self.out_features:
            raise ValueError("dense layer shape does not match structured layer")
        torch.manual_seed(seed)
        self.reset_parameters("fit_start")
        target = dense.weight.detach().to(device=self.factors.device, dtype=self.factors.dtype)
        target_power = target.square().mean().clamp_min(torch.finfo(target.dtype).eps)
        optimizer = torch.optim.Adam([self.factors], lr=learning_rate)
        initial_relative_mse = float(
            ((self.dense_weight() - target).square().mean() / target_power).detach()
        )
        loss = torch.tensor(float("nan"), device=target.device)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = (self.dense_weight() - target).square().mean() / target_power
            loss.backward()
            optimizer.step()
        if self.bias is not None:
            if dense.bias is None:
                self.bias.data.zero_()
            else:
                self.bias.data.copy_(dense.bias.detach().to(self.bias))
        return {
            "steps": steps,
            "learning_rate": learning_rate,
            "initial_relative_mse": initial_relative_mse,
            "final_relative_mse": float(loss.detach()),
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block_size={self.block_size}, bias={self.bias is not None}"
        )
