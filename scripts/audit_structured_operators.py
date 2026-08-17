#!/usr/bin/env python3
"""Execute the pre-registered H14 structured-operator contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mlxsim.structured import (
    HierarchicalButterflyLinear,
    chunked_fft_compress,
    chunked_fft_decompress,
    hierarchical_butterfly_weight_count,
)


def _check(name: str, actual: float, tolerance: float) -> dict[str, float | str | bool]:
    return {
        "name": name,
        "actual": actual,
        "tolerance": tolerance,
        "pass": actual <= tolerance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/structured-operators-run017.json"),
    )
    args = parser.parse_args()
    torch.manual_seed(17)
    checks: list[dict[str, object]] = []

    value = torch.randn(2, 3, 96, 8)
    compressed, context = chunked_fft_compress(
        value, chunk_length=32, compression_ratio=1.0, dim=-2
    )
    restored = chunked_fft_decompress(compressed, context, dim=-2)
    checks.append(_check("fft_s1_roundtrip_max_abs", float((restored - value).abs().max()), 1e-5))

    for ratio in (0.5, 0.75):
        constant = torch.ones(2, 64, 5)
        compressed, _ = chunked_fft_compress(
            constant, chunk_length=32, compression_ratio=ratio, dim=-2
        )
        checks.append(
            _check(
                f"fft_constant_amplitude_s{ratio}",
                float((compressed - 1.0).abs().max()),
                1e-5,
            )
        )

    base = torch.zeros(1, 64, 1)
    changed = base.clone()
    changed[:, :32] = torch.randn(1, 32, 1)
    base_compressed, _ = chunked_fft_compress(
        base, chunk_length=32, compression_ratio=0.5, dim=-2
    )
    changed_compressed, _ = chunked_fft_compress(
        changed, chunk_length=32, compression_ratio=0.5, dim=-2
    )
    isolation_error = float((base_compressed[:, 16:] - changed_compressed[:, 16:]).abs().max())
    checks.append(_check("fft_chunk_isolation_max_abs", isolation_error, 0.0))

    layer = HierarchicalButterflyLinear(64, 64, block_size=16, bias=True)
    factor_input = torch.randn(3, 7, 64, requires_grad=True)
    dense_output = layer(factor_input)
    factor_output = layer.factorized_forward(factor_input)
    checks.append(
        _check(
            "butterfly_dense_factorized_max_abs",
            float((dense_output - factor_output).abs().max().detach()),
            1e-5,
        )
    )
    dense_output.square().mean().backward()
    finite_gradients = bool(
        torch.isfinite(factor_input.grad).all() and torch.isfinite(layer.factors.grad).all()
    )
    checks.append(
        {
            "name": "butterfly_gradients_finite",
            "actual": finite_gradients,
            "expected": True,
            "pass": finite_gradients,
        }
    )

    identity_error = float(
        (layer(torch.eye(64)) - torch.eye(64)).abs().max().detach()
    )
    checks.append(_check("butterfly_identity_max_abs", identity_error, 1e-5))

    parameter_counts: list[dict[str, object]] = []
    for block_size in (16, 32, 64):
        counted = HierarchicalButterflyLinear(
            128, 128, block_size=block_size, bias=False
        ).structured_weight_count
        expected = hierarchical_butterfly_weight_count(128, 128, block_size)
        item = {
            "block_size": block_size,
            "counted": counted,
            "expected": expected,
            "density": counted / (128 * 128),
            "formula_density": 2 * torch.log2(torch.tensor(float(block_size))).item() / block_size,
            "pass": counted == expected,
        }
        parameter_counts.append(item)
        checks.append({"name": f"butterfly_parameter_count_b{block_size}", **item})

    report = {
        "run_id": "run_017",
        "hypothesis": "H14",
        "protocol": "experiments/h14-structured-operator-contract/protocol.md",
        "classification": "inferred-functional-operator-contract",
        "validation_eligible": False,
        "torch_version": torch.__version__,
        "checks": checks,
        "parameter_counts": parameter_counts,
        "summary": {
            "check_count": len(checks),
            "all_checks_pass": all(bool(check["pass"]) for check in checks),
            "max_numerical_error": max(
                float(check["actual"])
                for check in checks
                if isinstance(check.get("actual"), float)
                and "parameter_count" not in str(check["name"])
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
