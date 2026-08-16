from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .experiments import reproduce, write_json
from .schema import CalibrationConfig, HardwareConfig, Workload
from .simulator import MLXSimulator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxsim",
        description="Transparent tag/block simulator for the MLX paper reproduction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="simulate one structured kernel")
    simulate.add_argument("--hardware", required=True, type=Path)
    simulate.add_argument("--calibration", type=Path)
    simulate.add_argument(
        "--kernel",
        required=True,
        choices=["attention", "bsmm", "fft", "fft_cmp", "gemm", "swa", "transformer"],
    )
    simulate.add_argument("--n", required=True, type=int)
    simulate.add_argument("--d", required=True, type=int)
    simulate.add_argument("--output-dim", type=int)
    simulate.add_argument("--batch", type=int, default=1)
    simulate.add_argument("--block-size", type=int, default=32)
    simulate.add_argument("--compression-ratio", type=float, default=0.5)
    simulate.add_argument("--chunk-length", type=int, default=64)
    simulate.add_argument("--window", type=int, default=128)
    simulate.add_argument("--query-block", type=int, default=32)
    simulate.add_argument("--projections", type=int, default=1)
    simulate.add_argument("--trace-limit", type=int, default=0)
    simulate.add_argument("--output", type=Path)

    reproduce_parser = subparsers.add_parser("reproduce", help="run a paper figure manifest")
    reproduce_parser.add_argument(
        "--figure",
        required=True,
        choices=[
            "2",
            "3",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "tables",
            "h2-ablations",
            "all",
        ],
    )
    reproduce_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "simulate":
        hardware = HardwareConfig.from_yaml(args.hardware)
        calibration = CalibrationConfig.from_yaml(args.calibration) if args.calibration else None
        workload = Workload(
            kernel=args.kernel,
            n=args.n,
            d=args.d,
            output_dim=args.output_dim,
            batch=args.batch,
            block_size=args.block_size,
            compression_ratio=args.compression_ratio,
            chunk_length=args.chunk_length,
            window=args.window,
            query_block=args.query_block,
            projections=args.projections,
        )
        result = (
            MLXSimulator(hardware, calibration, trace_limit=args.trace_limit)
            .simulate(workload)
            .to_dict()
        )
    elif args.command == "reproduce":
        result = reproduce(args.figure)
    else:
        raise AssertionError(f"unhandled command: {args.command}")

    if args.output:
        write_json(result, args.output)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
