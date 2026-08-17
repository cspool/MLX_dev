#!/usr/bin/env python3
"""Compare pair-wise and aggregate B8 overlay traces after ID normalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairwise", type=Path)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def normalize_name(value: str) -> str:
    value = re.sub(r"_aggregate_stage(\d+)_slot(\d+)", r"_stage\1_pair\2", value)
    return re.sub(r"_s(\d+)_slot(\d+)_ready", r"_s\1_p\2_ready", value)


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(event))
    normalized["block"] = normalize_name(normalized.get("block", ""))
    detail = normalized.get("detail") or {}
    if "event" in detail:
        detail["event"] = normalize_name(detail["event"])
    return normalized


def load_trace(path: Path) -> list[dict[str, Any]]:
    return [
        normalize_event(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compare(pairwise: Path, aggregate: Path) -> dict[str, Any]:
    pairwise_events = load_trace(pairwise)
    aggregate_events = load_trace(aggregate)
    pairwise_canonical = json.dumps(pairwise_events, sort_keys=True, separators=(",", ":"))
    aggregate_canonical = json.dumps(aggregate_events, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "pairwise_path": str(pairwise),
        "aggregate_path": str(aggregate),
        "pairwise_event_count": len(pairwise_events),
        "aggregate_event_count": len(aggregate_events),
        "pairwise_normalized_sha256": hashlib.sha256(pairwise_canonical.encode()).hexdigest(),
        "aggregate_normalized_sha256": hashlib.sha256(
            aggregate_canonical.encode()
        ).hexdigest(),
        "normalized_events_identical": pairwise_canonical == aggregate_canonical,
    }


def main() -> int:
    args = parse_args()
    report = compare(args.pairwise, args.aggregate)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["normalized_events_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
