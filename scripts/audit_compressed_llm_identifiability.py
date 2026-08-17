#!/usr/bin/env python3
"""Run the frozen H29 compressed-LLM source-identifiability audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from mlxsim.identifiability import (
    fft_ambiguity_witness,
    layer_plan_combinatorics,
    missing_field_audit,
    qualify_file,
    qualify_line_segment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/compressed_llm_identifiability_v1.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    args = _parse_args()
    config = _load_yaml(args.config)
    output = PROJECT_ROOT / config["run"]["output"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite official result: {output}")
    started = time.perf_counter()

    sources = config["sources"]
    manuscript_path = PROJECT_ROOT / sources["manuscript"]["path"]
    figure_path = PROJECT_ROOT / sources["figure7"]["path"]
    source_qualification = {
        "manuscript": qualify_file(manuscript_path, sources["manuscript"]),
        "figure7": qualify_file(figure_path, sources["figure7"]),
        "segments": {
            name: qualify_line_segment(manuscript_path, name, expected)
            for name, expected in sources["manuscript_segments"].items()
        },
    }
    source_qualification["pass"] = bool(
        source_qualification["manuscript"]["pass"]
        and source_qualification["figure7"]["pass"]
        and all(item["pass"] for item in source_qualification["segments"].values())
    )

    gate_config = config["gate"]
    field_audit = missing_field_audit(
        config["necessary_fields"], gate_config["required_missing_domains"]
    )
    witness_config = config["ambiguity_witnesses"]
    witness = fft_ambiguity_witness(
        chunk_length=int(witness_config["chunk_length"]),
        compression_ratio=float(witness_config["compression_ratio"]),
        perturbation_index=int(witness_config["perturbation_index"]),
        perturbation_delta=float(witness_config["perturbation_delta"]),
    )
    combinatorics_config = config["combinatorics"]
    combinatorics = layer_plan_combinatorics(
        total_layers=int(combinatorics_config["llama_layers"]),
        minimum_modified_layers=int(combinatorics_config["minimum_modified_layers"]),
        chunk_lengths=list(combinatorics_config["power_of_two_chunk_lengths"]),
    )

    checks = {
        "all_source_checks": source_qualification["pass"],
        "all_missing_domains": field_audit["pass"],
        "future_perturbation_leakage": witness["maximum_earlier_absolute_change"]
        > float(witness_config["leakage_tolerance"]),
        "two_interpretations_differ": witness[
            "interpretation_maximum_absolute_difference"
        ]
        > float(witness_config["interpretation_difference_tolerance"]),
        "literal_prefix_complex_output": witness["literal_prefix_maximum_imaginary"]
        > float(witness_config["literal_imaginary_tolerance"]),
        "layer_subset_count": combinatorics["admissible_layer_subsets"]
        == int(combinatorics_config["expected_admissible_layer_subsets"]),
        "chunk_assignment_count": combinatorics[
            "minimum_chunk_assignments_at_minimum_layers"
        ]
        == int(combinatorics_config["expected_minimum_chunk_assignments_at_20_layers"]),
    }
    report = {
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "classification": config["classification"],
        "validation_eligible": bool(config["validation_eligible"]),
        "git_commit": _git_commit(),
        "protocol": config,
        "source_qualification": source_qualification,
        "necessary_field_audit": field_audit,
        "fft_ambiguity_witness": witness,
        "layer_plan_combinatorics": combinatorics,
        "checks": checks,
        "pass": all(checks.values()),
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "torch_version": str(torch.__version__),
        },
    }
    summary = {
        "pass": report["pass"],
        "missing_fields": field_audit["missing_field_count"],
        "registered_fields": field_audit["field_count"],
        "missing_domains": len(field_audit["domain_missing_counts"]),
        "maximum_earlier_absolute_change": witness[
            "maximum_earlier_absolute_change"
        ],
        "interpretation_maximum_absolute_difference": witness[
            "interpretation_maximum_absolute_difference"
        ],
        "admissible_layer_subsets": combinatorics["admissible_layer_subsets"],
        "validation_eligible": report["validation_eligible"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
