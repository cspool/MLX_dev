#!/usr/bin/env python3
"""Compare Figure 24/25 proxy FU work with complete batch-32 shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.schema import Workload
from mlxsim.workloads import compile_workload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig24_25_work_identity_v1.yaml"


def qualify(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    exists = path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    checks = {"is_file": exists, "sha256": digest == expected["sha256"]}
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "bytes": path.stat().st_size if exists else None,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def actual_work(operator: dict[str, Any], case: dict[str, Any], batch: int) -> dict[str, Any]:
    family = operator["family"]
    n = int(case["n"])
    d = int(case["d"])
    if family == "fft":
        retained = n // 2
        vectors = batch * d * 3
        pairs = vectors * (n // 2) * int(math.log2(n))
        pairs += vectors * (retained // 2) * int(math.log2(retained))
        return {
            "stage_count": int(math.log2(n)) + 1 + int(math.log2(retained)),
            "fu": {
                "fma": 4 * pairs,
                "add": 6 * pairs,
                "shuffle": vectors * retained,
            },
        }
    if family == "qkv_bsmm":
        profile = compile_workload(
            Workload(
                kernel="bsmm",
                n=n,
                d=d,
                batch=batch,
                projections=3,
                block_size=int(operator["block_size"]),
            )
        )
        fma = int(profile.operations / 2)
        return {
            "stage_count": int(math.log2(int(operator["block_size"]))),
            "fu": {"fma": fma},
            "execution_template_add_estimate": fma // 2,
        }
    if family == "swa":
        window = int(operator["window"])
        score_elements = batch * n * window
        return {
            "stage_count": 4,
            "fu": {
                "fma": 2 * score_elements * d,
                "fmax": score_elements,
                "fexp": score_elements,
                "add": score_elements,
                "fdiv": batch * n * d,
            },
            "query_tile": int(operator["query_tile"]),
        }
    raise ValueError(f"unsupported family: {family}")


def proxy_work(metadata: dict[str, Any], simd_width: int = 8) -> dict[str, Any]:
    return {
        "stage_count": int(metadata["stage_count"]),
        "fu": {
            operation: int(count) * simd_width
            for operation, count in metadata["operation_counts"].items()
        },
    }


def compare_surface(
    *,
    surface: str,
    config: dict[str, Any],
    manifest: dict[str, Any],
    batch: int,
    dimensions: dict[str, int],
) -> list[dict[str, Any]]:
    records = manifest["records"]
    comparisons = []
    for case_spec in config["cases"]:
        case_name = case_spec["name"]
        family_name = case_spec.get("family") or case_name.split("_", maxsplit=1)[0]
        case = {
            "n": int(case_spec.get("n") or case_spec["sequence"]),
            "d": int(case_spec.get("d") or dimensions[family_name]),
        }
        for operator in config["operators"]:
            key = f"{operator['name']}--{case_name}"
            record = records[key]
            if "metadata" in record:
                metadata = record["metadata"]
            else:
                path = PROJECT_ROOT / record["backends"]["column_port"]["output"]["path"]
                metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
            actual = actual_work(operator, case, batch)
            proxy = proxy_work(metadata)
            fractions = {
                operation: proxy["fu"].get(operation, 0) / required
                for operation, required in actual["fu"].items()
            }
            comparisons.append(
                {
                    "surface": surface,
                    "key": key,
                    "case": {"name": case_name, **case, "batch": batch},
                    "operator": operator,
                    "actual": actual,
                    "proxy": proxy,
                    "represented_fractions": fractions,
                    "stage_match": proxy["stage_count"] == actual["stage_count"],
                    "full_work_represented": all(
                        math.isclose(value, 1.0) for value in fractions.values()
                    ),
                }
            )
    return comparisons


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    reports = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
        for name, spec in config["frozen_inputs"].items()
        if name in {"fig24_runs", "fig25_runs"}
    }
    parent_checks = {
        name: reports[name]["hypothesis_status"]
        == config["frozen_inputs"][name]["required_status"]
        and reports[name]["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name in reports
    }
    fig24_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["fig24_config"]["path"]).read_text()
    )
    fig25_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["fig25_config"]["path"]).read_text()
    )
    fig24_manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["fig24_manifest"]["path"]).read_text()
    )
    fig25_manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["fig25_manifest"]["path"]).read_text()
    )
    batch = int(config["batch"])
    dimensions = {name: int(value) for name, value in config["model_dimensions"].items()}
    comparisons = [
        *compare_surface(
            surface="fig24",
            config=fig24_config,
            manifest=fig24_manifest,
            batch=batch,
            dimensions=dimensions,
        ),
        *compare_surface(
            surface="fig25",
            config=fig25_config,
            manifest=fig25_manifest,
            batch=batch,
            dimensions=dimensions,
        ),
    ]
    fractions = [
        value for item in comparisons for value in item["represented_fractions"].values()
    ]
    by_surface = {
        surface: {
            "comparison_count": sum(item["surface"] == surface for item in comparisons),
            "full_work_count": sum(
                item["surface"] == surface and item["full_work_represented"]
                for item in comparisons
            ),
            "stage_match_count": sum(
                item["surface"] == surface and item["stage_match"] for item in comparisons
            ),
        }
        for surface in ("fig24", "fig25")
    }
    source_next_steps = {
        "fft_cmp": "generalize H83 variable-depth compiler to every N/D/batch case",
        "qkv_bsmm": "generalize H92 five/variable-stage projection paths to B16/B32/B64",
        "swa": "generalize H94 grouped Attention to fixed W/Q windowed work",
    }
    summary = {
        "comparison_count": len(comparisons),
        "full_work_count": sum(item["full_work_represented"] for item in comparisons),
        "stage_match_count": sum(item["stage_match"] for item in comparisons),
        "minimum_represented_fraction": min(fractions),
        "maximum_represented_fraction": max(fractions),
        "by_surface": by_surface,
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "counts": len(comparisons) == 66
        and by_surface["fig24"]["comparison_count"] == 42
        and by_surface["fig25"]["comparison_count"] == 24,
        "mismatch_proven": all(not item["full_work_represented"] for item in comparisons),
        "fractions": 0 <= summary["minimum_represented_fraction"]
        <= summary["maximum_represented_fraction"] < 1,
        "targets_consumed": False,
    }
    integrity = all(
        value for key, value in integrity_checks.items() if key != "targets_consumed"
    ) and not integrity_checks["targets_consumed"]
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if integrity else "rejected",
        "audit_integrity": integrity,
        "frozen_inputs": files,
        "parent_checks": parent_checks,
        "comparisons": comparisons,
        "source_next_steps": source_next_steps,
        "summary": summary,
        "integrity_checks": integrity_checks,
        "paper_performance_targets_consumed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = ("hypothesis_status", "audit_integrity", "comparisons", "summary", "integrity_checks")
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
