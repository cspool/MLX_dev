#!/usr/bin/env python3
"""Generate or verify the frozen full-paper completion certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.experiments import reproduce

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/full_paper_completion_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return value


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def tracked_worktree_clean() -> bool:
    commands = (
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "diff", "--quiet"],
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "diff", "--cached", "--quiet"],
    )
    return all(
        subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode == 0
        for command in commands
    )


def resolve_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer}")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise KeyError(pointer)
    return current


def evaluate_assertions(document: Any, assertions: list[dict[str, Any]]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for assertion in assertions:
        pointer = str(assertion["pointer"])
        expected = assertion["equals"]
        error = None
        try:
            actual = resolve_pointer(document, pointer)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            actual = None
            error = f"{type(exc).__name__}: {exc}"
        reports.append(
            {
                "pointer": pointer,
                "expected": expected,
                "actual": actual,
                "error": error,
                "pass": error is None and actual == expected,
            }
        )
    return {"assertions": reports, "pass": all(item["pass"] for item in reports)}


def qualify_file(specification: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / specification["path"]
    is_file = path.is_file()
    size = path.stat().st_size if is_file else None
    digest = sha256_file(path) if is_file else None
    assertions = specification.get("assertions") or []
    assertion_report: dict[str, Any] = {"assertions": [], "pass": True}
    parse_error = None
    if is_file and assertions:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            assertion_report = evaluate_assertions(document, assertions)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            assertion_report = {"assertions": [], "pass": False}
    checks = {
        "is_file": is_file,
        "bytes": size == int(specification["bytes"]),
        "sha256": digest == specification["sha256"],
        "semantic_assertions": assertion_report["pass"] and parse_error is None,
    }
    return {
        "id": specification["id"],
        "path": str(path),
        "actual_bytes": size,
        "expected_bytes": int(specification["bytes"]),
        "actual_sha256": digest,
        "expected_sha256": specification["sha256"],
        "parse_error": parse_error,
        "assertion_report": assertion_report,
        "checks": checks,
        "pass": all(checks.values()),
    }


def parse_inventory_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 7 or cells[0] == "Paper item" or set(cells[0]) <= {"-", ":"}:
            continue
        labels.append(cells[0])
    return labels


def expected_status_facts(status: str) -> dict[str, bool]:
    common = {
        "numeric_targets_complete": True,
        "executable_audit_present": True,
        "exact_mlx_author_artifact_used": False,
    }
    variants = {
        "reproduced_within_10pct": {
            "full_measurement_gate_passed": True,
            "adequate_attempt_executed_and_failed": False,
            "target_guided_replay": False,
            "required_public_inputs_available": True,
        },
        "attempt_rejected": {
            "full_measurement_gate_passed": False,
            "adequate_attempt_executed_and_failed": True,
            "target_guided_replay": False,
            "required_public_inputs_available": True,
        },
        "calibration_replay_only": {
            "full_measurement_gate_passed": False,
            "adequate_attempt_executed_and_failed": False,
            "target_guided_replay": True,
            "required_public_inputs_available": True,
        },
        "publicly_blocked": {
            "full_measurement_gate_passed": False,
            "adequate_attempt_executed_and_failed": False,
            "target_guided_replay": False,
            "required_public_inputs_available": False,
        },
    }
    if status not in variants:
        raise KeyError(status)
    return {**common, **variants[status]}


def evaluate_items(config: dict[str, Any]) -> dict[str, Any]:
    items = config["items"]
    statuses = set(config["status_taxonomy"])
    evidence_ids = {item["id"] for item in config["frozen_files"]}
    suite_sections = set(config["suite"]["required_sections"])
    required_facts = set(config["required_fact_keys"])
    reports: list[dict[str, Any]] = []
    for item in items:
        status = item.get("status")
        facts = item.get("facts") or {}
        expected_facts = expected_status_facts(status) if status in statuses else {}
        checks = {
            "id_present": bool(str(item.get("id") or "").strip()),
            "paper_item_present": bool(str(item.get("paper_item") or "").strip()),
            "status_registered": status in statuses,
            "fact_keys_exact": set(facts) == required_facts,
            "facts_boolean": all(isinstance(value, bool) for value in facts.values()),
            "status_invariant": facts == expected_facts,
            "evidence_nonempty": bool(item.get("evidence")),
            "evidence_registered": set(item.get("evidence") or []).issubset(evidence_ids),
            "suite_sections_registered": set(item.get("suite_sections") or []).issubset(
                suite_sections
            ),
            "note_present": bool(str(item.get("note") or "").strip()),
        }
        reports.append(
            {"id": item.get("id"), "status": status, "checks": checks, "pass": all(checks.values())}
        )
    actual_counts = dict(Counter(item["status"] for item in items))
    expected_counts = config["expected_status_counts"]
    aggregate_checks = {
        "eighteen_items": len(items) == 18,
        "unique_ids": len({item["id"] for item in items}) == len(items),
        "unique_paper_items": len({item["paper_item"] for item in items}) == len(items),
        "all_statuses_present": set(actual_counts) == statuses,
        "status_counts": actual_counts == expected_counts,
        "all_items_pass": all(item["pass"] for item in reports),
    }
    return {
        "items": reports,
        "actual_status_counts": actual_counts,
        "expected_status_counts": expected_counts,
        "checks": aggregate_checks,
        "pass": all(aggregate_checks.values()),
    }


def evaluate_inventory(config: dict[str, Any]) -> dict[str, Any]:
    specification = next(item for item in config["frozen_files"] if item["id"] == "inventory")
    actual = parse_inventory_labels(PROJECT_ROOT / specification["path"])
    expected = [item["paper_item"] for item in config["items"]]
    checks = {
        "eighteen_rows": len(actual) == 18,
        "unique_rows": len(set(actual)) == len(actual),
        "ordered_labels": actual == expected,
    }
    return {
        "actual_labels": actual,
        "expected_labels": expected,
        "checks": checks,
        "pass": all(checks.values()),
    }


def evaluate_suite(config: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    required_sections = config["suite"]["required_sections"]
    assertion_report = evaluate_assertions(suite, config["suite"]["assertions"])
    regressions = (suite.get("h2_ablations") or {}).get("cycle_regression") or {}
    nonbaseline = [value for key, value in regressions.items() if key != "baseline"]
    checks = {
        "section_set_exact": set(suite) == set(required_sections),
        "section_count": len(suite) == 13,
        "all_sections_mappings": all(isinstance(suite.get(key), dict) for key in required_sections),
        "semantic_assertions": assertion_report["pass"],
        "h2_baseline_present": "baseline" in regressions,
        "h2_nonbaseline_present": bool(nonbaseline),
        "h2_nonbaseline_regressions_gte_one": bool(nonbaseline)
        and all(float(value) >= 1.0 for value in nonbaseline),
    }
    return {"checks": checks, "assertion_report": assertion_report, "pass": all(checks.values())}


def preflight(
    config: dict[str, Any], *, require_outputs_absent: bool, require_clean: bool
) -> dict[str, Any]:
    file_reports = [qualify_file(item) for item in config["frozen_files"]]
    inventory_report = evaluate_inventory(config)
    item_report = evaluate_items(config)
    suite_output = PROJECT_ROOT / config["run"]["suite_output"]
    output = PROJECT_ROOT / config["run"]["output"]
    output_state = not suite_output.exists() and not output.exists()
    checks = {
        "all_frozen_files": all(item["pass"] for item in file_reports),
        "inventory": inventory_report["pass"],
        "items": item_report["pass"],
        "protocol": (PROJECT_ROOT / config["run"]["protocol"]).is_file(),
        "tracked_worktree_clean": tracked_worktree_clean() if require_clean else True,
        "output_state": output_state if require_outputs_absent else True,
    }
    return {
        "frozen_files": file_reports,
        "inventory": inventory_report,
        "item_evaluation": item_report,
        "suite_output": str(suite_output),
        "output": str(output),
        "actual_outputs_absent": output_state,
        "checks": checks,
        "pass": all(checks.values()),
    }


def certificate_summary(config: dict[str, Any]) -> dict[str, Any]:
    counts = dict(Counter(item["status"] for item in config["items"]))
    full_passes = [item["facts"]["full_measurement_gate_passed"] for item in config["items"]]
    return {
        "inventory_item_count": len(config["items"]),
        "status_counts": counts,
        "reproduced_within_10pct_count": sum(full_passes),
        "not_fully_reproduced_count": len(full_passes) - sum(full_passes),
        "all_paper_experiments_reproduced_within_10pct": all(full_passes),
        "exact_mlx_author_artifact_used": any(
            item["facts"]["exact_mlx_author_artifact_used"] for item in config["items"]
        ),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_existing(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    preflight_report = preflight(config, require_outputs_absent=False, require_clean=False)
    suite_path = PROJECT_ROOT / config["run"]["suite_output"]
    output_path = PROJECT_ROOT / config["run"]["output"]
    suite = json.loads(suite_path.read_text(encoding="utf-8")) if suite_path.is_file() else {}
    result = json.loads(output_path.read_text(encoding="utf-8")) if output_path.is_file() else {}
    suite_evaluation = evaluate_suite(config, suite) if suite else {"pass": False}
    recorded_suite = result.get("suite_artifact") or {}
    recorded_config = result.get("config_artifact") or {}
    checks = {
        "frozen_preflight": preflight_report["pass"],
        "suite_exists": suite_path.is_file(),
        "certificate_exists": output_path.is_file(),
        "suite_evaluation": suite_evaluation["pass"],
        "run_id": result.get("run_id") == config["run"]["id"],
        "hypothesis": result.get("hypothesis") == config["run"]["hypothesis"],
        "audit_integrity": result.get("audit_integrity") is True,
        "hypothesis_status": result.get("hypothesis_status") == "supported",
        "summary": result.get("summary") == certificate_summary(config),
        "suite_bytes": suite_path.is_file()
        and recorded_suite.get("bytes") == suite_path.stat().st_size,
        "suite_sha256": suite_path.is_file()
        and recorded_suite.get("sha256") == sha256_file(suite_path),
        "config_bytes": recorded_config.get("bytes") == config_path.stat().st_size,
        "config_sha256": recorded_config.get("sha256") == sha256_file(config_path),
    }
    return {"checks": checks, "pass": all(checks.values()), "summary": result.get("summary")}


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    if args.verify_existing:
        report = verify_existing(config_path, config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["pass"] else 2

    preflight_report = preflight(
        config,
        require_outputs_absent=True,
        require_clean=not args.preflight_only,
    )
    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return 0 if preflight_report["pass"] else 2
    if not preflight_report["pass"]:
        print(json.dumps(preflight_report, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    suite = reproduce("all")
    suite_evaluation = evaluate_suite(config, suite)
    suite_path = PROJECT_ROOT / config["run"]["suite_output"]
    write_json(suite_path, suite)
    summary = certificate_summary(config)
    audit_checks = {
        "preflight": preflight_report["pass"],
        "suite": suite_evaluation["pass"],
        "source_commit_recorded": git_commit() is not None,
        "global_verdict_is_boolean": isinstance(
            summary["all_paper_experiments_reproduced_within_10pct"], bool
        ),
        "global_verdict_remains_false": summary["all_paper_experiments_reproduced_within_10pct"]
        is False,
        "one_full_measurement_pass": summary["reproduced_within_10pct_count"] == 1,
    }
    audit_integrity = all(audit_checks.values())
    result = {
        "schema_version": 1,
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "protocol_path": config["run"]["protocol"],
        "config_artifact": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
        },
        "suite_artifact": {
            "path": str(suite_path),
            "bytes": suite_path.stat().st_size,
            "sha256": sha256_file(suite_path),
        },
        "preflight": preflight_report,
        "suite_evaluation": suite_evaluation,
        "items": config["items"],
        "summary": summary,
        "audit_checks": audit_checks,
        "audit_integrity": audit_integrity,
        "hypothesis_status": "supported" if audit_integrity else "inconclusive",
    }
    output_path = PROJECT_ROOT / config["run"]["output"]
    write_json(output_path, result)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "audit_integrity": audit_integrity,
                "hypothesis_status": result["hypothesis_status"],
                "summary": summary,
                "suite_output": str(suite_path),
                "output": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit_integrity else 2


if __name__ == "__main__":
    raise SystemExit(main())
