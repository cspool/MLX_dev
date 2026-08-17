#!/usr/bin/env python3
"""Audit a manually recorded original-detail inspection of MLX Fig. 14."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/mlx_fig14_identity_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def qualify_file(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    is_file = path.is_file()
    size = path.stat().st_size if is_file else None
    digest = sha256_file(path) if is_file else None
    checks = {
        "is_file": is_file,
        "bytes": size == int(expected["bytes"]),
        "sha256": digest == expected["sha256"],
    }
    return {
        "path": str(path),
        "actual_bytes": size,
        "expected_bytes": int(expected["bytes"]),
        "actual_sha256": digest,
        "expected_sha256": expected["sha256"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def qualify_image(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    file_report = qualify_file(path, expected)
    actual = {"width": None, "height": None, "mode": None, "format": None}
    error = None
    if file_report["pass"]:
        try:
            with Image.open(path) as image:
                actual = {
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "format": image.format,
                }
        except Exception as exc:  # noqa: BLE001 - retain image-open failure
            error = f"{type(exc).__name__}: {exc}"
    checks = {
        **file_report["checks"],
        "width": actual["width"] == int(expected["width"]),
        "height": actual["height"] == int(expected["height"]),
        "mode": actual["mode"] == expected["mode"],
        "format": actual["format"] == expected["format"],
    }
    return {
        **file_report,
        "actual_image": actual,
        "expected_image": {key: expected[key] for key in ("width", "height", "mode", "format")},
        "error": error,
        "checks": checks,
        "pass": all(checks.values()),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def is_timezone_aware_iso8601(value: Any) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def preflight(config: dict[str, Any], *, require_observation_absent: bool) -> dict[str, Any]:
    paper = qualify_file(
        PROJECT_ROOT / config["local_sources"]["paper"]["path"], config["local_sources"]["paper"]
    )
    figure = qualify_image(
        PROJECT_ROOT / config["local_sources"]["figure14"]["path"],
        config["local_sources"]["figure14"],
    )
    h35_specification = config["local_sources"]["h35_result"]
    h35_file = qualify_file(PROJECT_ROOT / h35_specification["path"], h35_specification)
    h35 = json.loads(Path(h35_file["path"]).read_text(encoding="utf-8")) if h35_file["pass"] else {}
    observation = PROJECT_ROOT / config["run"]["observation"]
    output = PROJECT_ROOT / config["run"]["output"]
    checks = {
        "paper": paper["pass"],
        "figure14": figure["pass"],
        "h35_result": h35_file["pass"],
        "h35_run_id": h35.get("run_id") == h35_specification["run_id"],
        "h35_source_commit": h35.get("git_commit") == h35_specification["git_commit"],
        "h35_audit_integrity": h35.get("audit_integrity") is True,
        "h35_rejected": h35.get("hypothesis_status") == "rejected",
        "nine_unique_allowed_identifiers": len(config["allowed_identifier_tokens"]) == 9
        and len(config["allowed_identifier_tokens"])
        == len(set(config["allowed_identifier_tokens"])),
        "five_unique_exact_parent_candidates": len(config["exact_parent_candidate_tokens"]) == 5
        and len(config["exact_parent_candidate_tokens"])
        == len(set(config["exact_parent_candidate_tokens"])),
        "exact_parent_candidates_are_registered": set(
            config["exact_parent_candidate_tokens"]
        ).issubset(config["allowed_identifier_tokens"]),
        "ten_allowed_categories": len(config["allowed_categories"]) == 10,
        "protocol": (PROJECT_ROOT / config["run"]["protocol"]).is_file(),
        "observation_state": (not observation.exists())
        if require_observation_absent
        else observation.is_file(),
        "output_absent": not output.exists(),
    }
    return {
        "paper": paper,
        "figure14": figure,
        "h35_result": h35_file,
        "observation_path": str(observation),
        "checks": checks,
        "pass": all(checks.values()),
    }


def evaluate_observation(config: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    record = observation.get("observation") or {}
    image = record.get("image") or {}
    entries = record.get("clear_text_entries") or []
    allowed_categories = set(config["allowed_categories"])
    allowed_identifiers = set(config["allowed_identifier_tokens"])
    exact_parent_candidates = set(config["exact_parent_candidate_tokens"])
    numeric_categories = set(config["numeric_parent_categories"])
    generic_tokens = {item.casefold() for item in config["generic_identity_tokens"]}
    entry_checks = [
        {
            "text_nonempty": bool(str(entry.get("text") or "").strip()),
            "category_allowed": entry.get("category") in allowed_categories,
            "confidence_clear": entry.get("confidence") == "clear",
            "numeric_category_contains_digit": entry.get("category") not in numeric_categories
            or any(character.isdigit() for character in str(entry.get("text") or "")),
        }
        for entry in entries
    ]
    clear_entries = [
        entry for entry, checks in zip(entries, entry_checks, strict=True) if all(checks.values())
    ]
    non_generic_identifiers = [
        entry
        for entry in clear_entries
        if entry["category"] in {"chip_identifier", "project_or_family_identifier"}
        and str(entry["text"]).casefold() not in generic_tokens
    ]
    numeric_entries = [entry for entry in clear_entries if entry["category"] in numeric_categories]
    registered_candidate_labels = [
        entry for entry in non_generic_identifiers if entry["text"] in allowed_identifiers
    ]
    exact_parent_candidate_labels = [
        entry for entry in registered_candidate_labels if entry["text"] in exact_parent_candidates
    ]
    image_specification = config["local_sources"]["figure14"]
    image_checks = {
        "path": image.get("path") == image_specification["path"],
        "bytes": image.get("bytes") == image_specification["bytes"],
        "sha256": image.get("sha256") == image_specification["sha256"],
        "width": image.get("width") == image_specification["width"],
        "height": image.get("height") == image_specification["height"],
        "mode": image.get("mode") == image_specification["mode"],
        "inspection_tool": record.get("inspection_tool") == "view_image:original",
        "too_small_or_blurred_list_present": isinstance(
            record.get("too_small_or_blurred_text"), list
        ),
        "prohibited_inferences_acknowledged": record.get("layout_resemblance_used_for_identity")
        is False,
    }
    minimum_identifiers = int(config["decision"]["minimum_clear_non_generic_identifiers"])
    minimum_numeric = int(config["decision"]["minimum_clear_numeric_parent_values"])
    decision = {
        "clear_non_generic_identifier_count": len(non_generic_identifiers),
        "clear_numeric_parent_value_count": len(numeric_entries),
        "registered_candidate_label_count": len(registered_candidate_labels),
        "exact_parent_candidate_label_count": len(exact_parent_candidate_labels),
        "identifier_path": len(non_generic_identifiers) >= minimum_identifiers,
        "numeric_path": len(numeric_entries) >= minimum_numeric,
        "pass": len(non_generic_identifiers) >= minimum_identifiers
        or len(numeric_entries) >= minimum_numeric,
    }
    checks = {
        "run_id": record.get("run_id") == config["run"]["id"],
        "inspection_time": is_timezone_aware_iso8601(record.get("inspected_at_utc")),
        "image_binding": all(image_checks.values()),
        "all_entries_valid": all(all(item.values()) for item in entry_checks),
        "description_present": bool(str(record.get("neutral_description") or "").strip()),
    }
    return {
        "recorded_at_utc": record.get("inspected_at_utc"),
        "image_checks": image_checks,
        "entry_checks": entry_checks,
        "clear_text_entries": clear_entries,
        "non_generic_identifiers": non_generic_identifiers,
        "numeric_parent_values": numeric_entries,
        "registered_candidate_labels": registered_candidate_labels,
        "exact_parent_candidate_labels": exact_parent_candidate_labels,
        "too_small_or_blurred_text": record.get("too_small_or_blurred_text"),
        "decision": decision,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    preflight_report = preflight(config, require_observation_absent=args.preflight_only)
    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return 0 if preflight_report["pass"] else 2
    if not preflight_report["pass"]:
        print(json.dumps(preflight_report, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    observation_path = PROJECT_ROOT / config["run"]["observation"]
    observation = load_yaml(observation_path)
    evaluation = evaluate_observation(config, observation)
    audit_checks = {
        "preflight": preflight_report["pass"],
        "observation_schema": evaluation["pass"],
        "source_commit_recorded": git_commit() is not None,
        "output_absent": not (PROJECT_ROOT / config["run"]["output"]).exists(),
    }
    audit_integrity = all(audit_checks.values())
    if not audit_integrity:
        hypothesis_status = "inconclusive"
    elif evaluation["decision"]["pass"]:
        hypothesis_status = "supported"
    else:
        hypothesis_status = "rejected"
    exact_parent_supported = bool(evaluation["exact_parent_candidate_labels"])
    result = {
        "schema_version": 1,
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "protocol_path": config["run"]["protocol"],
        "observation": {
            "path": str(observation_path),
            "bytes": observation_path.stat().st_size,
            "sha256": sha256_file(observation_path),
        },
        "preflight": preflight_report,
        "evaluation": evaluation,
        "conclusions": {
            "figure_identifier": "supported"
            if evaluation["decision"]["pass"]
            else "not_present_or_not_legible",
            "architecture_family": "explicit_candidate_label"
            if exact_parent_supported
            else "unchanged_inconclusive",
            "exact_parent_chip": "supported" if exact_parent_supported else "unresolved",
            "code_provenance": "not_supported",
        },
        "audit_checks": audit_checks,
        "audit_integrity": audit_integrity,
        "hypothesis_status": hypothesis_status,
    }
    output = PROJECT_ROOT / config["run"]["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "audit_integrity": audit_integrity,
                "hypothesis_status": hypothesis_status,
                "decision": evaluation["decision"],
                "conclusions": result["conclusions"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit_integrity else 2


if __name__ == "__main__":
    raise SystemExit(main())
