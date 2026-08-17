#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training" / "quality_v1.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="hash multi-GB model shards in addition to checking their exact sizes",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_files(
    root: Path,
    files: list[dict[str, Any]],
    *,
    full_hash: bool,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for expected in files:
        path = root / expected["path"]
        item: dict[str, Any] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "present": path.is_file(),
            "expected_bytes": int(expected["bytes"]),
            "expected_sha256": expected["sha256"],
        }
        if path.is_file():
            actual_size = path.stat().st_size
            item["actual_bytes"] = actual_size
            item["size_matches"] = actual_size == int(expected["bytes"])
            if "records" in expected and item["size_matches"]:
                actual_records = len(json.loads(path.read_text(encoding="utf-8")))
                item["actual_records"] = actual_records
                item["expected_records"] = int(expected["records"])
                item["records_match"] = actual_records == int(expected["records"])
            if "questions" in expected and item["size_matches"]:
                payload = json.loads(path.read_text(encoding="utf-8"))
                actual_questions = sum(
                    len(paragraph["qas"])
                    for article in payload["data"]
                    for paragraph in article["paragraphs"]
                )
                item["actual_questions"] = actual_questions
                item["expected_questions"] = int(expected["questions"])
                item["questions_match"] = actual_questions == int(expected["questions"])
            should_hash = full_hash or actual_size < 100 * 1024 * 1024
            if should_hash and item["size_matches"]:
                actual_hash = _sha256(path)
                item["actual_sha256"] = actual_hash
                item["sha256_matches"] = actual_hash == expected["sha256"]
            else:
                item["sha256_matches"] = None
        else:
            item["size_matches"] = False
            item["sha256_matches"] = False
        results.append(item)
    return {
        "complete_by_size": all(item["size_matches"] for item in results),
        "complete_by_manifest": all(
            item["size_matches"]
            and item.get("records_match", True)
            and item.get("questions_match", True)
            for item in results
        ),
        "all_checked_hashes_match": all(item["sha256_matches"] is not False for item in results),
        "files": results,
    }


def main() -> int:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "config": str(args.config.relative_to(PROJECT_ROOT)),
        "full_hash": args.full_hash,
        "datasets": {},
        "models": {},
        "known_recipe_gaps": config["paper_recipe_gaps"],
    }
    for name in ("ada_leval", "wikitext", "squad"):
        entry = config["datasets"][name]
        report["datasets"][name] = _audit_files(
            PROJECT_ROOT / entry["local_root"],
            entry["files"],
            full_hash=True,
        )
    for name in ("bert_base_uncased", "internlm2_7b", "internlm2_chat_7b"):
        entry = config["models"][name]
        report["models"][name] = _audit_files(
            PROJECT_ROOT / entry["local_root"],
            entry["files"],
            full_hash=args.full_hash,
        )
    report["blocked_inputs"] = {
        "fgscr42": config["datasets"]["fgscr42"]["availability"],
        "llama2_7b": config["models"]["llama2_7b"]["availability"],
    }
    report["all_quality_inputs_complete"] = (
        all(
            item["complete_by_manifest"]
            for group in (report["datasets"], report["models"])
            for item in group.values()
        )
        and not report["blocked_inputs"]
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{encoded}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
