#!/usr/bin/env python3
"""Run H35's first-party primary-source recovery audit."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlxsim.lineage import (
    evaluate_candidate_features,
    evaluate_exact_parent_gate,
    evaluate_family_gate,
    normalize_text,
    reported_features,
    title_identity,
)
from scripts.audit_mlx_lineage import (
    strict_family_conflicts,
    strict_material_conflicts,
    strict_relation,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/mlx_primary_source_recovery_v1.yaml"
USER_AGENT = "MLX-reproduction-audit/1.0 (first-party lineage source recovery)"
DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class RecoveryEndpoint:
    key: str
    candidate_id: str
    url: str
    representation: str
    accept: str
    source_class: str
    required_transport: bool = False


class RedirectRecorder(urllib.request.HTTPRedirectHandler):
    """Retain redirect hops without changing urllib's normal behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.redirects: list[dict[str, Any]] = []

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        self.redirects.append(
            {
                "from_url": request.full_url,
                "to_url": new_url,
                "status": int(code),
            }
        )
        return super().redirect_request(request, file_pointer, code, message, headers, new_url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
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


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify_file(PROJECT_ROOT / specification["path"], specification)
        for name, specification in config["local_sources"].items()
    }
    h34 = (
        json.loads(Path(files["h34_result"]["path"]).read_text(encoding="utf-8"))
        if files["h34_result"]["pass"]
        else {}
    )
    h34_specification = config["local_sources"]["h34_result"]
    candidate_ids = [item["id"] for item in config["candidates"]]
    route_ids = [item["id"] for item in config["routes"]]
    route_candidate_ids = {item["candidate_id"] for item in config["routes"]}
    output = PROJECT_ROOT / config["run"]["output"]
    protocol = PROJECT_ROOT / config["run"]["protocol"]
    checks = {
        "local_sources": all(item["pass"] for item in files.values()),
        "h34_run_id": h34.get("run_id") == h34_specification["run_id"],
        "h34_source_commit": h34.get("git_commit") == h34_specification["git_commit"],
        "h34_audit_integrity": h34.get("audit_integrity") is True,
        "h34_hypothesis_inconclusive": h34.get("hypothesis_status") == "inconclusive",
        "h34_family_inconclusive": h34.get("conclusions", {})
        .get("architecture_family", {})
        .get("status")
        == "inconclusive",
        "seventeen_unique_routes": len(route_ids) == 17 and len(route_ids) == len(set(route_ids)),
        "six_unique_candidates": len(candidate_ids) == 6
        and len(candidate_ids) == len(set(candidate_ids)),
        "route_candidates_registered_or_institutional": route_candidate_ids
        <= {*candidate_ids, "institutional_bibliography"},
        "six_frozen_feature_classes": len(config["frozen_feature_classes"]) == 6,
        "h34_feature_classes_unchanged": set(config["frozen_feature_classes"])
        == set(h34.get("feature_matrices", {}).get("mlx", {}).get("feature_classes", {})),
        "protocol": protocol.is_file(),
        "output_absent": not output.exists(),
    }
    return {
        "local_files": files,
        "candidate_ids": candidate_ids,
        "route_ids": route_ids,
        "checks": checks,
        "pass": all(checks.values()),
    }


def build_endpoints(config: dict[str, Any]) -> list[RecoveryEndpoint]:
    return [
        RecoveryEndpoint(
            key=item["id"],
            candidate_id=item["candidate_id"],
            url=item["url"],
            representation=item["representation"],
            accept=item["accept"],
            source_class=item["source_class"],
            required_transport=bool(item.get("required_transport")),
        )
        for item in config["routes"]
    ]


def transient_transport_failure(status: int | None, error: str | None) -> bool:
    return error is not None and (
        status is None or status in {408, 425, 429} or 500 <= status < 600
    )


def fetch_endpoint(
    endpoint: RecoveryEndpoint,
    *,
    timeout: float,
    max_attempts: int,
    max_response_bytes: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    headers = {"User-Agent": USER_AGENT, "Accept": endpoint.accept}
    attempts = []
    max_attempts = max(1, max_attempts)
    for attempt_number in range(1, max_attempts + 1):
        attempt_started = time.perf_counter()
        retrieved = datetime.now(timezone.utc).isoformat()
        status: int | None = None
        final_url: str | None = None
        response_headers: dict[str, str] = {}
        redirects: list[dict[str, Any]] = []
        body = b""
        error: str | None = None
        payload_limit_blocked = False
        recorder = RedirectRecorder()
        opener = urllib.request.build_opener(recorder)
        request = urllib.request.Request(endpoint.url, headers=headers)
        try:
            with opener.open(request, timeout=timeout) as response:
                status = int(response.status)
                final_url = response.url
                response_headers = {
                    key.casefold(): value for key, value in response.headers.items()
                }
                declared_length = response_headers.get("content-length")
                if declared_length and int(declared_length) > max_response_bytes:
                    payload_limit_blocked = True
                    error = (
                        f"PayloadLimit: declared {declared_length} bytes exceeds "
                        f"{max_response_bytes}"
                    )
                else:
                    body = response.read(max_response_bytes)
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            final_url = exc.url
            response_headers = {key.casefold(): value for key, value in exc.headers.items()}
            body = exc.read(max_response_bytes)
            error = f"HTTPError: {exc.code} {exc.reason}"
        except Exception as exc:  # noqa: BLE001 - preserve all route failures
            error = f"{type(exc).__name__}: {exc}"
        redirects = recorder.redirects
        attempts.append(
            {
                "attempt": attempt_number,
                "retrieved_at_utc": retrieved,
                "status": status,
                "error": error,
                "redirects": redirects,
                "payload_limit_blocked": payload_limit_blocked,
                "elapsed_seconds": time.perf_counter() - attempt_started,
            }
        )
        transport_success = status is not None and 200 <= status < 300 and error is None
        if transport_success or not transient_transport_failure(status, error):
            break
        if attempt_number < max_attempts:
            time.sleep(min(0.5 * 2 ** (attempt_number - 1), 2.0))
    declared = response_headers.get("content-length")
    possible_truncation = (
        not payload_limit_blocked
        and len(body) == max_response_bytes
        and (declared is None or int(declared) > len(body))
    )
    return {
        "endpoint": endpoint,
        "body": body,
        "metadata": {
            "key": endpoint.key,
            "candidate_id": endpoint.candidate_id,
            "source_class": endpoint.source_class,
            "representation": endpoint.representation,
            "request_url": endpoint.url,
            "request_headers": headers,
            "retrieved_at_utc": retrieved,
            "status": status,
            "final_url": final_url,
            "redirects": redirects,
            "content_type": response_headers.get("content-type"),
            "content_length_header": declared,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest() if body else None,
            "possible_truncation": possible_truncation,
            "payload_limit_blocked": payload_limit_blocked,
            "error": error,
            "transport_success": transport_success,
            "required_transport": endpoint.required_transport,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def fetch_all(
    endpoints: list[RecoveryEndpoint],
    *,
    timeout: float,
    max_workers: int,
    max_attempts: int,
    max_response_bytes: int,
) -> dict[str, dict[str, Any]]:
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_endpoint,
                endpoint,
                timeout=timeout,
                max_attempts=max_attempts,
                max_response_bytes=max_response_bytes,
            ): endpoint
            for endpoint in endpoints
        }
        for future in as_completed(futures):
            result = future.result()
            results[result["endpoint"].key] = result
    return results


def decoded_body(result: dict[str, Any]) -> str:
    content_type = str(result["metadata"].get("content_type") or "")
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", maxsplit=1)[1].split(";", maxsplit=1)[0]
    try:
        return result["body"].decode(charset, errors="replace")
    except LookupError:
        return result["body"].decode("utf-8", errors="replace")


def pdf_text(body: bytes) -> tuple[str, dict[str, Any]]:
    if not body.startswith(b"%PDF"):
        return "", {"pass": False, "error": "response is not a PDF"}
    try:
        reader = PdfReader(io.BytesIO(body))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(pages)
        return text, {
            "pass": bool(text.strip()),
            "page_count": len(reader.pages),
            "text_characters": len(text),
            "word_count": len(normalize_text(text).split()),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - parse failure is audit evidence
        return "", {
            "pass": False,
            "page_count": None,
            "text_characters": 0,
            "word_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def candidate_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in config["candidates"]}


def exact_candidate_identity(text: str, candidate: dict[str, Any]) -> bool:
    return title_identity(
        text,
        title=candidate["title"],
        aliases=tuple(candidate.get("title_aliases") or []),
    )


def parse_sources(
    config: dict[str, Any], results: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates = candidate_map(config)
    texts: list[dict[str, Any]] = []
    summaries = {}
    for key, result in sorted(results.items()):
        endpoint: RecoveryEndpoint = result["endpoint"]
        metadata = result["metadata"]
        if not metadata["transport_success"]:
            text = ""
            extraction = {"pass": False, "error": metadata.get("error") or "transport failure"}
        elif endpoint.representation == "pdf":
            text, extraction = pdf_text(result["body"])
        else:
            text = decoded_body(result)
            words = len(normalize_text(text).split())
            extraction = {
                "pass": bool(text.strip()),
                "text_characters": len(text),
                "word_count": words,
                "error": None,
            }
        associated_ids = (
            list(candidates)
            if endpoint.candidate_id == "institutional_bibliography"
            else [endpoint.candidate_id]
        )
        identities = {}
        qualifications = {}
        for candidate_id in associated_ids:
            candidate = candidates[candidate_id]
            identity = exact_candidate_identity(text, candidate)
            identities[candidate_id] = identity
            word_count = int(extraction.get("word_count") or 0)
            normalized_source = normalize_text(text)
            substantive_html = word_count >= 150 and (
                "abstract" in normalized_source
                or ("introduction" in normalized_source and word_count >= 500)
            )
            substantive = (endpoint.representation == "pdf" and word_count >= 500) or (
                endpoint.representation == "html"
                and endpoint.source_class.startswith("publisher_")
                and substantive_html
            )
            feature_eligible = bool(
                identity
                and extraction.get("pass")
                and metadata["transport_success"]
                and not metadata["possible_truncation"]
                and endpoint.source_class.startswith("publisher_")
                and substantive
            )
            qualifications[candidate_id] = {
                "identity": identity,
                "substantive": substantive,
                "feature_eligible": feature_eligible,
            }
            texts.append(
                {
                    "source_id": key,
                    "candidate_id": candidate_id,
                    "source_class": endpoint.source_class,
                    "feature_eligible": feature_eligible,
                    "identity_pass": identity,
                    "text": text,
                }
            )
        summaries[key] = {
            "extraction": extraction,
            "candidate_qualifications": qualifications,
            "institutional_identity_only": endpoint.candidate_id == "institutional_bibliography",
        }
    return texts, summaries


def merge_h34_chronology(
    new_matrix: dict[str, Any], h34_result: dict[str, Any], candidate_id: str
) -> None:
    old_matrix = h34_result["feature_matrices"].get(candidate_id)
    if old_matrix:
        new_matrix["feature_classes"]["chronology_ownership"] = old_matrix["feature_classes"][
            "chronology_ownership"
        ]


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    preflight_report = preflight(config)
    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return 0 if preflight_report["pass"] else 2
    if not preflight_report["pass"]:
        print(json.dumps(preflight_report, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    source_commit = git_commit()
    max_bytes = int(config["max_primary_payload_bytes"])
    endpoints = build_endpoints(config)
    results = fetch_all(
        endpoints,
        timeout=args.timeout,
        max_workers=args.max_workers,
        max_attempts=args.max_attempts,
        max_response_bytes=max_bytes,
    )
    source_texts, source_summaries = parse_sources(config, results)
    h34_result = json.loads(
        (PROJECT_ROOT / config["local_sources"]["h34_result"]["path"]).read_text(encoding="utf-8")
    )
    paper_text = (PROJECT_ROOT / config["local_sources"]["paper"]["path"]).read_text(
        encoding="utf-8"
    )
    relation_texts = [paper_text] + [item["text"] for item in source_texts if item["text"]]

    matrices = {}
    candidate_gates = {}
    for candidate in config["candidates"]:
        candidate_id = candidate["id"]
        matrix = evaluate_candidate_features(
            candidate_id=candidate_id,
            source_texts=source_texts,
            frozen_feature_classes=config["frozen_feature_classes"],
        )
        merge_h34_chronology(matrix, h34_result, candidate_id)
        matrices[candidate_id] = matrix
        exact_relations = strict_relation(relation_texts, candidate, exact=True)
        family_relations = strict_relation(relation_texts, candidate, exact=False)
        hardware_conflicts = strict_material_conflicts(source_texts, candidate_id)
        family_conflicts = strict_family_conflicts(relation_texts, candidate)
        exact_gate = evaluate_exact_parent_gate(
            matrix,
            explicit_primary_link=bool(exact_relations),
            material_conflict=bool(hardware_conflicts),
            minimum_hardware_fingerprints=int(
                config["decision"]["exact_parent"]["minimum_hardware_fingerprints"]
            ),
            minimum_exact_numeric_fingerprints=int(
                config["decision"]["exact_parent"]["minimum_exact_numeric_fingerprints"]
            ),
        )
        ownership = (
            matrix["feature_classes"]["chronology_ownership"]["ict_or_ricore_ownership"]["status"]
            == "reported"
        )
        tapeout = "taped_out_verilog" in reported_features(matrix, "parent_hardware_fingerprints")
        family_gate = evaluate_family_gate(
            matrix,
            explicit_family_link=bool(family_relations),
            prior_tapeout_or_ownership=ownership or tapeout,
            material_conflict=bool(family_conflicts),
            minimum_high_specificity_cross_class_matches=int(
                config["decision"]["family_attribution"][
                    "minimum_high_specificity_cross_class_matches"
                ]
            ),
        )
        candidate_gates[candidate_id] = {
            "exact_primary_relations": exact_relations,
            "family_primary_relations": family_relations,
            "exact_hardware_conflicts": hardware_conflicts,
            "family_material_conflicts": family_conflicts,
            "exact_parent": exact_gate,
            "family_attribution": family_gate,
        }

    qualifying_ids = set(config["decision"]["source_recovery"]["qualifying_candidate_ids"])
    qualified_primary_texts = sorted(
        {
            (item["candidate_id"], item["source_id"])
            for item in source_texts
            if item["candidate_id"] in qualifying_ids and item["feature_eligible"]
        }
    )
    minimum_recovered = int(
        config["decision"]["source_recovery"]["minimum_qualified_primary_texts"]
    )
    source_recovery_pass = len(qualified_primary_texts) >= minimum_recovered
    parent_family_ids = sorted(qualifying_ids)
    family_supported = any(
        candidate_gates[candidate_id]["family_attribution"]["pass"]
        for candidate_id in parent_family_ids
    )
    exact_supported = any(
        candidate_gates[candidate_id]["exact_parent"]["pass"] for candidate_id in parent_family_ids
    )
    explicit_family_conflict = any(
        candidate_gates[candidate_id]["family_material_conflicts"]
        for candidate_id in parent_family_ids
    )

    response_metadata = {key: result["metadata"] for key, result in sorted(results.items())}
    required = [item for item in response_metadata.values() if item["required_transport"]]
    audit_checks = {
        "preflight": preflight_report["pass"],
        "source_commit_recorded": source_commit is not None,
        "all_seventeen_routes_attempted": set(response_metadata)
        == {item["id"] for item in config["routes"]},
        "all_required_transports_succeeded": all(item["transport_success"] for item in required),
        "exact_request_headers_recorded": all(
            item["request_headers"] == {"User-Agent": USER_AGENT, "Accept": route["accept"]}
            for route in config["routes"]
            for item in [response_metadata[route["id"]]]
        ),
        "no_download_exceeded_25_mib": all(
            int(item["bytes"]) <= max_bytes for item in response_metadata.values()
        ),
        "all_candidates_modeled": set(matrices) == {item["id"] for item in config["candidates"]},
        "all_feature_classes_unchanged": all(
            set(matrix["feature_classes"]) == set(config["frozen_feature_classes"])
            for matrix in matrices.values()
        ),
        "institutional_page_never_feature_eligible": all(
            not item["feature_eligible"]
            for item in source_texts
            if item["source_class"] == "primary_institutional_record"
        ),
        "generic_and_authorship_exclusions_preserved": all(
            not gate["family_attribution"]["checks"]["generic_terminology_counted"]
            and not gate["family_attribution"]["checks"]["shared_authorship_counted"]
            for gate in candidate_gates.values()
        ),
    }
    audit_integrity = all(audit_checks.values())
    if not audit_integrity:
        hypothesis_status = "inconclusive"
    elif source_recovery_pass:
        hypothesis_status = "supported"
    else:
        hypothesis_status = "rejected"
    if family_supported:
        family_status = "supported"
    elif explicit_family_conflict:
        family_status = "rejected"
    else:
        family_status = "inconclusive"

    result = {
        "schema_version": 1,
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": source_commit,
        "config_path": str(config_path),
        "protocol_path": config["run"]["protocol"],
        "cutoff_utc": config["cutoff_utc"],
        "max_primary_payload_bytes": max_bytes,
        "preflight": preflight_report,
        "response_metadata": response_metadata,
        "response_totals": {
            "route_count": len(response_metadata),
            "transport_success_count": sum(
                bool(item["transport_success"]) for item in response_metadata.values()
            ),
            "response_bytes": sum(int(item["bytes"]) for item in response_metadata.values()),
            "payload_limit_block_count": sum(
                bool(item["payload_limit_blocked"]) for item in response_metadata.values()
            ),
        },
        "source_summaries": source_summaries,
        "qualified_primary_texts": [
            {"candidate_id": candidate_id, "source_id": source_id}
            for candidate_id, source_id in qualified_primary_texts
        ],
        "feature_matrices": matrices,
        "candidate_gates": candidate_gates,
        "conclusions": {
            "source_recovery": {
                "status": "supported" if source_recovery_pass else "rejected",
                "qualified_primary_text_count": len(qualified_primary_texts),
                "minimum_required": minimum_recovered,
            },
            "architecture_family": {
                "status": family_status,
                "supported_candidate_ids": [
                    candidate_id
                    for candidate_id in parent_family_ids
                    if candidate_gates[candidate_id]["family_attribution"]["pass"]
                ],
            },
            "exact_parent_chip": {
                "status": "supported" if exact_supported else "unresolved",
                "supported_candidate_ids": [
                    candidate_id
                    for candidate_id in parent_family_ids
                    if candidate_gates[candidate_id]["exact_parent"]["pass"]
                ],
            },
            "simulator_ancestry": h34_result["conclusions"]["simulator_ancestry"],
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
                "source_recovery": result["conclusions"]["source_recovery"],
                "architecture_family": result["conclusions"]["architecture_family"],
                "exact_parent_chip": result["conclusions"]["exact_parent_chip"],
                "route_count": result["response_totals"]["route_count"],
                "response_bytes": result["response_totals"]["response_bytes"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit_integrity else 2


if __name__ == "__main__":
    raise SystemExit(main())
