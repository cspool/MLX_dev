#!/usr/bin/env python3
"""Collect hash-qualified H104 author/simulator lineage sources."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/mlx_author_simulator_lineage_v1.yaml"
USER_AGENT = "MLX-reproduction-audit/1.0 (author simulator lineage)"


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", unescape(value)).casefold()
    return "".join(value.split())


def extract_text(body: bytes, parser: str) -> str:
    if parser == "pdf":
        reader = PdfReader(io.BytesIO(body))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return body.decode("utf-8", errors="replace")


def fetch(spec: dict[str, Any], timeout: float, limit: int) -> dict[str, Any]:
    attempts = []
    body = b""
    status = None
    final_url = None
    content_type = None
    error = None
    for attempt in range(1, 4):
        started = time.perf_counter()
        request = urllib.request.Request(
            spec["url"],
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/pdf,text/html,application/json,text/plain,*/*",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                final_url = response.url
                content_type = response.headers.get("content-type")
                body = response.read(limit + 1)
                if len(body) > limit:
                    raise ValueError(f"payload exceeds {limit} bytes")
                error = None
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            final_url = exc.url
            content_type = exc.headers.get("content-type")
            body = exc.read(limit + 1)
            error = f"HTTPError: {exc.code} {exc.reason}"
        except Exception as exc:  # noqa: BLE001 - retain transport evidence
            error = f"{type(exc).__name__}: {exc}"
        attempts.append(
            {
                "attempt": attempt,
                "status": status,
                "error": error,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if error is None:
            break
        if status not in {408, 425, 429, 500, 502, 503, 504, None}:
            break
        time.sleep(0.25 * attempt)
    text = ""
    parse_error = None
    if body and error is None:
        try:
            text = extract_text(body, str(spec["parser"]))
        except Exception as exc:  # noqa: BLE001 - retain parser evidence
            parse_error = f"{type(exc).__name__}: {exc}"
    normalized = normalize(text)
    tokens = [str(token) for token in spec.get("tokens", [])]
    matched = {token: normalize(token) in normalized for token in tokens}
    return {
        "id": spec["id"],
        "tier": spec["tier"],
        "parser": spec["parser"],
        "required": bool(spec["required"]),
        "request_url": spec["url"],
        "final_url": final_url,
        "status": status,
        "content_type": content_type,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest() if body else None,
        "text_characters": len(text),
        "matched_tokens": matched,
        "all_tokens_match": bool(tokens) and all(matched.values()),
        "transport_success": error is None and status == 200,
        "parse_success": error is None and parse_error is None and bool(text),
        "error": error,
        "parse_error": parse_error,
        "attempts": attempts,
    }


def metadata_endpoints(config: dict[str, Any]) -> list[dict[str, Any]]:
    endpoints = []
    for name, doi in config["metadata_dois"].items():
        endpoints.extend(
            [
                {
                    "id": f"crossref_{name}",
                    "url": "https://api.crossref.org/works/"
                    + urllib.parse.quote(str(doi), safe=""),
                    "tier": "T1-metadata",
                    "parser": "json",
                    "required": False,
                    "tokens": [str(doi)],
                },
                {
                    "id": f"openalex_{name}",
                    "url": "https://api.openalex.org/works/https://doi.org/" + str(doi),
                    "tier": "T1-metadata",
                    "parser": "json",
                    "required": False,
                    "tokens": [str(doi)],
                },
            ]
        )
    return endpoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = PROJECT_ROOT / config["source_snapshot_path"]
    if output.exists():
        raise FileExistsError(output)
    endpoints = [*config["endpoints"], *metadata_endpoints(config)]
    records = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch,
                endpoint,
                args.timeout,
                int(config["max_primary_payload_bytes"]),
            ): endpoint["id"]
            for endpoint in endpoints
        }
        for future in as_completed(futures):
            record = future.result()
            records[record["id"]] = record
            print(
                json.dumps(
                    {
                        "id": record["id"],
                        "status": record["status"],
                        "bytes": record["bytes"],
                        "tokens": record["all_tokens_match"],
                    }
                ),
                flush=True,
            )
    ordered = {key: records[key] for key in sorted(records)}
    required = [item for item in ordered.values() if item["required"]]
    primary = [item for item in ordered.values() if item["tier"] == "T1-primary"]
    summary = {
        "endpoint_count": len(ordered),
        "transport_successes": sum(item["transport_success"] for item in ordered.values()),
        "parse_successes": sum(item["parse_success"] for item in ordered.values()),
        "required_count": len(required),
        "required_transport_successes": sum(item["transport_success"] for item in required),
        "primary_successes": sum(item["transport_success"] for item in primary),
        "primary_token_matches": sum(item["all_tokens_match"] for item in primary),
    }
    document = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": ordered,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["primary_successes"] >= 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
