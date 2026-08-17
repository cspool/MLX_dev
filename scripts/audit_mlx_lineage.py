#!/usr/bin/env python3
"""Run H34's evidence-bounded MLX architectural-lineage audit."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlxsim.lineage import (
    deduplicate_doi_records,
    evaluate_candidate_features,
    evaluate_exact_parent_gate,
    evaluate_family_gate,
    inverted_abstract_text,
    normalize_doi,
    normalize_text,
    reported_features,
    title_identity,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/mlx_architectural_lineage_v1.yaml"
USER_AGENT = "MLX-reproduction-audit/1.0 (architectural lineage; public metadata)"
DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class Endpoint:
    key: str
    candidate_ids: tuple[str, ...]
    channel: str
    url: str
    parser: str
    source_class: str
    tier: str
    feature_eligible: bool = False
    identity_required: bool = True
    required_transport: bool = False


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


def git_output(*arguments: str, cwd: Path = PROJECT_ROOT) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={cwd}", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_commit() -> str | None:
    return git_output("rev-parse", "HEAD")


def qualify_repository(path: Path, expected_commit: str) -> dict[str, Any]:
    actual_commit = git_output("rev-parse", "HEAD", cwd=path) if path.is_dir() else None
    checks = {
        "is_directory": path.is_dir(),
        "commit": actual_commit == expected_commit,
    }
    return {
        "path": str(path),
        "actual_commit": actual_commit,
        "expected_commit": expected_commit,
        "checks": checks,
        "pass": all(checks.values()),
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    local_files = {
        name: qualify_file(PROJECT_ROOT / specification["path"], specification)
        for name, specification in config["local_sources"].items()
    }
    repositories = {
        name: qualify_repository(PROJECT_ROOT / specification["path"], specification["commit"])
        for name, specification in config["control_repositories"].items()
    }
    browser_path = PROJECT_ROOT / config["local_sources"]["browser_discovery_snapshot"]["path"]
    browser = load_yaml(browser_path) if local_files["browser_discovery_snapshot"]["pass"] else {}
    snapshot_queries = [item["query"] for item in browser.get("registered_queries", [])]
    candidates = config["candidates"]
    candidate_ids = [item["id"] for item in candidates]
    feature_classes = config["frozen_feature_classes"]
    protocol = PROJECT_ROOT / config["run"]["protocol"]
    output = PROJECT_ROOT / config["run"]["output"]
    checks = {
        "local_sources": all(item["pass"] for item in local_files.values()),
        "control_repositories": all(item["pass"] for item in repositories.values()),
        "registered_browser_queries": snapshot_queries == config["registered_queries"],
        "fourteen_registered_queries": len(snapshot_queries) == 14,
        "nine_unique_candidates": len(candidate_ids) == 9
        and len(candidate_ids) == len(set(candidate_ids)),
        "six_frozen_feature_classes": len(feature_classes) == 6,
        "all_feature_lists_nonempty": all(feature_classes.values()),
        "protocol": protocol.is_file(),
        "output_absent": not output.exists(),
    }
    return {
        "local_files": local_files,
        "control_repositories": repositories,
        "browser_query_count": len(snapshot_queries),
        "candidate_ids": candidate_ids,
        "feature_class_counts": {name: len(features) for name, features in feature_classes.items()},
        "checks": checks,
        "pass": all(checks.values()),
    }


def api_url(base: str, parameters: dict[str, Any]) -> str:
    return f"{base}?{urllib.parse.urlencode(parameters)}"


def candidate_by_id(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in config["candidates"]}


def build_t1_endpoints(config: dict[str, Any]) -> list[Endpoint]:
    endpoints: list[Endpoint] = []
    for candidate in config["candidates"]:
        candidate_id = candidate["id"]
        if candidate_id.startswith("cn"):
            continue
        doi = normalize_doi(candidate.get("doi"))
        if doi:
            crossref_url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
            openalex_url = "https://api.openalex.org/works/https://doi.org/" + doi
            crossref_parser = "crossref_work"
            openalex_parser = "openalex_work"
        else:
            crossref_url = api_url(
                "https://api.crossref.org/works",
                {"query.title": candidate["title"], "rows": 10},
            )
            openalex_url = api_url(
                "https://api.openalex.org/works",
                {"search": candidate["title"], "per-page": 10},
            )
            crossref_parser = "crossref_search"
            openalex_parser = "openalex_search"
        endpoints.extend(
            [
                Endpoint(
                    f"crossref_{candidate_id}",
                    (candidate_id,),
                    "Crossref",
                    crossref_url,
                    crossref_parser,
                    "t1_bibliographic_metadata",
                    "T1",
                    required_transport=True,
                ),
                Endpoint(
                    f"openalex_{candidate_id}",
                    (candidate_id,),
                    "OpenAlex",
                    openalex_url,
                    openalex_parser,
                    "t1_bibliographic_metadata",
                    "T1",
                    required_transport=True,
                ),
            ]
        )
    for source in config["primary_endpoints"]:
        endpoints.append(
            Endpoint(
                source["id"],
                tuple(source["candidate_ids"]),
                source["source_class"],
                source["url"],
                source["parser"],
                source["source_class"],
                "T1-primary",
                bool(source.get("feature_eligible")),
                bool(source.get("identity_required", True)),
                bool(source.get("required_transport")),
            )
        )
    for source in config["patent_metadata_lookups"]:
        endpoints.append(
            Endpoint(
                source["id"],
                (source["candidate_id"],),
                "Google Patents exact identifier lookup",
                source["url"],
                "html",
                "patent_bibliographic_index",
                "T1-patent-index",
            )
        )
    return endpoints


SEMANTIC_FIELDS = (
    "paperId,title,authors,year,venue,externalIds,url,openAccessPdf,"
    "publicationTypes,publicationDate,journal,abstract"
)


def build_t2_endpoints(
    config: dict[str, Any], resolved_dois: dict[str, str | None]
) -> list[Endpoint]:
    endpoints = []
    for candidate in config["candidates"]:
        candidate_id = candidate["id"]
        if candidate_id.startswith("cn"):
            continue
        doi = resolved_dois.get(candidate_id)
        if doi:
            base = "https://api.semanticscholar.org/graph/v1/paper/DOI:" + urllib.parse.quote(
                doi, safe="/"
            )
            url = api_url(base, {"fields": SEMANTIC_FIELDS})
            parser = "semantic_work"
        else:
            url = api_url(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                {"query": candidate["title"], "limit": 10, "fields": SEMANTIC_FIELDS},
            )
            parser = "semantic_search"
        endpoints.append(
            Endpoint(
                f"semantic_{candidate_id}",
                (candidate_id,),
                "Semantic Scholar",
                url,
                parser,
                "t2_bibliographic_crosscheck",
                "T2",
            )
        )
    return endpoints


def request_headers(endpoint: Endpoint) -> dict[str, str]:
    accept = "application/json, application/pdf, text/html, text/plain, */*"
    return {"User-Agent": USER_AGENT, "Accept": accept}


def transient_transport_failure(status: int | None, error: str | None) -> bool:
    return error is not None and (
        status is None or status in {408, 425, 429} or 500 <= status < 600
    )


def fetch_endpoint(
    endpoint: Endpoint,
    *,
    timeout: float,
    max_attempts: int,
    max_response_bytes: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    request = urllib.request.Request(endpoint.url, headers=request_headers(endpoint))
    attempts: list[dict[str, Any]] = []
    max_attempts = max(1, max_attempts)
    for attempt_number in range(1, max_attempts + 1):
        attempt_started = time.perf_counter()
        retrieved = datetime.now(timezone.utc).isoformat()
        status: int | None = None
        final_url: str | None = None
        response_headers: dict[str, str] = {}
        body = b""
        error: str | None = None
        payload_limit_blocked = False
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
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
        except Exception as exc:  # noqa: BLE001 - retain all network evidence
            error = f"{type(exc).__name__}: {exc}"
        attempts.append(
            {
                "attempt": attempt_number,
                "retrieved_at_utc": retrieved,
                "status": status,
                "error": error,
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
            "candidate_ids": list(endpoint.candidate_ids),
            "channel": endpoint.channel,
            "source_class": endpoint.source_class,
            "tier": endpoint.tier,
            "request_url": endpoint.url,
            "retrieved_at_utc": retrieved,
            "status": status,
            "final_url": final_url,
            "content_type": response_headers.get("content-type"),
            "content_length_header": declared,
            "etag": response_headers.get("etag"),
            "last_modified": response_headers.get("last-modified"),
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
    endpoints: list[Endpoint],
    *,
    timeout: float,
    max_workers: int,
    max_attempts: int,
    max_response_bytes: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_endpoint,
                endpoint,
                timeout=timeout,
                max_attempts=max_attempts if endpoint.required_transport else 1,
                max_response_bytes=max_response_bytes,
            ): endpoint
            for endpoint in endpoints
        }
        for future in as_completed(futures):
            result = future.result()
            results[result["endpoint"].key] = result
    return results


def json_body(result: dict[str, Any]) -> Any:
    return json.loads(result["body"])


def decoded_body(result: dict[str, Any]) -> str:
    content_type = str(result["metadata"].get("content_type") or "")
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", maxsplit=1)[1].split(";", maxsplit=1)[0]
    try:
        return result["body"].decode(charset, errors="replace")
    except LookupError:
        return result["body"].decode("utf-8", errors="replace")


def candidate_titles(candidate: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    title = str(candidate.get("title") or candidate.get("identifier") or candidate["id"])
    aliases = list(candidate.get("title_aliases") or [])
    identifier = str(candidate.get("identifier") or "")
    if identifier.casefold().startswith("cn"):
        aliases.append(identifier[2:])
    return title, tuple(aliases)


def crossref_authors(record: dict[str, Any]) -> list[str]:
    return [
        " ".join(filter(None, (author.get("given"), author.get("family"))))
        for author in record.get("author") or []
    ]


def crossref_year(record: dict[str, Any]) -> int | None:
    for field in ("published", "published-online", "published-print", "issued"):
        parts = (record.get(field) or {}).get("date-parts") or []
        if parts and parts[0]:
            return int(parts[0][0])
    return None


def select_crossref_record(
    result: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any] | None:
    if not result["metadata"]["transport_success"]:
        return None
    data = json_body(result)["message"]
    records = data.get("items", []) if isinstance(data, dict) and "items" in data else [data]
    title, aliases = candidate_titles(candidate)
    for record in records:
        observed = " ".join(record.get("title") or [])
        if title_identity(observed, title=title, aliases=aliases):
            return record
    return None


def select_openalex_record(
    result: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any] | None:
    if not result["metadata"]["transport_success"]:
        return None
    data = json_body(result)
    records = data.get("results", []) if isinstance(data, dict) and "results" in data else [data]
    title, aliases = candidate_titles(candidate)
    for record in records:
        if title_identity(str(record.get("display_name", "")), title=title, aliases=aliases):
            return record
    return None


def select_semantic_record(
    result: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any] | None:
    if not result["metadata"]["transport_success"]:
        return None
    data = json_body(result)
    records = data.get("data", []) if isinstance(data, dict) and "data" in data else [data]
    title, aliases = candidate_titles(candidate)
    for record in records:
        if title_identity(str(record.get("title", "")), title=title, aliases=aliases):
            return record
    return None


def summarize_crossref(
    candidate_id: str, record: dict[str, Any] | None, source_id: str
) -> dict[str, Any]:
    if record is None:
        return {"source_id": source_id, "candidate_id": candidate_id, "identity": False}
    return {
        "source_id": source_id,
        "candidate_id": candidate_id,
        "identity": True,
        "doi": normalize_doi(record.get("DOI")),
        "title": " ".join(record.get("title") or []),
        "authors": crossref_authors(record),
        "year": crossref_year(record),
        "publisher": record.get("publisher"),
        "container_title": record.get("container-title"),
        "abstract": normalize_text(str(record.get("abstract") or "")),
        "alternative_ids": record.get("alternative-id") or [],
        "resource": record.get("resource"),
        "links": record.get("link") or [],
        "relation": record.get("relation") or {},
    }


def summarize_openalex(
    candidate_id: str, record: dict[str, Any] | None, source_id: str
) -> dict[str, Any]:
    if record is None:
        return {"source_id": source_id, "candidate_id": candidate_id, "identity": False}
    return {
        "source_id": source_id,
        "candidate_id": candidate_id,
        "identity": True,
        "id": record.get("id"),
        "doi": normalize_doi(record.get("doi")),
        "title": record.get("display_name"),
        "authors": [
            item.get("author", {}).get("display_name") for item in record.get("authorships") or []
        ],
        "institutions": sorted(
            {
                institution.get("display_name")
                for authorship in record.get("authorships") or []
                for institution in authorship.get("institutions") or []
                if institution.get("display_name")
            }
        ),
        "year": record.get("publication_year"),
        "publication_date": record.get("publication_date"),
        "abstract": inverted_abstract_text(record.get("abstract_inverted_index")),
        "open_access": record.get("open_access"),
        "primary_location": record.get("primary_location"),
        "best_oa_location": record.get("best_oa_location"),
        "locations": record.get("locations") or [],
    }


def summarize_semantic(
    candidate_id: str, record: dict[str, Any] | None, source_id: str
) -> dict[str, Any]:
    if record is None:
        return {"source_id": source_id, "candidate_id": candidate_id, "identity": False}
    return {
        "source_id": source_id,
        "candidate_id": candidate_id,
        "identity": True,
        "paper_id": record.get("paperId"),
        "doi": normalize_doi((record.get("externalIds") or {}).get("DOI")),
        "title": record.get("title"),
        "authors": [item.get("name") for item in record.get("authors") or []],
        "year": record.get("year"),
        "venue": record.get("venue"),
        "publication_date": record.get("publicationDate"),
        "abstract": record.get("abstract"),
        "open_access_pdf": record.get("openAccessPdf"),
        "url": record.get("url"),
    }


def parse_t1_metadata(
    config: dict[str, Any], results: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, str | None], list[dict[str, Any]]]:
    candidates = candidate_by_id(config)
    summaries: dict[str, dict[str, Any]] = {}
    resolved: dict[str, str | None] = {}
    records: list[dict[str, Any]] = []
    for candidate_id, candidate in candidates.items():
        if candidate_id.startswith("cn"):
            resolved[candidate_id] = None
            continue
        crossref_id = f"crossref_{candidate_id}"
        openalex_id = f"openalex_{candidate_id}"
        crossref = summarize_crossref(
            candidate_id,
            select_crossref_record(results[crossref_id], candidate),
            crossref_id,
        )
        openalex = summarize_openalex(
            candidate_id,
            select_openalex_record(results[openalex_id], candidate),
            openalex_id,
        )
        summaries[crossref_id] = crossref
        summaries[openalex_id] = openalex
        configured = normalize_doi(candidate.get("doi"))
        resolved[candidate_id] = configured or crossref.get("doi") or openalex.get("doi")
        records.extend(item for item in (crossref, openalex) if item.get("identity"))
    return summaries, resolved, records


def parse_t2_metadata(
    config: dict[str, Any], results: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    summaries = {}
    records = []
    for candidate in config["candidates"]:
        candidate_id = candidate["id"]
        if candidate_id.startswith("cn"):
            continue
        source_id = f"semantic_{candidate_id}"
        summary = summarize_semantic(
            candidate_id,
            select_semantic_record(results[source_id], candidate),
            source_id,
        )
        summaries[source_id] = summary
        if summary.get("identity"):
            records.append(summary)
    return summaries, records


PRIMARY_FULLTEXT_HOSTS = {
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "arxiv.org",
    "export.arxiv.org",
    "were.github.io",
    "islped.org",
    "jcst.ict.ac.cn",
}


def primary_fulltext_host(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").casefold()
    return host in PRIMARY_FULLTEXT_HOSTS or host.endswith((".edu", ".edu.cn", ".ac.cn"))


def dynamic_primary_endpoints(
    metadata: dict[str, dict[str, Any]], existing_urls: set[str]
) -> list[Endpoint]:
    candidates: dict[str, list[str]] = {}
    for summary in metadata.values():
        if not summary.get("identity"):
            continue
        candidate_id = summary["candidate_id"]
        for link in summary.get("links") or []:
            url = str(link.get("URL") or "")
            content_type = str(link.get("content-type") or "")
            if url and ("pdf" in content_type.casefold() or ".pdf" in url.casefold()):
                candidates.setdefault(candidate_id, []).append(url)
        for location_name in ("best_oa_location", "primary_location"):
            location = summary.get(location_name) or {}
            if location.get("pdf_url"):
                candidates.setdefault(candidate_id, []).append(location["pdf_url"])
        for location in summary.get("locations") or []:
            if location.get("pdf_url"):
                candidates.setdefault(candidate_id, []).append(location["pdf_url"])
    endpoints = []
    seen = set(existing_urls)
    for candidate_id in sorted(candidates):
        accepted = 0
        for url in candidates[candidate_id]:
            url = str(url)
            if url in seen or not url.startswith("https://") or not primary_fulltext_host(url):
                continue
            seen.add(url)
            endpoints.append(
                Endpoint(
                    f"t1_primary_followup_{candidate_id}_{accepted}",
                    (candidate_id,),
                    "T1 primary full-text follow-up",
                    url,
                    "pdf",
                    "publisher_or_author_full_text_followup",
                    "T1-primary-followup",
                    feature_eligible=True,
                )
            )
            accepted += 1
            if accepted == 3:
                break
    return endpoints


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
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - access gap must stay in the report
        return "", {
            "pass": False,
            "page_count": None,
            "text_characters": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def primary_source_texts(
    config: dict[str, Any], results: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates = candidate_by_id(config)
    texts = []
    summaries = {}
    for key, result in sorted(results.items()):
        endpoint: Endpoint = result["endpoint"]
        if endpoint.source_class in {
            "t1_bibliographic_metadata",
            "t2_bibliographic_crosscheck",
        }:
            continue
        metadata = result["metadata"]
        extraction: dict[str, Any]
        if not metadata["transport_success"]:
            text = ""
            extraction = {"pass": False, "error": metadata.get("error") or "transport failure"}
        elif endpoint.parser == "pdf":
            text, extraction = pdf_text(result["body"])
        else:
            text = decoded_body(result)
            extraction = {
                "pass": bool(text.strip()),
                "text_characters": len(text),
                "error": None,
            }
        candidate_identity = {}
        for candidate_id in endpoint.candidate_ids:
            if candidate_id not in candidates:
                candidate_identity[candidate_id] = False
                continue
            candidate = candidates[candidate_id]
            title, aliases = candidate_titles(candidate)
            candidate_identity[candidate_id] = title_identity(text, title=title, aliases=aliases)
            identity_ok = candidate_identity[candidate_id] or not endpoint.identity_required
            texts.append(
                {
                    "source_id": key,
                    "candidate_id": candidate_id,
                    "source_class": endpoint.source_class,
                    "feature_eligible": bool(
                        endpoint.feature_eligible
                        and metadata["transport_success"]
                        and extraction.get("pass")
                        and identity_ok
                    ),
                    "identity_pass": candidate_identity[candidate_id],
                    "text": text,
                }
            )
        summaries[key] = {
            "candidate_identity": candidate_identity,
            "feature_eligible_configured": endpoint.feature_eligible,
            "identity_required": endpoint.identity_required,
            "extraction": extraction,
        }
    return texts, summaries


def metadata_authors_and_years(
    metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, list[int]], dict[str, list[str]]]:
    authors: dict[str, set[str]] = {}
    years: dict[str, set[int]] = {}
    institutions: dict[str, set[str]] = {}
    for summary in metadata.values():
        if not summary.get("identity"):
            continue
        candidate_id = summary["candidate_id"]
        authors.setdefault(candidate_id, set()).update(
            str(item) for item in summary.get("authors") or [] if item
        )
        if summary.get("year"):
            years.setdefault(candidate_id, set()).add(int(summary["year"]))
        institutions.setdefault(candidate_id, set()).update(
            str(item) for item in summary.get("institutions") or [] if item
        )
    return (
        {key: sorted(value) for key, value in authors.items()},
        {key: sorted(value) for key, value in years.items()},
        {key: sorted(value) for key, value in institutions.items()},
    )


def set_observation(
    matrix: dict[str, Any], class_name: str, feature: str, evidence: list[dict[str, Any]]
) -> None:
    matrix["feature_classes"][class_name][feature] = {
        "status": "reported" if evidence else "not_reported",
        "evidence": evidence,
    }


def add_chronology_ownership(
    *,
    matrix: dict[str, Any],
    candidate: dict[str, Any],
    authors: dict[str, list[str]],
    years: dict[str, list[int]],
    institutions: dict[str, list[str]],
    primary_texts: list[dict[str, Any]],
    mlx_authors: list[str],
) -> None:
    candidate_id = candidate["id"]
    candidate_years = years.get(candidate_id, [])
    predates = [year for year in candidate_years if year < 2026]
    set_observation(
        matrix,
        "chronology_ownership",
        "predates_mlx",
        [
            {
                "source_id": "deduplicated_bibliographic_metadata",
                "source_class": "T1/T2 bibliographic identity",
                "paraphrase": f"Publication year {min(predates)} predates MLX (2026).",
            }
        ]
        if predates
        else [],
    )
    institution_values = institutions.get(candidate_id, [])
    normalized_institutions = normalize_text(" ".join(institution_values))
    institutional_page_match = any(
        item["candidate_id"] == candidate_id
        and item["source_id"] == "ucas_author_bibliography"
        and item["identity_pass"]
        for item in primary_texts
    )
    ownership = institutional_page_match or any(
        term in normalized_institutions
        for term in (
            "institute of computing technology",
            "chinese academy of sciences",
            "university of chinese academy of sciences",
            "ricore",
        )
    )
    set_observation(
        matrix,
        "chronology_ownership",
        "ict_or_ricore_ownership",
        [
            {
                "source_id": (
                    "ucas_author_bibliography"
                    if institutional_page_match
                    else "openalex_affiliations"
                ),
                "source_class": "primary institutional record or affiliation metadata",
                "paraphrase": "The work is recorded in the ICT/CAS institutional research line.",
            }
        ]
        if ownership
        else [],
    )
    normalized_mlx_authors = {normalize_text(name) for name in mlx_authors}
    overlap = sorted(
        author
        for author in authors.get(candidate_id, [])
        if normalize_text(author) in normalized_mlx_authors
    )
    set_observation(
        matrix,
        "chronology_ownership",
        "overlapping_hardware_authors",
        [
            {
                "source_id": "deduplicated_bibliographic_metadata",
                "source_class": "bibliographic identity only",
                "paraphrase": f"Overlapping authors: {', '.join(overlap)}.",
                "counts_toward_family_gate": False,
            }
        ]
        if overlap
        else [],
    )


def strict_relation(
    texts: list[str], candidate: dict[str, Any], *, exact: bool
) -> list[dict[str, Any]]:
    title, _ = candidate_titles(candidate)
    name = re.escape(normalize_text(title).split(":", maxsplit=1)[0])
    if exact:
        relation = r"(?:exact parent|parent chip|same chip|tape[d -]?out parent|implemented from)"
    else:
        relation = (
            r"(?:architectur(?:e|al) family|derived from|based on|subset of|variant of|extends)"
        )
    patterns = (
        rf"\bmlx\b.{{0,220}}{relation}.{{0,220}}\b{name}\b",
        rf"\b{name}\b.{{0,220}}{relation}.{{0,220}}\bmlx\b",
    )
    matches = []
    for index, text in enumerate(texts):
        normalized = normalize_text(text)
        for pattern in patterns:
            if match := re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL):
                matches.append(
                    {
                        "text_index": index,
                        "pattern": pattern,
                        "excerpt": " ".join(match.group(0).split()[:20]),
                    }
                )
                break
    return matches


def strict_material_conflicts(
    primary_texts: list[dict[str, Any]], candidate_id: str
) -> list[dict[str, Any]]:
    specifications = {
        "process_12nm": (r"(?:our|proposed|fabricated|implemented).{0,80}(\d+)\s*nm", "12"),
        "frequency_1ghz": (
            r"(?:our|proposed|fabricated|implemented).{0,80}([\d.]+)\s*ghz",
            "1",
        ),
        "simd_width_32": (r"(?:our|proposed).{0,80}simd\s*[-_ ]?(\d+)", "32"),
    }
    conflicts = []
    for source in primary_texts:
        if source["candidate_id"] != candidate_id or not source["feature_eligible"]:
            continue
        normalized = normalize_text(source["text"])
        for feature, (pattern, expected) in specifications.items():
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE | re.DOTALL):
                observed = match.group(1).lstrip("0") or "0"
                if float(observed) != float(expected):
                    conflicts.append(
                        {
                            "feature": feature,
                            "expected": expected,
                            "observed": observed,
                            "source_id": source["source_id"],
                            "excerpt": " ".join(match.group(0).split()[:20]),
                        }
                    )
    return conflicts


def strict_family_conflicts(texts: list[str], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect explicit family separation, not exact-chip numeric differences."""
    title, _ = candidate_titles(candidate)
    name = re.escape(normalize_text(title).split(":", maxsplit=1)[0])
    separation = r"(?:unrelated to|independent of|different architecture family from)"
    patterns = (
        rf"\bmlx\b.{{0,160}}{separation}.{{0,160}}\b{name}\b",
        rf"\b{name}\b.{{0,160}}{separation}.{{0,160}}\bmlx\b",
    )
    conflicts = []
    for index, text in enumerate(texts):
        normalized = normalize_text(text)
        for pattern in patterns:
            if match := re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL):
                conflicts.append(
                    {
                        "text_index": index,
                        "pattern": pattern,
                        "excerpt": " ".join(match.group(0).split()[:20]),
                    }
                )
                break
    return conflicts


def serialized_response_metadata(
    result_groups: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    output = {}
    for group in result_groups:
        for key, result in group.items():
            output[key] = result["metadata"]
    return dict(sorted(output.items()))


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

    phase_times: dict[str, dict[str, str]] = {}
    phase_times["t1_start"] = {"at_utc": datetime.now(timezone.utc).isoformat()}
    t1_endpoints = build_t1_endpoints(config)
    t1_results = fetch_all(
        t1_endpoints,
        timeout=args.timeout,
        max_workers=args.max_workers,
        max_attempts=args.max_attempts,
        max_response_bytes=max_bytes,
    )
    phase_times["t1_end"] = {"at_utc": datetime.now(timezone.utc).isoformat()}
    t1_metadata, resolved_dois, bibliographic_records = parse_t1_metadata(config, t1_results)

    existing_urls = {endpoint.url for endpoint in t1_endpoints}
    followup_endpoints = dynamic_primary_endpoints(t1_metadata, existing_urls)
    phase_times["t1_primary_followup_start"] = {"at_utc": datetime.now(timezone.utc).isoformat()}
    followup_results = fetch_all(
        followup_endpoints,
        timeout=args.timeout,
        max_workers=args.max_workers,
        max_attempts=args.max_attempts,
        max_response_bytes=max_bytes,
    )
    phase_times["t1_primary_followup_end"] = {"at_utc": datetime.now(timezone.utc).isoformat()}

    phase_times["t2_start"] = {"at_utc": datetime.now(timezone.utc).isoformat()}
    t2_endpoints = build_t2_endpoints(config, resolved_dois)
    t2_results = fetch_all(
        t2_endpoints,
        timeout=args.timeout,
        max_workers=args.max_workers,
        max_attempts=1,
        max_response_bytes=max_bytes,
    )
    phase_times["t2_end"] = {"at_utc": datetime.now(timezone.utc).isoformat()}
    t2_metadata, t2_records = parse_t2_metadata(config, t2_results)
    bibliographic_records.extend(t2_records)
    deduplicated_records = deduplicate_doi_records(bibliographic_records)
    all_metadata = {**t1_metadata, **t2_metadata}

    primary_results = {
        key: result
        for key, result in t1_results.items()
        if result["endpoint"].source_class != "t1_bibliographic_metadata"
    }
    primary_results.update(followup_results)
    primary_texts, primary_summaries = primary_source_texts(config, primary_results)
    paper_path = PROJECT_ROOT / config["local_sources"]["paper"]["path"]
    paper_text = paper_path.read_text(encoding="utf-8")
    primary_texts.append(
        {
            "source_id": "target_mlx_manuscript",
            "candidate_id": "mlx",
            "source_class": "target_primary_manuscript",
            "feature_eligible": True,
            "identity_pass": True,
            "text": paper_text,
        }
    )

    authors, years, institutions = metadata_authors_and_years(all_metadata)
    matrices = {}
    target_matrix = evaluate_candidate_features(
        candidate_id="mlx",
        source_texts=primary_texts,
        frozen_feature_classes=config["frozen_feature_classes"],
    )
    matrices["mlx"] = target_matrix
    candidate_gates = {}
    relation_texts = [paper_text] + [item["text"] for item in primary_texts if item["text"]]
    for candidate in config["candidates"]:
        candidate_id = candidate["id"]
        matrix = evaluate_candidate_features(
            candidate_id=candidate_id,
            source_texts=primary_texts,
            frozen_feature_classes=config["frozen_feature_classes"],
        )
        add_chronology_ownership(
            matrix=matrix,
            candidate=candidate,
            authors=authors,
            years=years,
            institutions=institutions,
            primary_texts=primary_texts,
            mlx_authors=config["local_sources"]["paper"]["authors"],
        )
        matrices[candidate_id] = matrix
        exact_relations = strict_relation(relation_texts, candidate, exact=True)
        family_relations = strict_relation(relation_texts, candidate, exact=False)
        hardware_conflicts = strict_material_conflicts(primary_texts, candidate_id)
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

    parent_family_ids = ["dfu_e", "m2_dfu"]
    family_supported = any(
        candidate_gates[candidate_id]["family_attribution"]["pass"]
        for candidate_id in parent_family_ids
    )
    exact_supported = any(
        candidate_gates[candidate_id]["exact_parent"]["pass"] for candidate_id in parent_family_ids
    )
    material_family_conflict = any(
        candidate_gates[candidate_id]["family_material_conflicts"]
        for candidate_id in parent_family_ids
    )

    simulator_citation_checks = {
        "simulator_sentence_cites_36": bool(
            re.search(r"tuned in (?:our|the) simulator\s*\[36\]", paper_text, flags=re.IGNORECASE)
        ),
        "reference_36_is_simict": bool(
            re.search(r"\[36\].{0,300}simict", paper_text, flags=re.IGNORECASE | re.DOTALL)
        ),
    }
    simulator_citation_supported = all(simulator_citation_checks.values())

    h33_data = json.loads(
        (PROJECT_ROOT / config["local_sources"]["h33_result"]["path"]).read_text(encoding="utf-8")
    )
    code_provenance_checks = {
        "h33_audit_integrity": h33_data.get("audit_integrity") is True,
        "h33_zero_qualifying_exact_artifacts": h33_data.get("qualifying_artifact_count") == 0,
        "primary_repository_or_reuse_statement_found": False,
    }
    code_provenance_pass = code_provenance_checks["primary_repository_or_reuse_statement_found"]

    response_metadata = serialized_response_metadata([t1_results, followup_results, t2_results])
    required_transports = [
        metadata for metadata in response_metadata.values() if metadata["required_transport"]
    ]
    candidate_attempts = {
        candidate["id"]: sorted(
            key
            for key, metadata in response_metadata.items()
            if candidate["id"] in metadata["candidate_ids"]
        )
        for candidate in config["candidates"]
    }
    allowed_feature_source_ids = {
        item["source_id"] for item in primary_texts if item["feature_eligible"]
    }
    cited_feature_source_ids = {
        evidence["source_id"]
        for matrix in matrices.values()
        for observations in matrix["feature_classes"].values()
        for observation in observations.values()
        for evidence in observation["evidence"]
        if "source_id" in evidence
        and evidence["source_id"]
        not in {
            "deduplicated_bibliographic_metadata",
            "openalex_affiliations",
            "ucas_author_bibliography",
        }
    }
    audit_checks = {
        "preflight": preflight_report["pass"],
        "source_commit_recorded": source_commit is not None,
        "all_required_transports_succeeded": all(
            item["transport_success"] for item in required_transports
        ),
        "all_candidates_have_attempt_coverage": all(candidate_attempts.values()),
        "t1_precedes_t2": phase_times["t1_primary_followup_end"]["at_utc"]
        <= phase_times["t2_start"]["at_utc"],
        "doi_deduplication_applied": bool(deduplicated_records),
        "no_download_exceeded_25_mib": all(
            int(item["bytes"]) <= max_bytes for item in response_metadata.values()
        ),
        "all_registered_candidates_modeled": set(matrices)
        == {"mlx", *(candidate["id"] for candidate in config["candidates"])},
        "all_six_feature_classes_modeled": all(
            set(matrix["feature_classes"]) == set(config["frozen_feature_classes"])
            for matrix in matrices.values()
        ),
        "feature_evidence_uses_eligible_primary_sources": cited_feature_source_ids
        <= allowed_feature_source_ids,
        "shared_authorship_never_counts_toward_family_gate": all(
            not gate["family_attribution"]["checks"]["shared_authorship_counted"]
            for gate in candidate_gates.values()
        ),
        "generic_terms_never_count_toward_family_gate": all(
            not gate["family_attribution"]["checks"]["generic_terminology_counted"]
            for gate in candidate_gates.values()
        ),
    }
    audit_integrity = all(audit_checks.values())
    if not audit_integrity:
        hypothesis_status = "inconclusive"
    elif family_supported:
        hypothesis_status = "supported"
    elif material_family_conflict:
        hypothesis_status = "rejected"
    else:
        hypothesis_status = "inconclusive"

    fulltext_access = {
        candidate["id"]: sorted(
            {
                item["source_id"]
                for item in primary_texts
                if item["candidate_id"] == candidate["id"]
                and item["feature_eligible"]
                and "full_text" in item["source_class"]
            }
        )
        for candidate in config["candidates"]
    }
    access_gaps = [
        candidate_id
        for candidate_id, source_ids in fulltext_access.items()
        if not source_ids and candidate_id in {"dfu_e", "m2_dfu", "dfgas", "transfer_latency"}
    ]

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
        "tool_environment": {
            "academic_search_mcp_available": False,
            "fallback": "T1 Crossref/OpenAlex and primary URLs, then T2 Semantic Scholar",
            "pdf_parser": "pypdf",
        },
        "preflight": preflight_report,
        "phase_times": phase_times,
        "response_metadata": response_metadata,
        "response_totals": {
            "endpoint_count": len(response_metadata),
            "transport_success_count": sum(
                bool(item["transport_success"]) for item in response_metadata.values()
            ),
            "response_bytes": sum(int(item["bytes"]) for item in response_metadata.values()),
            "payload_limit_block_count": sum(
                bool(item["payload_limit_blocked"]) for item in response_metadata.values()
            ),
        },
        "metadata": {
            "resolved_dois": resolved_dois,
            "t1": t1_metadata,
            "t2_crosscheck": t2_metadata,
            "deduplicated_records": deduplicated_records,
            "deduplication_key": "normalized DOI",
        },
        "primary_source_summaries": primary_summaries,
        "candidate_attempt_coverage": candidate_attempts,
        "fulltext_access": fulltext_access,
        "primary_fulltext_access_gaps": access_gaps,
        "feature_matrices": matrices,
        "candidate_gates": candidate_gates,
        "simulator_ancestry": {
            "citation_level_supported": simulator_citation_supported,
            "checks": simulator_citation_checks,
            "scope": "SimICT is the framework referenced by citation [36] at the simulator-tuning sentence.",
            "source_code_reuse_supported": False,
            "hardware_parent_supported": False,
        },
        "code_provenance": {
            "pass": code_provenance_pass,
            "status": "not_supported",
            "checks": code_provenance_checks,
            "scope": "No candidate repository is relabeled as MLX code.",
        },
        "conclusions": {
            "architecture_family": {
                "status": "supported" if family_supported else "inconclusive",
                "supported_candidate_ids": [
                    candidate_id
                    for candidate_id in parent_family_ids
                    if candidate_gates[candidate_id]["family_attribution"]["pass"]
                ],
                "access_gaps": access_gaps,
            },
            "exact_parent_chip": {
                "status": "supported" if exact_supported else "unresolved",
                "supported_candidate_ids": [
                    candidate_id
                    for candidate_id in parent_family_ids
                    if candidate_gates[candidate_id]["exact_parent"]["pass"]
                ],
            },
            "simulator_ancestry": {
                "status": "supported_at_citation_level"
                if simulator_citation_supported
                else "inconclusive",
                "code_reuse": "not_supported",
            },
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
                "architecture_family": result["conclusions"]["architecture_family"],
                "exact_parent_chip": result["conclusions"]["exact_parent_chip"],
                "simulator_ancestry": result["conclusions"]["simulator_ancestry"],
                "endpoint_count": result["response_totals"]["endpoint_count"],
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
