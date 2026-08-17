#!/usr/bin/env python3
"""Run H33's source-qualified public MLX artifact discovery audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlxsim.source_discovery import (
    critical_domain_matches,
    crossref_artifact_links,
    evaluate_artifact_candidate,
    exact_paper_identity,
    normalize_text,
    repository_identity_text,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/mlx_public_artifact_discovery_v1.yaml"
DOI = "10.1109/ISCA66397.2026.00017"
IEEE_DOCUMENT = "11617948"
USER_AGENT = "MLX-reproduction-audit/1.0 (public source qualification)"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class Endpoint:
    key: str
    channel: str
    url: str
    parser: str
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
    paper = qualify_file(PROJECT_ROOT / config["paper"]["path"], config["paper"])
    browser = qualify_file(
        PROJECT_ROOT / config["browser_snapshot"]["path"],
        config["browser_snapshot"],
    )
    browser_data = load_yaml(PROJECT_ROOT / config["browser_snapshot"]["path"])
    snapshot_queries = [item["query"] for item in browser_data["registered_queries"]]
    protocol = PROJECT_ROOT / config["run"]["protocol"]
    output = PROJECT_ROOT / config["run"]["output"]
    prior_inconclusive = None
    if prior_spec := config.get("prior_inconclusive"):
        prior_file = qualify_file(PROJECT_ROOT / prior_spec["path"], prior_spec)
        prior_data = (
            json.loads(Path(prior_file["path"]).read_text(encoding="utf-8"))
            if prior_file["pass"]
            else {}
        )
        prior_checks = {
            "file": prior_file["pass"],
            "run_id": prior_data.get("run_id") == prior_spec["run_id"],
            "git_commit": prior_data.get("git_commit") == prior_spec["git_commit"],
            "audit_integrity_false": prior_data.get("audit_integrity") is False,
            "hypothesis_inconclusive": prior_data.get("hypothesis_status") == "inconclusive",
            "zero_qualifying_artifacts": prior_data.get("qualifying_artifact_count") == 0,
        }
        prior_inconclusive = {
            "file": prior_file,
            "checks": prior_checks,
            "pass": all(prior_checks.values()),
        }
    checks = {
        "paper": paper["pass"],
        "browser_snapshot": browser["pass"],
        "registered_web_queries": snapshot_queries == config["web_queries"],
        "registered_repository_queries": len(config["repository_queries"]) == 6,
        "browser_has_no_prequalified_candidate": not browser_data["qualifying_artifact_candidates"],
        "protocol": protocol.is_file(),
        "output_absent": not output.exists(),
    }
    if prior_inconclusive is not None:
        checks["prior_inconclusive"] = prior_inconclusive["pass"]
    return {
        "paper": paper,
        "browser_snapshot": browser,
        "browser_query_counts": {
            "registered": len(browser_data["registered_queries"]),
            "identifier_and_index_followups": len(browser_data["identifier_and_index_followups"]),
            "author_repository_followups": len(browser_data["author_repository_followups"]),
        },
        "prior_inconclusive": prior_inconclusive,
        "checks": checks,
        "pass": all(checks.values()),
    }


def api_url(base: str, params: dict[str, Any]) -> str:
    return f"{base}?{urllib.parse.urlencode(params)}"


def build_endpoints(config: dict[str, Any], browser_data: dict[str, Any]) -> list[Endpoint]:
    title = config["paper"]["title"]
    title_fragment = "Multi-Layer Execution Structured LLM"
    endpoints = [
        Endpoint(
            "crossref_doi",
            "Crossref",
            f"https://api.crossref.org/works/{urllib.parse.quote(DOI, safe='')}",
            "crossref",
            True,
        ),
        Endpoint(
            "openalex_doi",
            "OpenAlex",
            f"https://api.openalex.org/works/https://doi.org/{DOI}",
            "openalex",
            True,
        ),
        Endpoint(
            "dblp_title",
            "DBLP",
            api_url(
                "https://dblp.org/search/publ/api",
                {"q": title, "format": "json", "h": 20},
            ),
            "dblp",
        ),
        Endpoint(
            "arxiv_title",
            "arXiv",
            api_url(
                "https://export.arxiv.org/api/query",
                {"search_query": f'ti:"{title}"', "start": 0, "max_results": 20},
            ),
            "arxiv",
        ),
        Endpoint(
            "semantic_scholar_doi",
            "Semantic Scholar",
            api_url(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{DOI}",
                {
                    "fields": (
                        "paperId,title,authors,year,venue,externalIds,url,"
                        "openAccessPdf,publicationTypes,publicationDate,journal"
                    )
                },
            ),
            "semantic_scholar",
        ),
        Endpoint(
            "zenodo_title",
            "Zenodo",
            api_url(
                "https://zenodo.org/api/records",
                {"q": f'"{title}"', "size": 25},
            ),
            "zenodo",
        ),
        Endpoint(
            "hf_models",
            "Hugging Face",
            api_url(
                "https://huggingface.co/api/models",
                {"search": title_fragment, "limit": 100},
            ),
            "hf_list",
            True,
        ),
        Endpoint(
            "hf_datasets",
            "Hugging Face",
            api_url(
                "https://huggingface.co/api/datasets",
                {"search": title_fragment, "limit": 100},
            ),
            "hf_list",
            True,
        ),
        Endpoint(
            "hf_spaces",
            "Hugging Face",
            api_url(
                "https://huggingface.co/api/spaces",
                {"search": title_fragment, "limit": 100},
            ),
            "hf_list",
            True,
        ),
        Endpoint(
            "github_were_repos",
            "GitHub author profile",
            api_url(
                "https://api.github.com/users/were/repos",
                {"type": "owner", "sort": "updated", "per_page": 100},
            ),
            "github_repos",
            True,
        ),
        Endpoint(
            "github_synthesys_repos",
            "GitHub author lab",
            api_url(
                "https://api.github.com/orgs/Synthesys-Lab/repos",
                {"type": "public", "sort": "updated", "per_page": 100},
            ),
            "github_repos",
            True,
        ),
        Endpoint(
            "gitee_title_search",
            "Gitee",
            api_url(
                "https://search.gitee.com/",
                {"q": title_fragment, "type": "repository"},
            ),
            "html_search",
        ),
        Endpoint(
            "modelscope_title_search",
            "ModelScope",
            api_url(
                "https://www.modelscope.cn/search",
                {"search": title_fragment},
            ),
            "html_search",
        ),
    ]
    for index, query in enumerate(config["repository_queries"]):
        github_query = f"{query} in:name,description,readme"
        endpoints.append(
            Endpoint(
                f"github_search_{index}",
                "GitHub repository search",
                api_url(
                    "https://api.github.com/search/repositories",
                    {"q": github_query, "per_page": 100},
                ),
                "github_search",
                True,
            )
        )
        endpoints.append(
            Endpoint(
                f"gitlab_search_{index}",
                "GitLab project search",
                api_url(
                    "https://gitlab.com/api/v4/projects",
                    {"search": query.replace('"', ""), "simple": "true", "per_page": 100},
                ),
                "gitlab_search",
                True,
            )
        )
    seen_urls: set[str] = set()
    for index, source in enumerate(browser_data["official_identity_sources"]):
        url = source["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        endpoints.append(
            Endpoint(
                f"official_page_{index}",
                source["classification"],
                url,
                "official_page",
                url
                in {
                    "https://www.iscaconf.org/isca2026/program/",
                    "https://were.github.io/",
                    "https://people.ucas.ac.cn/~liwenming",
                },
            )
        )
    return endpoints


def request_headers(endpoint: Endpoint) -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, application/atom+xml, text/html, */*",
    }
    if "api.github.com" in endpoint.url:
        headers["Accept"] = "application/vnd.github+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    return headers


def transient_transport_failure(status: int | None, error: str | None) -> bool:
    """Return whether a failed request is safe to retry without changing scope."""
    return error is not None and (
        status is None or status in {408, 425, 429} or 500 <= status < 600
    )


def fetch_endpoint(endpoint: Endpoint, *, timeout: float, max_attempts: int) -> dict[str, Any]:
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
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                final_url = response.url
                response_headers = {
                    key.casefold(): value for key, value in response.headers.items()
                }
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            final_url = exc.url
            response_headers = {key.casefold(): value for key, value in exc.headers.items()}
            body = exc.read(MAX_RESPONSE_BYTES + 1)
            error = f"HTTPError: {exc.code} {exc.reason}"
        except Exception as exc:  # noqa: BLE001 - retain network failures
            error = f"{type(exc).__name__}: {exc}"
        attempts.append(
            {
                "attempt": attempt_number,
                "retrieved_at_utc": retrieved,
                "status": status,
                "error": error,
                "elapsed_seconds": time.perf_counter() - attempt_started,
            }
        )
        transport_success = status is not None and 200 <= status < 300
        if transport_success or not transient_transport_failure(status, error):
            break
        if attempt_number < max_attempts:
            time.sleep(min(0.5 * 2 ** (attempt_number - 1), 2.0))
    truncated = len(body) > MAX_RESPONSE_BYTES
    if truncated:
        body = body[:MAX_RESPONSE_BYTES]
    return {
        "endpoint": endpoint,
        "body": body,
        "metadata": {
            "key": endpoint.key,
            "channel": endpoint.channel,
            "request_url": endpoint.url,
            "retrieved_at_utc": retrieved,
            "status": status,
            "final_url": final_url,
            "content_type": response_headers.get("content-type"),
            "content_length_header": response_headers.get("content-length"),
            "etag": response_headers.get("etag"),
            "last_modified": response_headers.get("last-modified"),
            "rate_limit_remaining": response_headers.get("x-ratelimit-remaining"),
            "rate_limit_reset": response_headers.get("x-ratelimit-reset"),
            "retry_after": response_headers.get("retry-after"),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest() if body else None,
            "truncated": truncated,
            "error": error,
            "transport_success": transport_success,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def fetch_all(
    endpoints: list[Endpoint], *, timeout: float, max_workers: int, max_attempts: int
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_endpoint,
                endpoint,
                timeout=timeout,
                max_attempts=max_attempts if endpoint.required_transport else 1,
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


def parse_arxiv(result: dict[str, Any]) -> dict[str, Any]:
    root = ET.fromstring(result["body"])
    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "open": "http://a9.com/-/spec/opensearch/1.1/",
    }
    entries = []
    for entry in root.findall("atom:entry", namespaces):
        entries.append(
            {
                "id": entry.findtext("atom:id", namespaces=namespaces),
                "title": entry.findtext("atom:title", namespaces=namespaces),
                "authors": [
                    author.findtext("atom:name", namespaces=namespaces)
                    for author in entry.findall("atom:author", namespaces)
                ],
            }
        )
    return {
        "total_results": int(
            root.findtext("open:totalResults", default="0", namespaces=namespaces)
        ),
        "entries": entries,
    }


def parse_results(
    config: dict[str, Any], results: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    title = config["paper"]["title"]
    authors = config["paper"]["authors"]
    summaries: dict[str, Any] = {}
    repositories: dict[str, dict[str, Any]] = {}
    exact_metadata_sources: list[dict[str, Any]] = []

    for key, result in results.items():
        endpoint: Endpoint = result["endpoint"]
        metadata = result["metadata"]
        if not metadata["transport_success"]:
            summaries[key] = {"parser": endpoint.parser, "available": False}
            continue
        try:
            if endpoint.parser == "crossref":
                record = json_body(result)["message"]
                identity = exact_paper_identity(
                    "\n".join(record.get("title") or [])
                    + "\n"
                    + "\n".join(
                        f"{item.get('given', '')} {item.get('family', '')}"
                        for item in record.get("author") or []
                    ),
                    title=title,
                    authors=authors,
                )
                summaries[key] = {
                    "available": True,
                    "identity": identity,
                    "doi": record.get("DOI"),
                    "publisher": record.get("publisher"),
                    "container_title": record.get("container-title"),
                    "published": record.get("published"),
                    "resource": record.get("resource"),
                    "license": record.get("license"),
                    "artifact_links": crossref_artifact_links(record),
                    "paper_links": record.get("link") or [],
                    "relation": record.get("relation"),
                }
                if identity["pass"]:
                    exact_metadata_sources.append({"source": key, "identifier": record.get("DOI")})
            elif endpoint.parser == "openalex":
                record = json_body(result)
                identity = exact_paper_identity(
                    record.get("display_name", "")
                    + "\n"
                    + "\n".join(
                        item.get("author", {}).get("display_name", "")
                        for item in record.get("authorships") or []
                    ),
                    title=title,
                    authors=authors,
                )
                summaries[key] = {
                    "available": True,
                    "identity": identity,
                    "id": record.get("id"),
                    "doi": record.get("doi"),
                    "publication_date": record.get("publication_date"),
                    "open_access": record.get("open_access"),
                    "locations": record.get("locations"),
                    "datasets": record.get("datasets"),
                }
                if identity["pass"]:
                    exact_metadata_sources.append({"source": key, "identifier": record.get("id")})
            elif endpoint.parser == "dblp":
                data = json_body(result)["result"]["hits"]
                hits = data.get("hit") or []
                summaries[key] = {
                    "available": True,
                    "total": int(data["@total"]),
                    "hits": [item["info"] for item in hits],
                }
            elif endpoint.parser == "arxiv":
                summaries[key] = {"available": True, **parse_arxiv(result)}
            elif endpoint.parser == "semantic_scholar":
                record = json_body(result)
                identity = exact_paper_identity(
                    record.get("title", "")
                    + "\n"
                    + "\n".join(item.get("name", "") for item in record.get("authors") or []),
                    title=title,
                    authors=authors,
                )
                summaries[key] = {
                    "available": True,
                    "identity": identity,
                    **{
                        name: record.get(name)
                        for name in (
                            "paperId",
                            "title",
                            "year",
                            "venue",
                            "externalIds",
                            "url",
                            "openAccessPdf",
                            "publicationTypes",
                            "publicationDate",
                            "journal",
                        )
                    },
                }
                if identity["pass"]:
                    exact_metadata_sources.append(
                        {"source": key, "identifier": record.get("paperId")}
                    )
            elif endpoint.parser == "zenodo":
                data = json_body(result)["hits"]
                hits = data.get("hits") or []
                summaries[key] = {
                    "available": True,
                    "total": int(data["total"]),
                    "hits": [
                        {
                            "id": item.get("id"),
                            "doi": item.get("doi"),
                            "title": item.get("metadata", {}).get("title"),
                            "creators": item.get("metadata", {}).get("creators"),
                            "resource_type": item.get("metadata", {}).get("resource_type"),
                            "license": item.get("metadata", {}).get("license"),
                            "files": [file.get("key") for file in item.get("files") or []],
                            "links": item.get("links"),
                        }
                        for item in hits
                    ],
                }
            elif endpoint.parser in {"github_search", "github_repos"}:
                data = json_body(result)
                items = (data.get("items") or []) if isinstance(data, dict) else data
                for item in items:
                    if item.get("full_name"):
                        repositories[item["full_name"]] = item
                summaries[key] = {
                    "available": True,
                    "total": int(data.get("total_count", len(items)))
                    if isinstance(data, dict)
                    else len(items),
                    "repositories": [item.get("full_name") for item in items],
                }
            elif endpoint.parser == "gitlab_search":
                items = json_body(result)
                summaries[key] = {
                    "available": True,
                    "total": len(items),
                    "repositories": [item.get("path_with_namespace") for item in items],
                    "exact_identity_candidates": [
                        item.get("path_with_namespace")
                        for item in items
                        if exact_paper_identity(
                            repository_identity_text(item),
                            title=title,
                            authors=authors,
                        )["pass"]
                    ],
                }
            elif endpoint.parser == "hf_list":
                items = json_body(result)
                summaries[key] = {
                    "available": True,
                    "total": len(items),
                    "ids": [item.get("id") for item in items],
                    "exact_identity_candidates": [
                        item.get("id")
                        for item in items
                        if exact_paper_identity(
                            json.dumps(item, ensure_ascii=False),
                            title=title,
                            authors=authors,
                        )["pass"]
                    ],
                }
            elif endpoint.parser == "official_page":
                identity = exact_paper_identity(decoded_body(result), title=title, authors=authors)
                summaries[key] = {
                    "available": True,
                    "identity": identity,
                    "url": metadata["final_url"],
                    "channel": endpoint.channel,
                }
                if identity["pass"]:
                    exact_metadata_sources.append(
                        {"source": key, "identifier": metadata["final_url"]}
                    )
            elif endpoint.parser == "html_search":
                summaries[key] = {
                    "available": True,
                    "status": metadata["status"],
                    "final_url": metadata["final_url"],
                    "body_contains_title_query": normalize_text(title)
                    in normalize_text(decoded_body(result)),
                    "discovery_only": True,
                }
            else:
                summaries[key] = {"available": True, "parser": endpoint.parser}
        except Exception as exc:  # noqa: BLE001 - parser failures remain evidence
            summaries[key] = {
                "available": False,
                "parse_error": f"{type(exc).__name__}: {exc}",
            }
    ordered_repositories = sorted(
        repositories.values(), key=lambda item: item["full_name"].casefold()
    )
    ordered_metadata_sources = sorted(
        exact_metadata_sources,
        key=lambda item: (item["source"], str(item["identifier"])),
    )
    return summaries, ordered_repositories, ordered_metadata_sources


def inspect_github_repositories(
    config: dict[str, Any],
    repositories: list[dict[str, Any]],
    *,
    timeout: float,
    max_workers: int,
    max_attempts: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    title = config["paper"]["title"]
    authors = config["paper"]["authors"]
    readme_endpoints = []
    by_key: dict[str, dict[str, Any]] = {}
    for index, repository in enumerate(repositories):
        full_name = repository["full_name"]
        branch = repository.get("default_branch") or "main"
        key = f"github_readme_{index}"
        by_key[key] = repository
        readme_endpoints.append(
            Endpoint(
                key,
                "GitHub repository README",
                f"https://raw.githubusercontent.com/{full_name}/{branch}/README.md",
                "raw_readme",
            )
        )
    readme_results = (
        fetch_all(
            readme_endpoints,
            timeout=timeout,
            max_workers=max_workers,
            max_attempts=max_attempts,
        )
        if readme_endpoints
        else {}
    )
    candidates: list[dict[str, Any]] = []
    for key, repository in by_key.items():
        readme_result = readme_results[key]
        readme = (
            decoded_body(readme_result) if readme_result["metadata"]["transport_success"] else ""
        )
        identity = exact_paper_identity(
            repository_identity_text(repository, readme),
            title=title,
            authors=authors,
        )
        if not identity["pass"]:
            continue
        full_name = repository["full_name"]
        branch = repository.get("default_branch") or "main"
        followups = [
            Endpoint(
                "detail",
                "GitHub candidate detail",
                f"https://api.github.com/repos/{full_name}",
                "json",
                True,
            ),
            Endpoint(
                "tree",
                "GitHub candidate tree",
                f"https://api.github.com/repos/{full_name}/git/trees/{branch}?recursive=1",
                "json",
                True,
            ),
            Endpoint(
                "commit",
                "GitHub candidate commit",
                f"https://api.github.com/repos/{full_name}/commits/{branch}",
                "json",
                True,
            ),
        ]
        followed = fetch_all(
            followups,
            timeout=timeout,
            max_workers=3,
            max_attempts=max_attempts,
        )
        detail = json_body(followed["detail"])
        tree = json_body(followed["tree"])
        commit = json_body(followed["commit"])
        paths = [item.get("path", "") for item in tree.get("tree") or []]
        matches = critical_domain_matches(text=readme, paths=paths)
        domains = [domain for domain, terms in matches.items() if terms]
        candidate = evaluate_artifact_candidate(
            {
                "source": "GitHub",
                "name": full_name,
                "url": detail.get("html_url"),
                "exact_paper_identity": identity["pass"],
                "identity": identity,
                "anonymous_retrieval": all(
                    result["metadata"]["transport_success"] for result in followed.values()
                ),
                "stable_identifier": commit.get("sha"),
                "critical_domains": domains,
                "critical_domain_matches": matches,
                "license": detail.get("license"),
                "dependencies": [
                    path
                    for path in paths
                    if Path(path).name.casefold()
                    in {
                        "requirements.txt",
                        "environment.yml",
                        "environment.yaml",
                        "pyproject.toml",
                        "dockerfile",
                        ".gitmodules",
                    }
                ],
                "license_and_dependencies_recorded": True,
                "excluded_noise": False,
            }
        )
        candidates.append(candidate)
    return candidates, readme_results


def zenodo_candidates(config: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for hit in summary.get("hits") or []:
        identity = exact_paper_identity(
            str(hit.get("title", ""))
            + "\n"
            + "\n".join(str(item.get("name", "")) for item in hit.get("creators") or []),
            title=config["paper"]["title"],
            authors=config["paper"]["authors"],
        )
        if not identity["pass"]:
            continue
        matches = critical_domain_matches(
            text=json.dumps(hit, ensure_ascii=False), paths=hit.get("files") or []
        )
        domains = [domain for domain, terms in matches.items() if terms]
        candidates.append(
            evaluate_artifact_candidate(
                {
                    "source": "Zenodo",
                    "name": hit.get("title"),
                    "url": (hit.get("links") or {}).get("html"),
                    "exact_paper_identity": True,
                    "identity": identity,
                    "anonymous_retrieval": True,
                    "stable_identifier": hit.get("doi") or hit.get("id"),
                    "critical_domains": domains,
                    "critical_domain_matches": matches,
                    "license": hit.get("license"),
                    "dependencies": [],
                    "license_and_dependencies_recorded": True,
                    "excluded_noise": False,
                }
            )
        )
    return candidates


def qualified_official_identity_sources(
    summaries: dict[str, Any], identity_config: dict[str, Any]
) -> dict[str, list[str]]:
    """Select exact official sources only from registered provenance classes."""
    venue_classes = set(
        identity_config.get("official_venue_classes") or ["venue_program_identity_only"]
    )
    author_classes = set(
        identity_config.get("author_controlled_classes")
        or [
            "coauthor_homepage_identity_only",
            "corresponding_author_homepage_identity_only",
        ]
    )

    def select(classes: set[str]) -> list[str]:
        return sorted(
            key
            for key, summary in summaries.items()
            if key.startswith("official_page_")
            and summary.get("channel") in classes
            and summary.get("identity", {}).get("pass") is True
        )

    return {
        "venue": select(venue_classes),
        "author_controlled": select(author_classes),
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    local_preflight = preflight(config)
    if args.preflight_only:
        print(json.dumps(local_preflight, indent=2, sort_keys=True))
        return 0 if local_preflight["pass"] else 1
    if not local_preflight["pass"]:
        print(json.dumps(local_preflight, indent=2, sort_keys=True))
        return 1

    browser_data = load_yaml(PROJECT_ROOT / config["browser_snapshot"]["path"])
    endpoints = build_endpoints(config, browser_data)
    started = time.perf_counter()
    fetched = fetch_all(
        endpoints,
        timeout=float(args.timeout),
        max_workers=int(args.max_workers),
        max_attempts=int(args.max_attempts),
    )
    required_failures = [
        endpoint.key
        for endpoint in endpoints
        if endpoint.required_transport
        and not fetched[endpoint.key]["metadata"]["transport_success"]
    ]
    if required_failures:
        print(
            json.dumps(
                {
                    "error": "required H33 transport failed before report serialization",
                    "required_failures": required_failures,
                    "responses": {key: fetched[key]["metadata"] for key in required_failures},
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    summaries, repositories, metadata_sources = parse_results(config, fetched)
    github_candidates, readme_results = inspect_github_repositories(
        config,
        repositories,
        timeout=float(args.timeout),
        max_workers=int(args.max_workers),
        max_attempts=int(args.max_attempts),
    )
    candidates = [
        *github_candidates,
        *zenodo_candidates(config, summaries["zenodo_title"]),
    ]
    qualifying = [candidate for candidate in candidates if candidate["pass"]]

    github_search_keys = sorted(key for key in summaries if key.startswith("github_search_"))
    gitlab_search_keys = sorted(key for key in summaries if key.startswith("gitlab_search_"))
    official_identity_sources = qualified_official_identity_sources(
        summaries, config.get("identity_integrity") or {}
    )
    official_venue_sources = official_identity_sources["venue"]
    official_author_sources = official_identity_sources["author_controlled"]
    registered_queries = [item["query"] for item in browser_data["registered_queries"]]
    followup_queries = [item["query"] for item in browser_data["identifier_and_index_followups"]]
    unresolved_exact_leads = {
        key: summary["exact_identity_candidates"]
        for key, summary in summaries.items()
        if (key.startswith("gitlab_search_") or key in {"hf_models", "hf_datasets", "hf_spaces"})
        and summary.get("exact_identity_candidates")
    }
    channel_coverage = {
        "general_web_registered_queries": len(browser_data["registered_queries"])
        == len(config["web_queries"]),
        "general_web_identifier_followups": len(browser_data["identifier_and_index_followups"])
        >= 12,
        "official_venue_identity": bool(official_venue_sources),
        "official_author_identity": bool(official_author_sources),
        "crossref": summaries["crossref_doi"].get("identity", {}).get("pass") is True,
        "openalex": summaries["openalex_doi"].get("identity", {}).get("pass") is True,
        "bibliographic_alternatives_attempted": all(
            key in fetched for key in ("dblp_title", "arxiv_title", "semantic_scholar_doi")
        ),
        "github_queries": len(github_search_keys) == len(config["repository_queries"])
        and all(summaries[key].get("available") for key in github_search_keys),
        "github_author_and_lab": all(
            summaries[key].get("available")
            for key in ("github_were_repos", "github_synthesys_repos")
        ),
        "gitlab_queries": len(gitlab_search_keys) == len(config["repository_queries"])
        and all(summaries[key].get("available") for key in gitlab_search_keys),
        "gitee_query_and_attempt": any("site:gitee.com" in query for query in registered_queries)
        and "gitee_title_search" in fetched,
        "zenodo_query_and_attempt": any("site:zenodo.org" in query for query in registered_queries)
        and "zenodo_title" in fetched,
        "huggingface_all_types": all(
            summaries[key].get("available") for key in ("hf_models", "hf_datasets", "hf_spaces")
        ),
        "modelscope_query_and_attempt": any(
            "site:modelscope.cn" in query for query in followup_queries
        )
        and "modelscope_title_search" in fetched,
    }
    checks = {
        "local_preflight": local_preflight["pass"],
        "all_registered_channels": all(channel_coverage.values()),
        "exact_doi": summaries["crossref_doi"].get("doi", "").casefold() == DOI.casefold(),
        "crossref_has_no_artifact_link": not summaries["crossref_doi"].get("artifact_links"),
        "openalex_has_no_repository_fulltext": not bool(
            summaries["openalex_doi"].get("open_access", {}).get("any_repository_has_fulltext")
        ),
        "browser_snapshot_no_qualified_candidate": not browser_data[
            "qualifying_artifact_candidates"
        ],
        "candidate_gates_reapplied": all(
            candidate == evaluate_artifact_candidate(candidate) for candidate in candidates
        ),
        "no_unresolved_exact_repository_or_model_leads": not unresolved_exact_leads,
    }
    audit_integrity = all(checks.values())
    if qualifying:
        hypothesis_status = "supported"
    elif audit_integrity:
        hypothesis_status = "rejected"
    else:
        hypothesis_status = "inconclusive"

    response_metadata = {key: value["metadata"] for key, value in sorted(fetched.items())}
    readme_metadata = {key: value["metadata"] for key, value in sorted(readme_results.items())}
    unavailable = [
        key for key, value in response_metadata.items() if not value["transport_success"]
    ]
    report = {
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "classification": config["classification"],
        "validation_eligible": bool(config["validation_eligible"]),
        "cutoff_utc": config["cutoff_utc"],
        "git_commit": git_commit(),
        "doi": DOI,
        "ieee_document": IEEE_DOCUMENT,
        "local_preflight": local_preflight,
        "browser_snapshot_summary": {
            "registered_queries": len(browser_data["registered_queries"]),
            "identifier_and_index_followups": len(browser_data["identifier_and_index_followups"]),
            "author_repository_followups": len(browser_data["author_repository_followups"]),
            "qualifying_artifact_candidates": browser_data["qualifying_artifact_candidates"],
            "lineage_leads_not_exact_artifacts": browser_data["lineage_leads_not_exact_artifacts"],
            "rejected_noise_classes": browser_data["rejected_noise_classes"],
        },
        "response_snapshots": response_metadata,
        "github_readme_snapshots": readme_metadata,
        "parsed_sources": summaries,
        "exact_metadata_sources": metadata_sources,
        "official_identity_sources": official_identity_sources,
        "repository_counts": {
            "unique_github_repositories_inspected": len(repositories),
            "exact_github_identity_candidates": len(github_candidates),
            "qualifying_github_artifacts": sum(
                candidate["pass"] for candidate in github_candidates
            ),
        },
        "artifact_candidates": candidates,
        "unresolved_exact_repository_or_model_leads": unresolved_exact_leads,
        "qualifying_artifacts": qualifying,
        "qualifying_artifact_count": len(qualifying),
        "unavailable_or_blocked_endpoints": unavailable,
        "channel_coverage": channel_coverage,
        "checks": checks,
        "audit_integrity": audit_integrity,
        "hypothesis_status": hypothesis_status,
        "pass": audit_integrity,
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "endpoint_count": len(endpoints),
            "timeout_seconds_per_attempt": float(args.timeout),
            "max_attempts": max(1, int(args.max_attempts)),
            "max_workers": int(args.max_workers),
            "response_bytes": sum(int(value["metadata"]["bytes"]) for value in fetched.values()),
        },
        "limitations": config["limitations"],
    }
    output = PROJECT_ROOT / config["run"]["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "audit_integrity": audit_integrity,
                "hypothesis_status": hypothesis_status,
                "qualifying_artifact_count": len(qualifying),
                "exact_metadata_sources": len(metadata_sources),
                "unavailable_or_blocked_endpoints": unavailable,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit_integrity else 1


if __name__ == "__main__":
    raise SystemExit(main())
