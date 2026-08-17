"""Sanitized public-input audit for the FGSCR-42 ViT experiments."""

from __future__ import annotations

import base64
import hashlib
import http.cookiejar
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fgscr42_input_audit_v1.yaml"
DEFAULT_USER_AGENT = "MLX-paper-reproduction/0.1 public-input-audit"


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_input_audit_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the frozen H21 manifest."""

    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "extends" not in config:
        return config
    base_path = PROJECT_ROOT / str(config["extends"])
    return _deep_merge(load_input_audit_config(base_path), config)


def sha256_file(path: str | Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_paper_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Check the frozen paper and search its ViT result paragraph for recipe fields."""

    path = PROJECT_ROOT / str(config["source"])
    text = path.read_text(encoding="utf-8")
    start = text.find("The ViT model")
    end = text.find("TABLE III", start)
    if start < 0 or end < 0:
        vit_section = ""
    else:
        vit_section = text[start:end]
    lowered = vit_section.lower()
    fields = {
        field: [term for term in terms if term.lower() in lowered]
        for field, terms in config["required_recipe_terms"].items()
    }
    actual_hash = sha256_file(path)
    phrases = {
        phrase: phrase in text for phrase in config["disclosed_vit_phrases"]
    }
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "expected_sha256": config["sha256"],
        "actual_sha256": actual_hash,
        "hash_pass": actual_hash == config["sha256"],
        "disclosed_phrase_checks": phrases,
        "disclosed_phrases_pass": all(phrases.values()),
        "recipe_term_matches_in_vit_paragraph": fields,
        "missing_recipe_fields": [field for field, matches in fields.items() if not matches],
        "exact_split_disclosed": bool(fields.get("data_split")),
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def inspect_official_repository(config: Mapping[str, Any]) -> dict[str, Any]:
    """Scan the pinned official repository, including every reachable commit."""

    root = PROJECT_ROOT / str(config["local_root"])
    revision = _git(root, "rev-parse", "HEAD")
    head_files = sorted(_git(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
    commits = _git(root, "rev-list", "--all").splitlines()
    branches = _git(root, "for-each-ref", "--format=%(refname)", "refs/heads").splitlines()
    tags_text = _git(root, "tag", "--list")
    tags = tags_text.splitlines() if tags_text else []
    history_files: set[str] = set()
    for commit in commits:
        history_files.update(
            _git(root, "ls-tree", "-r", "--name-only", commit).splitlines()
        )

    suffixes = tuple(str(item).lower() for item in config["forbidden_tree_suffixes"])
    patterns = [str(item).lower() for item in config["split_name_patterns"]]
    archive_or_manifest_files = sorted(
        path for path in history_files if path.lower().endswith(suffixes)
    )
    split_named_files = sorted(
        path for path in history_files if any(pattern in Path(path).name.lower() for pattern in patterns)
    )
    readme = root / "README.md"
    actual_readme_hash = sha256_file(readme)
    worktree_clean = (
        subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet"], check=False
        ).returncode
        == 0
        and subprocess.run(
            ["git", "-C", str(root), "diff", "--cached", "--quiet"], check=False
        ).returncode
        == 0
    )
    checks = {
        "revision": revision == config["revision"],
        "readme_sha256": actual_readme_hash == config["readme_sha256"],
        "head_tree": head_files == sorted(config["expected_head_files"]),
        "commit_count": len(commits) == int(config["expected_commit_count"]),
        "branch_count": len(branches) == int(config["expected_branch_count"]),
        "tag_count": len(tags) == int(config["expected_tag_count"]),
        "tracked_files_clean": worktree_clean,
    }
    return {
        "path": str(root.relative_to(PROJECT_ROOT)),
        "expected_revision": config["revision"],
        "actual_revision": revision,
        "expected_readme_sha256": config["readme_sha256"],
        "actual_readme_sha256": actual_readme_hash,
        "head_files": head_files,
        "commit_count": len(commits),
        "branch_count": len(branches),
        "tag_count": len(tags),
        "all_history_unique_file_count": len(history_files),
        "archive_or_manifest_files_in_history": archive_or_manifest_files,
        "split_or_label_named_files_in_history": split_named_files,
        "versioned_split_present": bool(split_named_files),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _request_bytes(
    url: str,
    *,
    opener: urllib.request.OpenerDirector | None = None,
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 20,
) -> tuple[int, bytes, Mapping[str, str]]:
    request_headers = {"User-Agent": DEFAULT_USER_AGENT, **dict(headers or {})}
    request = urllib.request.Request(url, data=data, headers=request_headers)
    client = opener or urllib.request.build_opener()
    try:
        with client.open(request, timeout=timeout) as response:
            return int(response.status), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as error:
        return int(error.code), error.read(), dict(error.headers.items())


def _request_json(
    url: str,
    *,
    opener: urllib.request.OpenerDirector | None = None,
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 20,
) -> tuple[int, Any]:
    status, body, _ = _request_bytes(
        url, opener=opener, data=data, headers=headers, timeout=timeout
    )
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        payload = {"parse_error": True, "body_bytes": len(body)}
    return status, payload


def inspect_independent_index(config: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    """Verify the pinned independent index without cloning it."""

    status, body, _ = _request_bytes(str(config["raw_readme_url"]), timeout=timeout)
    text = body.decode("utf-8", errors="replace")
    checks = {
        "http_status": status == 200,
        "baidu_link": str(config["required_baidu_fragment"]) in text,
        "empty_google_link": str(config["required_empty_google_link"]) in text,
    }
    return {
        "url": config["url"],
        "revision": config["revision"],
        "http_status": status,
        "readme_sha256": hashlib.sha256(body).hexdigest(),
        "checks": checks,
        "pass": all(checks.values()),
    }


def inspect_github_issues(config: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    """Capture a compact public issue snapshot."""

    status, payload = _request_json(
        str(config["api_url"]),
        headers={"Accept": "application/vnd.github+json"},
        timeout=timeout,
    )
    issues = [] if not isinstance(payload, list) else payload
    public_issues = [item for item in issues if "pull_request" not in item]
    by_number = {int(item["number"]): item for item in public_issues}
    expected = [int(item) for item in config["expected_issue_numbers"]]
    compact = [
        {
            "number": number,
            "state": by_number[number].get("state"),
            "title": by_number[number].get("title"),
            "comment_count": by_number[number].get("comments"),
        }
        for number in expected
        if number in by_number
    ]
    return {
        "http_status": status,
        "expected_issue_numbers": expected,
        "observed_issue_numbers": sorted(by_number),
        "expected_issues_present": all(number in by_number for number in expected),
        "download_or_completeness_issues": config["download_or_completeness_issues"],
        "label_issues": config["label_issues"],
        "issues": compact,
        "pass": status == 200 and all(number in by_number for number in expected),
    }


def inspect_huggingface_searches(urls: Sequence[str], *, timeout: float) -> dict[str, Any]:
    """Search the public Hugging Face dataset catalog and retain only identifiers."""

    searches = []
    for url in urls:
        status, payload = _request_json(str(url), timeout=timeout)
        rows = payload if isinstance(payload, list) else []
        searches.append(
            {
                "query": urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get(
                    "search", [""]
                )[0],
                "http_status": status,
                "match_count": len(rows),
                "dataset_ids": [str(row.get("id")) for row in rows if row.get("id")],
            }
        )
    return {
        "searches": searches,
        "all_requests_pass": all(item["http_status"] == 200 for item in searches),
        "total_match_count": sum(item["match_count"] for item in searches),
    }


def classify_download_value(value: Any) -> str:
    """Classify a Baidu download field without retaining its sensitive value."""

    if not isinstance(value, str) or not value:
        return "missing"
    if value.startswith(("https://", "http://")):
        return "https_url"
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error):
        return "opaque_string"
    return "base64_desktop_task" if decoded else "opaque_string"


def _download_value(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    direct = payload.get("dlink")
    if isinstance(direct, str):
        return direct
    rows = payload.get("list")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        value = rows[0].get("dlink")
        return value if isinstance(value, str) else None
    return None


def _errno(payload: Any) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("errno")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cookie_value(jar: http.cookiejar.CookieJar, name: str) -> str | None:
    for cookie in jar:
        if cookie.name == name:
            return cookie.value
    return None


def _query_url(base: str, params: Mapping[str, Any]) -> str:
    return f"{base}?{urllib.parse.urlencode(params)}"


def _share_file(payload: Any) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    rows = payload.get("list")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        return rows[0]
    return None


def compare_share_metadata(file_row: Mapping[str, Any] | None, share: Mapping[str, Any]) -> dict[str, Any]:
    """Compare a share-list row with the frozen public object identity."""

    observed = {
        "file_name": None if file_row is None else file_row.get("server_filename"),
        "fs_id": None if file_row is None else int(file_row.get("fs_id", -1)),
        "size_bytes": None if file_row is None else int(file_row.get("size", -1)),
        "server_ctime": None if file_row is None else int(file_row.get("server_ctime", -1)),
    }
    expected = {
        "file_name": share["file_name"],
        "fs_id": int(share["fs_id"]),
        "size_bytes": int(share["size_bytes"]),
        "server_ctime": int(share["server_ctime"]),
    }
    checks = {key: observed[key] == value for key, value in expected.items()}
    return {"expected": expected, "observed": observed, "checks": checks, "pass": all(checks.values())}


def parse_pcs_error(body: bytes) -> dict[str, Any]:
    """Extract only stable PCS error fields from a Range response."""

    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {"error_code": None, "error_msg": None}
    if not isinstance(payload, Mapping):
        return {"error_code": None, "error_msg": None}
    code = payload.get("error_code")
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = None
    return {"error_code": code, "error_msg": payload.get("error_msg")}


def _replace_host(url: str, host: str) -> str:
    parts = urllib.parse.urlsplit(url)
    port = f":{parts.port}" if parts.port else ""
    return urllib.parse.urlunsplit(
        (parts.scheme, f"{host}{port}", parts.path, parts.query, parts.fragment)
    )


def canonical_share_init(short_url: str) -> tuple[str, str]:
    """Convert `/s/1TOKEN` into the TOKEN-only share-init endpoint contract."""

    path = urllib.parse.urlsplit(short_url).path
    prefix = "/s/1"
    if not path.startswith(prefix) or len(path) <= len(prefix):
        raise ValueError("expected a Baidu short URL with /s/1TOKEN")
    verify_surl = path[len(prefix) :]
    init_url = f"https://pan.baidu.com/share/init?surl={verify_surl}"
    return verify_surl, init_url


def _probe_ranges(
    dlink: str,
    *,
    public_opener: urllib.request.OpenerDirector,
    baidu: Mapping[str, Any],
    timeout: float,
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for host in baidu["pcs_hosts"]:
        probe_url = _replace_host(dlink, str(host))
        for agent_name, user_agent in baidu["range_user_agents"].items():
            for use_public_cookie in (False, True):
                opener = public_opener if use_public_cookie else urllib.request.build_opener()
                status, body, headers = _request_bytes(
                    probe_url,
                    opener=opener,
                    headers={
                        "User-Agent": str(user_agent),
                        "Range": str(baidu["range_header"]),
                    },
                    timeout=timeout,
                )
                error = parse_pcs_error(body)
                probes.append(
                    {
                        "host": host,
                        "user_agent_class": agent_name,
                        "public_share_cookie": use_public_cookie,
                        "http_status": status,
                        "response_bytes": len(body),
                        "content_type": headers.get("Content-Type"),
                        **error,
                        "archive_byte_returned": status in {200, 206} and len(body) > 0,
                    }
                )
    return probes


def _zip_entries(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    rows = payload.get("list")
    if not isinstance(rows, list):
        data = payload.get("data")
        rows = data.get("list") if isinstance(data, Mapping) else []
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("path") or row.get("server_filename"))
        for row in rows
        if isinstance(row, Mapping) and (row.get("path") or row.get("server_filename"))
    ]


def audit_baidu_share(
    name: str,
    share: Mapping[str, Any],
    baidu: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the frozen anonymous checks and return no auth material or URLs."""

    timeout = float(baidu["timeout_seconds"])
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    if baidu.get("verify_surl_rule") == "strip_short_link_literal_prefix_1":
        verify_surl, init_url = canonical_share_init(str(share["url"]))
    else:
        verify_surl, init_url = str(share["surl"]), str(share["url"])
    _request_bytes(init_url, opener=opener, timeout=timeout)

    common = {
        "channel": "chunlei",
        "web": 1,
        "app_id": int(baidu["app_id"]),
        "clienttype": 0,
    }
    verify_url = _query_url(
        str(baidu["verify_endpoint"]),
        {
            **common,
            "surl": verify_surl,
            "t": int(time.time() * 1000),
            **baidu.get("verify_extra_query", {}),
        },
    )
    verify_data = urllib.parse.urlencode(
        {"pwd": share["public_passcode"], "vcode": "", "vcode_str": ""}
    ).encode()
    verify_status, verify_payload = _request_json(
        verify_url,
        opener=opener,
        data=verify_data,
        headers={
            "Referer": init_url,
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
        timeout=timeout,
    )

    list_url = _query_url(
        str(baidu["list_endpoint"]),
        {
            **common,
            "uk": int(share["share_uk"]),
            "shareid": int(share["share_id"]),
            "page": 1,
            "num": 100,
            "dir": "/",
            "order": "time",
            "desc": 1,
            "showempty": 0,
        },
    )
    list_status, list_payload = _request_json(list_url, opener=opener, timeout=timeout)
    metadata = compare_share_metadata(_share_file(list_payload), share)

    tpl_url = _query_url(
        str(baidu["tplconfig_endpoint"]),
        {"surl": share["surl"], "fields": "sign,timestamp", "view_mode": 1},
    )
    tpl_status, tpl_payload = _request_json(tpl_url, opener=opener, timeout=timeout)
    tpl_data = tpl_payload.get("data", tpl_payload) if isinstance(tpl_payload, Mapping) else {}
    sign = tpl_data.get("sign") if isinstance(tpl_data, Mapping) else None
    timestamp = tpl_data.get("timestamp") if isinstance(tpl_data, Mapping) else None
    sekey = _cookie_value(jar, "BDCLND")

    download_data = urllib.parse.urlencode(
        {
            "encrypt": 0,
            "product": "share",
            "uk": int(share["share_uk"]),
            "primaryid": int(share["share_id"]),
            "fid_list": json.dumps([int(share["fs_id"])]),
            "path_list": "",
            "extra": json.dumps(
                {"sekey": urllib.parse.unquote(sekey or "")}, separators=(",", ":")
            ),
        }
    ).encode()
    download_query = {
        **common,
        "sign": sign or "",
        "timestamp": timestamp or "",
        "bdstoken": "null",
    }
    individual_status, individual_payload = _request_json(
        _query_url(str(baidu["sharedownload_endpoint"]), download_query),
        opener=opener,
        data=download_data,
        timeout=timeout,
    )
    individual_value = _download_value(individual_payload)
    individual_kind = classify_download_value(individual_value)

    batch_status, batch_payload = _request_json(
        _query_url(
            str(baidu["sharedownload_endpoint"]), {**download_query, "type": "batch"}
        ),
        opener=opener,
        data=download_data,
        timeout=timeout,
    )
    batch_value = _download_value(batch_payload)
    batch_kind = classify_download_value(batch_value)
    range_probes = (
        _probe_ranges(batch_value, public_opener=opener, baidu=baidu, timeout=timeout)
        if batch_kind == "https_url" and batch_value is not None
        else []
    )

    zip_results = []
    zip_entries: list[str] = []
    for path_value in (str(share["file_name"]), f"/{share['file_name']}"):
        zip_url = _query_url(
            str(baidu["zip_list_endpoint"]),
            {
                **common,
                "shareid": int(share["share_id"]),
                "uk": int(share["share_uk"]),
                "fs_id": int(share["fs_id"]),
                "path": path_value,
                "page": 1,
                "num": 500,
            },
        )
        zip_status, zip_payload = _request_json(zip_url, opener=opener, timeout=timeout)
        entries = _zip_entries(zip_payload)
        zip_entries.extend(entries)
        zip_results.append(
            {
                "path_form": "rooted" if path_value.startswith("/") else "relative",
                "http_status": zip_status,
                "errno": _errno(zip_payload),
                "entry_count": len(entries),
            }
        )

    top_levels = {
        path.strip("/").split("/")[0] for path in zip_entries if path.strip("/")
    }
    split_terms = ("train", "test", "valid", "val")
    archive_split_named = sorted(
        path for path in zip_entries if any(term in path.lower() for term in split_terms)
    )
    return {
        "name": name,
        "public_identity": {
            "share_uk": int(share["share_uk"]),
            "share_id": int(share["share_id"]),
            "file_name": share["file_name"],
            "fs_id": int(share["fs_id"]),
            "size_bytes": int(share["size_bytes"]),
            "server_ctime": int(share["server_ctime"]),
        },
        "verify": {"http_status": verify_status, "errno": _errno(verify_payload)},
        "listing": {"http_status": list_status, "errno": _errno(list_payload)},
        "metadata": metadata,
        "signing": {
            "http_status": tpl_status,
            "errno": _errno(tpl_payload),
            "dynamic_sign_present": bool(sign),
            "timestamp_present": timestamp is not None,
            "public_share_cookie_present": sekey is not None,
        },
        "individual_download": {
            "http_status": individual_status,
            "errno": _errno(individual_payload),
            "payload_kind": individual_kind,
            "encoded_value_length": len(individual_value or ""),
        },
        "batch_download": {
            "http_status": batch_status,
            "errno": _errno(batch_payload),
            "payload_kind": batch_kind,
        },
        "range_probes": range_probes,
        "zip_listings": zip_results,
        "zip_top_level_count": len(top_levels),
        "archive_split_named_entries": archive_split_named,
        "archive_byte_retrievable": any(
            probe["archive_byte_returned"] for probe in range_probes
        ),
        "class_label_organization_exposed": len(top_levels) == 42,
        "archive_split_exposed": bool(archive_split_named),
    }


def evaluate_input_decision(
    *,
    paper: Mapping[str, Any],
    repository: Mapping[str, Any],
    shares: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply H21's two necessary input gates."""

    corpus_gate = any(
        bool(share["archive_byte_retrievable"])
        and bool(share["class_label_organization_exposed"])
        for share in shares
    )
    split_gate = bool(paper["exact_split_disclosed"]) or bool(
        repository["versioned_split_present"]
    ) or any(bool(share["archive_split_exposed"]) for share in shares)
    gates = {"corpus": corpus_gate, "experiment_split": split_gate}
    missing = [name for name, passed in gates.items() if not passed]
    return {
        "gates": gates,
        "missing_required_inputs": missing,
        "missing_required_inputs_fraction": len(missing) / len(gates),
        "input_sufficient": all(gates.values()),
        "verdict": "supported" if all(gates.values()) else "rejected",
    }


def _baidu_observations_match(
    shares: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for share in shares:
        range_probes = share["range_probes"]
        item = {
            "name": share["name"],
            "verify_errno": share["verify"]["errno"] == int(expected["verify_errno"]),
            "list_errno": share["listing"]["errno"] == int(expected["list_errno"]),
            "metadata": bool(share["metadata"]["pass"]),
            "individual_payload_kind": share["individual_download"]["payload_kind"]
            == expected["individual_payload_kind"],
            "range_probe_count": len(range_probes) == 18,
            "range_block": bool(range_probes)
            and all(
                probe["http_status"] == int(expected["range_http_status"])
                and probe["error_code"] == int(expected["range_error_code"])
                for probe in range_probes
            ),
            "zip_block": all(
                item["errno"] == int(expected["zip_errno"])
                for item in share["zip_listings"]
            ),
        }
        item["pass"] = all(value for key, value in item.items() if key != "name")
        checks.append(item)
    return {"shares": checks, "pass": all(item["pass"] for item in checks)}


def project_git_revision() -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_input_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the complete frozen H21 audit."""

    started = time.monotonic()
    timeout = float(config["baidu"]["timeout_seconds"])
    paper = inspect_paper_recipe(config["paper"])
    repository = inspect_official_repository(config["official_repository"])
    index = inspect_independent_index(config["independent_index"], timeout=timeout)
    issues = inspect_github_issues(config["github_issues"], timeout=timeout)
    huggingface = inspect_huggingface_searches(
        config["huggingface_searches"], timeout=timeout
    )
    shares = [
        audit_baidu_share(name, share, config["baidu"])
        for name, share in config["shares"].items()
    ]
    baidu_match = _baidu_observations_match(shares, config["baidu"]["expected"])
    decision = evaluate_input_decision(
        paper=paper, repository=repository, shares=shares
    )
    source_checks = {
        "paper": paper["hash_pass"] and paper["disclosed_phrases_pass"],
        "official_repository": repository["pass"],
        "independent_index": index["pass"],
        "github_issues": issues["pass"],
        "huggingface_requests": huggingface["all_requests_pass"],
        "baidu_reference_observations": baidu_match["pass"],
    }
    return {
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "protocol": config["run"]["protocol"],
        "classification": "public-input-availability-audit",
        "validation_eligible": False,
        "project_git_revision": project_git_revision(),
        "paper": paper,
        "official_repository": repository,
        "independent_index": index,
        "github_issues": issues,
        "huggingface_catalog": huggingface,
        "baidu_shares": shares,
        "baidu_observation_checks": baidu_match,
        "decision": decision,
        "audit_integrity": {
            "source_checks": source_checks,
            "pass": all(source_checks.values()),
            "secrets_serialized": False,
            "large_archive_downloaded": False,
        },
        "wall_time_seconds": time.monotonic() - started,
    }
