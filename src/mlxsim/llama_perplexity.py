"""Input qualification and metric helpers for native Llama perplexity runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def qualify_model_files(
    model_config: dict[str, Any], *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    model_path = project_root / model_config["path"]
    sha_checks: dict[str, Any] = {}
    for filename, expected in model_config["required_official_hashes"].items():
        path = model_path / filename
        actual = sha256_file(path) if path.is_file() else None
        sha_checks[filename] = {
            "bytes": path.stat().st_size if path.is_file() else None,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "pass": actual == expected,
        }

    blob_checks: dict[str, Any] = {}
    for filename, expected in model_config["required_official_git_blobs"].items():
        path = model_path / filename
        actual = git_blob_sha1(path) if path.is_file() else None
        blob_checks[filename] = {
            "bytes": path.stat().st_size if path.is_file() else None,
            "expected_git_blob_sha1": expected,
            "actual_git_blob_sha1": actual,
            "pass": actual == expected,
        }

    serialized_path = model_path / "config.json"
    serialized = (
        json.loads(serialized_path.read_text(encoding="utf-8"))
        if serialized_path.is_file()
        else {}
    )
    signature = {name: serialized.get(name) for name in model_config["config_signature"]}
    signature_pass = signature == model_config["config_signature"]
    all_sha_pass = all(item["pass"] for item in sha_checks.values())
    all_blob_pass = all(item["pass"] for item in blob_checks.values())
    return {
        "model_path": str(model_path.relative_to(project_root)),
        "official_source": model_config["official_source"],
        "official_revision": model_config["official_revision"],
        "mirror_source": model_config["mirror_source"],
        "mirror_revision": model_config["mirror_revision"],
        "files": sha_checks,
        "git_blobs": blob_checks,
        "config_signature": signature,
        "config_signature_pass": signature_pass,
        "pass": all_sha_pass and all_blob_pass and signature_pass,
    }


def window_accounting(token_count: int, sequence_length: int) -> dict[str, int]:
    if token_count < 0:
        raise ValueError("token_count cannot be negative")
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    windows = token_count // sequence_length
    return {
        "windows": windows,
        "predicted_tokens": windows * (sequence_length - 1),
        "discarded_tail_tokens": token_count - windows * sequence_length,
    }


def complete_window_ranges(token_count: int, sequence_length: int) -> list[tuple[int, int]]:
    """Return only full non-overlapping windows, excluding any short tail."""

    accounting = window_accounting(token_count, sequence_length)
    return [
        (index * sequence_length, (index + 1) * sequence_length)
        for index in range(accounting["windows"])
    ]


def audit_perplexity(
    *, total_nll: float, predicted_tokens: int, target: float, relative_error_gate: float
) -> dict[str, Any]:
    if not math.isfinite(total_nll) or total_nll < 0:
        raise ValueError("total_nll must be finite and non-negative")
    if predicted_tokens <= 0:
        raise ValueError("predicted_tokens must be positive")
    if target <= 0:
        raise ValueError("target must be positive")
    perplexity = math.exp(total_nll / predicted_tokens)
    relative_error = abs(perplexity - target) / target
    return {
        "total_negative_log_likelihood": total_nll,
        "predicted_tokens": predicted_tokens,
        "perplexity": perplexity,
        "paper_target_perplexity": target,
        "relative_error": relative_error,
        "relative_error_gate": relative_error_gate,
        "pass": relative_error <= relative_error_gate,
    }
