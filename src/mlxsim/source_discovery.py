"""Helpers for source-qualified public-artifact discovery audits."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def normalize_text(value: str) -> str:
    """Normalize punctuation, markup, case, and whitespace for identity checks."""
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def exact_paper_identity(
    text: str,
    *,
    title: str,
    authors: Sequence[str],
    minimum_authors: int = 1,
) -> dict[str, Any]:
    """Check exact normalized title plus a minimum number of named authors."""
    normalized = normalize_text(text)
    normalized_title = normalize_text(title)
    matched_authors = [author for author in authors if normalize_text(author) in normalized]
    checks = {
        "title": normalized_title in normalized,
        "authors": len(matched_authors) >= minimum_authors,
    }
    return {
        "matched_authors": matched_authors,
        "checks": checks,
        "pass": all(checks.values()),
    }


CRITICAL_DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "architecture_simulator_rtl_mapping": (
        "simulator",
        "simulation",
        "rtl",
        "verilog",
        "systemverilog",
        "mapper",
        "mapping",
        "cycle accurate",
        "trace",
    ),
    "structured_operator_model_training": (
        "butterfly",
        "bsmm",
        "fft compression",
        "fft cmp",
        "lora",
        "finetun",
        "checkpoint",
        "train",
    ),
    "dataset_evaluator_checkpoint_manifest": (
        "dataset",
        "evaluation",
        "evaluator",
        "manifest",
        "checkpoint",
        "winogrande",
        "ada leval",
        "wikitext",
        "fgscr",
    ),
    "native_trace_raw_measurement": (
        "nsight",
        "ncu",
        "trace",
        "raw result",
        "measurement",
        "benchmark result",
        "power report",
        "synthesis report",
    ),
}


def critical_domain_matches(*, text: str, paths: Iterable[str] = ()) -> dict[str, list[str]]:
    """Return transparent term matches for the four registered domains."""
    haystack = normalize_text("\n".join((text, *paths)))
    return {
        domain: [term for term in terms if normalize_text(term) in haystack]
        for domain, terms in CRITICAL_DOMAIN_TERMS.items()
    }


def evaluate_artifact_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Apply H33's five conjunctive artifact-qualification gates."""
    checks = {
        "exact_paper_identity": bool(candidate.get("exact_paper_identity")),
        "anonymous_retrieval": bool(candidate.get("anonymous_retrieval")),
        "stable_identifier": bool(candidate.get("stable_identifier")),
        "critical_domain": bool(candidate.get("critical_domains")),
        "license_and_dependencies_recorded": bool(
            candidate.get("license_and_dependencies_recorded")
        ),
        "not_excluded_noise": not bool(candidate.get("excluded_noise")),
    }
    return {
        **dict(candidate),
        "checks": checks,
        "pass": all(checks.values()),
    }


def crossref_artifact_links(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep non-paper Crossref links/relations as potential artifact leads."""
    links: list[dict[str, Any]] = []
    for item in record.get("link") or []:
        url = str(item.get("URL", ""))
        content_type = str(item.get("content-type", ""))
        if "pdf" not in url.casefold() and "pdf" not in content_type.casefold():
            links.append(dict(item))
    relation = record.get("relation") or {}
    for relation_name, values in relation.items():
        for value in values or []:
            links.append({"relation": relation_name, **dict(value)})
    return links


def repository_identity_text(repository: Mapping[str, Any], readme: str = "") -> str:
    """Build the text eligible for exact identity checks on a repository."""
    values = (
        repository.get("full_name"),
        repository.get("path_with_namespace"),
        repository.get("name"),
        repository.get("description"),
        repository.get("homepage"),
        readme,
    )
    return "\n".join(str(value) for value in values if value)
