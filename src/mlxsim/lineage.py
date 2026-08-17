"""Evidence-gated helpers for MLX architectural-lineage audits."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def normalize_text(value: str) -> str:
    """Normalize markup, Unicode punctuation, case, and whitespace."""
    value = html.unescape(value)
    value = re.sub(r"<script\b[^>]*>.*?</script>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<style\b[^>]*>.*?</style>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def normalize_doi(value: str | None) -> str | None:
    """Return a lower-case bare DOI or ``None`` for a non-DOI value."""
    if not value:
        return None
    normalized = html.unescape(str(value)).strip().casefold()
    normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized)
    normalized = re.sub(r"^doi:\s*", "", normalized)
    normalized = normalized.rstrip(". ,;)")
    return normalized if normalized.startswith("10.") and "/" in normalized else None


def inverted_abstract_text(index: Mapping[str, Sequence[int]] | None) -> str:
    """Reconstruct an OpenAlex inverted-index abstract deterministically."""
    if not index:
        return ""
    positioned: list[tuple[int, str]] = []
    for token, positions in index.items():
        positioned.extend((int(position), str(token)) for position in positions)
    positioned.sort()
    return " ".join(token for _, token in positioned)


def title_identity(
    observed: str,
    *,
    title: str,
    aliases: Sequence[str] = (),
) -> bool:
    """Check an exact normalized title against registered aliases."""
    normalized = normalize_text(observed)
    accepted = (title, *aliases)
    return any(normalize_text(candidate) in normalized for candidate in accepted)


def deduplicate_doi_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate bibliographic records by DOI while preserving source provenance."""
    grouped: dict[str, dict[str, Any]] = {}
    without_doi: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        doi = normalize_doi(item.get("doi"))
        if doi is None:
            without_doi.append(item)
            continue
        if doi not in grouped:
            grouped[doi] = {
                "doi": doi,
                "candidate_ids": [],
                "source_records": [],
            }
        candidate_id = item.get("candidate_id")
        if candidate_id and candidate_id not in grouped[doi]["candidate_ids"]:
            grouped[doi]["candidate_ids"].append(candidate_id)
        grouped[doi]["source_records"].append(item)
    output = []
    for doi in sorted(grouped):
        grouped[doi]["candidate_ids"].sort()
        grouped[doi]["source_records"].sort(
            key=lambda item: (str(item.get("source_id")), str(item.get("title")))
        )
        output.append(grouped[doi])
    output.extend(
        sorted(
            without_doi,
            key=lambda item: (
                str(item.get("candidate_id")),
                str(item.get("source_id")),
                str(item.get("title")),
            ),
        )
    )
    return output


# These expressions deliberately encode high-specificity phrases. A lone word
# such as "dataflow", "mesh", "reuse", or "RISC-V" cannot match a gate.
FEATURE_PATTERNS: dict[str, tuple[str, ...]] = {
    "taped_out_verilog": (
        r"(?:tape(?:d[- ]?|[- ])?out|fabricat(?:ed|ion)).{0,180}(?:verilog|rtl)",
        r"(?:verilog|rtl).{0,180}(?:tape(?:d[- ]?|[- ])?out|fabricat(?:ed|ion))",
    ),
    "process_12nm": (r"(?<!\d)12\s*nm(?!\d)",),
    "frequency_1ghz": (r"(?<![\d.])1(?:\.0)?\s*ghz(?!\d)",),
    "simd_width_32": (r"simd\s*[-_ ]?32\b", r"32\s*[- ]?(?:lane|wide).{0,30}simd"),
    "mesh_dimensions": (
        r"\b\d+\s*[x×]\s*\d+\s+(?:pe\s+)?(?:array|mesh)\b",
        r"(?:array|mesh).{0,30}\b\d+\s*[x×]\s*\d+\b",
    ),
    "peak_about_1tops": (
        r"(?:about|approximately|approx\.?|~)?\s*1(?:\.0)?\s*t(?:op|flop)/?s\b",
        r"(?:peak|performance).{0,50}\b1(?:\.0)?\s*t(?:op|flop)/?s\b",
    ),
    "pe_array_area_7_712_mm2": (r"7\.712\s*(?:mm\s*(?:2|²)|mm\^2)",),
    "pe_array_power_5_846_w": (
        r"5\.846(?:4)?\s*w\b",
        r"5846\.4\s*mw\b",
    ),
    "risc_v_host": (
        r"risc\s*[- ]?v.{0,50}(?:host|controller)",
        r"(?:host|controller).{0,50}risc\s*[- ]?v",
    ),
    "per_pe_or_dataflow_assembly": (
        r"dataflow(?:[- ]style)?\s+assembly",
        r"(?:per|each)\s*[- ]?pe.{0,70}(?:assembly|operations?)",
    ),
    "llvm_compiler": (
        r"llvm.{0,50}(?:compiler|compilation)",
        r"(?:compiler|compilation).{0,50}llvm",
    ),
    "spatial_assembler": (r"spatial\s+assembler",),
    "binary_header_configuration": (
        r"(?:binary|bitstream).{0,80}header\s+file.{0,80}configur",
        r"header\s+file.{0,80}(?:binary|bitstream).{0,80}configur",
    ),
    "programmable_spatial_dataflow_pes": (
        r"programmable.{0,90}(?:spatial|dataflow).{0,90}(?:processing element|pe\b)",
        r"(?:processing element|pe\b).{0,90}programmable.{0,90}(?:spatial|dataflow)",
    ),
    "heterogeneous_functional_units": (
        r"heterogeneous.{0,70}(?:functional units?|fu\b)",
        r"(?:functional units?|fu\b).{0,70}heterogeneous",
    ),
    "decoupled_load_compute_transfer": (
        r"decoupled.{0,100}(?:load|memory).{0,100}comput.{0,100}(?:transfer|store|communicat)",
        r"(?:load|memory).{0,100}comput.{0,100}(?:transfer|store|communicat).{0,100}decoupled",
    ),
    "explicit_operand_routing": (
        r"operands?.{0,80}(?:rout(?:e|ed|ing)|interconnect)",
        r"(?:rout(?:e|ed|ing)|interconnect).{0,80}operands?",
    ),
    "data_reuse": (
        r"data\s+reuse.{0,100}(?:register|buffer|on[- ]chip|mapping|instruction)",
        r"(?:register|buffer|on[- ]chip|mapping|instruction).{0,100}data\s+reuse",
        r"数据重用.{0,30}(?:指令|映射|缓存|寄存器)",
    ),
    "instruction_reuse_or_hierarchy": (
        r"instruction\s+reuse.{0,100}(?:buffer|mapping|cache|hierarch)",
        r"(?:task|instruction[- ]block|instruction|data)\s*[-,;/ ]+(?:level\s+)?parallelism.{0,180}(?:task|instruction[- ]block|instruction|data)",
        r"(?:multi|multiple).{0,40}(?:task|instruction[- ]block|instruction|data).{0,90}(?:level|parallel)",
        r"指令复用.{0,30}(?:数据|处理器|缓存|映射)",
    ),
    "closed_dependency_components": (r"closed\s+dependency\s+components?", r"\bcdcs?\b"),
    "tagged_instruction_blocks": (r"tag(?:ged|[- ]based).{0,80}(?:instruction|block|schedul)",),
    "bounded_active_layer_window": (r"bounded.{0,70}(?:active|layer).{0,70}(?:window|layers?)",),
    "skip_hop_links": (r"skip[- ]hop",),
    "semantic_fft_compression": (
        r"semantic.{0,70}fft.{0,70}compress",
        r"fft.{0,70}semantic.{0,70}compress",
    ),
    "hierarchical_bsmm": (
        r"hierarchical.{0,60}(?:bsmm|butterfly)",
        r"(?:bsmm|butterfly).{0,60}hierarchical",
    ),
    "explicit_simict_use": (
        r"(?:use[sd]?|using|implement(?:ed|ing)?|simulat(?:e|ed|or)).{0,100}simict",
        r"simict.{0,100}(?:use[sd]?|using|implement(?:ed|ing)?|simulat(?:e|ed|or))",
    ),
    "explicit_simict_derivation": (
        r"(?:derived|built|based).{0,80}simict",
        r"simict[- ]based",
    ),
}


NUMERIC_HARDWARE_FEATURES = {
    "process_12nm",
    "frequency_1ghz",
    "simd_width_32",
    "peak_about_1tops",
    "pe_array_area_7_712_mm2",
    "pe_array_power_5_846_w",
}

# The supplied text calls the parent mesh compact but does not expose its exact
# dimensions in machine-readable text. A candidate's arbitrary mesh dimensions
# therefore cannot count as a matching exact-parent fingerprint.
NONCOMPARABLE_HARDWARE_FEATURES = {"mesh_dimensions"}


def _excerpt(text: str, match: re.Match[str], *, maximum_words: int = 20) -> str:
    words = normalize_text(text).split()
    matched = normalize_text(match.group(0)).split()
    if not matched:
        return ""
    needle = " ".join(matched)
    joined = " ".join(words)
    start_character = joined.find(needle)
    if start_character < 0:
        return " ".join(matched[:maximum_words])
    start_word = joined[:start_character].count(" ")
    left = max(0, start_word - 4)
    return " ".join(words[left : left + maximum_words])


def scan_feature(text: str, feature: str) -> dict[str, Any] | None:
    """Return the first high-specificity match for a frozen feature."""
    patterns = FEATURE_PATTERNS.get(feature, ())
    normalized = normalize_text(text)
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return {
                "pattern": pattern,
                "excerpt": _excerpt(normalized, match),
            }
    return None


def evaluate_candidate_features(
    *,
    candidate_id: str,
    source_texts: Sequence[Mapping[str, Any]],
    frozen_feature_classes: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Build a source-qualified feature matrix for one candidate."""
    classes: dict[str, Any] = {}
    for class_name, features in frozen_feature_classes.items():
        observations = {}
        for feature in features:
            evidence = []
            for source in source_texts:
                if source.get("candidate_id") != candidate_id:
                    continue
                if not source.get("feature_eligible"):
                    continue
                if match := scan_feature(str(source.get("text", "")), str(feature)):
                    evidence.append(
                        {
                            "source_id": source["source_id"],
                            "source_class": source.get("source_class"),
                            "excerpt": match["excerpt"],
                            "pattern": match["pattern"],
                        }
                    )
            observations[str(feature)] = {
                "status": "reported" if evidence else "not_reported",
                "evidence": evidence,
            }
        classes[str(class_name)] = observations
    return {"candidate_id": candidate_id, "feature_classes": classes}


def reported_features(matrix: Mapping[str, Any], class_name: str) -> list[str]:
    """List reported features in a matrix class."""
    values = matrix["feature_classes"].get(class_name, {})
    return sorted(
        feature for feature, observation in values.items() if observation["status"] == "reported"
    )


def evaluate_exact_parent_gate(
    matrix: Mapping[str, Any],
    *,
    explicit_primary_link: bool,
    material_conflict: bool,
    minimum_hardware_fingerprints: int,
    minimum_exact_numeric_fingerprints: int,
) -> dict[str, Any]:
    """Apply the pre-registered exact-parent decision gate."""
    reported_hardware = reported_features(matrix, "parent_hardware_fingerprints")
    hardware = sorted(set(reported_hardware) - NONCOMPARABLE_HARDWARE_FEATURES)
    numeric = sorted(set(hardware) & NUMERIC_HARDWARE_FEATURES)
    fingerprint_path = (
        len(hardware) >= minimum_hardware_fingerprints
        and len(numeric) >= minimum_exact_numeric_fingerprints
    )
    checks = {
        "explicit_primary_link_or_fingerprint_path": explicit_primary_link or fingerprint_path,
        "minimum_hardware_fingerprints": len(hardware) >= minimum_hardware_fingerprints,
        "minimum_exact_numeric_fingerprints": len(numeric) >= minimum_exact_numeric_fingerprints,
        "no_material_conflict": not material_conflict,
    }
    return {
        "explicit_primary_link": explicit_primary_link,
        "reported_noncomparable_hardware_fields": sorted(
            set(reported_hardware) & NONCOMPARABLE_HARDWARE_FEATURES
        ),
        "reported_hardware_fingerprints": hardware,
        "reported_exact_numeric_fingerprints": numeric,
        "fingerprint_path": fingerprint_path,
        "material_conflict": material_conflict,
        "checks": checks,
        "pass": (explicit_primary_link or fingerprint_path) and not material_conflict,
    }


def evaluate_family_gate(
    matrix: Mapping[str, Any],
    *,
    explicit_family_link: bool,
    prior_tapeout_or_ownership: bool,
    material_conflict: bool,
    minimum_high_specificity_cross_class_matches: int,
) -> dict[str, Any]:
    """Apply the architecture-family gate without counting authorship or generic words."""
    software = reported_features(matrix, "software_interface")
    substrate = reported_features(matrix, "execution_substrate")
    matches = sorted(
        [
            *(f"software_interface:{item}" for item in software),
            *(f"execution_substrate:{item}" for item in substrate),
        ]
    )
    cross_class = bool(software) and bool(substrate)
    evidence_path = (
        prior_tapeout_or_ownership
        and cross_class
        and len(matches) >= minimum_high_specificity_cross_class_matches
    )
    checks = {
        "explicit_family_link_or_evidence_path": explicit_family_link or evidence_path,
        "prior_tapeout_or_ownership": prior_tapeout_or_ownership,
        "software_and_substrate_both_represented": cross_class,
        "minimum_high_specificity_matches": len(matches)
        >= minimum_high_specificity_cross_class_matches,
        "shared_authorship_counted": False,
        "generic_terminology_counted": False,
        "no_material_conflict": not material_conflict,
    }
    return {
        "explicit_family_link": explicit_family_link,
        "prior_tapeout_or_ownership": prior_tapeout_or_ownership,
        "software_interface_matches": software,
        "execution_substrate_matches": substrate,
        "high_specificity_cross_class_matches": matches,
        "material_conflict": material_conflict,
        "checks": checks,
        "pass": (explicit_family_link or evidence_path) and not material_conflict,
    }
