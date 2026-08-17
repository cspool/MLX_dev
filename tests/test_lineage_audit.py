from __future__ import annotations

from email.message import Message
from typing import Self

from scripts import audit_mlx_lineage as audit


class _OversizeResponse:
    status = 200
    url = "https://example.test/final"

    def __init__(self, declared_bytes: int) -> None:
        self.headers = Message()
        self.headers["Content-Type"] = "application/pdf"
        self.headers["Content-Length"] = str(declared_bytes)
        self.read_called = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        self.read_called = True
        return b"forbidden"


def test_t1_and_t2_endpoint_builders_are_separate() -> None:
    config = {
        "candidates": [
            {"id": "known", "title": "Known Paper", "doi": "10.1/known"},
            {"id": "unknown", "title": "Unknown Paper"},
            {"id": "cn_patent", "identifier": "CN1"},
        ],
        "primary_endpoints": [],
        "patent_metadata_lookups": [],
    }
    t1 = audit.build_t1_endpoints(config)
    assert {item.channel for item in t1} == {"Crossref", "OpenAlex"}
    assert all(item.tier == "T1" for item in t1)
    assert all("semanticscholar" not in item.url for item in t1)

    t2 = audit.build_t2_endpoints(config, {"known": "10.1/known", "unknown": None})
    assert len(t2) == 2
    assert all(item.tier == "T2" for item in t2)
    assert all("semanticscholar" in item.url for item in t2)


def test_declared_oversize_payload_is_not_downloaded(monkeypatch) -> None:
    response = _OversizeResponse(declared_bytes=101)
    monkeypatch.setattr(audit.urllib.request, "urlopen", lambda *_args, **_kwargs: response)
    endpoint = audit.Endpoint(
        "pdf",
        ("candidate",),
        "publisher",
        "https://example.test/paper.pdf",
        "pdf",
        "publisher_full_text",
        "T1-primary",
    )
    result = audit.fetch_endpoint(
        endpoint,
        timeout=1.0,
        max_attempts=1,
        max_response_bytes=100,
    )
    assert response.read_called is False
    assert result["body"] == b""
    assert result["metadata"]["payload_limit_blocked"] is True
    assert result["metadata"]["transport_success"] is False


def test_patent_candidate_title_adds_unprefixed_application_alias() -> None:
    title, aliases = audit.candidate_titles({"id": "patent", "identifier": "CN202510992730.2"})
    assert title == "CN202510992730.2"
    assert aliases == ("202510992730.2",)


def test_exact_chip_numeric_difference_is_not_a_family_conflict() -> None:
    candidate = {"id": "dfu_e", "title": "DFU-E: A Dataflow Architecture"}
    numeric_difference = "The proposed DFU-E chip uses 16 nm, while MLX uses 12 nm."
    explicit_separation = "MLX is from a different architecture family from DFU-E."
    assert audit.strict_family_conflicts([numeric_difference], candidate) == []
    assert len(audit.strict_family_conflicts([explicit_separation], candidate)) == 1
