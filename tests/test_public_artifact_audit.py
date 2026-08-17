from __future__ import annotations

from email.message import Message
from typing import Self
from urllib.parse import parse_qs, urlparse

from scripts import audit_mlx_public_artifacts as audit


class _Response:
    status = 200
    url = "https://example.test/final"

    def __init__(self) -> None:
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return b"{}"


def test_transient_transport_classification() -> None:
    assert audit.transient_transport_failure(None, "TimeoutError") is True
    assert audit.transient_transport_failure(429, "rate limited") is True
    assert audit.transient_transport_failure(503, "unavailable") is True
    assert audit.transient_transport_failure(403, "forbidden") is False
    assert audit.transient_transport_failure(200, None) is False


def test_fetch_endpoint_retries_transient_failures(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(_request, *, timeout: float):
        nonlocal calls
        assert timeout == 1.0
        calls += 1
        if calls < 3:
            raise TimeoutError("temporary")
        return _Response()

    monkeypatch.setattr(audit.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(audit.time, "sleep", lambda _seconds: None)
    endpoint = audit.Endpoint("test", "test", "https://example.test", "json")
    result = audit.fetch_endpoint(endpoint, timeout=1.0, max_attempts=3)

    assert result["body"] == b"{}"
    assert result["metadata"]["transport_success"] is True
    assert result["metadata"]["attempt_count"] == 3
    assert [item["status"] for item in result["metadata"]["attempts"]] == [
        None,
        None,
        200,
    ]


def test_zenodo_endpoint_uses_registered_exact_title_phrase() -> None:
    title = "MLX: Multi-Layer Execution for Structured LLM Workload Acceleration"
    endpoints = audit.build_endpoints(
        {"paper": {"title": title}, "repository_queries": []},
        {"official_identity_sources": []},
    )
    endpoint = next(item for item in endpoints if item.key == "zenodo_title")
    query = parse_qs(urlparse(endpoint.url).query)

    assert query == {"q": [f'"{title}"'], "size": ["25"]}
    assert endpoint.required_transport is False
