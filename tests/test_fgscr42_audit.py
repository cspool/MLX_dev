import base64
import urllib.error
from pathlib import Path
from unittest.mock import Mock

from mlxsim import fgscr42_audit
from mlxsim.fgscr42_audit import (
    canonical_share_init,
    classify_download_value,
    compare_share_metadata,
    evaluate_input_decision,
    load_input_audit_config,
    parse_pcs_error,
)


def test_download_value_classification_does_not_need_raw_value() -> None:
    task = base64.b64encode(b"desktop-client-task").decode()
    assert classify_download_value(task) == "base64_desktop_task"
    assert classify_download_value("https://example.invalid/file") == "https_url"
    assert classify_download_value(None) == "missing"
    assert classify_download_value("not base64!") == "opaque_string"


def test_v2_manifest_extends_v1_without_losing_frozen_sources() -> None:
    config = load_input_audit_config(
        Path("configs/analysis/fgscr42_input_audit_v2.yaml")
    )
    assert config["run"]["id"] == "run_025"
    assert config["shares"]["official"]["fs_id"] == 90324070156283
    assert config["baidu"]["verify_surl_rule"] == "strip_short_link_literal_prefix_1"


def test_short_link_normalizes_to_token_only_init_url() -> None:
    surl, init_url = canonical_share_init(
        "https://pan.baidu.com/s/1eXplDfB5fCBPm7WMcFKZkg"
    )
    assert surl == "eXplDfB5fCBPm7WMcFKZkg"
    assert init_url == "https://pan.baidu.com/share/init?surl=eXplDfB5fCBPm7WMcFKZkg"


def test_share_metadata_matches_frozen_object() -> None:
    share = {
        "file_name": "FGSCR.zip",
        "fs_id": 90324070156283,
        "size_bytes": 5112338338,
        "server_ctime": 1608263618,
    }
    row = {
        "server_filename": "FGSCR.zip",
        "fs_id": "90324070156283",
        "size": "5112338338",
        "server_ctime": "1608263618",
    }
    assert compare_share_metadata(row, share)["pass"]
    row["size"] = "1"
    assert not compare_share_metadata(row, share)["pass"]


def test_pcs_error_parser_retains_only_stable_fields() -> None:
    parsed = parse_pcs_error(
        b'{"error_code":31064,"error_msg":"file is not authorized","request_id":123}'
    )
    assert parsed == {"error_code": 31064, "error_msg": "file is not authorized"}
    assert parse_pcs_error(b"not-json") == {"error_code": None, "error_msg": None}


def test_transport_timeout_is_a_sanitized_observation() -> None:
    opener = Mock()
    opener.open.side_effect = urllib.error.URLError(TimeoutError("timed out"))
    status, body, headers = fgscr42_audit._request_bytes(
        "https://example.invalid", opener=opener
    )
    assert status == 0
    assert body == b""
    assert headers == {"X-Audit-Transport-Error": "TimeoutError"}


def test_decision_requires_corpus_and_exact_split() -> None:
    paper = {"exact_split_disclosed": False}
    repository = {"versioned_split_present": False}
    blocked = {
        "archive_byte_retrievable": False,
        "class_label_organization_exposed": False,
        "archive_split_exposed": False,
    }
    decision = evaluate_input_decision(
        paper=paper, repository=repository, shares=[blocked]
    )
    assert decision["verdict"] == "rejected"
    assert decision["missing_required_inputs_fraction"] == 1.0

    available = {
        "archive_byte_retrievable": True,
        "class_label_organization_exposed": True,
        "archive_split_exposed": True,
    }
    decision = evaluate_input_decision(
        paper=paper, repository=repository, shares=[available]
    )
    assert decision["verdict"] == "supported"
    assert decision["missing_required_inputs_fraction"] == 0.0
