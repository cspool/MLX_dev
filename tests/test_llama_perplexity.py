import hashlib
import json
import math
from pathlib import Path

import pytest

from mlxsim.llama_perplexity import (
    audit_perplexity,
    complete_window_ranges,
    git_blob_sha1,
    qualify_model_files,
    window_accounting,
)


def test_h19_frozen_window_accounting() -> None:
    accounting = window_accounting(341468, 1024)
    assert accounting == {
        "windows": 333,
        "predicted_tokens": 340659,
        "discarded_tail_tokens": 476,
    }
    ranges = complete_window_ranges(341468, 1024)
    assert len(ranges) == 333
    assert ranges[0] == (0, 1024)
    assert ranges[-1] == (339968, 340992)
    assert all(end - start == 1024 for start, end in ranges)


def test_perplexity_audit_uses_token_weighted_nll() -> None:
    target = 6.62
    report = audit_perplexity(
        total_nll=math.log(target) * 100,
        predicted_tokens=100,
        target=target,
        relative_error_gate=0.10,
    )
    assert report["perplexity"] == pytest.approx(target)
    assert report["relative_error"] == pytest.approx(0.0)
    assert report["pass"] is True


def test_model_qualification_checks_hash_blobs_and_signature(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "weight.bin"
    weight.write_bytes(b"weights")
    metadata = model / "metadata.json"
    metadata.write_text('{"hidden_size": 16}\n', encoding="utf-8")
    config_path = model / "config.json"
    config_path.write_text(json.dumps({"hidden_size": 16}), encoding="utf-8")
    cfg = {
        "path": "model",
        "official_source": "official",
        "official_revision": "revision",
        "mirror_source": "mirror",
        "mirror_revision": "mirror-revision",
        "required_official_hashes": {
            "weight.bin": hashlib.sha256(b"weights").hexdigest(),
        },
        "required_official_git_blobs": {
            "metadata.json": git_blob_sha1(metadata),
        },
        "config_signature": {"hidden_size": 16},
    }
    report = qualify_model_files(cfg, project_root=tmp_path)
    assert report["pass"] is True
    weight.write_bytes(b"changed")
    assert qualify_model_files(cfg, project_root=tmp_path)["pass"] is False


@pytest.mark.parametrize("token_count,sequence_length", [(-1, 1024), (10, 1)])
def test_window_accounting_rejects_invalid_inputs(token_count: int, sequence_length: int) -> None:
    with pytest.raises(ValueError):
        window_accounting(token_count, sequence_length)
