"""Physical address ownership, not a claim of initialized model memory."""
import json
import random
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/mlx-model-storage"
BASE = 2**32


@pytest.fixture(scope="module")
def binary():
    for command in (["cmake", "-S", str(ROOT / "simulator_ext/model_storage"), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake", "--build", str(BUILD), "--target", "model-storage-contract", "-j4"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    return BUILD / "model-storage-contract"


def run(binary, path, actions, *, size=1024, base=BASE, alignment=64, error=None):
    path.mkdir(); job = {"base": base, "bytes": size, "alignment": alignment, "actions": actions}
    (path / "job.json").write_text(json.dumps(job))
    process = subprocess.run([str(binary), str(path / "job.json"), str(path / "result.json")], capture_output=True, text=True, timeout=60)
    if error:
        assert process.returncode != 0 and error in process.stderr, process.stdout + process.stderr
        return
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((path / "result.json").read_text())
    for event in report["events"]:
        state = event["state"]; ranges = sorted((r["base"], r["reserved_bytes"]) for r in state["allocations"])
        assert all(start >= base and length % alignment == 0 and start % alignment == 0 and start + length <= base + size for start, length in ranges)
        assert all(a + n <= b for (a, n), (b, _) in zip(ranges, ranges[1:]))
        assert sum(n for _, n in ranges) == state["reserved_bytes"]
        assert state["free_bytes"] + state["reserved_bytes"] == size
    assert not report["mlx_system_verified"] and not report["inference_performance_eligible"]
    return report


def alloc(name, elements, dtype="f32", **kwargs):
    return {"op": "allocate", "name": name, "dtype": dtype, "shape": [elements], **kwargs}


def test_builtin_pin_copy_and_arena_wrapper_lifetime(binary):
    process = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert process.returncode == 0 and "MODEL_STORAGE_CONTRACT_PASS" in process.stdout, process.stderr


def test_alias_and_inflight_pin_delay_reuse(binary, tmp_path):
    actions = [alloc("root", 17), {"op": "view", "name": "alias", "source": "root", "shape": [4], "strides": [2], "offset": 1},
        {"op": "release", "name": "root"}, {"op": "pin", "name": "read", "source": "alias"},
        {"op": "release", "name": "alias"}, alloc("other", 16),
        alloc("cannot_fit", 17, error="exhausted"), {"op": "unpin", "name": "read"}, alloc("reused", 17)]
    report = run(binary, tmp_path / "alias", actions, size=256); events = report["events"]
    first, other, reused = (events[n]["allocation"] for n in (0, 5, 8))
    assert first["base"] != other["base"] and reused["base"] == first["base"] and reused["id"] > first["id"]
    assert events[4]["state"]["allocations"][0]["pins"] == 1


def test_weight_permission_seal_requires_drained_pins(binary, tmp_path):
    report = run(binary, tmp_path / "seal", [alloc("w", 9), {"op": "pin", "name": "loader", "source": "w", "write": True},
        {"op": "seal", "name": "w", "error": "outstanding pins"}, {"op": "unpin", "name": "loader"},
        {"op": "seal", "name": "w"}, {"op": "pin", "name": "write", "source": "w", "write": True, "error": "read-only"},
        {"op": "pin", "name": "read", "source": "w"}])
    assert report["events"][-1]["allocation"]["writable"] is False


def test_fragmentation_failure_is_transactional_and_free_ranges_coalesce(binary, tmp_path):
    report = run(binary, tmp_path / "fragmented", [alloc("a", 16), alloc("b", 16), alloc("c", 16), alloc("d", 16),
        {"op": "release", "name": "b"}, {"op": "release", "name": "d"}, alloc("failed", 17, error="fragmented"),
        {"op": "release", "name": "c"}, alloc("wide", 48), {"op": "release", "name": "a"}, {"op": "release", "name": "wide"}], size=256)
    assert report["final"]["free_ranges"] == [{"base": BASE, "bytes": 256}]


def test_empty_tensors_have_distinct_identity_without_data_bytes(binary, tmp_path):
    report = run(binary, tmp_path / "empty", [alloc("a", 0), alloc("b", 0)], size=128)
    a, b = (e["allocation"] for e in report["events"])
    assert a["bytes"] == b["bytes"] == 0 and a["base"] != b["base"] and a["id"] != b["id"]
    assert report["final"]["live_bytes"] == 0 and report["final"]["reserved_bytes"] == 128


def test_foreign_storage_and_invalid_views_are_rejected(binary, tmp_path):
    run(binary, tmp_path / "invalid", [alloc("a", 4), {"op": "foreign", "name": "x", "error": "virtual tensor"},
        {"op": "clone_storage", "name": "clone", "source": "a", "error": "does not belong"},
        {"op": "view", "name": "bad", "source": "a", "shape": [3], "strides": [2], "error": "storage"},
        {"op": "view", "name": "negative", "source": "a", "offset": -1, "error": "invalid physical view"}])


@pytest.mark.parametrize("base,size,alignment,message", [(BASE, 1024, 3, "power of two"), (BASE, 1024, 4, "at least 8"),
    (BASE+1, 1024, 64, "not aligned"), (BASE, 65, 64, "not aligned"), (2**64-64, 128, 64, "overflow"), (BASE, 0, 64, "not aligned")])
def test_arena_extent_validation(binary, tmp_path, base, size, alignment, message):
    run(binary, tmp_path / "extent", [], base=base, size=size, alignment=alignment, error=message)


def test_large_model_bytes_are_address_metadata_not_host_allocations(binary, tmp_path):
    report = run(binary, tmp_path / "large", [alloc("full_weights", 6738415616, "f16", writable=False)], size=16*2**30)
    assert report["final"]["live_bytes"] == 6738415616*2
    assert report["events"][0]["allocation"]["base"] == BASE


def test_byte_multiplication_overflow_does_not_allocate(binary, tmp_path):
    run(binary, tmp_path / "overflow", [alloc("too_big", 2**62, "i64", error="byte extent overflow")])


def test_randomized_lifetimes_preserve_ownership_and_capacity(binary, tmp_path):
    rng = random.Random(906); live = []; actions = []
    for index in range(500):
        if live and (len(live) > 40 or rng.random() < 0.48):
            name = rng.choice(live); live.remove(name); actions.append({"op": "release", "name": name})
        else:
            name = f"v{index}"; live.append(name); actions.append(alloc(name, rng.randrange(0, 65)))
    actions.extend({"op": "release", "name": name} for name in live)
    report = run(binary, tmp_path / "random", actions, size=16384)
    assert report["final"]["allocations"] == [] and report["final"]["free_ranges"] == [{"base": BASE, "bytes": 16384}]
