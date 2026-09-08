"""Native sparse byte-storage contract; does not certify hardware memory."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sparse_cpp_backing_preserves_bytes_validity_and_generations():
    build = ROOT / "build/mlx-spike-matrix"
    for command in (
        ["cmake", "-S", str(ROOT / "system_sim/physical_device"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build), "--target", "mapped-memory-contract", "-j4"],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(build / "mapped-memory-contract")], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["randomized_steps"] == 3000
    backing = report["memory"]
    assert backing["capacity_bytes"] == 16 * 2**30
    assert backing["resident_pages"] == 5 and backing["writer_pages"] == 4
    assert backing["resident_data_bytes"] == 5 * 4096
    assert backing["validity_bytes"] == 5 * 512
    assert backing["writer_bytes"] == 4 * 4096 * 8
    assert not report["mlx_system_verified"] and not report["inference_performance_eligible"]
