"""Shared M4 entry gate and logged tool execution."""

import json
import subprocess
from pathlib import Path

from scripts.run_mlx_native_chipyard import ROOT, digest, validate_build
from scripts.verify_mlx_native_stage2 import fingerprints as system_fingerprints


def entry_gate():
    native = json.loads((ROOT / "artifacts/tagged/stage1-acceptance/certificate.json").read_text())
    system = json.loads((ROOT / "artifacts/tagged/stage2-acceptance/certificate.json").read_text())
    if native["status"] != "native_stage_passed" or any(
        digest(ROOT / path) != sha for path, sha in native["source_fingerprints"].items()
    ):
        raise RuntimeError("RTL entry requires a current native-stage certificate")
    if (
        system["status"] != "system_stage_passed"
        or system["source_fingerprints"] != system_fingerprints()
    ):
        raise RuntimeError("RTL entry requires a current system-stage certificate")
    attempt = Path(system["attempt"])
    if any(digest(attempt / path) != sha for path, sha in system["evidence_sha256"].items()):
        raise RuntimeError("system-stage evidence changed")
    chipyard = ROOT / "build/chipyard-native"
    simulator = chipyard / "sims/verilator/simulator-chipyard-MLXNativeRocketConfig"
    if validate_build(chipyard, simulator) != system["build"]:
        raise RuntimeError("system-stage build no longer matches its certificate")


def execute(command, log, *, timeout=180):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False
        )
    if result.returncode:
        raise RuntimeError(f"RTL command failed: {log}")
    return log.read_text()
