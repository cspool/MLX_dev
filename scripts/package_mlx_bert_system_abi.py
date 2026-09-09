"""Package checked primitive ABI evidence; no whole-model/system promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

from scripts.mlx_system_attempt import digest, record

ROOT = Path(__file__).resolve().parents[1]
TAGGED = ROOT / "artifacts/tagged"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise ValueError("use a fresh publication directory")
    checks = {"system-bert-abi-regression-002.xml": 99,
              "system-bert-abi-graph-regression-001.xml": 29,
              "system-bert-primitive-chain-002/regression.xml": 3,
              "system-memory-v2-safety-002/regression.xml": 45}
    for name, expected in checks.items():
        suites = ET.parse(TAGGED / name).getroot().findall(".//testsuite")
        if (sum(int(s.attrib["tests"]) for s in suites) != expected or
                any(int(s.get(key, "0")) for s in suites for key in ("failures", "errors", "skipped"))):
            raise ValueError("required regression is incomplete")
    control = json.loads((TAGGED / "host-mask-control-002/report.json").read_text())
    safety = json.loads((TAGGED / "system-memory-v2-safety-002/report.json").read_text())
    if not control["all_passed"] or control["cases"] != 71 or control["serialized_commands_match_c_abi"] != 36:
        raise ValueError("RV64 control evidence incomplete")
    if len(safety["cases"]) != 34 or len(safety["memory_v2_decoder_replays"]) != 26:
        raise ValueError("required safety replay coverage incomplete")
    chain = TAGGED / "system-bert-primitive-chain-002/pytest"
    reports = [control, safety]
    for i in range(3):
        directory = chain / f"test_cpu_dispatches_real_index{i}" / "run"
        execution = json.loads((directory / "execution.json").read_text())
        if execution["status"] != "exited" or execution["exit_code"] != (10 if i == 2 else 0):
            raise ValueError("RV64 primitive chain not at expected terminal")
        device = json.loads((directory / "device.json").read_text())
        if i == 2:
            if device["launches"] or device["windows"] or not device["memory_idle"] or (directory / "report.json").exists():
                raise ValueError("wrong guard reached device work")
        else:
            report = json.loads((directory / "report.json").read_text())
            if report["source_calls"] != 8 or report["device_windows"] != 3 or not report["output_bytes_checked_in_elf"]:
                raise ValueError("primitive chain coverage incomplete")
            reports.append(report)
    sources = {}
    for report in reports:
        if report.get("full_model_execution_verified", False) or report["mlx_system_verified"]:
            raise ValueError("primitive evidence promoted to full model")
        for name, value in report["sources"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts or digest(ROOT / name) != value:
                raise ValueError(f"source identity differs: {name}")
            if name in sources and sources[name] != value:
                raise ValueError("evidence source snapshots differ")
            sources[name] = value
    out.mkdir(parents=True)
    snapshot = out / "snapshot"
    snapshot.mkdir()
    copied = {}
    def copy(source, name):
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        value = digest(source)
        if digest(target) != value:
            raise ValueError("publication copy changed bytes")
        copied[name] = value
    for name in sources:
        copy(ROOT / name, "sources/" + name)
    for name in ("tests/test_system_mask_control.py", "tests/test_system_memory_v2.py",
                 "tests/test_system_bert_primitive_chain.py", "scripts/package_mlx_bert_system_abi.py",
                 "docs/mlx-bert-system-abi.md", "docs/mlx-dfu-simulator-reference.md", "docs/mlx-mainline-acceptance.md"):
        copy(ROOT / name, "sources/" + name)
    for name in checks:
        copy(TAGGED / name, name)
    for group in ("host-mask-control-002", "system-memory-v2-safety-002", "system-bert-primitive-chain-002"):
        for source in sorted((TAGGED / group).rglob("*")):
            if (source.is_symlink() or any(p.is_symlink() for p in source.parents) or not source.is_file()
                    or source.suffix not in {".json", ".xml", ".log", ".c", ".txt", ".bin", ".elf"}):
                continue
            copy(source, str(source.relative_to(TAGGED)))
    manifest = {"classification": "tested_system_primitives_not_full_bert_or_chipyard_acceptance",
                "regressions": checks, "rv64_control_cases": 71, "cpu_command_encoding_matches": 36,
                "sanitizer_bus_replays": 34, "sanitizer_decoder_replays": 26,
                "primitive_chain_variants": ["normal", "changed_data", "guard_mismatch"],
                "full_model_execution_verified": False, "mlx_system_verified": False,
                "inference_performance_eligible": False, "files_sha256": copied,
                "excluded": ["host_binaries", "object_files", "model_checkpoints", "private_reference_card_bodies"]}
    record(snapshot / "manifest.json", manifest)
    archive = out / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(snapshot.rglob("*")):
            if path.is_file(): stream.add(path, arcname=str(path.relative_to(snapshot)), recursive=False)
    with tarfile.open(archive, "r:gz") as stream:
        expected = {**copied, "manifest.json": digest(snapshot / "manifest.json")}
        if {member.name for member in stream} != set(expected):
            raise ValueError("archive members differ from manifest")
        for name, value in expected.items():
            with stream.extractfile(name) as entry:
                if hashlib.file_digest(entry, "sha256").hexdigest() != value:
                    raise ValueError("archive payload digest differs")
    manifest.update(archive="evidence.tar.gz", archive_sha256=digest(archive), payload_files=len(copied), archive_bytes=archive.stat().st_size)
    record(out / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("classification", "payload_files", "archive_bytes", "archive_sha256")}, indent=2))


if __name__ == "__main__":
    main()
