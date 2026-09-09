"""Package accepted GELU component evidence without model weights/binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from scripts.run_mlx_tensor_semantics import ROOT, sha


def require(value, message):
    if not value: raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--safety", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); tests = args.tests.resolve(); safety = args.safety.resolve(); out = args.output.resolve()
    require(not out.exists(), "choose a fresh GELU evidence package")
    report = json.loads((safety / "report.json").read_text())
    attempt = json.loads((safety / "attempt.json").read_text())
    require(attempt["status"] == "completed" and len(report["replays"]) == 14 and report["regression_tests"] == 392
            and report["legacy_full_model_compilation_equal"], "GELU verification is incomplete")
    require(sum(r["expected_exit"] == 0 for r in report["replays"]) == 7, "GELU replay set differs")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name, digest in report["sources"].items():
        require(sha(ROOT / name) == digest, "tested source changed")
        committed = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        require(hashlib.sha256(committed).hexdigest() == digest, "tested sources differ from publication commit")
    copies = {}
    for name, digest in report["files"].items():
        path = Path(name)
        require(sha(path) == digest, "tested input/result changed")
        if path.is_relative_to(tests): copies[path] = Path("tests") / path.relative_to(tests)
    for path in safety.rglob("*"):
        if path.is_file() and not path.is_symlink():
            require(path.suffix in {".json", ".bin", ".log"}, "unregistered safety payload type")
            copies[path] = Path("safety") / path.relative_to(safety)
    for index, replay in enumerate(report["replays"]):
        log = safety / f"asan-{index}.log"
        require(sha(log) == replay["log_sha256"] and not any(s in log.read_text() for s in ("ERROR: AddressSanitizer", "runtime error:", "LeakSanitizer", "DEADLYSIGNAL")), "sanitizer log changed or failed")
        require((Path(replay["output"]) / "result.json").exists() == (replay["expected_exit"] == 0), "GELU terminal output set differs")
    source_record = ROOT / "artifacts/tagged/gelu-reference-source-001/record.json"
    copies[source_record] = Path("formula-source.json")
    failed = ROOT / "artifacts/tagged/gelu-safety-001/watchdog-outcome.json"
    if failed.exists(): copies[failed] = Path("prior-attempt-watchdog.json")
    out.mkdir(parents=True); entries = []
    for source, relative in sorted(copies.items()):
        destination = out / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source); shutil.copy2(source, destination)
        require(sha(destination) == digest == sha(source), "publication copy changed")
        entries.append(dict(path=str(relative), source_path=str(source), bytes=source.stat().st_size, sha256=digest))
    manifest = dict(classification="accepted_gelu_component_numerical_and_safety_evidence_not_full_model", tested_code_commit=commit,
                    tested_sources=report["sources"], regression_tests=392, safety_replays=14, successful_replays=7, expected_rejections=7,
                    numeric_domains=report["numeric_domains"], historical_framework_difference_preserved=True,
                    legacy_full_model_compilation_equal=True, files=entries, file_count=len(entries), total_bytes=sum(e["bytes"] for e in entries),
                    omitted_external_inputs={p: h for p, h in report["files"].items() if not Path(p).is_relative_to(tests)},
                    full_model_execution_verified=False, mlx_system_verified=False, packager_sha256=sha(Path(__file__)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("file_count", "total_bytes", "tested_code_commit")}))


if __name__ == "__main__": main()
