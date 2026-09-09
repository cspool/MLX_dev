"""Publish complete QA correctness and component safety, excluding running results."""
import argparse
import json
from pathlib import Path
import shutil

from scripts.run_mlx_qa_model import require, full_bert_contract, audit_outputs, ROOT
from scripts.run_mlx_tensor_semantics import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "inventory", "safety", "performance", "output"): parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(); run = args.run.resolve(); safety = args.safety.resolve(); out = args.output.resolve()
    require(not out.exists(), "choose a fresh QA publication directory")
    state = json.loads((run / "execution.json").read_text())
    require(state["status"] == "exited" and state["exit_code"] == 0 and state["mode"] == "microcode", "only complete QA correctness can be published")
    require(sha(run / "mlx-tensor-semantics") == state["binary_sha256"] and all(sha(Path(p)) == h for p, h in {**state["inputs"], **state["runtime_libraries"]}.items()), "QA execution bytes changed")
    inventory = json.loads(args.inventory.read_text()); program = json.loads((run / "program.json").read_text()); native = json.loads((run / "native/result.json").read_text())
    contract = full_bert_contract(inventory, program); comparison = audit_outputs(program, inventory, native)
    require(all(c["major_correctness_passed"] for c in comparison), "complete QA correctness failed")
    safe = json.loads((safety / "report.json").read_text())
    require(safe["regression_tests"] == 138 and len(safe["replays"]) == 19 and sum(r["expected_exit"] == 0 for r in safe["replays"]) == 7, "QA safety set incomplete")
    require(all(sha(Path(p)) == h for p, h in safe["files"].items()), "QA safety inputs changed")
    copies = {}
    for name in ("program.json", "coverage.json", "options.json", "acceptance-policy.json", "execution.json", "comparison.json", "build.log", "native.log", "native/result.json"):
        copies[run / name] = Path("full-model") / name
    for name, digest in state["sources"].items():
        snapshot = run / "sources" / name; require(sha(snapshot) == digest, "executed QA source snapshot changed")
        copies[snapshot] = Path("full-model/sources") / name
    copies[args.inventory.resolve()] = Path("reference/inventory.json")
    for check, actual in zip(inventory["reference_checks"], native["outputs"], strict=True):
        for role in ("start_logits", "end_logits"):
            source = Path(check["outputs"][role]["file"]); copies[source] = Path("reference") / source.name
            source = Path(actual["outputs"][role]["file"]); copies[source] = Path("full-model/native") / source.name
    for path in safety.rglob("*"):
        if path.is_file() and not path.is_symlink(): copies[path] = Path("safety") / path.relative_to(safety)
    for name in safe["files"]:
        source = Path(name)
        if source.is_relative_to(ROOT / "artifacts/tagged/qa-final-regression-001"):
            copies[source] = Path("tests") / source.relative_to(ROOT / "artifacts/tagged/qa-final-regression-001")
    for name, digest in safe["sources"].items():
        source = ROOT / name; require(sha(source) == digest, "QA safety sources changed")
        copies[source] = Path("safety/sources") / name
    copies[args.performance.resolve()] = Path("llama2-native-performance.json")
    perf = json.loads(args.performance.read_text())
    require(perf["native_device_performance_experiment_eligible"] and not perf["mlx_system_verified"], "performance scope differs")
    require(sha(Path(perf["provenance"]["run"]) / "native/result.json") == perf["provenance"]["native_report_sha256"], "performance native report changed")
    # Preserve terminal configuration-limit failures, but no running outputs.
    for case in ("bert-qa-physical-full-001", "bert-qa-paired-full-001"):
        base = run.parent / case
        failed = json.loads((base / "execution.json").read_text())
        require(failed["status"] == "exited" and failed["exit_code"] == 1, "expected prior limit failure is not terminal")
        for name in ("execution.json", "native.log", "options.json"): copies[base / name] = Path("prior-limit-failures") / case / name
    require(len(set(copies.values())) == len(copies), "duplicate publication destination")
    out.mkdir(parents=True); entries = []
    for source, relative in sorted(copies.items()):
        require(source.suffix in {".json", ".bin", ".log", ".xml", ".py", ".c", ".cc", ".h", ".sv", ".scala"} or source.name == "CMakeLists.txt", "unregistered QA publication payload")
        digest = sha(source); target = out / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        require(sha(target) == digest == sha(source), "QA publication copy changed")
        entries.append(dict(path=str(relative), source_path=str(source), bytes=source.stat().st_size, sha256=digest))
    manifest = dict(classification="complete_dense_bert_qa_native_correctness_and_component_safety_not_system",
                    full_bert_native_major_correctness_passed=True, input_contract=contract, actual_answers=[c["actual_span"]["text"] for c in comparison],
                    maximum_absolute_logit_error=max(v["max_abs_error"] for c in comparison for v in c["outputs"].values()),
                    regression_tests=138, safety_replays=19, mlx_system_verified=False, full_cycle_comparison_complete=False,
                    executed_sources_are_owned_snapshots_not_current_development_version=True,
                    files=entries, file_count=len(entries), total_bytes=sum(e["bytes"] for e in entries), packager_sha256=sha(Path(__file__)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("file_count", "total_bytes", "actual_answers", "maximum_absolute_logit_error")}))


if __name__ == "__main__": main()
