"""Package audited real-GPU baselines without checkpoints or native libraries."""
import argparse
import json
from pathlib import Path
import shutil

from scripts.audit_mlx_gpu_benchmark import audit
from scripts.benchmark_mlx_gpu_models import require, ROOT
from scripts.run_mlx_tensor_semantics import sha
from scripts.mlx_system_attempt import record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bert-run", "bert-inventory", "llama-run", "llama-inventory", "tests", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); require(not out.exists(), "choose a fresh GPU package")
    copies = {}; audits = {}
    for label, directory, inventory_path in (("bert", args.bert_run.resolve(), args.bert_inventory.resolve()),
                                              ("llama2", args.llama_run.resolve(), args.llama_inventory.resolve())):
        audits[label] = audit(directory, inventory_path)
        provenance = json.loads((directory / "provenance.json").read_text())
        # All measured JSON, output vectors and environment snapshots; omit .so.
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".bin", ".log"} and not path.is_symlink():
                copies[path] = Path(label) / path.relative_to(directory)
        copies[inventory_path] = Path(label) / "reference/inventory.json"
        inventory = json.loads(inventory_path.read_text())
        for i, check in enumerate(inventory["reference_checks"]):
            if label == "llama2": copies[Path(check["logits_file"])] = Path(label) / f"reference/forward-{i}.f16.bin"
            else:
                for role, spec in check["outputs"].items(): copies[Path(spec["file"])] = Path(label) / f"reference/forward-{i}-{role}.f32.bin"
        for name, digest in provenance["sources"].items():
            source = Path(name)
            if source.is_relative_to(ROOT) and not source.is_relative_to(ROOT / ".venv"):
                require(sha(source) == digest, "measured benchmark source changed")
                copies[source] = Path("sources") / source.relative_to(ROOT)
    copies[args.tests.resolve()] = Path("regression.xml")
    for name in ("scripts/audit_mlx_gpu_benchmark.py", "scripts/package_mlx_gpu_benchmarks.py", "tests/test_gpu_model_benchmark.py"):
        copies[ROOT / name] = Path("sources") / name
    out.mkdir(parents=True); entries = []
    for source, relative in sorted(copies.items()):
        require(source.suffix != ".safetensors" and source.stat().st_size < 64 * 1024 * 1024, "unregistered large/checkpoint payload")
        digest = sha(source); target = out / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        require(sha(target) == digest == sha(source), "GPU package copy changed")
        entries.append(dict(path=str(relative), source_path=str(source), bytes=source.stat().st_size, sha256=digest))
    for label, result in audits.items(): record(out / f"{label}-audit.json", result)
    record(out / "manifest.json", dict(classification="full_model_real_rtx4090_measurements_not_simulator_accuracy_results",
        audits=audits, measured_full_models=["BERT-base-QA", "Llama2-7B"], file_count=len(entries), total_bytes=sum(e["bytes"] for e in entries),
        files=entries, model_weights_included=False, runtime_binaries_included=False, gpu_simulator_verified=False,
        mlx_event_vs_end_to_end_error_available=False, packager_sha256=sha(Path(__file__))))
    print(json.dumps(dict(file_count=len(entries), total_bytes=sum(e["bytes"] for e in entries), output=str(out))))


if __name__ == "__main__": main()
