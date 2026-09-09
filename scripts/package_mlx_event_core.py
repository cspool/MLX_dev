"""Package accepted concurrent event components and full-model resource inventories."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from scripts.run_mlx_tensor_semantics import ROOT, sha
from scripts.mlx_system_attempt import record
from scripts.verify_mlx_event_schedule import require, audit_trace
from mlxsim.model_event_resources import resource_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("tests", "safety", "resources", "output"): parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--include-loops",action="store_true")
    parser.add_argument("--include-ports",action="store_true")
    parser.add_argument("--matrix-alignment",type=Path)
    args = parser.parse_args(); out = args.output.resolve(); tests = args.tests.resolve(); safety = args.safety.resolve()
    test_count,replay_count,success_count=(95,94,67) if args.include_ports else (55,58,37) if args.include_loops else (35,29,18)
    require(not out.exists(), "choose a fresh event publication")
    suite = ET.parse(tests / "regression.xml").getroot().find("testsuite")
    require(suite is not None and suite.get("tests") == str(test_count) and all(suite.get(k) == "0" for k in ("errors", "failures", "skipped")), "event regression not accepted")
    report = json.loads((safety / "report.json").read_text())
    require(report["regression_tests"] == test_count and len(report["replays"]) == replay_count and sum(r["expected_exit"] == 0 for r in report["replays"]) == success_count, "event safety scope incomplete")
    copies = {tests / "regression.xml": Path("tests/regression.xml")}
    for name, digest in report["sources"].items():
        p = ROOT / name; require(sha(p) == digest, "event tested source changed"); copies[p] = Path("sources") / name
    for row in report["replays"]:
        p = Path(row["program"]); require(sha(p) == report["inputs"][str(p)], "event test input changed")
        for source in p.parent.iterdir():
            if source.is_file() and source.suffix in {".json", ".log"}: copies[source] = Path("tests") / source.relative_to(tests)
        if row["expected_exit"] == 0:
            result = json.loads(Path(row["result"]).read_text())
            require(result == json.loads((p.parent / "result.json").read_text()), "event result differs after safety replay")
            audit_trace(json.loads(p.read_text()), result)
    for p in safety.iterdir():
        if p.is_file(): copies[p] = Path("safety") / p.name
    contracts = {}
    for p in args.resources.resolve().glob("*.json"):
        r = json.loads(p.read_text()); program = Path(r["program_path"])
        require(sha(program) == r["program_file_sha256"], "full event source program changed")
        fresh = resource_contract(json.loads(program.read_text()))
        require(all(r[k] == v for k,v in fresh.items()), "full event resource contract does not reproduce")
        contracts[p.stem] = dict(source_calls=r["source_calls"], lowered_calls=r["lowered_calls"], full_event_lowering_complete=False)
        copies[p] = Path("resources") / p.name
    require(set(contracts) == {"bert", "llama2"}, "both full-model resource inventories required")
    for name in ("scripts/package_mlx_event_core.py", "scripts/compile_mlx_event_resources.py"):
        copies[ROOT / name] = Path("sources") / name
    if args.include_loops or args.include_ports:copies[ROOT/"docs/mlx-event-loop-ir.md"]=Path("sources/docs/mlx-event-loop-ir.md")
    if args.include_ports:
        require(args.matrix_alignment is not None,"port publication needs actual matrix execution alignment")
        alignment=args.matrix_alignment.resolve();bound=json.loads((alignment/"report.json").read_text())
        require(len(bound["cases"])==22 and bound["all_window_cycles_and_work_equal"],"matrix alignment incomplete")
        for name,value in bound["sources"].items():
            require(sha(ROOT/name)==value,"matrix native/alignment source changed");copies[ROOT/name]=Path("sources")/name
        for p in alignment.rglob("*"):
            if p.is_file() and p.suffix in {".json",".bin",".log"}:copies[p]=Path("matrix-alignment")/p.relative_to(alignment)
        for row in bound["cases"]:
            job=Path(row["job"]);require(sha(job)==row["job_sha256"],"matrix input changed")
            copies[job]=Path("matrix-inputs")/job.parent.parent.name/"job.json"
        for name in ("docs/mlx-event-port-contract.md","scripts/compile_mlx_matrix_events.py"):
            copies[ROOT/name]=Path("sources")/name
    out.mkdir(parents=True); files = []
    for source, relative in sorted(copies.items()):
        target = out / relative; target.parent.mkdir(parents=True, exist_ok=True); digest = sha(source); shutil.copy2(source, target)
        require(sha(target) == digest == sha(source), "event publication copy changed")
        files.append(dict(path=str(relative), source_path=str(source), bytes=source.stat().st_size, sha256=digest))
    record(out / "manifest.json", dict(classification="concurrent_event_core_and_complete_resource_inventory_not_full_model_event_execution",
        regression_tests=test_count, sanitizer_replays=replay_count, successful_replays=success_count, expected_rejections=replay_count-success_count, full_program_resources=contracts,
        full_event_lowering_complete=False, event_vs_end_to_end_error_available=False, files=files,
        file_count=len(files), total_bytes=sum(f["bytes"] for f in files), packager_sha256=sha(Path(__file__))))
    print(json.dumps(dict(file_count=len(files), total_bytes=sum(f["bytes"] for f in files))))


if __name__ == "__main__": main()
