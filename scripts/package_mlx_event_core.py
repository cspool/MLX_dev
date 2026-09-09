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
    parser.add_argument("--include-vectors",action="store_true")
    parser.add_argument("--include-memory",action="store_true")
    parser.add_argument("--include-control",action="store_true")
    parser.add_argument("--include-source-order",action="store_true")
    parser.add_argument("--matrix-alignment",type=Path)
    parser.add_argument("--vector-alignment",type=Path)
    parser.add_argument("--memory-alignment",type=Path)
    parser.add_argument("--control-alignment",type=Path)
    parser.add_argument("--group-alignment",type=Path)
    args = parser.parse_args(); out = args.output.resolve(); tests = args.tests.resolve(); safety = args.safety.resolve()
    if args.include_source_order:args.include_control=True
    test_count,replay_count,success_count=(261,272,236) if args.include_source_order else (221,202,168) if args.include_control else (176,166,136) if args.include_memory else (148,142,115) if args.include_vectors else (95,94,67) if args.include_ports else (55,58,37) if args.include_loops else (35,29,18)
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
    if args.include_loops or args.include_ports or args.include_vectors or args.include_memory or args.include_control:copies[ROOT/"docs/mlx-event-loop-ir.md"]=Path("sources/docs/mlx-event-loop-ir.md")
    if args.include_ports or args.include_vectors or args.include_memory or args.include_control:
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
    if args.include_vectors or args.include_memory or args.include_control:
        require(args.vector_alignment is not None,"vector publication needs numerical window alignment")
        alignment=args.vector_alignment.resolve();bound=json.loads((alignment/"report.json").read_text())
        require(bound.get("family")=="vector" and len(bound["cases"])==48 and bound["all_window_cycles_and_work_equal"],"vector alignment incomplete")
        for name,value in bound["sources"].items():
            require(sha(ROOT/name)==value,"vector native/alignment source changed");copies[ROOT/name]=Path("sources")/name
        for p in alignment.rglob("*"):
            if p.is_file() and p.suffix in {".json",".bin",".log"}:copies[p]=Path("vector-alignment")/p.relative_to(alignment)
        for row in bound["cases"]:
            job=Path(row["job"]);require(sha(job)==row["job_sha256"],"vector input changed")
            copies[job]=Path("vector-inputs")/job.parent.parent.name/"job.json"
        copies[ROOT/"docs/mlx-vector-event-lowering.md"]=Path("sources/docs/mlx-vector-event-lowering.md")
    if args.include_memory or args.include_control:
        require(args.memory_alignment is not None,"memory publication needs numerical controller alignment")
        alignment=args.memory_alignment.resolve();bound=json.loads((alignment/"report.json").read_text())
        require(bound.get("family")=="memory" and len(bound["cases"])==18 and bound["all_window_cycles_and_work_equal"],"memory alignment incomplete")
        for name,value in bound["sources"].items():
            require(sha(ROOT/name)==value,"memory native/alignment source changed");copies[ROOT/name]=Path("sources")/name
        for p in alignment.rglob("*"):
            if p.is_file() and p.suffix in {".json",".bin",".log"}:copies[p]=Path("memory-alignment")/p.relative_to(alignment)
        for row in bound["cases"]:
            job=Path(row["job"]);native_job=Path(row["native_job"])
            require(sha(job)==row["job_sha256"] and sha(native_job)==row["native_job_sha256"],"memory input changed")
            native_data=json.loads(native_job.read_text())
            require(all(asset.get("kind")=="literal" for asset in native_data["assets"].values()),"memory publication requires self-contained test assets")
            destination=Path("memory-inputs")/job.parent.name
            copies[job]=destination/"memory-event-job.json"
            copies[native_job]=destination/"native/job.json"
        copies[ROOT/"docs/mlx-memory-event-lowering.md"]=Path("sources/docs/mlx-memory-event-lowering.md")
    if args.include_control:
        require(args.control_alignment is not None,"control publication needs numerical RV64 window alignment")
        alignment=args.control_alignment.resolve();bound=json.loads((alignment/"report.json").read_text())
        require(bound.get("family")=="control" and len(bound["cases"])==28 and bound["all_window_cycles_and_work_equal"],"control alignment incomplete")
        for name,value in bound["sources"].items():
            require(sha(ROOT/name)==value,"control native/alignment source changed");copies[ROOT/name]=Path("sources")/name
        for p in alignment.rglob("*"):
            if p.is_file() and p.suffix in {".json",".bin",".log"}:copies[p]=Path("control-alignment")/p.relative_to(alignment)
        for row in bound["cases"]:
            job=Path(row["job"]);native_job=Path(row["native_job"])
            require(sha(job)==row["job_sha256"] and sha(native_job)==row["native_job_sha256"],"control input changed")
            destination=Path("control-inputs")/job.parent.name
            copies[job]=destination/"control-event-job.json";copies[native_job]=destination/"native/job.json"
            for path,value in row["input_files"].items():
                p=Path(path);require(sha(p)==value,"control raw input changed")
                require(p.parent==job.parent,"control input outside test directory")
                copies[p]=destination/p.name
        copies[ROOT/"docs/mlx-control-event-lowering.md"]=Path("sources/docs/mlx-control-event-lowering.md")
    if args.include_source_order:
        require(args.group_alignment is not None,"source order publication needs numerical concurrent groups")
        alignment=args.group_alignment.resolve();bound=json.loads((alignment/"report.json").read_text())
        require(len(bound["cases"])==32 and bound["all_source_tick_cycles_equal"],"concurrent group alignment incomplete")
        for name,value in bound["sources"].items():
            require(sha(ROOT/name)==value,"group alignment source changed");copies[ROOT/name]=Path("sources")/name
        for p in alignment.rglob("*"):
            if p.is_file() and p.suffix in {".json",".bin",".log"}:copies[p]=Path("group-alignment")/p.relative_to(alignment)
        for row in bound["cases"]:
            job=Path(row["job"]);require(sha(job)==row["job_sha256"],"group input changed")
            destination=Path("group-inputs")/job.parent.name;copies[job]=destination/"group-job.json"
            for i,value in enumerate(row["numerical_output_sha256"]):
                solo=job.parent/f"solo{i}/out";require(sha(solo/"output.bin")==value,"standalone numerical reference changed")
                for p in solo.iterdir():
                    if p.is_file() and p.suffix in {".json",".bin"}:copies[p]=destination/f"solo{i}/out"/p.name
        copies[ROOT/"docs/mlx-event-source-order.md"]=Path("sources/docs/mlx-event-source-order.md")
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
