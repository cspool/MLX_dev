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
    args = parser.parse_args(); out = args.output.resolve(); tests = args.tests.resolve(); safety = args.safety.resolve()
    require(not out.exists(), "choose a fresh event publication")
    suite = ET.parse(tests / "regression.xml").getroot().find("testsuite")
    require(suite is not None and suite.get("tests") == "35" and all(suite.get(k) == "0" for k in ("errors", "failures", "skipped")), "event regression not accepted")
    report = json.loads((safety / "report.json").read_text())
    require(report["regression_tests"] == 35 and len(report["replays"]) == 29 and sum(r["expected_exit"] == 0 for r in report["replays"]) == 18, "event safety scope incomplete")
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
    out.mkdir(parents=True); files = []
    for source, relative in sorted(copies.items()):
        target = out / relative; target.parent.mkdir(parents=True, exist_ok=True); digest = sha(source); shutil.copy2(source, target)
        require(sha(target) == digest == sha(source), "event publication copy changed")
        files.append(dict(path=str(relative), source_path=str(source), bytes=source.stat().st_size, sha256=digest))
    record(out / "manifest.json", dict(classification="concurrent_event_core_and_complete_resource_inventory_not_full_model_event_execution",
        regression_tests=35, sanitizer_replays=29, successful_replays=18, expected_rejections=11, full_program_resources=contracts,
        full_event_lowering_complete=False, event_vs_end_to_end_error_available=False, files=files,
        file_count=len(files), total_bytes=sum(f["bytes"] for f in files), packager_sha256=sha(Path(__file__))))
    print(json.dumps(dict(file_count=len(files), total_bytes=sum(f["bytes"] for f in files))))


if __name__ == "__main__": main()
