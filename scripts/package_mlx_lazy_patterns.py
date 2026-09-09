"""Package accepted lazy-parser execution and a full-shape model-pattern replay."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from scripts.mlx_system_attempt import digest,record
from scripts.verify_mlx_event_schedule import require,audit_trace

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tests","safety","pattern-run","output"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--rom-recycling",action="store_true")
    parser.add_argument("--compact",action="store_true")
    args=parser.parse_args();tests=args.tests.resolve();safety=args.safety.resolve();run=args.pattern_run.resolve();out=args.output.resolve()
    tests_expected,replays_expected,success_expected=(369,364,303) if args.compact else (344,328,277) if args.rom_recycling else (342,324,273)
    require(not out.exists(),"choose fresh lazy publication")
    suite=ET.parse(tests/"regression.xml").getroot().find("testsuite")
    require(suite is not None and suite.get("tests")==str(tests_expected) and all(suite.get(k)=="0" for k in ("errors","failures","skipped")),"lazy regression not accepted")
    report=json.loads((safety/"report.json").read_text());require(report["regression_tests"]==tests_expected and len(report["replays"])==replays_expected and sum(r["expected_exit"]==0 for r in report["replays"])==success_expected,"lazy safety scope differs")
    copies={tests/"regression.xml":Path("tests/regression.xml")}
    for name,sha in report["sources"].items():require(digest(ROOT/name)==sha,"lazy tested source changed");copies[ROOT/name]=Path("sources")/name
    for path,sha in report["inputs"].items():
        p=Path(path);require(digest(p)==sha,"lazy input changed");copies[p]=Path("tests")/p.relative_to(tests)
    require(all(not Path(p).exists() for p in report["missing_pattern_paths"]),"missing-file rejection input changed")
    require(all(digest(Path(p))==h for p,h in report["runtime_libraries"].items()),"sanitizer runtime changed")
    for row in report["replays"]:
        p=Path(row["program"])
        if row["expected_exit"]==0:
            result=json.loads(Path(row["result"]).read_text());require(result==json.loads((p.parent/"result.json").read_text()),"safety replay differs from release")
            audit_trace(json.loads(p.read_text()),result);copies[p.parent/"result.json"]=Path("tests")/(p.parent/"result.json").relative_to(tests)
        copies[p.parent/"run.log"]=Path("tests")/(p.parent/"run.log").relative_to(tests)
    for p in safety.iterdir():
        if p.is_file():copies[p]=Path("safety")/p.name
    full=json.loads((run/"report.json").read_text());require(full["full_result_equal"] and full["events"]==724992 and full["blocks"]==12288,"full-shape execution scope differs")
    for name,sha in full["sources"].items():require(digest(ROOT/name)==sha,"full-shape source changed");copies[ROOT/name]=Path("sources")/name
    for mode in (("original","eager","lazy","compact") if args.compact else ("original","eager","lazy")):
        state=json.loads((run/(mode+"-execution.json")).read_text());require(state["status"]=="exited" and state["exit_code"]==0,"full-shape run not terminal")
        require(digest(run/"event-schedule")==state["binary_sha256"] and all(digest(Path(p))==h for p,h in {**state["inputs"],**state["runtime_libraries"]}.items()),"full-shape provenance changed")
    actual=json.loads((run/"lazy-result.json").read_text());audit_trace(json.loads((run/"lazy.json").read_text()),actual)
    actual.pop("lazy_patterns");require(actual==json.loads((run/"eager-result.json").read_text()),"lazy/eager report differs")
    if args.compact:
        require(full["compact_result_equal"] and full["compact_blocks"]["logical_blocks"]==12288 and full["compact_blocks"]["run_descriptors"]==1,"full-shape compact scope differs")
        result=json.loads((run/"compact-result.json").read_text());audit_trace(json.loads((run/"compact.json").read_text()),result)
        for field in ("cycles","events","blocks","counts","integrated_usage","declared_work","peak","capacity","source_intervals"):
            require(result[field]==actual[field],"compact full-shape execution changed")
        copies[ROOT/"docs/mlx-compact-block-streams.md"]=Path("sources/docs/mlx-compact-block-streams.md")
    for p in run.rglob("*"):
        if p.is_file() and p.suffix in {".json",".log"}:copies[p]=Path("full-shape")/p.relative_to(run)
    for name in ("scripts/package_mlx_lazy_patterns.py","docs/mlx-lazy-event-patterns.md","tests/test_event_model_windows.py","src/mlxsim/model_event_windows.py"):
        copies[ROOT/name]=Path("sources")/name
    out.mkdir(parents=True);files=[]
    for source,relative in sorted(copies.items()):
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);sha=digest(source);shutil.copy2(source,target);require(digest(target)==sha==digest(source),"lazy publication copy changed")
        files.append(dict(path=str(relative),source_path=str(source),sha256=sha,bytes=source.stat().st_size))
    record(out/"manifest.json",dict(classification="lazy_event_state_loading_not_full_model_execution",regression_tests=tests_expected,sanitizer_replays=replays_expected,successful_replays=success_expected,expected_rejections=replays_expected-success_expected,
        full_shape_pattern=full,block_descriptors_still_eager=not args.compact,full_event_lowering_complete=False,event_model_performance_error_available=False,
        files=files,file_count=len(files),total_bytes=sum(f["bytes"] for f in files)))
    print(f"LAZY_EVENT_PUBLICATION_PASS files={len(files)} bytes={sum(f['bytes'] for f in files)}")


if __name__=="__main__":main()
