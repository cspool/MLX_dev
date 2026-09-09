"""Publish streaming pair implementation, numerical alignment and safety evidence."""
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
    for key in ("tests","safety","alignment","output"):parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();tests=args.tests.resolve();safety=args.safety.resolve();alignment=args.alignment.resolve();out=args.output.resolve()
    require(not out.exists(),"choose fresh streaming publication")
    suite=ET.parse(tests/"regression.xml").getroot().find("testsuite");require(suite is not None and suite.get("tests")=="406" and all(suite.get(k)=="0" for k in ("errors","failures","skipped")),"streaming regression not accepted")
    safe=json.loads((safety/"report.json").read_text());bound=json.loads((alignment/"report.json").read_text())
    require(safe["regression_tests"]==406 and len(safe["replays"])==401 and sum(r["expected_exit"]==0 for r in safe["replays"])==329,"streaming safety scope differs")
    require(len(bound["cases"])==25 and bound["all_pair_cycles_and_events_equal"],"streaming numerical scope differs")
    copies={tests/"regression.xml":Path("tests/regression.xml")}
    for report in (safe,bound):
        for name,sha in report["sources"].items():require(digest(ROOT/name)==sha,"streaming source changed");copies[ROOT/name]=Path("sources")/name
        require(all(digest(Path(p))==h for p,h in report["runtime_libraries"].items()),"streaming runtime library changed")
    for path,sha in safe["inputs"].items():
        p=Path(path);require(digest(p)==sha,"streaming replay input changed");copies[p]=Path("tests")/p.relative_to(tests)
    require(all(not Path(p).exists() for p in safe["missing_pattern_paths"]),"expected missing input changed")
    for row in safe["replays"]:
        p=Path(row["program"])
        if row["expected_exit"]==0:
            result=json.loads(Path(row["result"]).read_text());require(result==json.loads((p.parent/"result.json").read_text()),"streaming safety/release result differs");audit_trace(json.loads(p.read_text()),result)
            copies[p.parent/"result.json"]=Path("tests")/(p.parent/"result.json").relative_to(tests)
        copies[p.parent/"run.log"]=Path("tests")/(p.parent/"run.log").relative_to(tests)
    for directory,label in ((safety,"safety"),(alignment,"alignment")):
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix in {".json",".log",".bin"}:copies[p]=Path(label)/p.relative_to(directory)
    for index,row in enumerate(bound["cases"]):
        p=Path(row["program"]);require(digest(p)==row["program_sha256"] and digest(p.parent/"options.json")==row["options_sha256"],"numerical streaming input changed")
        require(digest(alignment/f"case-{index:02d}/native/result.json")==row["native_result_sha256"] and digest(alignment/f"case-{index:02d}/event.json")==row["event_result_sha256"],"numerical streaming result changed")
        copies[p]=Path("native-inputs")/f"case-{index:02d}/program.json";copies[p.parent/"options.json"]=Path("native-inputs")/f"case-{index:02d}/options.json"
    for name in ("scripts/package_mlx_streaming_events.py","docs/mlx-streaming-pair-events.md"):
        copies[ROOT/name]=Path("sources")/name
    out.mkdir(parents=True);files=[]
    for source,relative in sorted(copies.items()):
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);sha=digest(source);shutil.copy2(source,target);require(digest(target)==sha==digest(source),"streaming publication copy changed")
        files.append(dict(path=str(relative),source_path=str(source),sha256=sha,bytes=source.stat().st_size))
    record(out/"manifest.json",dict(classification="streaming_pair_components_not_full_model_event_or_chipyard_execution",regression_tests=406,sanitizer_replays=401,successful_replays=329,expected_rejections=72,numerical_pair_cases=25,
        full_event_lowering_complete=False,model_performance_error_available=False,files=files,file_count=len(files),total_bytes=sum(f["bytes"] for f in files)))
    print(f"STREAMING_PUBLICATION_PASS files={len(files)} bytes={sum(f['bytes'] for f in files)}")


if __name__=="__main__":main()
