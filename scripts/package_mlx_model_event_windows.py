"""Package audited full-program window compilation and actual Boolean witnesses."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from scripts.mlx_system_attempt import digest,record
from scripts.verify_mlx_event_schedule import require

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("compiled","audits","witness","tests","output"):parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();require(not out.exists(),"choose fresh window publication")
    suite=ET.parse(args.tests/"regression.xml").getroot().find("testsuite")
    require(suite is not None and suite.get("tests")=="320" and all(suite.get(k)=="0" for k in ("errors","failures","skipped")),"window regression not accepted")
    copies={args.tests/"regression.xml":Path("tests/regression.xml")};models={}
    for name,expected,compiled in (("bert",3838,3838),("llama2",12133,12127)):
        directory=args.compiled/name;m=json.loads((directory/"manifest.json").read_text());a=json.loads((args.audits/(name+".json")).read_text())
        require(a["manifest_sha256"]==digest(directory/"manifest.json") and a["all_patterns_rebuilt"] and a["all_layout_bindings_rebuilt"] and a["expected_windows"]==expected and a["compiled_windows"]==compiled,"full-window audit scope differs")
        require(digest(ROOT/"scripts/verify_mlx_model_event_windows.py")==a["auditor_sha256"],"window auditor changed")
        for source,sha in m["sources"].items():require(digest(ROOT/source)==sha,"tested window source changed");copies[ROOT/source]=Path("sources")/source
        require(digest(Path(m["program_path"]))==m["program_file_sha256"],"full model compiled program changed")
        copies[Path(m["program_path"])]=Path("programs")/(name+".json")
        for key,pattern in a["patterns"].items():
            require(digest(directory/"patterns"/(key+".json"))==pattern["event_file_sha256"] and digest(directory/"patterns"/(key+"-job.json"))==pattern["job_file_sha256"],"audited pattern changed")
        require(all(digest(Path(path))==sha for path,sha in m["witness_files"].items()),"bound witness changed")
        for p in directory.rglob("*.json"):copies[p]=Path("compiled")/name/p.relative_to(directory)
        copies[args.audits/(name+".json")]=Path("audits")/(name+".json")
        models[name]={k:a[k] for k in ("compiled_windows","expected_windows","unique_patterns","all_window_patterns_compiled","matrix_macs")}
    witness=args.witness.resolve();state=json.loads((witness/"execution.json").read_text());bundle=json.loads((witness/"witnesses.json").read_text())
    require(state["status"]=="exited" and state["exit_code"]==0 and bundle["full_numerical_execution_equal"] and len(bundle["witnesses"])==5,"full Boolean witness run not accepted")
    require(digest(witness/"native/result.json")==bundle["native_report_sha256"] and digest(witness/"mlx-tensor-semantics")==state["binary_sha256"] and digest(ROOT/"scripts/run_mlx_event_witness.py")==state["runner_sha256"],"owned witness execution changed")
    require(all(digest(Path(path))==sha for path,sha in {**state["inputs"],**state["runtime_libraries"]}.items()),"witness input/library provenance changed")
    for name,sha in state["sources"].items():require(digest(witness/"sources"/name)==sha,"witness numerical source snapshot changed")
    for p in witness.rglob("*"):
        if p.is_file() and ("sources" in p.relative_to(witness).parts or p.suffix in {".json",".bin",".log"}):copies[p]=Path("witness")/p.relative_to(witness)
    baseline=json.loads((witness/"reference-result.json").read_text())
    for row in baseline["outputs"]:
        for role,data in row["outputs"].items():copies[Path(data["file"])]=Path("witness/reference-outputs")/(str(row["forward_id"])+"-"+role+".bin")
    prior=ROOT/"artifacts/tagged/event-source-graph-publication-001/manifest.json";core=json.loads(prior.read_text())
    for f in core["files"]:
        if f["path"].startswith("sources/simulator_ext/event_schedule/"):require(digest(ROOT/f["path"].removeprefix("sources/"))==f["sha256"],"prior sanitized event core changed")
    copies[prior]=Path("prior-core/manifest.json")
    for name in ("scripts/package_mlx_model_event_windows.py","scripts/verify_mlx_model_event_windows.py","tests/test_event_model_windows.py","docs/mlx-full-model-event-windows.md"):
        copies[ROOT/name]=Path("sources")/name
    out.mkdir(parents=True);files=[]
    for source,relative in sorted(copies.items()):
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);sha=digest(source);shutil.copy2(source,target);require(digest(target)==sha==digest(source),"publication copy changed")
        files.append(dict(path=str(relative),source_path=str(source),sha256=sha,bytes=source.stat().st_size))
    record(out/"manifest.json",dict(classification="full_model_window_patterns_and_cpp_witnesses_not_full_event_execution",regression_tests=320,unchanged_core_prior_sanitizer_replays=291,
        prior_core_manifest_sha256=digest(prior),models=models,full_bert_numeric_witness_rerun_equal=True,full_event_lowering_complete=False,event_simulated_cycles=None,
        model_performance_error_available=False,files=files,file_count=len(files),total_bytes=sum(f["bytes"] for f in files)))
    print(f"MODEL_WINDOWS_PACKAGED files={len(files)} bytes={sum(f['bytes'] for f in files)}")


if __name__=="__main__":main()
