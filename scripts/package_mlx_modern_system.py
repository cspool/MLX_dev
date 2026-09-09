"""Archive accepted modern-system code, small-system evidence and full image.

The running full BERT system attempt is deliberately excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

from scripts.mlx_system_attempt import digest,record

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts/tagged"


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():raise ValueError("choose a fresh modern-system publication")
    regressions={"modern-system-regression-002.xml":227,"modern-system-qa-tests-002.xml":28,"bert-system-audit-tests-001.xml":7}
    for name,count in regressions.items():
        suites=ET.parse(BASE/name).getroot().findall('.//testsuite')
        if sum(int(s.get('tests',0)) for s in suites)!=count or any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped')):
            raise ValueError("required regression did not pass")
    system=json.loads((BASE/"modern-chipyard-002/report.json").read_text())
    if not system["actual_rocket_execution"] or len(system["cases"])!=2 or system["full_model_execution_verified"]:
        raise ValueError("small-system scope differs")
    for name,value in system["sources"].items():
        if digest(BASE/"modern-chipyard-002/sources"/name)!=value:raise ValueError("executed system source snapshot differs")
    for name in ("qa-tuple-paired","qa-gelu-grouped-paired"):
        audit=json.loads((BASE/"modern-chipyard-audit-002"/name/"report.json").read_text())
        if not audit["registered_graph_execution_verified"] or not audit["complete_elf_rebuild_equal"] or not audit["actual_output_dump_emitted"] or audit["full_model_execution_verified"]:
            raise ValueError("system output/ELF audit incomplete")
    prepared=json.loads((BASE/"bert-paired-system-image-001/report.json").read_text())
    if not prepared["scheduled_program_recompiled_equal"] or prepared["original_source_calls"]!=1141 or prepared["lowered_calls"]!=3046 or prepared["actual_rocket_execution"]:
        raise ValueError("full image preparation scope differs")
    rejection=json.loads((BASE/"bert-system-audit-reject-small-001/failure.json").read_text())
    if rejection["error"]!="audit input was not bound before the system run":raise ValueError("native/small-system result substitution was not rejected")
    out.mkdir(parents=True);snapshot=out/"snapshot";snapshot.mkdir();files={}
    def copy(path,name):
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        value=digest(path)
        if digest(target)!=value:raise ValueError("archive copy differs")
        files[name]=value
    roots=("modern-chipyard-002","modern-chipyard-audit-002","bert-paired-system-image-001","modern-system-qa-tests-001","bert-system-audit-reject-small-001")
    for directory in roots:
        for file in sorted((BASE/directory).rglob('*')):
            if file.is_symlink() or any(p.is_symlink() for p in file.parents) or not file.is_file():continue
            relative=file.relative_to(BASE/directory)
            source_snapshot='sources' in relative.parts
            if not source_snapshot and file.suffix not in {'.json','.xml','.log','.c','.txt','.bin','.elf'}:continue
            copy(file,str(file.relative_to(BASE)))
    for file in regressions:copy(BASE/file,file)
    for name in ("scripts/package_mlx_modern_system.py","scripts/prepare_mlx_bert_system_image.py","scripts/verify_mlx_bert_system_model.py",
                 "tests/test_modern_system_graph.py","tests/test_bert_system_audit.py","tests/fixtures/modern_graph_runtime_contract.c",
                 "tests/model_storage_contract.cc","docs/mlx-modern-system-graph.md","docs/mlx-mainline-acceptance.md"):
        copy(ROOT/name,"current_sources/"+name)
    manifest={"classification":"accepted_modern_system_components_and_prepared_full_BERT_not_completed_model",
              "regressions":regressions,"actual_rocket_small_graphs":2,"small_graph_lowered_calls":45,"small_graph_original_calls":15,
              "full_BERT_image":{"original_calls":1141,"lowered_calls":3046,"pairs":693,"tasks":2749,"device_windows":2094,"executed":False},
              "full_model_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False,
              "files_sha256":files,"excluded":["ongoing_full_BERT_system_attempt","host_binaries","object_files","checkpoints","private_simulator_card_bodies"]}
    record(snapshot/"manifest.json",manifest);archive=out/"evidence.tar.gz"
    with tarfile.open(archive,'w:gz') as stream:
        for file in sorted(snapshot.rglob('*')):
            if file.is_file():stream.add(file,arcname=str(file.relative_to(snapshot)),recursive=False)
    with tarfile.open(archive,'r:gz') as stream:
        expected={**files,"manifest.json":digest(snapshot/"manifest.json")}
        if {item.name for item in stream}!=set(expected):raise ValueError("archive member list differs")
        for name,value in expected.items():
            with stream.extractfile(name) as member:
                if hashlib.file_digest(member,'sha256').hexdigest()!=value:raise ValueError("archive payload hash differs")
    manifest.update(archive="evidence.tar.gz",archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,payload_files=len(files))
    record(out/"manifest.json",manifest)
    print(json.dumps({key:manifest[key] for key in ("classification","payload_files","archive_bytes","archive_sha256")},indent=2))


if __name__=="__main__":main()
