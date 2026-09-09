"""Package an accepted full physical QA result, never a running/paired result."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from mlxsim.model_physical_evidence import verify_physical_execution
from scripts.run_mlx_qa_model import full_bert_contract,audit_outputs
from scripts.mlx_system_attempt import digest,record
from scripts.verify_mlx_event_schedule import require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','reference','inventory','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();base=args.run.resolve();reference=args.reference.resolve();out=args.output.resolve()
    require(not out.exists(),'choose a fresh physical QA package')
    state=json.loads((base/'execution.json').read_text());comparison=json.loads((base/'comparison.json').read_text())
    require(state['status']=='exited' and state['exit_code']==0 and state['mode']=='physical','not completed physical QA')
    require(comparison['major_correctness_passed'] and comparison['full_microcode_execution_bitwise_equal'],'numerical acceptance incomplete')
    require(digest(base/'native/result.json')==comparison['native_report_sha256'],'result changed')
    require(digest(Path(state['command'][0]))==state['binary_sha256'],'binary changed')
    provenance={**state['inputs'],**state['runtime_libraries']}
    require(provenance.get(str(args.inventory.resolve()))==digest(args.inventory),'different inventory')
    require(all(digest(Path(p))==h for p,h in provenance.items()),'execution inputs/libraries changed')
    program=json.loads((base/'program.json').read_text());options=json.loads((base/'options.json').read_text())
    native=json.loads((base/'native/result.json').read_text());prior=json.loads((reference/'native/result.json').read_text())
    require(provenance.get(str(reference/'native/result.json'))==digest(reference/'native/result.json'),'different numerical baseline')
    inventory=json.loads(args.inventory.read_text());contract=full_bert_contract(inventory,program)
    actual=audit_outputs(program,inventory,native);coverage=verify_physical_execution(program,native,options)
    require(actual==comparison['comparison'] and all(c['major_correctness_passed'] for c in actual),'output re-audit differs')
    copies={}
    for name in ('program.json','options.json','coverage.json','acceptance-policy.json','execution.json','comparison.json','native/result.json','native.log'):
        copies[base/name]=Path('physical')/name
    copies[args.inventory.resolve()]=Path('reference/inventory.json')
    for name in ('execution.json','comparison.json','native/result.json'):
        copies[reference/name]=Path('numeric-reference')/name
    for name,sha in state['sources'].items():
        source=base/'sources'/name;require(digest(source)==sha,'executed source snapshot changed')
        copies[source]=Path('physical/sources')/name
    for a,b in zip(native['outputs'],prior['outputs'],strict=True):
        require(a['forward_id']==b['forward_id'],'forward coverage differs')
        for role in ('start_logits','end_logits'):
            source=Path(a['outputs'][role]['file']);other=Path(b['outputs'][role]['file'])
            require(source.read_bytes()==other.read_bytes(),'actual logits not bitwise equal')
            copies[source]=Path('physical/native')/source.name;copies[other]=Path('numeric-reference/native')/other.name
    require(len(set(copies.values()))==len(copies),'package path collision')
    out.mkdir(parents=True);snapshot=out/'snapshot';entries=[]
    for source,relative in sorted(copies.items()):
        require(source.is_file() and not source.is_symlink(),'invalid package source')
        sha=digest(source);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        require(digest(target)==sha==digest(source),'package copy changed')
        entries.append(dict(path=str(relative),source=str(source),bytes=target.stat().st_size,sha256=sha))
    manifest=dict(classification='full_physical_qa_numerical_acceptance_not_concurrent_or_chipyard',
        full_physical_major_correctness=True,full_microcode_outputs_bitwise_equal=True,contract=contract,coverage=coverage,
        answers=[r['actual_span']['text'] for r in actual],device_component_cycles=native['device_component_cycles'],
        host_readback_cycles=native['host_readback_cycles'],shared_elapsed_cycles=native['shared_elapsed_cycles'],
        host_elapsed_seconds=state['host_elapsed_seconds'],formal_concurrent_performance_error_available=False,
        mlx_system_verified=False,checkpoint_weights_included=False,binary_included=False,files=entries,packager_sha256=digest(Path(__file__)))
    record(out/'manifest.json',manifest)
    archive=out/'physical-qa-evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for row in entries:tar.add(snapshot/row['path'],arcname=row['path'],recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        import hashlib
        require(set(tar.getnames())=={r['path'] for r in entries},'archive coverage differs')
        for row in entries:require(hashlib.sha256(tar.extractfile(row['path']).read()).hexdigest()==row['sha256'],'archive bytes differ')
    record(out/'archive.json',dict(file=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,files=len(entries)))
    print('PHYSICAL_QA_PACKAGE_PASS files='+str(len(entries)),flush=True)


if __name__=='__main__':main()
