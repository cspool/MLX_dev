"""Execute asset-free original control/view closures to observe where predicates."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil

from scripts.mlx_system_attempt import digest,record,run_process,linked_libraries
from scripts.verify_mlx_event_schedule import require


def references(value):
    if isinstance(value,dict):
        if 'value' in value:yield value['value']
        else:
            for child in value.values():yield from references(child)
    elif isinstance(value,list):
        for child in value:yield from references(child)


def closure(program):
    nodes={n['id']:n for n in program['nodes']};selected=set();visiting=set()
    requests=[n for n in program['nodes'] if n['kind']=='where']
    def visit(name):
        require(name in nodes,'predicate closure requires external data; not an asset-free replay')
        require(name not in visiting,'cyclic predicate dependency')
        if name in selected:return
        visiting.add(name);n=nodes[name]
        require(n['kind'] in {'arange','add','le','unsqueeze','expand'},'unsupported predicate closure operation')
        require('control_program' in n or 'memory_program' in n,'missing C++ route')
        for field in ('args','kwargs','control_dependencies'):
            for parent in references(n.get(field,[])):visit(parent)
        visiting.remove(name);selected.add(name)
    for n in requests:visit(n['args'][0]['value'])
    require(requests,'no where predicates')
    result=copy.deepcopy(program);result['nodes']=[copy.deepcopy(n) for n in program['nodes'] if n['id'] in selected]
    # Releases refer to the original whole graph; keep closure values alive
    # until observation instead of importing foreign lifetime actions.
    for n in result['nodes']:n['release']=[]
    result['assets']={};result['outputs']=[];result.pop('block_pipeline_plan',None)
    require(result['schema']=='mlx_tensor_semantics_v1','closure currently supports ungrouped original profile only')
    for f,b in zip(('matrix','vector','memory','control'),('microcode','microcode','planned','rv64_leaf')):
        result[f+'_backend']=b
    return result,requests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('program','binary','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();require(not out.exists(),'choose fresh predicate replay')
    target=json.loads(args.program.read_text());p,requests=closure(target);nodes={n['id']:n for n in p['nodes']}
    ids=sorted({nodes[n['args'][0]['value']]['source_operator_id'] for n in requests})
    out.mkdir(parents=True);record(out/'program.json',p);record(out/'observation-ids.json',ids)
    binary=out/'mlx-tensor-semantics';shutil.copy2(args.binary,binary);sha=digest(binary);libraries=linked_libraries(binary)
    inputs={str(path.resolve()):digest(path) for path in (args.program,out/'program.json',out/'observation-ids.json')}
    runner=digest(Path(__file__))
    run_process([str(binary),str(out/'program.json'),str(out/'native'),'none','1',str(out/'observation-ids.json')],
                out/'run.log',out/'execution.json',timeout=300,metadata=dict(mode='asset_free_original_predicate_closure_not_full_model',
                inputs=inputs,binary_sha256=sha,runtime_libraries=libraries,runner_sha256=runner))
    r=json.loads((out/'native/result.json').read_text())
    require(r['executed_source_calls']==len(p['nodes']),'closure execution incomplete')
    observations={o['value_id']:o for o in r['observations']};require(len(observations)==len(ids),'observation coverage')
    witnesses=[]
    for n in requests:
        name=n['args'][0]['value'];o=observations[name];raw=Path(o['file']).read_bytes()
        require(o['dtype']=='bool' and o['shape']==n['output']['shape'] and len(raw)==math.prod(o['shape']) and set(raw)<={0,1},'predicate shape/encoding mismatch')
        witnesses.append(dict(source_operator_id=n['source_operator_id'],kind='actual_where_predicate',producer=nodes[name]['source_operator_id'],
                              value_data=[bool(v) for v in raw],observed_file=o['file'],observed_sha256=digest(Path(o['file']))))
    require(all(digest(Path(p))==h for p,h in inputs.items()) and digest(binary)==sha and digest(Path(__file__))==runner and
            all(digest(Path(p))==h for p,h in libraries.items()),'closure provenance changed')
    record(out/'witnesses.json',dict(classification='actual_cpp_asset_free_predicate_closure_not_full_model_execution',
        target_program_sha256=digest(args.program),closure_sources=[n['source_operator_id'] for n in p['nodes']],witnesses=witnesses,
        inputs=inputs,full_model_execution_verified=False,paired_execution_verified=False,mlx_system_verified=False))
    print('PREDICATE_CLOSURE_REPLAY_PASS sources='+str(len(p['nodes'])),flush=True)


if __name__=='__main__':main()
