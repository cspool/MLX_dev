"""Check replayed control evidence without declaring a new full numerical run."""
import argparse
import json
from pathlib import Path

from mlxsim.model_event_windows import WindowCatalog
from scripts.replay_mlx_predicate_closure import closure
from scripts.mlx_system_attempt import digest,record
from scripts.verify_mlx_event_schedule import require


def audit(program_path,argmax_path,predicate_path):
    target=json.loads(program_path.read_text());sha=digest(program_path);catalog=WindowCatalog(target)
    requirements={r['source_operator_id']:r for r in catalog.plan['timing_witness_requirements']}
    files={str(program_path.resolve()):sha};values={}
    def checked(path,expected=None):
        path=path.resolve();actual=digest(path)
        require(expected is None or actual==expected,'replay evidence changed')
        files[str(path)]=actual;return path
    def execution(path):
        state=json.loads(checked(path).read_text())
        require(state['status']=='exited' and state['exit_code']==0,'replay not successfully completed')
        checked(Path(state['command'][0]),state['binary_sha256'])
        for p,h in {**state['inputs'],**state['runtime_libraries']}.items():checked(Path(p),h)
        return state
    for path in (argmax_path,predicate_path):
        bundle=json.loads(checked(path).read_text());require(bundle['target_program_sha256']==sha,'wrong replay target')
        for p,h in bundle['inputs'].items():checked(Path(p),h)
        predicate=path==predicate_path
        if predicate:
            execution(path.parent/'execution.json')
            expected,_=closure(target)
            require(json.loads(checked(path.parent/'program.json').read_text())==expected,'predicate closure differs from target')
            result=json.loads(checked(path.parent/'native/result.json').read_text())
            require(result['executed_source_calls']==len(expected['nodes']),'incomplete predicate closure')
            observations={r['source_operator_id']:r for r in result['observations']}
        for row in bundle['witnesses']:
            sid=row['source_operator_id'];require(sid in requirements and sid not in values,'foreign or duplicated witness')
            requirement=requirements[sid]
            if predicate:
                require(requirement['kind']=='actual_where_predicate','wrong predicate witness family')
                observed=observations[row['producer']]
                raw=checked(Path(row['observed_file']),row['observed_sha256']).read_bytes()
                node=next(n for n in target['nodes'] if n['source_operator_id']==sid)
                require(observed['file']==row['observed_file'] and observed['value_id']==node['args'][0]['value'] and
                        observed['shape']==node['output']['shape'] and observed['dtype']=='bool' and set(raw)<={0,1},'predicate observation binding')
                value=[bool(v) for v in raw]
            else:
                require(requirement['kind']=='actual_argmax_branch_taken','wrong argmax witness family')
                case=path.parent/str(sid);execution(case/'execution.json')
                result=json.loads(checked(case/'out/result.json',row['replay_report_sha256']).read_text())
                require(result['done'] and result['functional_oracle_equal'] and not result['trace_truncated'],'incomplete RV64 path')
                selects=[e for e in result['trace'] if e['event']=='instruction_issue' and e['phase']=='select']
                branch=[e for e in selects if e['pc']==0];updates={(e['row'],e['column']) for e in selects if e['pc']==1}
                require(len({(e['row'],e['column']) for e in branch})==len(branch),'duplicate branch site')
                value=[(e['row'],e['column']) not in updates for e in branch]
                require(sum(value)==result['branches_taken'],'RV64 taken counter differs')
            require(len(value)==requirement['count'] and value==row['value_data'],'replayed witness values differ')
            # Exercise the real six window lowering routes, not just a count.
            job,binding=catalog.window(sid,witness=value);catalog.compile(job)
            values[sid]=value
    require(set(values)==set(requirements),'missing model timing witnesses')
    require(all(digest(Path(p))==h for p,h in files.items()),'evidence changed during audit')
    return values,files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('program','argmax','predicates','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'choose fresh replay audit')
    values,files=audit(args.program,args.argmax,args.predicates)
    record(args.output,dict(classification='complete_timing_witness_replay_audit_not_full_model_execution',
           target_program_sha256=digest(args.program),witnesses=[dict(source_operator_id=k,value_data=v) for k,v in sorted(values.items())],
           files=files,all_required_timing_witnesses_checked=True,full_numerical_execution_equal=False,
           paired_execution_verified=False,mlx_system_verified=False,auditor_sha256=digest(Path(__file__))))
    print('REPLAY_WITNESS_AUDIT_PASS sources='+str(len(values)),flush=True)


if __name__=='__main__':main()
