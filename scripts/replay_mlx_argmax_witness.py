"""Replay actual accepted full-run logits through C++ RV64 control execution."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.mlx_system_attempt import digest,record,run_process,linked_libraries
from scripts.verify_mlx_event_schedule import require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','target-program','binary','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();base=args.reference.resolve();out=args.output.resolve()
    require(not out.exists(),'choose fresh branch replay')
    state=json.loads((base/'execution.json').read_text());proof=json.loads((base/'numeric-conformance-001.json').read_text())
    require(state['status']=='completed' and state['returncode']==0 and state['source_and_program_identity_unchanged'],'reference not completed unchanged')
    require(proof['numeric_conformance_passed'] and proof['native_report_sha256']==digest(base/'native/result.json') and
            proof['program_sha256']==digest(base/'program.json'),'reference numerical evidence changed')
    program=json.loads((base/'program.json').read_text());target=json.loads(args.target_program.read_text())
    omitted={'block_pipeline_plan',*(f+s for f in ('matrix','vector','memory','control') for s in ('_backend','_schedule_options'))}
    require({k:v for k,v in program.items() if k not in omitted}=={k:v for k,v in target.items() if k not in omitted},'target differs beyond scheduling')
    native=json.loads((base/'native/result.json').read_text());nodes={n['id']:n for n in target['nodes']}
    comparisons={r['forward_id']:r for r in proof['comparisons']}
    inputs={str(p.resolve()):digest(p) for p in (base/'execution.json',base/'numeric-conformance-001.json',base/'native/result.json',base/'program.json',args.target_program)}
    jobs=[]
    for spec,row in zip(target['outputs'],native['outputs'],strict=True):
        require(spec['forward_id']==row['forward_id'],'forward binding mismatch')
        node=nodes[spec['token']];value=nodes[spec['logits']];path=Path(row['logits_file'])
        require(node['kind']=='argmax' and node['args'][0]=={'value':spec['logits']},'logits not actual argmax input')
        require(value['output']==dict(dtype=row['dtype'],shape=row['shape']),'logit layout changed')
        check=comparisons[spec['forward_id']]
        require(check['bitwise_equal'] and digest(path)==check['logits_sha256'],'actual logits changed')
        inputs[str(path)]=digest(path)
        asset=dict(kind='mapped_file',path=str(path),dtype=row['dtype'],shape=row['shape'],bytes=path.stat().st_size,byte_offset=0)
        jobs.append((node,row,dict(node=copy.deepcopy(node),assets={spec['logits']:asset},external=False,
                                  options=dict(trace=True,trace_limit=1000000,max_cycles=10000000))))
    out.mkdir(parents=True);binary=out/'control-external-memory';shutil.copy2(args.binary,binary)
    sha=digest(binary);libraries=linked_libraries(binary);witnesses=[]
    root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(root)):digest(p) for directory in ('control_schedule','control_model')
             for p in (root/'simulator_ext'/directory).iterdir() if p.is_file()}
    sources['scripts/replay_mlx_argmax_witness.py']=digest(Path(__file__))
    for node,row,job in jobs:
        case=out/str(node['source_operator_id']);case.mkdir();record(case/'job.json',job)
        run_process([str(binary),str(case/'job.json'),str(case/'out')],case/'run.log',case/'execution.json',timeout=300,
                    metadata=dict(mode='actual_full_logits_rv64_branch_replay',inputs={**inputs,str(case/'job.json'):digest(case/'job.json')},
                                  sources=sources,binary_sha256=sha,runtime_libraries=libraries))
        result=json.loads((case/'out/result.json').read_text())
        require(result['done'] and result['functional_oracle_equal'] and not result['trace_truncated'],'control replay incomplete')
        actual=np.fromfile(case/'out/output.bin',dtype='<i8').tolist()
        require(actual==row['tokens'],'actual RV64 replay token differs')
        selects=[e for e in result['trace'] if e['event']=='instruction_issue' and e['phase']=='select']
        updated={(e['row'],e['column']) for e in selects if e['pc']==1}
        branches=[(e['row'],e['column']) not in updated for e in selects if e['pc']==0]
        count=int(np.prod(row['shape'][:-1]))*(row['shape'][-1]-1)
        require(len(branches)==count and sum(branches)==result['branches_taken'],'branch path coverage differs')
        witnesses.append(dict(source_operator_id=node['source_operator_id'],kind='actual_argmax_branch_path',value_data=branches,
                              logits_sha256=inputs[row['logits_file']],replay_report_sha256=digest(case/'out/result.json')))
    require(all(digest(Path(p))==h for p,h in inputs.items()) and digest(binary)==sha and
            all(digest(root/p)==h for p,h in sources.items()) and all(digest(Path(p))==h for p,h in libraries.items()),'replay provenance changed')
    record(out/'witnesses.json',dict(classification='actual_full_native_logits_rv64_branch_replay_not_paired_acceptance',
           target_program_sha256=digest(args.target_program),witnesses=witnesses,inputs=inputs,
           paired_execution_verified=False,mlx_system_verified=False,full_model_witness_coverage=False))
    print('ARGMAX_BRANCH_REPLAY_PASS count='+str(len(witnesses)),flush=True)


if __name__=='__main__':main()
