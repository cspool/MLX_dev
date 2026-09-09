"""Independently rebuild every replay-backed full-model window and binding."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from mlxsim.model_event_windows import WindowCatalog
from scripts.compile_mlx_model_event_windows import encoded,work
from scripts.mlx_system_attempt import digest,record
from scripts.verify_mlx_event_schedule import require


def check_argmax_binding(node,job,report,logits_sha):
    require(job['node']==node,'replayed control node differs from target')
    name=node['args'][0]['value'];require(set(job['assets'])=={name},'replayed operands differ')
    asset=job['assets'][name]
    require(asset['kind']=='mapped_file' and asset['byte_offset']==0 and digest(Path(asset['path']))==logits_sha,
            'replayed logits file/range differs')
    width=asset['shape'][-1];rows=math.prod(asset['shape'][:-1])
    sites=[(e['row'],e['column']) for e in report['trace'] if e['event']=='instruction_issue' and e['phase']=='select' and e['pc']==0]
    require(sites==[(r,c) for r in range(rows) for c in range(1,width)],'argmax branch order/coverage differs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('compiled','witness-audit','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();base=args.compiled.resolve();require(not args.output.exists(),'choose fresh full rebuild audit')
    m=json.loads((base/'manifest.json').read_text());w=json.loads(args.witness_audit.read_text());root=Path(__file__).resolve().parents[1]
    require(w['all_required_timing_witnesses_checked'] and not w['full_numerical_execution_equal'] and
            w['target_program_sha256']==m['program_file_sha256'],'replay witness contract differs')
    require(w['auditor_sha256']==digest(root/'scripts/audit_mlx_replayed_witnesses.py'),'witness auditor changed')
    require(all(digest(root/p)==h for p,h in m['sources'].items()),'compiler source changed')
    files={**m['witness_files'],str(base/'manifest.json'):digest(base/'manifest.json')}
    require(files.get(str(args.witness_audit.resolve()))==digest(args.witness_audit),'different witness audit')
    require(all(digest(Path(p))==h for p,h in files.items()),'replay provenance changed')
    program=json.loads(Path(m['program_path']).read_text());require(digest(Path(m['program_path']))==m['program_file_sha256'],'model changed')
    nodes={n['source_operator_id']:n for n in program['nodes']}
    for name in w['files']:
        if not name.endswith('/witnesses.json'):continue
        bundle=json.loads(Path(name).read_text())
        if bundle['classification']!='actual_full_native_logits_rv64_branch_replay_not_paired_acceptance':continue
        for row in bundle['witnesses']:
            case=Path(name).parent/str(row['source_operator_id'])
            check_argmax_binding(nodes[row['source_operator_id']],json.loads((case/'job.json').read_text()),
                                 json.loads((case/'out/result.json').read_text()),row['logits_sha256'])
    values={r['source_operator_id']:r['value_data'] for r in w['witnesses']};catalog=WindowCatalog(program)
    rows={(r['binding']['source_operator_id'],r['binding']['batch_index']):r for r in m['windows']}
    require(len(rows)==len(m['windows'])==catalog.plan['total_windows'] and not m['blocked_windows'],'full coverage differs')
    checked={};counts=Counter();macs=0
    for source in catalog.plan['sources']:
        sid=source['source_operator_id']
        for batch in range(source['window_count']):
            job,binding=catalog.window(sid,batch,values.get(sid));row=rows[(sid,batch)]
            require(row['binding']==binding,'original layout/source/batch binding changed')
            key=hashlib.sha256(encoded(job).encode()).hexdigest();require(row['pattern']==key,'different compiled job')
            if key not in checked:
                path=base/'patterns'/(key+'.json');job_path=base/'patterns'/(key+'-job.json');meta=m['patterns'][key]
                require(digest(path)==meta['event_sha256'] and json.loads(job_path.read_text())==job,'compiled pattern/job changed')
                rebuilt=catalog.compile(job);require(json.loads(path.read_text())==rebuilt,'full pattern does not rebuild')
                summary=work(rebuilt);require(all(meta[k]==v for k,v in summary.items()),'pattern work changed')
                checked[key]=dict(**summary,event_file_sha256=digest(path),job_file_sha256=digest(job_path))
            counts.update(checked[key]['declared_work'])
            if source['family']=='matrix':
                required=binding['matrix']['m']*binding['matrix']['n']*binding['matrix']['k']
                require(checked[key]['declared_work'].get('mul_lanes',0)==required,'matrix MAC coverage differs')
                macs+=required
    require(dict(counts)==m['declared_work'] and set(checked)==set(m['patterns']),'full work/pattern coverage differs')
    require(all(digest(Path(p))==h for p,h in files.items()),'inputs changed during rebuild')
    record(args.output,dict(classification='full_replayed_window_rebuild_not_model_execution',manifest_sha256=digest(base/'manifest.json'),
        all_patterns_rebuilt=True,all_layout_bindings_rebuilt=True,all_window_patterns_compiled=True,
        patterns=checked,compiled_windows=len(rows),matrix_macs=macs,files=files,
        full_numerical_execution_equal=False,model_performance_error_available=False,auditor_sha256=digest(Path(__file__))))
    print('REPLAYED_FULL_WINDOW_AUDIT_PASS windows='+str(len(rows)),flush=True)


if __name__=='__main__':main()
