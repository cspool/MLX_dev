"""Complete an audited partial catalogue using separately audited replay witnesses."""
import argparse
import json
import hashlib
from pathlib import Path
import shutil
from collections import Counter

from mlxsim.model_event_windows import WindowCatalog
from scripts.compile_mlx_model_event_windows import encoded,work
from scripts.mlx_system_attempt import digest,record
from scripts.verify_mlx_event_schedule import require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('compiled','window-audit','witness-audit','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();base=args.compiled.resolve();out=args.output.resolve();root=Path(__file__).resolve().parents[1]
    require(not out.exists(),'choose fresh completed catalogue')
    m=json.loads((base/'manifest.json').read_text());a=json.loads(args.window_audit.read_text());w=json.loads(args.witness_audit.read_text())
    require(a['manifest_sha256']==digest(base/'manifest.json') and a['all_patterns_rebuilt'] and a['all_layout_bindings_rebuilt'],'partial catalogue not audited')
    require(w['all_required_timing_witnesses_checked'] and w['target_program_sha256']==m['program_file_sha256'] and
            w['auditor_sha256']==digest(root/'scripts/audit_mlx_replayed_witnesses.py'),'witness audit differs')
    require(digest(Path(m['program_path']))==m['program_file_sha256'],'target changed')
    require(all(digest(root/p)==h for p,h in m['sources'].items()),'original compiler changed')
    files={**w['files'],str(args.witness_audit.resolve()):digest(args.witness_audit),
           str(args.window_audit.resolve()):digest(args.window_audit),str(base/'manifest.json'):digest(base/'manifest.json')}
    require(all(digest(Path(p))==h for p,h in files.items()),'witness inputs changed')
    values={r['source_operator_id']:r['value_data'] for r in w['witnesses']}
    require({r['source_operator_id'] for r in m['blocked_windows']}==set(values),'blocked window coverage differs')
    c=WindowCatalog(json.loads(Path(m['program_path']).read_text()));out.mkdir(parents=True);(out/'patterns').mkdir()
    for key,meta in m['patterns'].items():
        source=base/'patterns'/(key+'.json');require(digest(source)==a['patterns'][key]['event_file_sha256'],'original window changed')
        for suffix in ('.json','-job.json'):shutil.copy2(base/'patterns'/(key+suffix),out/'patterns'/(key+suffix))
        meta['event_file']=str(out/'patterns'/(key+'.json'))
    for missing in m['blocked_windows']:
        sid=missing['source_operator_id'];batch=missing['batch_index'];job,binding=c.window(sid,batch,values[sid])
        key=hashlib.sha256(encoded(job).encode()).hexdigest();events=c.compile(job);path=out/'patterns'/(key+'.json')
        record(path,events);record(out/'patterns'/(key+'-job.json'),job)
        m['patterns'][key]=dict(**work(events),event_file=str(path),event_sha256=digest(path),job_sha256=key)
        m['windows'].append(dict(binding=binding,pattern=key))
    m['windows'].sort(key=lambda r:(r['binding']['source_operator_id'],r['binding']['batch_index']))
    counts=Counter()
    for row in m['windows']:counts.update(m['patterns'][row['pattern']]['declared_work'])
    require(len(m['windows'])==c.plan['total_windows'],'full window coverage incomplete')
    m.update(compiled_windows=len(m['windows']),blocked_windows=[],all_window_patterns_compiled=True,
             declared_work=dict(counts),witness_files=files,witness_kind='mixed_actual_replays_not_full_numerical_rerun')
    for name in ('scripts/complete_mlx_replayed_windows.py','scripts/audit_mlx_replayed_witnesses.py'):m['sources'][name]=digest(root/name)
    require(all(digest(Path(p))==h for p,h in files.items()),'inputs changed during completion')
    record(out/'manifest.json',m)
    print('REPLAYED_MODEL_WINDOWS_COMPLETE windows='+str(len(m['windows'])),flush=True)


if __name__=='__main__':main()
