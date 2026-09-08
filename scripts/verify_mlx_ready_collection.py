"""Verify lifetime-triggered host collection without changing simulated work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

from scripts.run_mlx_ready_model import ROOT,sources
from scripts.run_mlx_tensor_semantics import sha
from scripts.verify_mlx_host_offer_cache import run,equivalent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('tests','baseline','candidate','asan','full-program','full-options','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();test=args.tests.resolve();out=args.output.resolve()
    if out.exists():raise RuntimeError('choose a fresh collection equivalence directory')
    suite=ET.parse(test/'regression.xml').getroot().find('testsuite')
    if any(suite.get(k)!='0' for k in ('failures','errors','skipped')):raise RuntimeError('collection regressions failed')
    out.mkdir(parents=True);before=sources()
    for file in (Path(__file__).resolve(),ROOT/'scripts/verify_mlx_host_offer_cache.py',ROOT/'tests/test_ready_collection.py'):before[str(file.relative_to(ROOT))]=sha(file)
    hashes={str(p.resolve()):sha(p) for p in (args.baseline,args.candidate,args.asan,args.full_program,args.full_options)}
    env=os.environ.copy();env['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1';env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1'
    selected=[]
    for file in sorted((test/'pytest').rglob('native/result.json')):
        if any(p.is_symlink() for p in file.parents):continue
        base=file.parent.parent
        if (base/'execution.json').exists():selected.append((base,file))
    for base in sorted((test/'pytest').glob('test_persistent_asset_regions_[0-9]*')):
        selected.append((base,base/'out/result.json'))
    if len(selected)!=13:raise RuntimeError('missing complete collection/lifetime cases')
    results=[]
    for index,(base,result) in enumerate(selected):
        old=out/f'old-{index}';san=out/f'asan-{index}'
        run(args.baseline,base/'program.json',base/'options.json',old,out/f'old-{index}.log')
        run(args.asan,base/'program.json',base/'options.json',san,out/f'asan-{index}.log',env)
        equivalent(result,old/'result.json');equivalent(result,san/'result.json')
        results.append({'case':str(base.relative_to(test)),'program_sha256':sha(base/'program.json'),'options_sha256':sha(base/'options.json'),
                        'candidate_report_sha256':sha(result),'baseline_report_sha256':sha(old/'result.json'),'sanitizer_report_sha256':sha(san/'result.json')})
    # This intentionally rejects at the limit, retaining the entire input graph.
    program=json.loads(args.full_program.read_text());options=json.loads(args.full_options.read_text())
    if len(program['nodes'])!=6181 or options['max_cycles']!=20000000:raise RuntimeError('bounded full graph profiling scope changed')
    timings={}
    for name,binary in (('old',args.baseline),('candidate',args.candidate)):
        start=time.perf_counter()
        with (out/f'full-{name}.log').open('w') as log:result=subprocess.run(list(map(str,[binary,args.full_program.resolve(),args.full_options.resolve(),out/f'full-{name}'])),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        timings[name]=time.perf_counter()-start
        if result.returncode!=1 or (out/f'full-{name}/result.json').exists() or 'ready graph exceeded global cycle budget' not in (out/f'full-{name}.log').read_text():raise RuntimeError('full graph diagnostic did not remain a limit rejection')
    if (out/'full-old.log').read_bytes()!=(out/'full-candidate.log').read_bytes():raise RuntimeError('bounded full graph source event log changed')
    if any(sha(ROOT/p)!=h for p,h in before.items()) or any(sha(Path(p))!=h for p,h in hashes.items()):raise RuntimeError('collection equivalence source/input changed')
    report={'classification':'host_collection_equivalence_not_full_model_execution_or_inference_performance','sources':before,'files':hashes,'regression_tests':int(suite.get('tests')),'regression_xml_sha256':sha(test/'regression.xml'),
            'complete_target_equivalence_cases':results,'bounded_full_graph':{'source_calls_in_program':6181,'cycle_limit':20000000,'exit_code':1,'source_event_logs_equal':True,'target_state_dumped':False,'host_seconds_single_sample':timings,'log_sha256':sha(out/'full-old.log')},
            'simulated_cycles_skipped':0,'full_model_execution_verified':False,'inference_performance_eligible':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(f'READY_COLLECTION_EQUIVALENCE_PASS {out/"report.json"}')


if __name__=='__main__':main()
