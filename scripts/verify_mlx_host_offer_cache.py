"""Check host-only admission memoization against the original executable."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

from scripts.run_mlx_ready_model import ROOT,sources
from scripts.run_mlx_tensor_semantics import sha
from scripts.verify_mlx_ready_graph import normalized


def run(binary,program,options,destination,log,environment=None):
    start=time.perf_counter()
    with log.open('w') as output:result=subprocess.run(list(map(str,[binary,program,options,destination])),cwd=ROOT,env=environment,stdout=output,stderr=subprocess.STDOUT,timeout=300)
    if result.returncode:raise RuntimeError(f'host implementation regression: {log}')
    return time.perf_counter()-start


def equivalent(a,b):
    left=json.loads(a.read_text());right=json.loads(b.read_text())
    if normalized(left)!=normalized(right):raise RuntimeError('host optimization changed complete target report')
    for x,y in zip(left['outputs'],right['outputs'],strict=True):
        if Path(x['logits_file']).read_bytes()!=Path(y['logits_file']).read_bytes():raise RuntimeError('host optimization changed actual output bytes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('tests','baseline','candidate','asan','output','profile-case'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();test=args.tests.resolve()
    if out.exists():raise RuntimeError('choose a fresh host equivalence directory')
    suite=ET.parse(test/'regression.xml').getroot().find('testsuite')
    if any(suite.get(k)!='0' for k in ('failures','errors','skipped')):raise RuntimeError('host equivalence requires passing regressions')
    out.mkdir(parents=True);before=sources()
    for file in (Path(__file__).resolve(),ROOT/'tests/shared_offer_cache_contract.cc',ROOT/'tests/test_shared_offer_cache.py'):before[str(file.relative_to(ROOT))]=sha(file)
    binaries={str(p.resolve()):sha(p) for p in (args.baseline,args.candidate,args.asan)}
    env=os.environ.copy();env['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1';env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1'
    contract=args.asan.resolve().parent/'shared-offer-cache-contract';binaries[str(contract)]=sha(contract)
    with (out/'asan-contract.log').open('w') as log:subprocess.run([str(contract)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    cases=[]
    # The model-runner fixture supplies complete pair, broadcast, generation,
    # perturbation and tail graphs; reference-failure cases still execute data.
    for file in sorted((test/'pytest').rglob('native/result.json')):
        if any(p.is_symlink() for p in file.parents):continue
        base=file.parent.parent
        if not (base/'execution.json').exists():continue
        state=json.loads((base/'execution.json').read_text())
        if state.get('classification')!='owned_native_paired_model_attempt_not_system_certificate':continue
        index=len(cases);old=out/f'old-{index}';san=out/f'asan-{index}'
        run(args.baseline,base/'program.json',base/'options.json',old,out/f'old-{index}.log')
        run(args.asan,base/'program.json',base/'options.json',san,out/f'asan-{index}.log',env)
        equivalent(file,old/'result.json');equivalent(file,san/'result.json')
        cases.append({'case':str(base.relative_to(test)),'program_sha256':sha(base/'program.json'),'options_sha256':sha(base/'options.json'),
                      'candidate_report_sha256':sha(file),'baseline_report_sha256':sha(old/'result.json'),'sanitizer_report_sha256':sha(san/'result.json')})
    if len(cases)!=10:raise RuntimeError('missing complete native runner comparison cases')
    benchmark=args.profile_case.resolve();timings=[]
    for repeat in range(4):
        row={}
        for label,binary in ((('old',args.baseline),('candidate',args.candidate)) if repeat%2==0 else (('candidate',args.candidate),('old',args.baseline))):
            target=out/f'host-{repeat}-{label}'
            row[label]=run(binary,benchmark/'program.json',benchmark/'options.json',target,out/f'host-{repeat}-{label}.log')
        equivalent(out/f'host-{repeat}-old/result.json',out/f'host-{repeat}-candidate/result.json');timings.append(row)
    if any(sha(ROOT/p)!=h for p,h in before.items()) or any(sha(Path(p))!=h for p,h in binaries.items()):raise RuntimeError('host verification sources/binaries changed')
    report={'classification':'host_implementation_equivalence_and_component_wall_time_not_model_inference_performance','sources':before,'binaries':binaries,
            'regression_tests':int(suite.get('tests')),'regression_xml_sha256':sha(test/'regression.xml'),'complete_target_equivalence_cases':cases,'alternating_host_seconds':timings,
            'sanitizer_contract_sha256':sha(out/'asan-contract.log'),
            'host_component_program_sha256':sha(benchmark/'program.json'),'host_component_options_sha256':sha(benchmark/'options.json'),
            'simulated_cycles_skipped':0,'full_model_execution_verified':False,'inference_performance_eligible':False,'rtl_verified':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(f'HOST_OFFER_CACHE_EQUIVALENCE_PASS {out/"report.json"}')


if __name__=='__main__':main()
