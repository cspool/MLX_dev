"""Replay completed native-model runner cases with a sanitizer build."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from scripts.run_mlx_ready_model import ROOT,sources
from scripts.run_mlx_tensor_semantics import sha
from scripts.verify_mlx_ready_graph import normalized
from mlxsim.model_ready_evidence import verify_ready_execution


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tests',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--asan-binary',type=Path,required=True)
    args=parser.parse_args();test=args.tests.resolve();out=args.output.resolve();asan=args.asan_binary.resolve()
    if out.exists():raise RuntimeError('choose a fresh native runner verification directory')
    suite=ET.parse(test/'regression.xml').getroot().find('testsuite')
    if any(suite.get(k)!='0' for k in ('failures','errors','skipped')):raise RuntimeError('native runner regressions did not all pass')
    out.mkdir(parents=True);before=sources();extra={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).resolve(),ROOT/'tests/test_native_ready_model.py',ROOT/'scripts/verify_mlx_ready_graph.py')};binary_hash=sha(asan)
    env=os.environ.copy();env['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1';env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1';cases=[]
    for file in sorted((test/'pytest').rglob('execution.json')):
        if any(p.is_symlink() for p in file.parents):continue
        state=json.loads(file.read_text())
        if state.get('classification')!='owned_native_paired_model_attempt_not_system_certificate':continue
        if state['status']!='exited' or state['sources']!=before:raise RuntimeError('native case is not terminal or has changed sources')
        base=file.parent
        for name,digest in {**state['inputs'],**state['runtime_libraries']}.items():
            if sha(Path(name))!=digest:raise RuntimeError('native inputs/runtime changed')
        if sha(base/'mlx-ready-graph')!=state['binary_sha256']:raise RuntimeError('native owned binary changed')
        for name,digest in state['sources'].items():
            if sha(base/'sources'/name)!=digest:raise RuntimeError('native source snapshot changed')
        destination=out/f'case-{len(cases):02d}'
        with (out/f'case-{len(cases):02d}.log').open('w') as log:
            process=subprocess.run([str(asan),str(base/'program.json'),str(base/'options.json'),str(destination)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if process.returncode!=state['exit_code']:raise RuntimeError('sanitizer process terminal differs')
        entry={'case':str(base.relative_to(test)),'execution_sha256':sha(file),'program_sha256':sha(base/'program.json'),'options_sha256':sha(base/'options.json'),'exit_code':state['exit_code']}
        if process.returncode:
            if process.returncode!=1 or (destination/'result.json').exists() or 'ready graph exceeded global cycle budget' not in (out/f'case-{len(cases):02d}.log').read_text():raise RuntimeError('expected cycle rejection did not remain a failed attempt')
        else:
            result=json.loads((base/'native/result.json').read_text());observed=json.loads((destination/'result.json').read_text());program=json.loads((base/'program.json').read_text());options=json.loads((base/'options.json').read_text())
            verify_ready_execution(program,observed,options)
            if normalized(result)!=normalized(observed):raise RuntimeError('sanitizer changed source work, clocks, events or resources')
            for a,b in zip(result['outputs'],observed['outputs'],strict=True):
                if Path(a['logits_file']).read_bytes()!=Path(b['logits_file']).read_bytes():raise RuntimeError('sanitizer changed actual logits')
            entry.update(result_sha256=sha(base/'native/result.json'),sanitizer_result_sha256=sha(destination/'result.json'),numeric_reference_passed=json.loads((base/'comparison.json').read_text())['numeric_reference_bitwise_passed'])
        cases.append(entry)
    if len(cases)!=15 or sum(c['exit_code']==0 for c in cases)!=10 or sum(c.get('numeric_reference_passed',False) for c in cases)!=5:raise RuntimeError('missing normal, numerical-rejection or cycle-rejection cases')
    if sources()!=before or sha(asan)!=binary_hash or any(sha(ROOT/p)!=h for p,h in extra.items()):raise RuntimeError('verification sources/binary changed')
    report={'classification':'native_paired_model_runner_component_validation_not_full_model_execution','sources':{**before,**extra},'regression_tests':int(suite.get('tests')),'regression_xml_sha256':sha(test/'regression.xml'),'cases':cases,'asan_binary_sha256':binary_hash,
            'asan_ubsan_lsan_cases':len(cases),'full_model_execution_verified':False,'actual_cpu_execution':False,'inference_performance_eligible':False,'rtl_verified':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(f'NATIVE_READY_RUNNER_CHECKS_PASS {out/"report.json"}')


if __name__=='__main__':main()
