"""Bind exact arithmetic unit tests and sanitizer projection reproductions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from scripts.run_mlx_tensor_semantics import ROOT,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('tests','diagnosis','sanitized-diagnosis','asan-binary','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();test=args.tests.resolve();binary=args.asan_binary.resolve()
    if out.exists():raise RuntimeError('choose a fresh exact arithmetic verification directory')
    suite=ET.parse(test/'regression.xml').getroot().find('testsuite')
    if suite.get('tests')!='14' or any(suite.get(k)!='0' for k in ('failures','errors','skipped')):raise RuntimeError('exact rounding regressions failed')
    out.mkdir(parents=True);sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/diagnose_mlx_exact_projections.py',ROOT/'tests/test_exact_f16_dot.py',ROOT/'simulator_ext/exact_numeric/exact_f16_dot.cc',ROOT/'simulator_ext/exact_numeric/CMakeLists.txt',ROOT/'src/mlxsim/model_matrix_reference.py',ROOT/'scripts/run_mlx_tensor_semantics.py']}
    binary_hash=sha(binary);environment=os.environ.copy();environment['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1';environment['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1';cases=[]
    for job in sorted((test/'pytest').rglob('job.json')):
        if any(p.is_symlink() for p in job.parents):continue
        expected=job.parent/'out/report.json';target=out/f'case-{len(cases):02d}'
        with (out/f'case-{len(cases):02d}.log').open('w') as log:result=subprocess.run([str(binary),str(job),str(target)],env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if result.returncode!=(0 if expected.exists() else 1):raise RuntimeError('exact sanitizer terminal differs')
        row={'job':str(job.relative_to(test)),'job_sha256':sha(job),'exit_code':result.returncode}
        if expected.exists():
            if json.loads(expected.read_text())!=json.loads((target/'report.json').read_text()):raise RuntimeError('sanitizer changed exact arithmetic results')
            for name in ('exact.f16.bin','kasc.f16.bin'):
                if (expected.parent/name).exists() and (expected.parent/name).read_bytes()!=(target/name).read_bytes():raise RuntimeError('sanitizer changed exact output bytes')
            row['expected_report_sha256']=sha(expected);row['sanitizer_report_sha256']=sha(target/'report.json')
        elif (target/'report.json').exists():raise RuntimeError('invalid arithmetic input generated a report')
        cases.append(row)
    if len(cases)!=14:raise RuntimeError('missing exact arithmetic sanitizer cases')
    release=json.loads((args.diagnosis/'report.json').read_text());sanitized=json.loads((args.sanitized_diagnosis/'report.json').read_text())
    if release['projections']!=sanitized['projections'] or len(release['projections'])!=3:raise RuntimeError('full projection sanitizer comparison differs')
    for report in (release,sanitized):
        for file,digest in report['files'].items():
            if sha(Path(file))!=digest:raise RuntimeError('diagnostic evidence changed')
        if report['model_outputs_injected'] or report['original_gpu_tolerance_changed'] or report['full_model_execution_verified']:raise RuntimeError('diagnostic scope was relabelled')
    if any(sha(ROOT/p)!=h for p,h in sources.items()) or sha(binary)!=binary_hash:raise RuntimeError('exact arithmetic verification sources changed')
    report={'classification':'exact_integer_diagnostic_validation_not_full_model_acceptance','sources':sources,'tests':14,'regression_xml_sha256':sha(test/'regression.xml'),'sanitizer_binary_sha256':binary_hash,'cases':cases,
            'finite_half_roundtrip_cases':63488,'midpoint_and_adjacent_cases':190464,'random_exact_rounding_cases':4000,'full_projections':3,'full_projection_exact_products':402653184,
            'diagnosis_sha256':sha(args.diagnosis/'report.json'),'sanitized_diagnosis_sha256':sha(args.sanitized_diagnosis/'report.json'),'full_model_execution_verified':False,'inference_performance_eligible':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(f'EXACT_NUMERIC_CHECKS_PASS {out/"report.json"}')


if __name__=='__main__':main()
