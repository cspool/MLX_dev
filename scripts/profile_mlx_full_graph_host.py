"""Inspect a bounded prefix of an unchanged full graph; never certify inference."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from scripts.run_mlx_tensor_semantics import ROOT,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('program','options','profile-binary','profile-provenance','frozen-source-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError('choose a fresh bounded host profile directory')
    inputs={str(p.resolve()):sha(p) for p in (args.program,args.options,args.profile_binary,args.profile_provenance)}
    provenance=json.loads(args.profile_provenance.read_text());frozen=args.frozen_source_root.resolve();source={}
    if sha(args.profile_binary)!=provenance['binary_sha256']:raise RuntimeError('profiling binary provenance differs')
    for name,digest in provenance['sources'].items():
        if name.startswith('simulator_ext/'):
            if sha(frozen/name)!=digest:raise RuntimeError('profile C++ sources differ from actual frozen native execution')
            source[name]=digest
    program=json.loads(args.program.read_text());options=json.loads(args.options.read_text())
    if len(program['nodes'])!=6181 or not options['tile_pipeline'] or not options['template_load_timing'] or options['max_cycles']!=20000000:raise RuntimeError('unexpected full graph profiling scope')
    out.mkdir(parents=True);binary=out/'profile-binary';shutil.copy2(args.profile_binary,binary)
    with (out/'run.log').open('w') as log:result=subprocess.run([str(binary),str(args.program.resolve()),str(args.options.resolve()),str(out/'partial')],cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if result.returncode!=1 or 'ready graph exceeded global cycle budget' not in (out/'run.log').read_text() or (out/'partial/result.json').exists():raise RuntimeError('bounded diagnostic did not end at its explicit cycle limit')
    with (out/'gprof.txt').open('w') as log:subprocess.run(['gprof','-b',str(binary),str(out/'gmon.out')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    if any(sha(Path(p))!=h for p,h in inputs.items()) or any(sha(frozen/p)!=h for p,h in source.items()):raise RuntimeError('bounded profile sources or inputs changed')
    report={'classification':'bounded_full_graph_host_hotspots_not_full_model_execution_or_performance','inputs':inputs,'frozen_cpp_sources':source,'source_calls_in_program':6181,'cycle_limit':options['max_cycles'],'exit_code':result.returncode,
            'gprof_sha256':sha(out/'gprof.txt'),'gmon_sha256':sha(out/'gmon.out'),'full_model_execution_verified':False,'inference_performance_eligible':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(f'BOUNDED_FULL_GRAPH_HOST_PROFILE_COMPLETE {out/"report.json"}')


if __name__=='__main__':main()
