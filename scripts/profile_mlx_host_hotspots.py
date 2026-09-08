"""Profile host C++ implementation overhead, never model inference speed."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.run_mlx_ready_model import ROOT,sources
from scripts.run_mlx_tensor_semantics import sha
from mlxsim.model_ready_evidence import verify_ready_execution


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError('choose a fresh host profiling directory')
    out.mkdir(parents=True);before=sources();before[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    sys.path.insert(0,str(ROOT/'tests'));from test_block_pipeline import matrix_pair
    program,expected,tokens=matrix_pair(m=8,n=128,k=256,pes=16,precision='f16',slots=32)
    for kind in ('matrix','vector'):program[kind+'_schedule_options']['trace']=False
    options={'base':2**32,'bytes':1048576,'max_cycles':100000000,'max_active_nodes':32,'tile_pipeline':True,'template_load_timing':True,'memory':{'latency':8,'trace_limit':0}}
    (out/'program.json').write_text(json.dumps(program)+'\n');(out/'options.json').write_text(json.dumps(options)+'\n');(out/'expected.bin').write_bytes(expected.tobytes())
    owned=out/'mlx-ready-graph-profile';shutil.copy2(args.binary,owned);binary_hash=sha(owned)
    with (out/'run.log').open('w') as log:subprocess.run([str(owned),str(out/'program.json'),str(out/'options.json'),str(out/'result')],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    result=json.loads((out/'result/result.json').read_text());verify_ready_execution(program,result,options)
    if Path(result['outputs'][0]['logits_file']).read_bytes()!=expected.tobytes() or result['outputs'][0]['tokens']!=tokens:raise RuntimeError('profiling changed component numerical results')
    with (out/'gprof.txt').open('w') as log:subprocess.run(['gprof','-b',str(owned),str(out/'gmon.out')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    if any(sha(ROOT/p)!=h for p,h in before.items()) or sha(owned)!=binary_hash:raise RuntimeError('profiling sources/binary changed')
    report={'classification':'instrumented_host_hotspots_for_a_component_not_full_model_or_inference_performance','sources':before,'binary_sha256':binary_hash,
            'program_sha256':sha(out/'program.json'),'options_sha256':sha(out/'options.json'),'gmon_sha256':sha(out/'gmon.out'),'gprof_sha256':sha(out/'gprof.txt'),'result_sha256':sha(out/'result/result.json'),
            'matrix_mac_lanes':8*128*256,'full_model_execution_verified':False,'inference_performance_eligible':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(f'HOST_HOTSPOTS_RECORDED {out/"report.json"}')


if __name__=='__main__':main()
