"""Compare complete same-input Q/K/V projections to an integer exact oracle.

Saved intermediate tensors are used only as diagnostic inputs here. These
results never enter a model run and cannot certify end-to-end inference.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from scripts.run_mlx_tensor_semantics import ROOT,sha


def compare(file,oracle):
    a=np.fromfile(file,dtype=np.uint16);b=np.fromfile(oracle,dtype=np.uint16)
    if a.shape!=b.shape:raise RuntimeError('projection diagnostic output extent differs')
    x=a.view(np.float16).astype(np.float64);y=b.view(np.float16).astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise RuntimeError('nonfinite real projection cannot pass this diagnosis')
    rank=lambda v:np.where(v&0x8000,(~v)&0xffff,v|0x8000).astype(np.int64)
    distance=np.abs(rank(a)-rank(b));values,counts=np.unique(distance,return_counts=True)
    return {'elements':int(a.size),'bitwise_changed':int(np.count_nonzero(a!=b)),'value_changed':int(np.count_nonzero(x!=y)),
            'max_abs_error':float(np.max(np.abs(x-y))),'max_binary16_steps':int(distance.max()),'binary16_step_histogram':{str(int(k)):int(n) for k,n in zip(values,counts)},'actual_sha256':sha(Path(file)),'oracle_sha256':sha(Path(oracle))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('gpu','blas-run','microcode-run','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--binary',type=Path)
    args=parser.parse_args();out=args.output.resolve();gpu=args.gpu.resolve();blas=args.blas_run.resolve();micro=args.microcode_run.resolve()
    if out.exists():raise RuntimeError('choose a fresh exact diagnostic directory')
    out.mkdir(parents=True);files={}
    def bind(file,expected=None):
        file=Path(file).resolve();digest=files.get(str(file)) or sha(file)
        if expected is not None and digest!=expected:raise RuntimeError(f'diagnostic evidence identity differs: {file}')
        files[str(file)]=digest;return digest
    def load(file):bind(file);return json.loads(Path(file).read_text())
    sources={str(p.relative_to(ROOT)):bind(p) for p in (Path(__file__).resolve(),ROOT/'scripts/run_mlx_tensor_semantics.py',ROOT/'simulator_ext/exact_numeric/CMakeLists.txt',ROOT/'simulator_ext/exact_numeric/exact_f16_dot.cc')}
    inventory=load(gpu/'inventory.json');gpu_observations=load(gpu/'diagnostics/observations.json')
    observed={r['source_operator_id']:r for r in gpu_observations['observations']}
    native_reports={label:load(base/'native/result.json') for label,base in (('blas',blas),('microcode',micro))}
    native={label:{r['source_operator_id']:r for r in report['observations']} for label,report in native_reports.items()}
    programs={label:load(base/'program.json') for label,base in (('blas',blas),('microcode',micro))}
    for label,base in (('blas',blas),('microcode',micro)):
        comparison=load(base/'comparison.json')
        if comparison['program_sha256']!=bind(base/'program.json') or comparison['executed_source_calls']!=6181:raise RuntimeError('historical full native scope/program differs')
        historical=load(base/('gpu-numerical-diagnosis.json' if label=='blas' else 'kasc-numerical-diagnosis.json'))
        bind(base/'native/result.json',historical['native_report_sha256'])
        old={r['value_id']:r for r in historical['observations']}
        for sid in (49,50,53,56):bind(native[label][sid]['file'],old[f'v{sid}']['native_sha256'])
    input_sha=bind(observed[49]['file'],observed[49]['sha256'])
    if any(bind(rows[49]['file'])!=input_sha for rows in native.values()):raise RuntimeError('Q/K/V diagnostic operands are not the same actual input')
    if observed[49]['shape']!=[1,8,4096] or observed[49]['dtype']!='torch.float16':raise RuntimeError('unregistered full first-layer input shape/precision')
    build=ROOT/'build/mlx-exact-numeric'
    if args.binary is None:
        with (out/'build.log').open('w') as log:
            for command in (['cmake','-S',ROOT/'simulator_ext/exact_numeric','-B',build,'-DCMAKE_BUILD_TYPE=Release'],['cmake','--build',build,'-j4']):subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    binary=args.binary.resolve() if args.binary else build/'mlx-exact-f16-dot';binary_hash=bind(binary);rows=[]
    for source_id,projection in ((50,'q'),(53,'k'),(56,'v')):
        source=inventory['operations'][source_id];node=programs['microcode']['nodes'][source_id]
        if source['operator_id']!=source_id or source['operator']!='aten.linear.default' or node['source_operator_id']!=source_id or node['kind']!='linear' or node['args'][0]!={'value':'v49'}:raise RuntimeError('projection source identity or operand route differs')
        weight=programs['microcode']['assets'][node['args'][1]['value']]
        if weight['parameter_name']!=f'model.layers.0.self_attn.{projection}_proj.weight' or weight['shape']!=[4096,4096] or weight['dtype']!='f16' or weight['bytes']!=4096*4096*2:raise RuntimeError('projection checkpoint shape/precision changed')
        bind(weight['path'],weight['file_sha256']);bind(observed[source_id]['file'],observed[source_id]['sha256'])
        if observed[source_id]['shape']!=[1,8,4096] or observed[source_id]['dtype']!='torch.float16' or observed[source_id]['forward_id']!=0 or observed[source_id]['layer_idx']!=0:raise RuntimeError('GPU diagnostic source shape/identity differs')
        for label,native_rows in native.items():
            old_node=programs[label]['nodes'][source_id]
            if old_node['args']!=node['args'] or programs[label]['assets'][old_node['args'][1]['value']]!=weight:raise RuntimeError('historical projection did not bind the same weights')
            if native_rows[source_id]['shape']!=[1,8,4096] or native_rows[source_id]['dtype']!='f16':raise RuntimeError('historical projection result shape/precision differs')
            bind(native_rows[source_id]['file'])
        directory=out/projection;directory.mkdir();job={'mode':'linear_f16_no_bias','m':8,'n':4096,'k':4096,'a':{'file':observed[49]['file'],'offset':0},'b':{'file':weight['path'],'offset':weight['byte_offset']}}
        (directory/'job.json').write_text(json.dumps(job)+'\n');bind(directory/'job.json')
        with (directory/'run.log').open('w') as log:subprocess.run([str(binary),str(directory/'job.json'),str(directory/'exact')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        exact=directory/'exact/exact.f16.bin';kasc=directory/'exact/kasc.f16.bin';result=load(directory/'exact/report.json')
        if result['exact_products']!=8*4096*4096 or result['output_elements']!=32768:raise RuntimeError('exact oracle omitted matrix work')
        bind(exact);bind(kasc)
        ordered=compare(native['microcode'][source_id]['file'],kasc)
        if ordered['bitwise_changed']:raise RuntimeError('declared K-ascending C++ microcode differs from independent same-input FP32 diagnostic')
        rows.append({'source_operator_id':source_id,'projection':projection,'same_input_sha256':input_sha,'weight_parameter':weight['parameter_name'],'elements':32768,'exact_products':result['exact_products'],
                     'gpu_vs_exact':compare(observed[source_id]['file'],exact),'blas_vs_exact':compare(native['blas'][source_id]['file'],exact),'microcode_vs_exact':compare(native['microcode'][source_id]['file'],exact),
                     'microcode_vs_independent_kasc':ordered,'gpu_vs_microcode':compare(observed[source_id]['file'],native['microcode'][source_id]['file'])})
    if any(sha(Path(p))!=h for p,h in files.items()):raise RuntimeError('exact diagnostic inputs or outputs changed')
    report={'classification':'same_input_full_projection_exact_diagnosis_not_full_model_execution_or_acceptance','sources':sources,'files':files,'binary_sha256':binary_hash,'reference_runtime':inventory['runtime'],'projections':rows,
            'oracle':'exact integer dot over finite FP16 inputs; final IEEE binary16 RN ties-to-even','model_outputs_injected':False,'original_gpu_tolerance_changed':False,
            'full_model_execution_verified':False,'mlx_system_verified':False,'inference_performance_eligible':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(f'EXACT_PROJECTION_DIAGNOSIS_COMPLETE {out/"report.json"}')


if __name__=='__main__':main()
