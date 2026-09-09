"""Add C++ modeled output-readback time to a successfully audited full event run."""
import argparse
import json
from pathlib import Path
import shutil

from mlxsim.model_readback_timing import readback_job
from scripts.mlx_system_attempt import digest,record,run_process,linked_libraries
from scripts.verify_mlx_event_schedule import require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('event-run','program','runtime-options','binary','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();run=args.event_run.resolve()
    require(not out.exists(),'choose fresh readback output')
    require((run/'report.json').is_file(),'full graph has no successful audited terminal report')
    report=json.loads((run/'report.json').read_text());execution=json.loads((run/'execution.json').read_text())
    require(execution['status']=='exited' and execution['exit_code']==0 and report['event_graph_execution_verified'],
            'full graph has not successfully completed and passed its audit')
    require(report['result_sha256']==digest(run/'result.json'),'event result changed')
    for p in (args.program.resolve(),args.runtime_options.resolve()):
        require(execution['inputs'].get(str(p))==digest(p),'program/options differ from full graph')
    require(all(digest(Path(p))==h for p,h in execution['inputs'].items()),'graph inputs changed')
    graph=json.loads((run/'result.json').read_text())
    job=readback_job(json.loads(args.program.read_text()),json.loads(args.runtime_options.read_text()),graph['cycles'])
    root=Path(__file__).resolve().parents[1]
    sources={name:digest(root/name) for name in ('simulator_ext/readback_timing/main.cc','simulator_ext/readback_timing/CMakeLists.txt',
             'src/mlxsim/model_readback_timing.py','src/mlxsim/model_value_outputs.py','src/mlxsim/model_result_contract.py',
             'src/mlxsim/model_memory_program.py','scripts/complete_mlx_event_readback.py')}
    out.mkdir(parents=True);record(out/'job.json',job);binary=out/'readback-timing';shutil.copy2(args.binary,binary)
    inputs={str(p.resolve()):digest(p) for p in (args.program,args.runtime_options,run/'report.json',run/'result.json',run/'execution.json',out/'job.json')}
    sha=digest(binary);libraries=linked_libraries(binary)
    run_process([str(binary),str(out/'job.json'),str(out/'result.json')],out/'run.log',out/'execution.json',timeout=300,
                metadata=dict(mode='post_graph_readback_timing',inputs=inputs,sources=sources,binary_sha256=sha,runtime_libraries=libraries))
    require(digest(binary)==sha and all(digest(Path(p))==h for p,h in inputs.items()) and
            all(digest(root/p)==h for p,h in sources.items()) and all(digest(Path(p))==h for p,h in libraries.items()),'readback provenance changed')
    result=json.loads((out/'result.json').read_text())
    require(result['graph_cycles']==graph['cycles'],'readback start differs')
    record(out/'report.json',dict(classification='event_graph_plus_modeled_scalar_readback_not_numerical_or_chipyard_acceptance',
           event_graph_cycles=result['graph_cycles'],event_readback_cycles=result['host_readback_cycles'],
           event_total_cycles=result['shared_elapsed_cycles'],readback_requests=result['host_readback_requests'],
           modeled_final_readback_included=True,tensor_values_executed=False,initialization_included=False,
           cpu_postprocessing_timed=False,model_performance_error_available=False,mlx_system_verified=False))
    print('EVENT_READBACK_COMPLETED cycles='+str(result['shared_elapsed_cycles']),flush=True)


if __name__=='__main__':main()
