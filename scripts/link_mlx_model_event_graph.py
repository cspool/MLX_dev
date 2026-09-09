"""Link a fully audited model window directory, preserving all original pairs."""
import argparse
import json
from pathlib import Path

from mlxsim.model_event_linker import ModelEventLinker
from scripts.mlx_system_attempt import digest,record
from scripts.verify_mlx_event_schedule import require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("compiled","audit","runtime-options","output"):parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();base=args.compiled.resolve();out=args.output.resolve();require(not out.exists(),"choose fresh full event link directory")
    path=base/"manifest.json";m=json.loads(path.read_text());audit=json.loads(args.audit.read_text());program_path=Path(m["program_path"])
    require(audit["manifest_sha256"]==digest(path) and audit["all_patterns_rebuilt"] and audit["all_layout_bindings_rebuilt"] and audit["all_window_patterns_compiled"],"window directory is not fully rebuilt/audited")
    require(digest(program_path)==m["program_file_sha256"],"compiled full model source changed")
    root=Path(__file__).resolve().parents[1];require(all(digest(root/name)==sha for name,sha in m["sources"].items()),"window compiler source changed")
    require(all(digest(Path(name))==sha for name,sha in m["witness_files"].items()),"actual window witness changed")
    files={str(path):digest(path),str(program_path):digest(program_path),str(args.audit.resolve()):digest(args.audit),str(args.runtime_options.resolve()):digest(args.runtime_options)}
    def read_pattern(key):
        p=base/"patterns"/(key+".json");require(digest(p)==audit["patterns"][key]["event_file_sha256"],"audited pattern changed");files[str(p)]=digest(p);return json.loads(p.read_text())
    program=json.loads(program_path.read_text());options=json.loads(args.runtime_options.read_text());linker=ModelEventLinker(program,m,read_pattern,out/"patterns",options)
    result=linker.link();record(out/"graph.json",result)
    require(all(digest(Path(p))==h for p,h in files.items()),"full event linking inputs changed")
    record(out/"manifest.json",dict(classification="full_model_compact_event_graph_link_not_execution",program_file_sha256=m["program_file_sha256"],inputs=files,witness_files=m["witness_files"],
        graph_sha256=digest(out/"graph.json"),runtime_options=options,original_source_calls=m["original_source_calls"],lowered_sources=len(result["source_streams"]),windows=len(m["windows"]),
        logical_blocks=linker.logical_blocks,dynamic_events=linker.dynamic_events,streaming_pairs=len(result["streaming_pairs"]),global_templates=len(result["templates"]),block_patterns=len(result["event_patterns"]),
        run_descriptors=sum(len(w["runs"]) for s in result["source_streams"] for w in s["windows"]),unique_window_blocks_scanned=linker.unique_window_blocks_scanned,
        sources={name:digest(root/name) for name in ("src/mlxsim/model_event_linker.py","scripts/link_mlx_model_event_graph.py")},
        full_model_execution_verified=False,model_performance_error_available=False,addresses_executed=False,final_readback_included=False))
    print(f"FULL_MODEL_EVENT_GRAPH_LINKED sources={len(result['source_streams'])} blocks={linker.logical_blocks} pairs={len(result['streaming_pairs'])}",flush=True)


if __name__=="__main__":main()
