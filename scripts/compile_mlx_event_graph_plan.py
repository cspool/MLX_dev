"""Audit every original/lowered source, dependency and batch in full model IR."""
import argparse
import json
from pathlib import Path

from mlxsim.model_event_graph_plan import graph_plan
from scripts.mlx_system_attempt import digest,record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError("choose a fresh full graph plan path")
    before=digest(args.program);plan=graph_plan(json.loads(args.program.read_text()))
    if digest(args.program)!=before:raise ValueError("full graph source changed")
    root=Path(__file__).resolve().parents[1]
    names=["src/mlxsim/model_event_graph_plan.py","src/mlxsim/model_event_resources.py","src/mlxsim/model_block_pipeline.py","src/mlxsim/model_value_outputs.py","src/mlxsim/model_source_groups.py","src/mlxsim/model_result_contract.py","scripts/compile_mlx_event_graph_plan.py"]
    plan.update(program_path=str(args.program.resolve()),program_file_sha256=before,sources_sha256={name:digest(root/name) for name in names})
    args.output.parent.mkdir(parents=True,exist_ok=True);record(args.output,plan)
    print(json.dumps({k:plan[k] for k in ("source_calls","lowered_calls","family_windows","total_windows","dependency_edges")}))


if __name__=="__main__":main()
