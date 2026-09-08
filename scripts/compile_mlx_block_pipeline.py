"""Add validated bounded-pair scheduling metadata without changing model math."""
import argparse
import json
from pathlib import Path

from mlxsim.model_block_pipeline import compile_block_pipelines


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--event-slots",type=int,default=32)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError("choose a fresh pipeline program path")
    result=compile_block_pipelines(json.loads(args.program.read_text()),event_slots=args.event_slots)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(f"BLOCK_PIPELINE_COMPILED pairs={len(result['block_pipeline_plan']['pairs'])} (not execution) {args.output}")


if __name__=="__main__":main()
