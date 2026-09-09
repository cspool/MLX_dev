"""Bind complete model IR to the event model's finite resource requirements."""
import argparse
import json
from pathlib import Path

from mlxsim.model_event_resources import resource_contract
from scripts.run_mlx_tensor_semantics import sha
from scripts.mlx_system_attempt import record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise RuntimeError("choose a fresh event resource contract")
    digest = sha(args.program); result = resource_contract(json.loads(args.program.read_text()))
    if sha(args.program) != digest: raise RuntimeError("full source program changed during resource audit")
    result.update(program_path=str(args.program.resolve()), program_file_sha256=digest, compiler_sha256=sha(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True); record(args.output, result)
    print(json.dumps({k: result[k] for k in ("source_calls", "lowered_calls", "full_event_lowering_complete")}))


if __name__ == "__main__": main()
