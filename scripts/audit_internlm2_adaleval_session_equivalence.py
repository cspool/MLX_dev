#!/usr/bin/env python3
"""Run H32 exact-seed session-cap equivalence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_internlm2_adaleval import (
    preflight as h30_preflight,
)
from evaluate_internlm2_adaleval import (
    project_git_commit,
    response_finish_reason,
)

from mlxsim.adaleval import (
    build_stackselect_prompt,
    extract_stackselect_answer,
    fixed_replication_seed,
    load_stackselect,
    qualify_file,
    select_length_quantiles,
    sha256_file,
    validate_generation_result,
    wrap_internlm2_prompt,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/analysis/internlm2_adaleval_session_equivalence_v1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def selected_baseline_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = PROJECT_ROOT / config["sources"]["h31_rank1_samples"]["path"]
    selection = config["selection"]
    pool = [
        record
        for record in load_jsonl(source)
        if int(record["replicate"]) == int(selection["baseline_replicate"])
        and int(record["rank"]) == int(selection["baseline_rank"])
    ]
    return [
        dict(record)
        for record in select_length_quantiles(pool, count=int(selection["count"]))
    ]


def selection_stream_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(int(record["dataset_position"]).to_bytes(4, "big"))
        digest.update(int(record["input_token_len"]).to_bytes(4, "big"))
        digest.update(bytes.fromhex(record["prompt_utf8_sha256"]))
        digest.update(int(record["random_seed"]).to_bytes(8, "big"))
    return digest.hexdigest()


def gpu_inventory() -> list[dict[str, str | int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    inventory = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, name, uuid = (part.strip() for part in line.split(",", maxsplit=2))
        inventory.append({"index": int(index), "name": name, "uuid": uuid})
    return inventory


def preflight(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_reports = {
        name: qualify_file(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["sources"].items()
    }
    h30_config = load_yaml(PROJECT_ROOT / config["sources"]["h30_config"]["path"])
    h30_report = h30_preflight(h30_config)

    h31_path = PROJECT_ROOT / config["sources"]["h31_result"]["path"]
    h31 = json.loads(h31_path.read_text(encoding="utf-8"))
    h31_checks = {
        "run_id": h31["run_id"] == "run_035",
        "hypothesis": h31["hypothesis"] == "H31",
        "preflight": h31["checks"]["preflight"] is True,
        "records": h31["aggregate"]["total_records"] == 3000,
        "rank1_hash": h31["rank_sample_logs"][1]["sha256"]
        == config["sources"]["h31_rank1_samples"]["sha256"],
    }

    selected = selected_baseline_records(config)
    selection = config["selection"]
    expected_positions = [int(value) for value in selection["positions"]]
    expected_lengths = [int(value) for value in selection["input_token_lengths"]]
    dataset_root = PROJECT_ROOT / h30_config["datasets"]["root"]
    items = load_stackselect(
        dataset_root / h30_config["datasets"]["files"]["4k"]["path"]
    )
    prompt_checks = all(
        record["prompt_utf8_sha256"]
        == hashlib.sha256(
            build_stackselect_prompt(items[int(record["dataset_position"])]).encode()
        ).hexdigest()
        for record in selected
    )
    seed_checks = all(
        int(record["random_seed"])
        == fixed_replication_seed(
            "H31|internlm2-adaleval-4k",
            int(selection["baseline_replicate"]),
            int(record["dataset_position"]),
        )
        for record in selected
    )
    actual_selection_hash = selection_stream_sha256(selected)
    selection_checks = {
        "count": len(selected) == int(selection["count"]),
        "positions": [int(record["dataset_position"]) for record in selected]
        == expected_positions,
        "lengths": [int(record["input_token_len"]) for record in selected]
        == expected_lengths,
        "minimum": min(expected_lengths) == 3455,
        "maximum": max(expected_lengths) == 4451,
        "prompt_hashes": prompt_checks,
        "seeds": seed_checks,
        "stream_hash": actual_selection_hash == selection["stream_sha256"],
    }

    inventory = gpu_inventory()
    physical_gpu = int(selection["physical_gpu"])
    selected_gpu = next(
        (item for item in inventory if int(item["index"]) == physical_gpu), None
    )
    gpu_checks = {
        "inventory_count": len(inventory) == int(h30_config["runtime"]["gpu_count"]),
        "physical_gpu_present": selected_gpu is not None,
        "physical_gpu_name": bool(
            selected_gpu
            and selected_gpu["name"] == h30_config["runtime"]["gpu_name"]
        ),
        "baseline_rank_maps_to_gpu": int(selection["baseline_rank"])
        == physical_gpu,
    }
    candidate = config["candidate"]
    capacity_checks = {
        "official_session": int(candidate["session_len"]) == 160000,
        "cache_ratio": math.isclose(
            float(candidate["cache_max_entry_count"]),
            0.20,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "predicted_internal_capacity": int(candidate["expected_minimum_internal_tokens"])
        >= int(candidate["required_tokens"]),
        "candidate_covers_requests": int(candidate["session_len"])
        >= max(expected_lengths) + int(config["inference"]["max_new_tokens"]),
    }
    checks = {
        "sources": all(report["pass"] for report in source_reports.values()),
        "h30_preflight": h30_report["pass"],
        "h31": all(h31_checks.values()),
        "selection": all(selection_checks.values()),
        "gpu": all(gpu_checks.values()),
        "capacity": all(capacity_checks.values()),
    }
    report = {
        "sources": source_reports,
        "h30_preflight": h30_report,
        "h31_checks": h31_checks,
        "selection": {
            "actual_sha256": actual_selection_hash,
            "expected_sha256": selection["stream_sha256"],
            "positions": expected_positions,
            "input_token_lengths": expected_lengths,
            "checks": selection_checks,
            "pass": all(selection_checks.values()),
        },
        "gpu_inventory": inventory,
        "selected_gpu": selected_gpu,
        "gpu_checks": gpu_checks,
        "capacity_checks": capacity_checks,
        "checks": checks,
        "pass": all(checks.values()),
    }
    return report, h30_config


def assert_outputs_absent(config: dict[str, Any]) -> None:
    paths = [
        PROJECT_ROOT / config["run"]["output"],
        PROJECT_ROOT / config["run"]["samples"],
    ]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite H32 outputs: {existing}")


def generation_config(config: dict[str, Any], seed: int) -> Any:
    from lmdeploy import GenerationConfig

    inference = config["inference"]
    return GenerationConfig(
        max_new_tokens=int(inference["max_new_tokens"]),
        top_k=int(inference["top_k"]),
        top_p=float(inference["top_p"]),
        temperature=float(inference["temperature"]),
        repetition_penalty=float(inference["repetition_penalty"]),
        ignore_eos=bool(inference["ignore_eos"]),
        random_seed=seed,
    )


def run_worker(config: dict[str, Any], h30_config: dict[str, Any]) -> None:
    import torch
    from lmdeploy import Tokenizer, TurbomindEngineConfig, pipeline

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(
        config["selection"]["physical_gpu"]
    ):
        raise RuntimeError("H32 worker physical-GPU mask mismatch")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"H32 worker must see one GPU, got {torch.cuda.device_count()}")
    if torch.cuda.get_device_name(0) != h30_config["runtime"]["gpu_name"]:
        raise RuntimeError("H32 worker GPU identity mismatch")
    torch.cuda.set_device(0)

    model_root = PROJECT_ROOT / h30_config["model"]["historical_view_root"]
    candidate = config["candidate"]
    inference = config["inference"]
    backend_config = TurbomindEngineConfig(
        session_len=int(candidate["session_len"]),
        cache_max_entry_count=float(candidate["cache_max_entry_count"]),
        rope_scaling_factor=float(inference["rope_scaling_factor"]),
        max_prefill_token_num=int(inference["max_prefill_token_num"]),
    )
    pipe = pipeline(
        str(model_root),
        backend_config=backend_config,
        log_level=str(inference["log_level"]),
    )
    tokenizer = Tokenizer(str(model_root))

    baseline = selected_baseline_records(config)
    dataset_root = PROJECT_ROOT / h30_config["datasets"]["root"]
    items = load_stackselect(
        dataset_root / h30_config["datasets"]["files"]["4k"]["path"]
    )
    output = PROJECT_ROOT / config["run"]["samples"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", buffering=1) as handle:
        for selection_index, base in enumerate(baseline):
            position = int(base["dataset_position"])
            item = items[position]
            prompt = build_stackselect_prompt(item)
            expected_input_token_len = len(
                tokenizer.encode(wrap_internlm2_prompt(prompt))
            )
            started = time.perf_counter()
            with torch.no_grad():
                response = pipe(
                    prompt,
                    gen_config=generation_config(config, int(base["random_seed"])),
                )
            finish_reason = response_finish_reason(response)
            validate_generation_result(
                text=response.text,
                input_token_len=int(response.input_token_len),
                generate_token_len=int(response.generate_token_len),
                finish_reason=finish_reason,
                expected_input_token_len=expected_input_token_len,
                max_new_tokens=int(inference["max_new_tokens"]),
            )
            prediction = response.text
            extracted = extract_stackselect_answer(
                prediction, len(item["all_answers"])
            )
            record = {
                "selection_index": selection_index,
                "dataset_position": position,
                "physical_gpu": int(config["selection"]["physical_gpu"]),
                "index": item["index"],
                "answer": item["answer"],
                "num_choice": len(item["all_answers"]),
                "prompt_utf8_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "random_seed": int(base["random_seed"]),
                "input_token_len": int(response.input_token_len),
                "generate_token_len": int(response.generate_token_len),
                "finish_reason": finish_reason,
                "prediction": prediction,
                "extracted": extracted,
                "correct": extracted == item["answer"],
                "wall_time_seconds": time.perf_counter() - started,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if (selection_index + 1) % 8 == 0:
                print(f"H32 completed={selection_index + 1}/32", flush=True)


def launch_worker(config_path: Path, physical_gpu: int) -> int:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--worker",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    ).returncode


def write_report(
    config: dict[str, Any],
    preflight_report: dict[str, Any],
    *,
    wall_time_seconds: float,
) -> bool:
    baseline = selected_baseline_records(config)
    samples_path = PROJECT_ROOT / config["run"]["samples"]
    candidates = load_jsonl(samples_path)
    candidate_by_position = {
        int(record["dataset_position"]): record for record in candidates
    }
    comparisons = []
    fields = (
        "prediction",
        "extracted",
        "input_token_len",
        "generate_token_len",
        "finish_reason",
    )
    match_counts = {field: 0 for field in fields}
    mismatch_positions = {field: [] for field in fields}
    for base in baseline:
        position = int(base["dataset_position"])
        candidate = candidate_by_position[position]
        matches = {}
        for field in fields:
            matches[field] = candidate[field] == base[field]
            match_counts[field] += int(matches[field])
            if not matches[field]:
                mismatch_positions[field].append(position)
        comparisons.append(
            {
                "dataset_position": position,
                "input_token_len": int(base["input_token_len"]),
                "random_seed": int(base["random_seed"]),
                "baseline_prediction_sha256": hashlib.sha256(
                    base["prediction"].encode()
                ).hexdigest(),
                "candidate_prediction_sha256": hashlib.sha256(
                    candidate["prediction"].encode()
                ).hexdigest(),
                "matches": matches,
            }
        )

    selection = config["selection"]
    prompt_seed_checks = all(
        candidate["prompt_utf8_sha256"] == base["prompt_utf8_sha256"]
        and int(candidate["random_seed"]) == int(base["random_seed"])
        and int(candidate["physical_gpu"]) == int(selection["physical_gpu"])
        for base, candidate in zip(baseline, candidates, strict=True)
    )
    reextraction = all(
        candidate["extracted"]
        == extract_stackselect_answer(
            candidate["prediction"], int(candidate["num_choice"])
        )
        and bool(candidate["correct"])
        == (candidate["extracted"] == candidate["answer"])
        for candidate in candidates
    )
    checks = {
        "preflight": preflight_report["pass"],
        "sample_count": len(candidates) == int(selection["count"]),
        "unique_positions": len(candidate_by_position) == int(selection["count"]),
        "position_order": [int(record["dataset_position"]) for record in candidates]
        == [int(value) for value in selection["positions"]],
        "prompt_seed_gpu": prompt_seed_checks,
        "reextraction": reextraction,
        **{f"exact_{field}": match_counts[field] == int(selection["count"]) for field in fields},
    }
    report = {
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "classification": config["classification"],
        "validation_eligible": bool(config["validation_eligible"]),
        "git_commit": project_git_commit(),
        "qualification": preflight_report,
        "baseline": config["baseline"],
        "candidate": config["candidate"],
        "selection": {
            "count": int(selection["count"]),
            "minimum_input_tokens": min(int(r["input_token_len"]) for r in baseline),
            "maximum_input_tokens": max(int(r["input_token_len"]) for r in baseline),
            "stream_sha256": selection_stream_sha256(baseline),
        },
        "sample_log": {
            "path": str(samples_path),
            "bytes": samples_path.stat().st_size,
            "sha256": sha256_file(samples_path),
            "records": len(candidates),
        },
        "match_counts": match_counts,
        "mismatch_positions": mismatch_positions,
        "comparisons": comparisons,
        "baseline_correct": sum(bool(record["correct"]) for record in baseline),
        "candidate_correct": sum(bool(record["correct"]) for record in candidates),
        "checks": checks,
        "pass": all(checks.values()),
        "runtime": {
            "wall_time_seconds": wall_time_seconds,
            "sample_wall_time_seconds": sum(
                float(record["wall_time_seconds"]) for record in candidates
            ),
            "generated_tokens": sum(
                int(record["generate_token_len"]) for record in candidates
            ),
        },
        "limitations": config["limitations"],
    }
    output = PROJECT_ROOT / config["run"]["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "match_counts": match_counts,
                "mismatch_positions": mismatch_positions,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return bool(report["pass"])


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    h30_config = load_yaml(PROJECT_ROOT / config["sources"]["h30_config"]["path"])
    if args.worker:
        run_worker(config, h30_config)
        return 0

    preflight_report, _ = preflight(config)
    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return 0 if preflight_report["pass"] else 1
    if not preflight_report["pass"]:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return 1

    assert_outputs_absent(config)
    started = time.perf_counter()
    return_code = launch_worker(
        config_path,
        physical_gpu=int(config["selection"]["physical_gpu"]),
    )
    if return_code != 0:
        print(f"H32 worker failed: {return_code}", file=sys.stderr)
        return 1
    passed = write_report(
        config,
        preflight_report,
        wall_time_seconds=time.perf_counter() - started,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
