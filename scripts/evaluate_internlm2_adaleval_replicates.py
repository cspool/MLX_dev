#!/usr/bin/env python3
"""Run H31 fixed-seed replications of the historical Ada-LEval 4k stack."""

from __future__ import annotations

import argparse
import collections
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
    aggregate_replicate_records,
    build_stackselect_prompt,
    extract_stackselect_answer,
    fixed_replication_seed,
    load_stackselect,
    qualify_file,
    replication_seed_stream_sha256,
    sha256_file,
    validate_generation_result,
    wrap_internlm2_prompt,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/analysis/internlm2_adaleval_seed_replicates_v1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker-rank", type=int, choices=[0, 1])
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def qualify_h30_result(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "run_id": report["run_id"] == "run_034",
        "hypothesis": report["hypothesis"] == "H30",
        "preflight": report["checks"]["preflight"] is True,
        "records": report["aggregate"]["total_records"] == 3000,
        "4k_accuracy": math.isclose(
            float(report["aggregate"]["settings"]["4k"]["accuracy_pct"]),
            27.4,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    rank_logs = []
    for item in report["rank_sample_logs"]:
        path = Path(item["path"])
        actual_sha256 = sha256_file(path) if path.is_file() else None
        rank_logs.append(
            {
                "rank": int(item["rank"]),
                "path": str(path),
                "actual_bytes": path.stat().st_size if path.is_file() else None,
                "expected_bytes": int(item["bytes"]),
                "actual_sha256": actual_sha256,
                "expected_sha256": item["sha256"],
                "pass": bool(
                    path.is_file()
                    and path.stat().st_size == int(item["bytes"])
                    and actual_sha256 == item["sha256"]
                ),
            }
        )
    checks["rank_logs"] = all(item["pass"] for item in rank_logs)
    return {
        "checks": checks,
        "rank_logs": rank_logs,
        "pass": all(checks.values()),
    }


def preflight(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    base_specs = config["base_h30"]
    base_config_path = PROJECT_ROOT / base_specs["config"]["path"]
    base_result_path = PROJECT_ROOT / base_specs["formal_result"]["path"]
    base_files = {
        "config": qualify_file(base_config_path, base_specs["config"]),
        "formal_result": qualify_file(
            base_result_path, base_specs["formal_result"]
        ),
    }
    h30_config = load_yaml(base_config_path)
    h30_report = h30_preflight(h30_config)
    h30_result = qualify_h30_result(base_result_path)

    replicates = config["replicates"]
    namespace = str(replicates["seed_namespace"])
    replicate_count = int(replicates["count"])
    rows = int(config["dataset"]["rows"])
    seeds = [
        fixed_replication_seed(namespace, replicate, position)
        for replicate in range(replicate_count)
        for position in range(rows)
    ]
    actual_seed_hash = replication_seed_stream_sha256(
        namespace,
        replicate_count=replicate_count,
        rows_per_replicate=rows,
    )
    seed_schedule = {
        "rows": len(seeds),
        "unique": len(set(seeds)),
        "actual_sha256": actual_seed_hash,
        "expected_sha256": replicates["seed_stream_sha256"],
        "checks": {
            "rows": len(seeds) == replicate_count * rows,
            "unique": len(set(seeds)) == int(replicates["require_unique_seeds"]),
            "sha256": actual_seed_hash == replicates["seed_stream_sha256"],
        },
    }
    seed_schedule["pass"] = all(seed_schedule["checks"].values())

    checks = {
        "base_files": all(item["pass"] for item in base_files.values()),
        "h30_preflight": h30_report["pass"],
        "h30_result": h30_result["pass"],
        "seed_schedule": seed_schedule["pass"],
        "setting": config["dataset"]["setting"] == "4k",
        "rows": rows == 1000,
    }
    report = {
        "base_files": base_files,
        "h30_preflight": h30_report,
        "h30_result": h30_result,
        "seed_schedule": seed_schedule,
        "checks": checks,
        "pass": all(checks.values()),
    }
    return report, h30_config


def rank_output_path(config: dict[str, Any], rank: int) -> Path:
    return PROJECT_ROOT / config["run"]["rank_outputs"][rank]


def assert_formal_outputs_absent(config: dict[str, Any]) -> None:
    paths = [
        PROJECT_ROOT / config["run"]["output"],
        *(PROJECT_ROOT / path for path in config["run"]["rank_outputs"]),
    ]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite formal outputs: {existing}")


def generation_config(config: dict[str, Any], seed: int) -> Any:
    from lmdeploy import GenerationConfig

    inference = config["inference"]
    return GenerationConfig(
        max_new_tokens=int(inference["request_output_len"]),
        top_k=int(inference["top_k"]),
        top_p=float(inference["top_p"]),
        temperature=float(inference["temperature"]),
        repetition_penalty=float(inference["repetition_penalty"]),
        ignore_eos=bool(inference["ignore_eos"]),
        random_seed=seed,
    )


def run_worker(
    config: dict[str, Any], h30_config: dict[str, Any], *, rank: int
) -> None:
    import torch
    from lmdeploy import Tokenizer, TurbomindEngineConfig, pipeline

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated H31 worker must see one GPU, got {torch.cuda.device_count()}"
        )
    if torch.cuda.get_device_name(0) != h30_config["runtime"]["gpu_name"]:
        raise RuntimeError("isolated H31 worker GPU identity mismatch")
    torch.cuda.set_device(0)

    model_root = PROJECT_ROOT / h30_config["model"]["historical_view_root"]
    h30_inference = h30_config["inference"]
    backend_config = TurbomindEngineConfig(
        rope_scaling_factor=float(h30_inference["rope_scaling_factor"]),
        session_len=int(h30_inference["effective_session_len"]),
    )
    pipe = pipeline(
        str(model_root),
        backend_config=backend_config,
        log_level=str(config["inference"]["formal_log_level"]),
    )
    tokenizer = Tokenizer(str(model_root))

    setting = str(config["dataset"]["setting"])
    source = h30_config["datasets"]["files"][setting]
    dataset_root = PROJECT_ROOT / h30_config["datasets"]["root"]
    items = load_stackselect(dataset_root / source["path"])
    rows = int(config["dataset"]["rows"])
    if len(items) != rows:
        raise ValueError(f"expected {rows} items, got {len(items)}")

    replicate_count = int(config["replicates"]["count"])
    namespace = str(config["replicates"]["seed_namespace"])
    world_size = int(config["inference"]["nproc"])
    expected_total = replicate_count * (rows // world_size)
    progress_interval = int(config["inference"]["progress_interval"])
    output = rank_output_path(config, rank)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with output.open("x", encoding="utf-8", buffering=1) as handle:
        for replicate in range(replicate_count):
            for position in range(rank, rows, world_size):
                item = items[position]
                prompt = build_stackselect_prompt(item)
                expected_input_token_len = len(
                    tokenizer.encode(wrap_internlm2_prompt(prompt))
                )
                seed = fixed_replication_seed(namespace, replicate, position)
                started = time.perf_counter()
                with torch.no_grad():
                    response = pipe(
                        prompt,
                        gen_config=generation_config(h30_config, seed),
                    )
                finish_reason = response_finish_reason(response)
                validate_generation_result(
                    text=response.text,
                    input_token_len=int(response.input_token_len),
                    generate_token_len=int(response.generate_token_len),
                    finish_reason=finish_reason,
                    expected_input_token_len=expected_input_token_len,
                    max_new_tokens=int(h30_inference["request_output_len"]),
                )
                prediction = response.text
                extracted = extract_stackselect_answer(
                    prediction, len(item["all_answers"])
                )
                record = {
                    "setting": setting,
                    "replicate": replicate,
                    "dataset_position": position,
                    "rank": rank,
                    "index": item["index"],
                    "answer": item["answer"],
                    "num_choice": len(item["all_answers"]),
                    "prompt_utf8_sha256": hashlib.sha256(
                        prompt.encode()
                    ).hexdigest(),
                    "random_seed": seed,
                    "input_token_len": int(response.input_token_len),
                    "generate_token_len": int(response.generate_token_len),
                    "finish_reason": finish_reason,
                    "prediction": prediction,
                    "extracted": extracted,
                    "correct": extracted == item["answer"],
                    "wall_time_seconds": time.perf_counter() - started,
                }
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                completed += 1
                if completed % progress_interval == 0:
                    print(
                        f"rank={rank} completed={completed}/{expected_total} "
                        f"replicate={replicate} position={position}",
                        flush=True,
                    )


def launch_isolated_workers(
    *, config_path: Path, worker_count: int
) -> list[int]:
    processes: list[subprocess.Popen[str]] = []
    for rank in range(worker_count):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(rank)
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
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(config_path.resolve()),
            "--worker-rank",
            str(rank),
        ]
        processes.append(
            subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, text=True)
        )
    return [process.wait() for process in processes]


def read_rank_records(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for rank, configured_path in enumerate(config["run"]["rank_outputs"]):
        path = PROJECT_ROOT / configured_path
        rank_records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        records.extend(rank_records)
        files.append(
            {
                "rank": rank,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "records": len(rank_records),
            }
        )
    return records, files


def seed_stream_hash_from_records(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(
        records,
        key=lambda item: (int(item["replicate"]), int(item["dataset_position"])),
    ):
        digest.update(int(record["random_seed"]).to_bytes(8, "big"))
    return digest.hexdigest()


def compare_with_h30(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    result_path = PROJECT_ROOT / config["base_h30"]["formal_result"]["path"]
    report = json.loads(result_path.read_text(encoding="utf-8"))
    baseline_records: list[dict[str, Any]] = []
    for item in report["rank_sample_logs"]:
        baseline_records.extend(
            json.loads(line)
            for line in Path(item["path"]).read_text(encoding="utf-8").splitlines()
            if line
        )
    baseline_4k = {
        int(record["dataset_position"]): record
        for record in baseline_records
        if record["setting"] == "4k"
    }
    replicate_reports: dict[str, Any] = {}
    for replicate in range(int(config["replicates"]["count"])):
        selected = {
            int(record["dataset_position"]): record
            for record in records
            if int(record["replicate"]) == replicate
        }
        pairs = [(baseline_4k[position], selected[position]) for position in range(1000)]
        replicate_reports[str(replicate)] = {
            "prediction_agreement": sum(
                base["extracted"] == current["extracted"] for base, current in pairs
            ),
            "correctness_agreement": sum(
                bool(base["correct"]) == bool(current["correct"])
                for base, current in pairs
            ),
            "fixes": sum(
                not bool(base["correct"]) and bool(current["correct"])
                for base, current in pairs
            ),
            "regressions": sum(
                bool(base["correct"]) and not bool(current["correct"])
                for base, current in pairs
            ),
        }
    return {
        "run_id": report["run_id"],
        "accuracy_pct": report["aggregate"]["settings"]["4k"]["accuracy_pct"],
        "replicates": replicate_reports,
    }


def write_report(
    config: dict[str, Any],
    h30_config: dict[str, Any],
    preflight_report: dict[str, Any],
    *,
    wall_time_seconds: float,
) -> bool:
    records, rank_files = read_rank_records(config)
    replicate_count = int(config["replicates"]["count"])
    rows = int(config["dataset"]["rows"])
    tolerance = float(config["targets"]["relative_error_tolerance"])
    aggregate = aggregate_replicate_records(
        records,
        replicate_count=replicate_count,
        rows_per_replicate=rows,
        paper_target=float(config["targets"]["mlx_accuracy_pct"]),
        official_target=float(config["targets"]["official_accuracy_pct"]),
        tolerance=tolerance,
    )

    namespace = str(config["replicates"]["seed_namespace"])
    expected_prompt_hashes = {
        position: hashlib.sha256(build_stackselect_prompt(item).encode()).hexdigest()
        for position, item in enumerate(
            load_stackselect(
                PROJECT_ROOT
                / h30_config["datasets"]["root"]
                / h30_config["datasets"]["files"]["4k"]["path"]
            )
        )
    }
    rank_partition = all(
        int(record["dataset_position"]) % int(config["inference"]["nproc"])
        == int(record["rank"])
        for record in records
    )
    seeds = [int(record["random_seed"]) for record in records]
    seed_values = all(
        int(record["random_seed"])
        == fixed_replication_seed(
            namespace,
            int(record["replicate"]),
            int(record["dataset_position"]),
        )
        for record in records
    )
    record_correctness = all(
        record["extracted"]
        == extract_stackselect_answer(record["prediction"], int(record["num_choice"]))
        and bool(record["correct"]) == (record["extracted"] == record["answer"])
        for record in records
    )
    max_new_tokens = int(h30_config["inference"]["request_output_len"])
    response_bounds = all(
        1 <= int(record["generate_token_len"]) <= max_new_tokens
        and (
            (
                record["finish_reason"] == "length"
                and int(record["generate_token_len"]) == max_new_tokens
            )
            or (
                record["finish_reason"] == "stop"
                and int(record["generate_token_len"]) < max_new_tokens
            )
        )
        for record in records
    )
    prompt_hashes = all(
        record["prompt_utf8_sha256"]
        == expected_prompt_hashes[int(record["dataset_position"])]
        for record in records
    )
    per_replicate_rank_counts = all(
        sum(
            int(record["replicate"]) == replicate and int(record["rank"]) == rank
            for record in records
        )
        == rows // int(config["inference"]["nproc"])
        for replicate in range(replicate_count)
        for rank in range(int(config["inference"]["nproc"]))
    )
    checks = {
        "preflight": preflight_report["pass"],
        "total_records": len(records) == replicate_count * rows,
        "rank_file_counts": all(
            item["records"] == int(config["gate"]["require_records_per_rank"])
            for item in rank_files
        ),
        "per_replicate_rank_counts": per_replicate_rank_counts,
        "rank_partition": rank_partition,
        "unique_seeds": len(set(seeds))
        == int(config["replicates"]["require_unique_seeds"]),
        "seed_values": seed_values,
        "seed_stream": seed_stream_hash_from_records(records)
        == config["replicates"]["seed_stream_sha256"],
        "prompt_hashes": prompt_hashes,
        "response_bounds": response_bounds,
        "record_correctness": record_correctness,
        "sample_mean": math.isclose(
            aggregate["mean_accuracy_pct"],
            aggregate["sample_mean_accuracy_pct"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "primary_targets": aggregate["primary_pass"],
    }
    generated_lengths = [int(record["generate_token_len"]) for record in records]
    input_lengths = [int(record["input_token_len"]) for record in records]
    report = {
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "classification": config["classification"],
        "validation_eligible": bool(config["validation_eligible"]),
        "git_commit": project_git_commit(),
        "qualification": preflight_report,
        "rank_sample_logs": rank_files,
        "aggregate": aggregate,
        "h30_secondary_comparison": compare_with_h30(config, records),
        "checks": checks,
        "pass": all(checks.values()),
        "runtime": {
            "wall_time_seconds": wall_time_seconds,
            "total_sample_wall_time_seconds": sum(
                float(record["wall_time_seconds"]) for record in records
            ),
            "generated_tokens": sum(generated_lengths),
            "generate_token_len_min": min(generated_lengths),
            "generate_token_len_max": max(generated_lengths),
            "input_token_len_min": min(input_lengths),
            "input_token_len_max": max(input_lengths),
            "finish_reasons": dict(
                collections.Counter(record["finish_reason"] for record in records)
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
                "mean_accuracy_pct": aggregate["mean_accuracy_pct"],
                "replicate_accuracy_pct": {
                    key: value["accuracy_pct"]
                    for key, value in aggregate["replicates"].items()
                },
                "paper_relative_error": aggregate["paper_relative_error"],
                "official_relative_error": aggregate["official_relative_error"],
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
    base_config_path = PROJECT_ROOT / config["base_h30"]["config"]["path"]
    h30_config = load_yaml(base_config_path)

    if args.worker_rank is not None:
        run_worker(config, h30_config, rank=args.worker_rank)
        return 0

    preflight_report, h30_config = preflight(config)
    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return 0 if preflight_report["pass"] else 1
    if not preflight_report["pass"]:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return 1

    assert_formal_outputs_absent(config)
    started = time.perf_counter()
    worker_count = int(config["inference"]["nproc"])
    return_codes = launch_isolated_workers(
        config_path=config_path,
        worker_count=worker_count,
    )
    if any(code != 0 for code in return_codes):
        print(f"H31 workers failed: {return_codes}", file=sys.stderr)
        return 1
    passed = write_report(
        config,
        h30_config,
        preflight_report,
        wall_time_seconds=time.perf_counter() - started,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
