#!/usr/bin/env python3
"""Run H30 with historical LMDeploy and exact Ada-LEval semantics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlxsim.adaleval import (
    aggregate_records,
    audit_prompt_stream,
    build_stackselect_prompt,
    extract_stackselect_answer,
    git_revision,
    line_segment_sha256,
    load_stackselect,
    prompt_canary_checks,
    qualify_file,
    sha256_file,
    validate_generation_result,
    wrap_internlm2_prompt,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/internlm2_adaleval_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker-rank", type=int, choices=[0, 1])
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def project_git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def nvidia_driver_version() -> str | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return next(iter(versions)) if len(versions) == 1 else None


def package_versions() -> dict[str, str | None]:
    names = [
        "lmdeploy",
        "torch",
        "transformers",
        "triton",
        "numpy",
        "sentencepiece",
        "protobuf",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def qualify_lmdeploy_sources(
    lmdeploy_root: Path, expected: dict[str, str]
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for relative, expected_sha256 in expected.items():
        relative_path = Path(relative)
        if relative_path.parts[0] == "lmdeploy":
            relative_path = Path(*relative_path.parts[1:])
        path = lmdeploy_root / relative_path
        actual_sha256 = sha256_file(path) if path.is_file() else None
        files[relative] = {
            "path": str(path),
            "actual_sha256": actual_sha256,
            "expected_sha256": expected_sha256,
            "pass": actual_sha256 == expected_sha256,
        }
    return {"files": files, "pass": all(item["pass"] for item in files.values())}


def qualify_model_view(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    history_root = PROJECT_ROOT / model["source_history_root"]
    view_root = PROJECT_ROOT / model["historical_view_root"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for group_name, specs in (
        ("historical_files", model["historical_files"]),
        ("binary_files", model["binary_files"]),
    ):
        reports = []
        for spec in specs:
            path = view_root / spec["path"]
            report = qualify_file(path, spec)
            report["is_symlink"] = path.is_symlink()
            report["pass"] = bool(report["pass"] and report["is_symlink"])
            reports.append(report)
        groups[group_name] = reports
    revision = git_revision(history_root)
    checks = {
        "historical_revision": revision == model["historical_revision"],
        "all_historical_files": all(item["pass"] for item in groups["historical_files"]),
        "all_binary_files": all(item["pass"] for item in groups["binary_files"]),
    }
    return {
        "historical_revision": revision,
        "expected_historical_revision": model["historical_revision"],
        "view_root": str(view_root),
        **groups,
        "checks": checks,
        "pass": all(checks.values()),
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    import lmdeploy
    import torch
    from lmdeploy import Tokenizer

    target = config["paper_targets"]["manifest"]
    paper_target = qualify_file(PROJECT_ROOT / target["path"], target)

    official = config["official_reference"]
    official_root = PROJECT_ROOT / official["repository"]["path"]
    official_files = {
        relative: qualify_file(official_root / relative, expected)
        for relative, expected in official["files"].items()
    }
    official_revision = git_revision(official_root)
    segment = official["readme_result_segment"]
    segment_actual_sha256 = line_segment_sha256(
        official_root / "README.md",
        int(segment["start_line"]),
        int(segment["end_line"]),
    )
    official_source = {
        "revision": official_revision,
        "expected_revision": official["repository"]["revision"],
        "files": official_files,
        "readme_result_segment": {
            "actual_sha256": segment_actual_sha256,
            "expected_sha256": segment["sha256"],
            "pass": segment_actual_sha256 == segment["sha256"],
        },
    }
    official_source["pass"] = bool(
        official_revision == official["repository"]["revision"]
        and all(item["pass"] for item in official_files.values())
        and official_source["readme_result_segment"]["pass"]
    )

    dataset_root = PROJECT_ROOT / config["datasets"]["root"]
    datasets = {
        setting: qualify_file(dataset_root / expected["path"], expected)
        for setting, expected in config["datasets"]["files"].items()
    }

    model_view = qualify_model_view(config)
    lmdeploy_root = Path(lmdeploy.__file__).resolve().parent
    lmdeploy_sources = qualify_lmdeploy_sources(
        lmdeploy_root, config["lmdeploy"]["source_hashes"]
    )

    versions = package_versions()
    runtime_expected = config["runtime"]
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    runtime_checks = {
        "python": platform.python_version() == str(runtime_expected["python"]),
        "lmdeploy": versions["lmdeploy"] == str(config["lmdeploy"]["version"]),
        "torch": versions["torch"] == str(runtime_expected["torch"]),
        "transformers": versions["transformers"] == str(runtime_expected["transformers"]),
        "triton": versions["triton"] == str(runtime_expected["triton"]),
        "numpy": versions["numpy"] == str(runtime_expected["numpy"]),
        "sentencepiece": versions["sentencepiece"]
        == str(runtime_expected["sentencepiece"]),
        "protobuf": versions["protobuf"] == str(runtime_expected["protobuf"]),
        "gpu_count": len(gpu_names) == int(runtime_expected["gpu_count"]),
        "gpu_names": all(name == runtime_expected["gpu_name"] for name in gpu_names),
        "driver": nvidia_driver_version() == str(runtime_expected["driver"]),
    }
    runtime = {
        "python": platform.python_version(),
        "packages": versions,
        "torch_cuda": torch.version.cuda,
        "gpu_names": gpu_names,
        "driver": nvidia_driver_version(),
        "checks": runtime_checks,
        "pass": all(runtime_checks.values()),
    }

    tokenizer = Tokenizer(str(PROJECT_ROOT / config["model"]["historical_view_root"]))
    prompt_audits: dict[str, Any] = {}
    for setting in config["datasets"]["settings"]:
        expected_file = config["datasets"]["files"][setting]
        items = load_stackselect(dataset_root / expected_file["path"])
        audit = audit_prompt_stream(items, tokenizer)
        checks = prompt_canary_checks(
            audit, config["datasets"]["prompt_stream_canaries"][setting]
        )
        audit["checks"] = checks
        audit["pass"] = all(checks.values())
        prompt_audits[setting] = audit

    inference = config["inference"]
    maximum_prompt_tokens = max(
        int(audit["wrapped_token_length"]["max"])
        for audit in prompt_audits.values()
    )
    required_session_tokens = maximum_prompt_tokens + int(
        inference["request_output_len"]
    )
    capacity_checks = {
        "effective_covers_every_request": int(inference["effective_session_len"])
        >= required_session_tokens,
        "effective_not_above_official": int(inference["effective_session_len"])
        <= int(inference["official_session_len"]),
        "every_prompt_is_single_prefill": maximum_prompt_tokens
        <= int(inference["max_prefill_token_num"]),
    }
    inference_capacity = {
        "official_session_len": int(inference["official_session_len"]),
        "effective_session_len": int(inference["effective_session_len"]),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "request_output_len": int(inference["request_output_len"]),
        "required_session_tokens": required_session_tokens,
        "max_prefill_token_num": int(inference["max_prefill_token_num"]),
        "checks": capacity_checks,
        "pass": all(capacity_checks.values()),
    }

    checks = {
        "paper_target": paper_target["pass"],
        "official_source": official_source["pass"],
        "datasets": all(item["pass"] for item in datasets.values()),
        "model_view": model_view["pass"],
        "lmdeploy_sources": lmdeploy_sources["pass"],
        "runtime": runtime["pass"],
        "prompt_audits": all(item["pass"] for item in prompt_audits.values()),
        "inference_capacity": inference_capacity["pass"],
    }
    return {
        "paper_target": paper_target,
        "official_source": official_source,
        "datasets": datasets,
        "model_view": model_view,
        "lmdeploy_sources": lmdeploy_sources,
        "runtime": runtime,
        "prompt_audits": prompt_audits,
        "inference_capacity": inference_capacity,
        "checks": checks,
        "pass": all(checks.values()),
    }


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


def response_finish_reason(response: Any) -> str | None:
    reason = getattr(response, "finish_reason", None)
    return None if reason is None else str(reason)


def run_inference(
    config: dict[str, Any],
    *,
    rank: int,
    world_size: int,
    smoke: bool,
) -> None:
    import torch
    from lmdeploy import GenerationConfig, Tokenizer, TurbomindEngineConfig, pipeline

    model_root = PROJECT_ROOT / config["model"]["historical_view_root"]
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated H30 worker must see one GPU, got {torch.cuda.device_count()}"
        )
    if torch.cuda.get_device_name(0) != config["runtime"]["gpu_name"]:
        raise RuntimeError("isolated H30 worker GPU identity mismatch")
    torch.cuda.set_device(0)
    backend_config = TurbomindEngineConfig(
        rope_scaling_factor=float(config["inference"]["rope_scaling_factor"]),
        session_len=int(config["inference"]["effective_session_len"]),
    )
    pipe = pipeline(str(model_root), backend_config=backend_config, log_level="INFO")
    tokenizer = Tokenizer(str(model_root))
    dataset_root = PROJECT_ROOT / config["datasets"]["root"]

    if smoke:
        setting = config["datasets"]["settings"][0]
        source = config["datasets"]["files"][setting]
        items = load_stackselect(dataset_root / source["path"])
        position = rank
        item = items[position]
        seed = 30_000 + rank
        generation = GenerationConfig(
            max_new_tokens=int(config["inference"]["request_output_len"]),
            top_k=int(config["inference"]["top_k"]),
            top_p=float(config["inference"]["top_p"]),
            temperature=float(config["inference"]["temperature"]),
            repetition_penalty=float(config["inference"]["repetition_penalty"]),
            ignore_eos=bool(config["inference"]["ignore_eos"]),
            random_seed=seed,
        )
        prompt = build_stackselect_prompt(item)
        expected_input_token_len = len(
            tokenizer.encode(wrap_internlm2_prompt(prompt))
        )
        with torch.no_grad():
            response = pipe(prompt, gen_config=generation)
        finish_reason = response_finish_reason(response)
        validate_generation_result(
            text=response.text,
            input_token_len=int(response.input_token_len),
            generate_token_len=int(response.generate_token_len),
            finish_reason=finish_reason,
            expected_input_token_len=expected_input_token_len,
            max_new_tokens=int(config["inference"]["request_output_len"]),
        )
        print(
            json.dumps(
                {
                    "smoke": True,
                    "rank": rank,
                    "setting": setting,
                    "dataset_position": position,
                    "seed": seed,
                    "input_token_len": int(response.input_token_len),
                    "generate_token_len": int(response.generate_token_len),
                    "finish_reason": finish_reason,
                    "extracted": extract_stackselect_answer(
                        response.text, len(item["all_answers"])
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    output = rank_output_path(config, rank)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with output.open("x", encoding="utf-8", buffering=1) as handle:
        for setting in config["datasets"]["settings"]:
            source = config["datasets"]["files"][setting]
            items = load_stackselect(dataset_root / source["path"])
            for position in range(rank, len(items), world_size):
                item = items[position]
                seed = random.getrandbits(64)
                generation = GenerationConfig(
                    max_new_tokens=int(config["inference"]["request_output_len"]),
                    top_k=int(config["inference"]["top_k"]),
                    top_p=float(config["inference"]["top_p"]),
                    temperature=float(config["inference"]["temperature"]),
                    repetition_penalty=float(config["inference"]["repetition_penalty"]),
                    ignore_eos=bool(config["inference"]["ignore_eos"]),
                    random_seed=seed,
                )
                prompt = build_stackselect_prompt(item)
                expected_input_token_len = len(
                    tokenizer.encode(wrap_internlm2_prompt(prompt))
                )
                started = time.perf_counter()
                with torch.no_grad():
                    response = pipe(prompt, gen_config=generation)
                finish_reason = response_finish_reason(response)
                validate_generation_result(
                    text=response.text,
                    input_token_len=int(response.input_token_len),
                    generate_token_len=int(response.generate_token_len),
                    finish_reason=finish_reason,
                    expected_input_token_len=expected_input_token_len,
                    max_new_tokens=int(config["inference"]["request_output_len"]),
                )
                prediction = response.text
                extracted = extract_stackselect_answer(
                    prediction, len(item["all_answers"])
                )
                record = {
                    "setting": setting,
                    "dataset_position": position,
                    "rank": rank,
                    "index": item["index"],
                    "answer": item["answer"],
                    "num_choice": len(item["all_answers"]),
                    "prompt_utf8_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "random_seed": seed,
                    "input_token_len": int(response.input_token_len),
                    "generate_token_len": int(response.generate_token_len),
                    "finish_reason": finish_reason,
                    "prediction": prediction,
                    "extracted": extracted,
                    "correct": extracted == item["answer"],
                    "wall_time_seconds": time.perf_counter() - started,
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                completed += 1
                if completed % 25 == 0:
                    print(
                        f"rank={rank} completed={completed}/1500 "
                        f"setting={setting} position={position}",
                        flush=True,
                    )


def read_rank_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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


def write_aggregate_report(
    config: dict[str, Any],
    preflight_report: dict[str, Any],
    *,
    wall_time_seconds: float,
) -> bool:
    records, rank_files = read_rank_records(config)
    rows_per_setting = int(config["datasets"]["rows_per_setting"])
    tolerance = float(config["paper_targets"]["relative_error_tolerance"])
    aggregate = aggregate_records(
        records,
        settings=config["datasets"]["settings"],
        rows_per_setting=rows_per_setting,
        paper_targets=config["paper_targets"]["accuracy_pct"],
        official_targets=config["official_reference"]["accuracy_pct"],
        tolerance=tolerance,
    )
    rank_partition = all(
        int(record["dataset_position"]) % int(config["inference"]["nproc"])
        == int(record["rank"])
        for record in records
    )
    correctness = all(
        record["extracted"]
        == extract_stackselect_answer(record["prediction"], int(record["num_choice"]))
        and bool(record["correct"]) == (record["extracted"] == record["answer"])
        for record in records
    )
    sample_means = all(
        math.isclose(
            value["accuracy_pct"],
            value["sample_mean_accuracy_pct"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for value in aggregate["settings"].values()
    )
    checks = {
        "preflight": preflight_report["pass"],
        "total_records": aggregate["total_records"] == 3 * rows_per_setting,
        "rank_file_counts": all(item["records"] == 1500 for item in rank_files),
        "rank_partition": rank_partition,
        "record_correctness": correctness,
        "sample_means": sample_means,
        "paper_targets": aggregate["paper_pass"],
    }
    report = {
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "classification": config["classification"],
        "validation_eligible": bool(config["validation_eligible"]),
        "git_commit": project_git_commit(),
        "source_and_runtime_qualification": preflight_report,
        "rank_sample_logs": rank_files,
        "aggregate": aggregate,
        "checks": checks,
        "pass": all(checks.values()),
        "official_reference_pass": aggregate["official_pass"],
        "runtime": {
            "wall_time_seconds": wall_time_seconds,
            "total_sample_wall_time_seconds": sum(
                float(record["wall_time_seconds"]) for record in records
            ),
            "generated_tokens": sum(int(record["generate_token_len"]) for record in records),
            "maximum_input_tokens": max(int(record["input_token_len"]) for record in records),
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
                "official_reference_pass": report["official_reference_pass"],
                "paper_maximum_relative_error": aggregate[
                    "paper_maximum_relative_error"
                ],
                "official_maximum_relative_error": aggregate[
                    "official_maximum_relative_error"
                ],
                "accuracy_pct": {
                    setting: values["accuracy_pct"]
                    for setting, values in aggregate["settings"].items()
                },
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return bool(report["pass"])


def launch_isolated_workers(
    *,
    config_path: Path,
    smoke: bool,
    worker_count: int,
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
        if smoke:
            command.append("--smoke")
        processes.append(
            subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
            )
        )
    return [process.wait() for process in processes]


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)

    if args.preflight_only:
        report = preflight(config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["pass"] else 1

    expected_world_size = int(config["inference"]["nproc"])
    if args.worker_rank is not None:
        run_inference(
            config,
            rank=int(args.worker_rank),
            world_size=expected_world_size,
            smoke=bool(args.smoke),
        )
        return 0

    started = time.perf_counter()
    if not args.smoke:
        assert_formal_outputs_absent(config)
    preflight_report = preflight(config)
    if not preflight_report["pass"]:
        raise SystemExit("H30 preflight qualification gate failed")
    return_codes = launch_isolated_workers(
        config_path=args.config,
        smoke=bool(args.smoke),
        worker_count=expected_world_size,
    )
    if any(return_code != 0 for return_code in return_codes):
        raise SystemExit(f"H30 isolated workers failed: {return_codes}")
    if args.smoke:
        return 0
    passed = write_aggregate_report(
        config,
        preflight_report,
        wall_time_seconds=time.perf_counter() - started,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
