#!/usr/bin/env python3
"""Evaluate the frozen H27 Llama2-7B WinoGrande protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import datasets
import lm_eval
import torch
import transformers
import yaml
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable

from mlxsim.llama_perplexity import qualify_model_files, sha256_file
from mlxsim.winogrande import (
    audit_accuracy,
    canonical_rows_sha256,
    qualify_parquet_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/llama2_winogrande_v1.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true", help="score one example only")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _distribution_metadata_path(distribution_name: str) -> Path:
    distribution = importlib.metadata.distribution(distribution_name)
    matches = [
        Path(distribution.locate_file(item))
        for item in distribution.files or []
        if item.name == "METADATA" and ".dist-info" in str(item)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"could not identify {distribution_name} METADATA: {matches}")
    return matches[0]


def _qualify_harness(config: Mapping[str, Any]) -> dict[str, Any]:
    harness = config["harness"]
    metadata_path = _distribution_metadata_path(harness["distribution"])
    lm_eval_root = Path(lm_eval.__file__).resolve().parent
    paths = {
        "distribution_metadata": metadata_path,
        "upstream_task_yaml": lm_eval_root / "tasks/winogrande/default.yaml",
        "upstream_preprocessor": lm_eval_root / "tasks/winogrande/preprocess_winogrande.py",
        "task_yaml": PROJECT_ROOT / harness["task_yaml"],
        "preprocessor": PROJECT_ROOT / harness["preprocessor"],
    }
    expected_hashes = {
        "distribution_metadata": harness["distribution_metadata_sha256"],
        "upstream_task_yaml": harness["upstream_task_yaml_sha256"],
        "upstream_preprocessor": harness["upstream_preprocessor_sha256"],
        "task_yaml": harness["task_yaml_sha256"],
        "preprocessor": harness["preprocessor_sha256"],
    }
    file_checks: dict[str, Any] = {}
    for name, path in paths.items():
        actual = sha256_file(path) if path.is_file() else None
        file_checks[name] = {
            "path": _relative_or_absolute(path),
            "expected_sha256": expected_hashes[name],
            "actual_sha256": actual,
            "pass": actual == expected_hashes[name],
        }

    actual_version = importlib.metadata.version(harness["distribution"])
    version_pass = actual_version == str(harness["version"])
    return {
        "distribution": harness["distribution"],
        "expected_version": str(harness["version"]),
        "actual_version": actual_version,
        "version_pass": version_pass,
        "files": file_checks,
        "pass": version_pass and all(item["pass"] for item in file_checks.values()),
    }


def _qualify_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = config["runtime"]
    expected = {
        "python_version": str(runtime["python_version"]),
        "torch_version": str(runtime["torch_version"]),
        "transformers_version": str(runtime["transformers_version"]),
        "datasets_version": str(runtime["datasets_version"]),
        "cuda_visible_devices": str(runtime["required_cuda_visible_devices"]),
        "cuda_device_name": str(runtime["expected_cuda_device_name"]),
    }
    cuda_available = torch.cuda.is_available()
    actual = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }
    checks = {
        name: actual[name] == expected[name]
        for name in (
            "python_version",
            "torch_version",
            "transformers_version",
            "datasets_version",
            "cuda_visible_devices",
            "cuda_device_name",
        )
    }
    checks["cuda_available"] = cuda_available
    checks["one_visible_cuda_device"] = actual["cuda_device_count"] == 1
    return {
        "expected": expected,
        "actual": actual,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _qualify_loaded_task(task: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    expected_dataset = config["dataset"]
    expected_eval = config["evaluation"]
    expected_harness = config["harness"]
    rows = task.eval_docs
    content_hash = canonical_rows_sha256(dict(row) for row in rows)

    semantic_mismatches: list[int] = []
    for index, row in enumerate(rows):
        sentence = row["sentence"]
        underscore = sentence.index("_")
        expected_label = {"1": 0, "2": 1}[row["answer"]]
        expected_choices = [
            sentence[:underscore] + row["option1"],
            sentence[:underscore] + row["option2"],
        ]
        expected_target = sentence[underscore + 1 :].strip()
        if (
            task.doc_to_text(row) != expected_label
            or task.doc_to_choice(row) != expected_choices
            or task.doc_to_target(row) != expected_target
        ):
            semantic_mismatches.append(index)

    task_config = task.config
    checks = {
        "task_name": task.task_name == expected_harness["task_name"],
        "dataset_path": task_config.dataset_path == "allenai/winogrande",
        "dataset_name": task_config.dataset_name == expected_dataset["dataset_name"],
        "dataset_revision": task_config.dataset_kwargs.get("revision")
        == expected_dataset["official_revision"],
        "validation_split": task_config.validation_split == expected_dataset["split"],
        "output_type": task_config.output_type == expected_eval["output_type"],
        "num_fewshot": task_config.num_fewshot == expected_eval["num_fewshot"],
        "row_count": len(rows) == expected_dataset["expected_rows"],
        "content_sha256": content_hash == expected_dataset["canonical_content_sha256"],
        "partial_scoring_semantics": not semantic_mismatches,
    }
    first = rows[0]
    return {
        "task_name": task.task_name,
        "rows": len(rows),
        "canonical_content_sha256": content_hash,
        "semantic_mismatch_count": len(semantic_mismatches),
        "first_semantic_mismatch_indices": semantic_mismatches[:10],
        "first_example": {
            "doc_to_text": task.doc_to_text(first),
            "doc_to_choice": task.doc_to_choice(first),
            "doc_to_target": task.doc_to_target(first),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _resolve_target(manifest: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = manifest
    for key in dotted_key.split("."):
        value = value[key]
    return value


def _sample_log_text(samples: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(
            sample,
            default=handle_non_serializable,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for sample in samples
    )


def main() -> int:
    args = _parse_args()
    config = _load_yaml(args.config)
    official = not args.smoke
    output = PROJECT_ROOT / config["run"]["output"]
    samples_output = PROJECT_ROOT / config["run"]["samples_output"]
    if official:
        existing = [path for path in (output, samples_output) if path.exists()]
        if existing:
            raise SystemExit(f"refusing to overwrite official results: {existing}")

    started = time.perf_counter()
    harness_qualification = _qualify_harness(config)
    runtime_qualification = _qualify_runtime(config)
    model_qualification = qualify_model_files(config["model"])
    dataset_path = PROJECT_ROOT / config["dataset"]["qualification_path"]
    dataset_qualification = qualify_parquet_dataset(dataset_path, config["dataset"])
    preflight = {
        "harness": harness_qualification["pass"],
        "runtime": runtime_qualification["pass"],
        "model": model_qualification["pass"],
        "dataset": dataset_qualification["pass"],
    }
    if not all(preflight.values()):
        raise RuntimeError(f"H27 preflight qualification failed: {preflight}")

    task_manager = TaskManager(include_path=PROJECT_ROOT / config["harness"]["include_path"])
    loaded = task_manager.load([config["harness"]["task_name"]])
    task = loaded["tasks"][config["harness"]["task_name"]]
    task_qualification = _qualify_loaded_task(task, config)
    if not task_qualification["pass"]:
        raise RuntimeError(f"H27 task qualification failed: {task_qualification}")

    target_manifest = _load_yaml(PROJECT_ROOT / config["target"]["source"])
    canonical_target = float(_resolve_target(target_manifest, config["target"]["key"]))
    if canonical_target != float(config["target"]["accuracy_pct"]):
        raise RuntimeError(
            f"target mismatch: config {config['target']['accuracy_pct']}, "
            f"canonical {canonical_target}"
        )

    model_path = (PROJECT_ROOT / config["model"]["path"]).resolve()
    evaluation = config["evaluation"]
    results = simple_evaluate(
        model="hf",
        model_args={
            "pretrained": str(model_path),
            "revision": config["model"]["official_revision"],
            "dtype": config["runtime"]["model_dtype"],
            "max_length": int(evaluation["max_length"]),
            "use_fast_tokenizer": bool(evaluation["use_fast_tokenizer"]),
            "add_bos_token": evaluation["add_bos_token"],
            "parallelize": False,
            "trust_remote_code": False,
            "local_files_only": True,
        },
        tasks=[config["harness"]["task_name"]],
        num_fewshot=int(evaluation["num_fewshot"]),
        batch_size=int(evaluation["batch_size"]),
        device=config["runtime"]["device"],
        cache_requests=bool(evaluation["cache_requests"]),
        limit=1 if args.smoke else None,
        bootstrap_iters=int(evaluation["bootstrap_iters"]),
        write_out=bool(evaluation["write_out"]),
        log_samples=bool(evaluation["log_samples"]),
        system_instruction=evaluation["system_instruction"],
        apply_chat_template=bool(evaluation["apply_chat_template"]),
        fewshot_as_multiturn=bool(evaluation["fewshot_as_multiturn"]),
        task_manager=task_manager,
        verbosity="INFO",
        random_seed=int(evaluation["random_seed"]),
        numpy_random_seed=int(evaluation["numpy_random_seed"]),
        torch_random_seed=int(evaluation["torch_random_seed"]),
        fewshot_random_seed=int(evaluation["fewshot_random_seed"]),
    )
    if results is None:
        raise RuntimeError("lm-eval returned no rank-zero result")

    task_name = config["harness"]["task_name"]
    samples = results["samples"][task_name]
    aggregate = float(results["results"][task_name]["acc,none"])
    audit = audit_accuracy(
        sample_values=[float(sample["acc"]) for sample in samples],
        aggregate_accuracy=aggregate,
        paper_target_pct=canonical_target,
        relative_error_gate=float(config["target"]["relative_error_gate"]),
    )
    expected_samples = 1 if args.smoke else int(evaluation["expected_samples"])
    sample_count_pass = len(samples) == expected_samples
    audit["expected_sample_count"] = expected_samples
    audit["sample_count_pass"] = sample_count_pass
    audit["pass"] = bool(audit["pass"] and sample_count_pass)

    sample_text = _sample_log_text(samples)
    sample_sha256 = hashlib.sha256(sample_text.encode("utf-8")).hexdigest()
    harness_results = {key: value for key, value in results.items() if key != "samples"}
    report = {
        "run_id": config["run"]["id"] if official else "smoke_h27",
        "hypothesis": config["run"]["hypothesis"],
        "classification": (
            config["classification"] if official else "runtime_smoke_not_an_experiment"
        ),
        "validation_eligible": official,
        "git_commit": _git_commit(),
        "protocol": config,
        "preflight": preflight,
        "model_qualification": model_qualification,
        "dataset_qualification": dataset_qualification,
        "harness_qualification": harness_qualification,
        "task_qualification": task_qualification,
        "target_qualification": {
            "source": config["target"]["source"],
            "key": config["target"]["key"],
            "config_accuracy_pct": float(config["target"]["accuracy_pct"]),
            "canonical_accuracy_pct": canonical_target,
            "pass": canonical_target == float(config["target"]["accuracy_pct"]),
        },
        "audit": audit,
        "sample_log": {
            "path": config["run"]["samples_output"] if official else None,
            "records": len(samples),
            "sha256": sample_sha256,
        },
        "harness_results": harness_results,
        "runtime": {
            **runtime_qualification,
            "wall_time_seconds": time.perf_counter() - started,
        },
    }
    summary = {
        "accuracy_pct": audit["accuracy_pct"],
        "correct_count": audit["correct_count"],
        "sample_count": audit["sample_count"],
        "paper_target_accuracy_pct": audit["paper_target_accuracy_pct"],
        "relative_error": audit["relative_error"],
        "pass": audit["pass"] if official else None,
        "validation_eligible": official,
        "sample_log_sha256": sample_sha256,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if official:
        output.parent.mkdir(parents=True, exist_ok=True)
        samples_output.parent.mkdir(parents=True, exist_ok=True)
        samples_output.write_text(sample_text, encoding="utf-8")
        output.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=handle_non_serializable,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0 if args.smoke or audit["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
