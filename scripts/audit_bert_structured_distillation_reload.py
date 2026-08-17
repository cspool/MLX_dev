#!/usr/bin/env python3
"""Audit full-validation reload equivalence for H38 BERT checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import datasets
import safetensors
import torch
import transformers
import yaml
from datasets import Dataset
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlxsim.bert_structured import structured_parameter_summary
from scripts.train_bert_squad import (
    _load_squad,
    _prepare_validation_features,
)
from scripts.train_bert_structured_distillation import (
    canonicalize_bert_layernorm_keys,
    evaluate_qa,
    inject_structured_topology,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/bert_structured_distillation_reload_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def qualify_file(specification: dict[str, Any], *, hash_required: bool = True) -> dict[str, Any]:
    path = resolve(specification["path"])
    is_file = path.is_file()
    size = path.stat().st_size if is_file else None
    digest = sha256_file(path) if is_file and hash_required else None
    checks = {"is_file": is_file}
    if "bytes" in specification:
        checks["bytes"] = size == int(specification["bytes"])
    if "sha256" in specification:
        checks["sha256"] = digest == specification["sha256"]
    return {
        "path": str(path),
        "actual_bytes": size,
        "actual_sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def runtime_versions() -> dict[str, str | None]:
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "safetensors": safetensors.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def tracked_worktree_clean() -> bool:
    commands = (
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "diff", "--quiet"],
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "diff", "--cached", "--quiet"],
    )
    return all(
        subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode == 0
        for command in commands
    )


def preflight(
    config: dict[str, Any], *, require_output_absent: bool, require_clean: bool
) -> dict[str, Any]:
    result_file = qualify_file(config["h38"]["result"])
    h38_result = (
        json.loads(Path(result_file["path"]).read_text(encoding="utf-8"))
        if result_file["pass"]
        else {}
    )
    source_files = {
        "h38_result": result_file,
        "h38_config": qualify_file(config["h38"]["config"]),
        "h38_script": qualify_file(config["h38"]["script"]),
        "validation_dataset": qualify_file(config["validation_dataset"]),
        "model_config": qualify_file(config["model_assets"]["config"]),
        "tokenizer": qualify_file(config["model_assets"]["tokenizer"]),
        "tokenizer_config": qualify_file(config["model_assets"]["tokenizer_config"]),
    }
    checkpoints = {
        int(k): qualify_file(specification) for k, specification in config["checkpoints"].items()
    }
    h38_by_k = {int(item["modified_last_k_layers"]): item for item in h38_result.get("results", [])}
    result_bindings = {
        k: h38_by_k.get(k, {}).get("checkpoint_sha256") == config["checkpoints"][k]["sha256"]
        and h38_by_k.get(k, {}).get("final_metrics_pct") == config["expected_metrics_pct"][k]
        for k in checkpoints
    }
    output = resolve(config["run"]["output"])
    output_absent = not output.exists()
    expected_k = [1, 3, 6, 9, 12]
    h38_specification = config["h38"]["result"]
    checks = {
        "source_files": all(item["pass"] for item in source_files.values()),
        "checkpoints": all(item["pass"] for item in checkpoints.values()),
        "result_bindings": all(result_bindings.values()),
        "h38_run": h38_result.get("run_id") == h38_specification["run_id"],
        "h38_hypothesis": h38_result.get("hypothesis") == h38_specification["hypothesis"],
        "h38_commit": h38_result.get("git_commit") == h38_specification["git_commit"],
        "h38_status": h38_result.get("hypothesis_status") == h38_specification["hypothesis_status"],
        "h38_integrity": h38_result.get("audit_integrity") is True,
        "five_k_values": sorted(checkpoints) == expected_k
        and config["structured"]["modified_last_k_layers"] == expected_k,
        "metric_k_values": sorted(config["expected_metrics_pct"]) == expected_k,
        "runtime": runtime_versions() == config["runtime"],
        "cuda": torch.cuda.is_available(),
        "protocol": resolve(config["run"]["protocol"]).is_file(),
        "tracked_worktree_clean": tracked_worktree_clean() if require_clean else True,
        "output_state": output_absent if require_output_absent else True,
    }
    return {
        "source_files": source_files,
        "checkpoints": checkpoints,
        "result_bindings": result_bindings,
        "runtime": runtime_versions(),
        "output": str(output),
        "actual_output_absent": output_absent,
        "checks": checks,
        "pass": all(checks.values()),
    }


def metric_differences(actual: dict[str, float], expected: dict[str, float]) -> dict[str, float]:
    return {name: abs(float(actual[name]) - float(value)) for name, value in expected.items()}


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    preflight_report = preflight(
        config,
        require_output_absent=True,
        require_clean=not args.preflight_only,
    )
    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return 0 if preflight_report["pass"] else 2
    if not preflight_report["pass"]:
        print(json.dumps(preflight_report, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    validation_examples = Dataset.from_list(
        _load_squad(resolve(config["validation_dataset"]["path"]))
    )
    tokenizer = AutoTokenizer.from_pretrained(
        resolve(config["model_assets"]["root"]), use_fast=True
    )
    preprocessing = config["preprocessing"]
    validation_features = validation_examples.map(
        _prepare_validation_features,
        batched=True,
        remove_columns=validation_examples.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": int(preprocessing["max_sequence_length"]),
            "stride": int(preprocessing["document_stride"]),
        },
        desc="tokenizing H39 SQuAD validation",
    )
    model_validation_features = validation_features.remove_columns(["example_id", "offset_mapping"])
    collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=int(preprocessing["pad_to_multiple_of"])
    )
    evaluation = config["evaluation"]
    device = torch.device(evaluation["device"])
    audit_started = time.perf_counter()
    settings: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mlx-h39-reload-") as temporary_root:
        for k in config["structured"]["modified_last_k_layers"]:
            setting_started = time.perf_counter()
            model_config = AutoConfig.from_pretrained(resolve(config["model_assets"]["root"]))
            model = AutoModelForQuestionAnswering.from_config(model_config)
            topology_config = {"structured": config["structured"]}
            inject_structured_topology(model, topology_config, int(k))
            state = load_file(resolve(config["checkpoints"][k]["path"]), device="cpu")
            state, alias_count = canonicalize_bert_layernorm_keys(
                state, config["structured"]["layernorm_key_aliases"]
            )
            incompatible = model.load_state_dict(state, strict=True)
            del state
            strict = not incompatible.missing_keys and not incompatible.unexpected_keys
            model = model.to(device)
            training_arguments = TrainingArguments(
                output_dir=str(Path(temporary_root) / f"k{k}"),
                do_train=False,
                per_device_eval_batch_size=int(evaluation["per_device_batch_size"]),
                bf16=evaluation["dtype"] == "bfloat16",
                tf32=bool(evaluation["tf32"]),
                report_to="none",
            )
            trainer = Trainer(
                model=model,
                args=training_arguments,
                data_collator=collator,
                processing_class=tokenizer,
            )
            evaluation_config = {"evaluation": evaluation}
            metrics, prediction_metrics = evaluate_qa(
                trainer,
                validation_examples,
                validation_features,
                model_validation_features,
                evaluation_config,
            )
            expected = config["expected_metrics_pct"][k]
            differences = metric_differences(metrics, expected)
            parameter_summary = structured_parameter_summary(model)
            checks = {
                "strict": strict,
                "layernorm_alias_count": alias_count
                == int(config["structured"]["expected_layernorm_alias_count"]),
                "projection_count": parameter_summary["structured_projection_count"] == 3 * k,
                "density": abs(
                    parameter_summary["weight_density"]
                    - float(config["structured"]["expected_density"])
                )
                <= 1e-12,
                "metrics": max(differences.values())
                <= float(evaluation["absolute_metric_tolerance_pct"]),
            }
            settings.append(
                {
                    "modified_last_k_layers": k,
                    "metrics_pct": metrics,
                    "expected_metrics_pct": expected,
                    "absolute_differences_pct": differences,
                    "maximum_absolute_difference_pct": max(differences.values()),
                    "prediction_metrics": prediction_metrics,
                    "layernorm_alias_count": alias_count,
                    "parameter_summary": parameter_summary,
                    "checks": checks,
                    "pass": all(checks.values()),
                    "wall_time_seconds": time.perf_counter() - setting_started,
                }
            )
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()

    integrity_checks = {
        "preflight": preflight_report["pass"],
        "five_settings": [item["modified_last_k_layers"] for item in settings] == [1, 3, 6, 9, 12],
        "source_commit_recorded": git_commit() is not None,
        "all_structural_checks": all(
            all(value for name, value in item["checks"].items() if name != "metrics")
            for item in settings
        ),
    }
    audit_integrity = all(integrity_checks.values())
    all_metrics_pass = all(item["checks"]["metrics"] for item in settings)
    if not audit_integrity:
        hypothesis_status = "inconclusive"
    elif all_metrics_pass:
        hypothesis_status = "supported"
    else:
        hypothesis_status = "rejected"
    report = {
        "schema_version": 1,
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "preflight": preflight_report,
        "settings": settings,
        "summary": {
            "setting_count": len(settings),
            "metric_count": len(settings) * 2,
            "maximum_absolute_difference_pct": max(
                item["maximum_absolute_difference_pct"] for item in settings
            ),
            "all_metrics_pass": all_metrics_pass,
        },
        "integrity_checks": integrity_checks,
        "audit_integrity": audit_integrity,
        "hypothesis_status": hypothesis_status,
        "runtime": runtime_versions(),
        "validation_examples": len(validation_examples),
        "validation_features": len(validation_features),
        "wall_time_seconds": time.perf_counter() - audit_started,
    }
    output = resolve(config["run"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "audit_integrity": audit_integrity,
                "hypothesis_status": hypothesis_status,
                "summary": report["summary"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit_integrity else 2


if __name__ == "__main__":
    raise SystemExit(main())
