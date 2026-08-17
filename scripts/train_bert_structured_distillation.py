#!/usr/bin/env python3
"""Run H38 patient distillation for the structured BERT sweep."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
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
    set_seed,
)

from mlxsim.bert_structured import CompressedBertSelfAttention, structured_parameter_summary
from mlxsim.distillation import qa_distillation_losses
from mlxsim.quality import squad_metrics

try:
    from scripts.train_bert_squad import (
        _load_squad,
        _postprocess_answers,
        _prepare_train_features,
        _prepare_validation_features,
    )
except ModuleNotFoundError:  # Direct script execution puts scripts/ on sys.path.
    from train_bert_squad import (  # type: ignore[no-redef]
        _load_squad,
        _postprocess_answers,
        _prepare_train_features,
        _prepare_validation_features,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/training/bert_structured_distillation_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--k", type=int, action="append", dest="selected_k")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--smoke", action="store_true")
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


def qualify_file(specification: dict[str, Any]) -> dict[str, Any]:
    path = resolve(specification["path"])
    is_file = path.is_file()
    size = path.stat().st_size if is_file else None
    digest = sha256_file(path) if is_file else None
    checks = {
        "is_file": is_file,
        "bytes": size == int(specification["bytes"]),
        "sha256": digest == specification["sha256"],
    }
    return {
        "path": str(path),
        "actual_bytes": size,
        "expected_bytes": int(specification["bytes"]),
        "actual_sha256": digest,
        "expected_sha256": specification["sha256"],
        "checks": checks,
        "pass": all(checks.values()),
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


def runtime_versions() -> dict[str, str | None]:
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "safetensors": safetensors.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def preflight(
    config: dict[str, Any], *, require_outputs_absent: bool, require_clean: bool
) -> dict[str, Any]:
    h15_result_specification = config["parent"]["result"]
    h15_result_file = qualify_file(h15_result_specification)
    h15_result = (
        json.loads(Path(h15_result_file["path"]).read_text(encoding="utf-8"))
        if h15_result_file["pass"]
        else {}
    )
    h11_result_specification = config["teacher"]["result"]
    h11_result_file = qualify_file(h11_result_specification)
    h11_result = (
        json.loads(Path(h11_result_file["path"]).read_text(encoding="utf-8"))
        if h11_result_file["pass"]
        else {}
    )
    file_reports = {
        "literature": qualify_file(config["literature_basis"]),
        "parent_result": h15_result_file,
        "parent_config": qualify_file(config["parent"]["config"]),
        "teacher_weights": qualify_file(config["teacher"]["weights"]),
        "teacher_result": h11_result_file,
        "train_dataset": qualify_file(config["dataset"]["train"]),
        "validation_dataset": qualify_file(config["dataset"]["validation"]),
    }
    checkpoint_reports = {
        int(k): qualify_file(specification)
        for k, specification in config["students"]["checkpoints"].items()
    }
    h15_by_k = {int(item["modified_last_k_layers"]): item for item in h15_result.get("results", [])}
    checkpoint_result_checks = {
        int(k): h15_by_k.get(int(k), {}).get("checkpoint_sha256") == specification["sha256"]
        for k, specification in config["students"]["checkpoints"].items()
    }
    output = resolve(config["outputs"]["report"])
    checkpoint_root = resolve(config["outputs"]["checkpoint_root"])
    outputs_absent = not output.exists() and not checkpoint_root.exists()
    weights = config["distillation"]["weights"]
    expected_k = [1, 3, 6, 9, 12]
    checks = {
        "all_source_files": all(item["pass"] for item in file_reports.values()),
        "all_student_checkpoints": all(item["pass"] for item in checkpoint_reports.values()),
        "checkpoint_hashes_match_parent_result": all(checkpoint_result_checks.values()),
        "parent_run": h15_result.get("run_id") == h15_result_specification["run_id"],
        "parent_hypothesis": h15_result.get("hypothesis") == h15_result_specification["hypothesis"],
        "parent_commit": h15_result.get("git_commit") == h15_result_specification["git_commit"],
        "parent_rejected": h15_result.get("summary", {}).get("all_points_pass") is False,
        "parent_max_error": h15_result.get("summary", {}).get("max_relative_error")
        == h15_result_specification["maximum_relative_error"],
        "teacher_experiment": h11_result.get("experiment_id")
        == h11_result_specification["experiment_id"],
        "teacher_gate": h11_result.get("passes_10pct_gate")
        is h11_result_specification["passes_10pct_gate"],
        "five_k_values": config["structured"]["modified_last_k_layers"] == expected_k,
        "checkpoint_k_values": sorted(checkpoint_reports) == expected_k,
        "parent_metric_k_values": sorted(config["parent_metrics_pct"]) == expected_k,
        "target_k_values": config["paper_targets"]["modified_last_k_layers"] == expected_k,
        "loss_weights_sum_to_one": abs(sum(float(value) for value in weights.values()) - 1.0)
        <= 1e-12,
        "loss_weights_frozen": weights == {"hard_qa": 0.5, "output_kl": 0.25, "hidden_mse": 0.25},
        "temperature": float(config["distillation"]["temperature"]) == 2.0,
        "runtime_versions": runtime_versions() == config["runtime"],
        "cuda_available": torch.cuda.is_available(),
        "protocol": resolve(config["protocol"]).is_file(),
        "tracked_worktree_clean": tracked_worktree_clean() if require_clean else True,
        "output_state": outputs_absent if require_outputs_absent else True,
    }
    return {
        "files": file_reports,
        "student_checkpoints": checkpoint_reports,
        "checkpoint_result_checks": checkpoint_result_checks,
        "runtime": runtime_versions(),
        "output": str(output),
        "checkpoint_root": str(checkpoint_root),
        "actual_outputs_absent": outputs_absent,
        "checks": checks,
        "pass": all(checks.values()),
    }


def inject_structured_topology(model: torch.nn.Module, config: dict[str, Any], k: int) -> None:
    structured = config["structured"]
    layers = model.bert.encoder.layer
    for layer_index in range(len(layers) - k, len(layers)):
        original = layers[layer_index].attention.self
        layers[layer_index].attention.self = CompressedBertSelfAttention(
            original,
            compression_ratio=float(structured["compression_ratio"]),
            chunk_length=int(structured["chunk_length"]),
            block_size=int(structured["butterfly_block_size"]),
        )


def canonicalize_bert_layernorm_keys(
    state: dict[str, torch.Tensor], aliases: dict[str, str]
) -> tuple[dict[str, torch.Tensor], int]:
    """Normalize legacy LayerNorm suffixes without changing tensor values."""

    canonical: dict[str, torch.Tensor] = {}
    renamed = 0
    for key, value in state.items():
        target_key = key
        for old_suffix, new_suffix in aliases.items():
            suffix = f".{old_suffix}"
            if key.endswith(suffix):
                target_key = f"{key[: -len(suffix)]}.{new_suffix}"
                renamed += 1
                break
        if target_key in canonical:
            raise ValueError(f"LayerNorm alias collision: {key} -> {target_key}")
        canonical[target_key] = value
    return canonical, renamed


def restore_student(
    config: dict[str, Any], k: int, *, device: torch.device
) -> tuple[torch.nn.Module, dict[str, int | bool]]:
    model_config = AutoConfig.from_pretrained(resolve(config["teacher"]["path"]))
    model = AutoModelForQuestionAnswering.from_config(model_config)
    inject_structured_topology(model, config, k)
    checkpoint = resolve(config["students"]["checkpoints"][k]["path"])
    state = load_file(checkpoint, device="cpu")
    compatibility = config["students"]["serialization_compatibility"]
    state, renamed = canonicalize_bert_layernorm_keys(state, compatibility["layernorm_key_aliases"])
    expected_renamed = int(compatibility["expected_renamed_key_count"])
    if renamed != expected_renamed:
        raise RuntimeError(f"expected {expected_renamed} LayerNorm aliases, restored {renamed}")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict student restore failed: {incompatible}")
    del state
    return model.to(device), {
        "strict": True,
        "layernorm_alias_count": renamed,
        "expected_layernorm_alias_count": expected_renamed,
    }


class QADistillationTrainer(Trainer):
    """Trainer with the frozen H38 hard/output/hidden objective."""

    def __init__(
        self,
        *args: Any,
        teacher: torch.nn.Module,
        modified_layer_indices: list[int],
        distillation: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.modified_layer_indices = modified_layer_indices
        self.distillation = distillation
        self.component_sums: dict[str, torch.Tensor] = {}
        self.component_count = 0

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        student_outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        teacher_inputs = {
            key: inputs[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in inputs
        }
        with torch.inference_mode():
            teacher_outputs = self.teacher(
                **teacher_inputs, output_hidden_states=True, return_dict=True
            )
        weights = self.distillation["weights"]
        losses = qa_distillation_losses(
            hard_loss=student_outputs.loss,
            student_start_logits=student_outputs.start_logits,
            student_end_logits=student_outputs.end_logits,
            teacher_start_logits=teacher_outputs.start_logits,
            teacher_end_logits=teacher_outputs.end_logits,
            student_hidden_states=student_outputs.hidden_states,
            teacher_hidden_states=teacher_outputs.hidden_states,
            attention_mask=inputs["attention_mask"],
            encoder_layer_indices=self.modified_layer_indices,
            temperature=float(self.distillation["temperature"]),
            hard_weight=float(weights["hard_qa"]),
            output_weight=float(weights["output_kl"]),
            hidden_weight=float(weights["hidden_mse"]),
        )
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise RuntimeError("non-finite H38 distillation loss")
        for name, value in losses.items():
            detached = value.detach().float()
            self.component_sums[name] = (
                self.component_sums.get(name, detached.new_zeros(())) + detached
            )
        self.component_count += 1
        total = losses["total_loss"]
        return (total, student_outputs) if return_outputs else total

    def component_means(self) -> dict[str, float | int | bool]:
        return {
            "microbatch_count": self.component_count,
            "all_finite": self.component_count > 0
            and all(bool(torch.isfinite(value)) for value in self.component_sums.values()),
            **{
                name: float(value.cpu() / self.component_count)
                for name, value in self.component_sums.items()
            },
        }


def evaluate_qa(
    trainer: Trainer,
    validation_examples: Dataset,
    validation_features: Dataset,
    model_validation_features: Dataset,
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    prediction_output = trainer.predict(model_validation_features)
    start_logits, end_logits = prediction_output.predictions[:2]
    evaluation = config["evaluation"]
    predictions = _postprocess_answers(
        validation_examples,
        validation_features,
        start_logits,
        end_logits,
        n_best_size=int(evaluation["n_best_size"]),
        max_answer_length=int(evaluation["max_answer_length"]),
    )
    references = {example["id"]: example["answers"]["text"] for example in validation_examples}
    return squad_metrics(predictions, references), prediction_output.metrics


def metric_errors(actual: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    return {name: abs(float(actual[name]) - value) / value for name, value in target.items()}


def build_training_arguments(
    config: dict[str, Any], *, output_dir: Path, seed: int, smoke: bool
) -> TrainingArguments:
    optimization = config["optimization"]
    evaluation = config["evaluation"]
    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        per_device_train_batch_size=int(optimization["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(optimization["gradient_accumulation_steps"]),
        num_train_epochs=float(optimization["continuation_epochs"]),
        max_steps=2 if smoke else -1,
        learning_rate=float(optimization["learning_rate"]),
        lr_scheduler_type=optimization["scheduler"],
        warmup_steps=float(optimization["warmup_ratio"]),
        optim=optimization["optimizer"],
        weight_decay=float(optimization["weight_decay"]),
        max_grad_norm=float(optimization["max_gradient_norm"]),
        bf16=optimization["dtype"] == "bfloat16",
        tf32=bool(optimization["tf32"]),
        seed=seed,
        data_seed=seed,
        per_device_eval_batch_size=int(evaluation["per_device_batch_size"]),
        dataloader_num_workers=int(optimization["dataloader_workers"]),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1 if smoke else 100,
        report_to="none",
    )


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    preflight_report = preflight(
        config,
        require_outputs_absent=not args.smoke,
        require_clean=not args.smoke and not args.preflight_only,
    )
    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return 0 if preflight_report["pass"] else 2
    if not preflight_report["pass"]:
        print(json.dumps(preflight_report, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    all_k = [int(value) for value in config["structured"]["modified_last_k_layers"]]
    selected_k = args.selected_k or all_k
    if any(k not in all_k for k in selected_k):
        raise SystemExit(f"--k must be selected from {all_k}")
    if args.smoke:
        selected_k = selected_k[:1]
    official = not args.smoke and selected_k == all_k
    report_path = args.report.resolve() if args.report else resolve(config["outputs"]["report"])
    if official and report_path != resolve(config["outputs"]["report"]):
        raise SystemExit("formal H38 must use the frozen report path")
    if official and report_path.exists():
        raise SystemExit(f"refusing to overwrite official report: {report_path}")

    train_examples_list = _load_squad(resolve(config["dataset"]["train"]["path"]))
    validation_examples_list = _load_squad(resolve(config["dataset"]["validation"]["path"]))
    if args.smoke:
        train_examples_list = train_examples_list[:128]
        validation_examples_list = validation_examples_list[:64]
    train_examples = Dataset.from_list(train_examples_list)
    validation_examples = Dataset.from_list(validation_examples_list)
    tokenizer = AutoTokenizer.from_pretrained(resolve(config["teacher"]["path"]), use_fast=True)
    preprocessing = config["preprocessing"]
    train_features = train_examples.map(
        _prepare_train_features,
        batched=True,
        remove_columns=train_examples.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": int(preprocessing["max_sequence_length"]),
            "stride": int(preprocessing["document_stride"]),
        },
        desc="tokenizing H38 SQuAD train",
    )
    validation_features = validation_examples.map(
        _prepare_validation_features,
        batched=True,
        remove_columns=validation_examples.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": int(preprocessing["max_sequence_length"]),
            "stride": int(preprocessing["document_stride"]),
        },
        desc="tokenizing H38 SQuAD validation",
    )
    model_validation_features = validation_features.remove_columns(["example_id", "offset_mapping"])
    collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=int(preprocessing["pad_to_multiple_of"])
    )

    device = torch.device(config["optimization"]["device"])
    teacher = AutoModelForQuestionAnswering.from_pretrained(resolve(config["teacher"]["path"])).to(
        device
    )
    teacher.eval()
    teacher.requires_grad_(False)
    run_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    target_index = {k: index for index, k in enumerate(all_k)}
    checkpoint_root = resolve(config["outputs"]["checkpoint_root"])

    for k in selected_k:
        setting_started = time.perf_counter()
        seed = int(config["optimization"]["seed_base"]) + k
        set_seed(seed)
        student, restore_report = restore_student(config, k, device=device)
        modified_indices = list(range(12 - k, 12))
        output_dir = (
            Path("/tmp/mlx-h38-smoke") / f"k{k}" if args.smoke else checkpoint_root / f"k{k}"
        )
        if not args.smoke and output_dir.exists() and any(output_dir.iterdir()):
            raise SystemExit(f"refusing to overwrite checkpoint: {output_dir}")
        trainer = QADistillationTrainer(
            model=student,
            args=build_training_arguments(
                config, output_dir=output_dir, seed=seed, smoke=args.smoke
            ),
            train_dataset=train_features,
            data_collator=collator,
            processing_class=tokenizer,
            teacher=teacher,
            modified_layer_indices=modified_indices,
            distillation=config["distillation"],
        )
        initial_metrics, initial_prediction_metrics = evaluate_qa(
            trainer,
            validation_examples,
            validation_features,
            model_validation_features,
            config,
        )
        parent_metrics = config["parent_metrics_pct"][k]
        initial_replay_errors = (
            {}
            if args.smoke
            else {
                name: abs(float(initial_metrics[name]) - float(value))
                for name, value in parent_metrics.items()
            }
        )
        initial_replay_pass = (
            None
            if args.smoke
            else max(initial_replay_errors.values())
            <= float(config["evaluation"]["initial_replay_absolute_tolerance_pct"])
        )
        if initial_replay_pass is False:
            raise RuntimeError(f"k={k} does not replay H15 before distillation")
        set_seed(seed)
        train_result = trainer.train()
        final_metrics, final_prediction_metrics = evaluate_qa(
            trainer,
            validation_examples,
            validation_features,
            model_validation_features,
            config,
        )
        index = target_index[k]
        targets = {
            "f1": float(config["paper_targets"]["f1_pct"][index]),
            "exact_match": float(config["paper_targets"]["exact_match_pct"][index]),
        }
        relative_errors = metric_errors(final_metrics, targets)
        checkpoint_sha256 = None
        if not args.smoke:
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(output_dir)
            checkpoint_sha256 = sha256_file(output_dir / "model.safetensors")
        result = {
            "modified_last_k_layers": k,
            "modified_layer_indices": modified_indices,
            "seed": seed,
            "strict_checkpoint_restore": restore_report["strict"],
            "layernorm_alias_count": restore_report["layernorm_alias_count"],
            "expected_layernorm_alias_count": restore_report["expected_layernorm_alias_count"],
            "initial_metrics_pct": initial_metrics,
            "parent_metrics_pct": parent_metrics,
            "initial_replay_absolute_errors_pct": initial_replay_errors,
            "initial_replay_pass": initial_replay_pass,
            "final_metrics_pct": final_metrics,
            "paper_targets_pct": targets,
            "relative_errors": relative_errors,
            "maximum_relative_error": max(relative_errors.values()),
            "passes_10pct_gate": None
            if args.smoke
            else max(relative_errors.values())
            <= float(config["paper_targets"]["maximum_relative_error"]),
            "target_metrics_excluded": args.smoke,
            "metric_changes_from_parent_pct": {
                name: float(final_metrics[name]) - float(parent_metrics[name]) for name in targets
            },
            "distillation_component_means": trainer.component_means(),
            "parameter_summary": structured_parameter_summary(student),
            "train_metrics": train_result.metrics,
            "initial_prediction_metrics": initial_prediction_metrics,
            "final_prediction_metrics": final_prediction_metrics,
            "checkpoint_sha256": checkpoint_sha256,
            "wall_time_seconds": time.perf_counter() - setting_started,
        }
        results.append(result)
        if not args.smoke:
            progress_path = report_path.with_suffix(".partial.json")
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(
                json.dumps({"completed_settings": results}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        del trainer, student, train_result
        gc.collect()
        torch.cuda.empty_cache()

    all_errors = [error for result in results for error in result["relative_errors"].values()]
    integrity_checks = {
        "preflight": preflight_report["pass"],
        "all_settings": [result["modified_last_k_layers"] for result in results] == selected_k,
        "all_strict_restores": all(result["strict_checkpoint_restore"] for result in results),
        "layernorm_alias_counts": all(
            result["layernorm_alias_count"] == result["expected_layernorm_alias_count"] == 50
            for result in results
        ),
        "all_initial_replays": args.smoke
        or all(result["initial_replay_pass"] is True for result in results),
        "all_losses_finite": all(
            result["distillation_component_means"]["all_finite"] for result in results
        ),
        "projection_counts": all(
            result["parameter_summary"]["structured_projection_count"]
            == 3 * result["modified_last_k_layers"]
            for result in results
        ),
        "density": all(
            abs(result["parameter_summary"]["weight_density"] - 0.3125) <= 1e-12
            for result in results
        ),
        "source_commit_recorded": git_commit() is not None,
    }
    audit_integrity = all(integrity_checks.values())
    all_points_pass = None if args.smoke else all(result["passes_10pct_gate"] for result in results)
    if args.smoke:
        hypothesis_status = "not_evaluated"
    elif not audit_integrity:
        hypothesis_status = "inconclusive"
    elif all_points_pass:
        hypothesis_status = "supported"
    else:
        hypothesis_status = "rejected"
    report = {
        "schema_version": 1,
        "run_id": config["run_id"] if official else None,
        "hypothesis": config["experiment_id"],
        "classification": config["classification"]
        if official
        else "smoke_or_partial_not_an_experiment",
        "validation_eligible": bool(config["validation_eligible"]) if official else False,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "preflight": preflight_report,
        "results": results,
        "summary": {
            "point_count": len(all_errors),
            "mape": None if args.smoke else sum(all_errors) / len(all_errors),
            "max_relative_error": None if args.smoke else max(all_errors),
            "parent_max_relative_error": float(
                config["parent"]["result"]["maximum_relative_error"]
            ),
            "max_relative_error_delta": None
            if args.smoke
            else max(all_errors) - float(config["parent"]["result"]["maximum_relative_error"]),
            "all_points_pass": all_points_pass,
            "target_metrics_excluded": args.smoke,
        },
        "integrity_checks": integrity_checks,
        "audit_integrity": audit_integrity,
        "hypothesis_status": hypothesis_status,
        "runtime": runtime_versions(),
        "train_examples": len(train_examples),
        "train_features": len(train_features),
        "validation_examples": len(validation_examples),
        "validation_features": len(validation_features),
        "wall_time_seconds": time.perf_counter() - run_started,
        "smoke": args.smoke,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if official or args.report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(encoded + "\n", encoding="utf-8")
    if official:
        partial = report_path.with_suffix(".partial.json")
        if partial.exists():
            partial.unlink()
    return 0 if audit_integrity else 2


if __name__ == "__main__":
    raise SystemExit(main())
