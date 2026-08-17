from pathlib import Path

import torch
from peft import PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from mlxsim.llama_fft import make_lora_config
from mlxsim.winogrande_training import (
    EncodedChoice,
    EncodedExample,
    audit_lora_training_model,
    collate_pairwise,
    continuation_choice_scores,
    encode_example,
    qualify_adapter_checkpoint,
    qualify_reloaded_adapter,
    shuffled_selection,
    tokenization_audit,
)


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [1, *(2 + ord(character) % 29 for character in text)]


def _tiny_model() -> LlamaForCausalLM:
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
    )


def _lora_config() -> dict:
    return {
        "rank": 2,
        "alpha": 4,
        "dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "bias": "none",
        "layers_to_transform": [0, 1],
        "layers_pattern": "layers",
    }


def test_pairwise_encoding_uses_filled_prefix_and_common_suffix() -> None:
    row = {
        "sentence": "Alice thanked Bob because _ had helped.",
        "option1": "Alice",
        "option2": "Bob",
        "answer": "2",
    }
    encoded = encode_example(row, _Tokenizer())
    assert encoded.target == 1
    assert encoded.choices[0].context_tokens == len(_Tokenizer().encode("Alice thanked Bob because Alice"))
    assert encoded.choices[1].token_ids == tuple(
        _Tokenizer().encode("Alice thanked Bob because Bob had helped.")
    )


def test_tokenization_and_shuffle_hashes_are_order_sensitive() -> None:
    first = EncodedExample(
        choices=(EncodedChoice((1, 2, 3), 2), EncodedChoice((1, 4, 3), 2)), target=0
    )
    second = EncodedExample(
        choices=(EncodedChoice((1, 5, 6), 2), EncodedChoice((1, 7, 6), 2)), target=1
    )
    assert tokenization_audit([first, second])["sha256"] != tokenization_audit(
        [second, first]
    )["sha256"]
    selection = shuffled_selection(10, 8, 42)
    assert len(selection["selected_indices"]) == 8
    assert len(selection["dropped_indices"]) == 2
    assert set(selection["selected_indices"] + selection["dropped_indices"]) == set(range(10))


def test_pairwise_collation_and_scores_are_differentiable() -> None:
    examples = [
        EncodedExample(
            choices=(EncodedChoice((1, 2, 3), 2), EncodedChoice((1, 4, 3, 2), 2)),
            target=0,
        ),
        EncodedExample(
            choices=(EncodedChoice((1, 3, 2), 1), EncodedChoice((1, 2, 4), 1)),
            target=1,
        ),
    ]
    batch = collate_pairwise(examples, pad_token_id=0)
    assert batch["input_ids"].shape == (4, 4)
    logits = torch.randn(4, 4, 8, requires_grad=True)
    scores = continuation_choice_scores(
        logits, batch["input_ids"], batch["continuation_mask"]
    )
    assert scores.shape == (2, 2)
    loss = torch.nn.functional.cross_entropy(scores, batch["targets"])
    loss.backward()
    assert logits.grad is not None


def test_lora_model_and_saved_checkpoint_are_fully_audited(tmp_path: Path) -> None:
    model = get_peft_model(_tiny_model(), make_lora_config(_lora_config()))
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    trainable_tensors = sum(parameter.requires_grad for parameter in model.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    report = audit_lora_training_model(
        model,
        expected_layers=[0, 1],
        expected_modules=_lora_config()["target_modules"],
        expected_trainable_parameters=trainable,
        expected_trainable_tensors=trainable_tensors,
        expected_total_parameters=total,
        maximum_trainable_fraction=1.0,
    )
    assert report["pass"] is True

    model.save_pretrained(tmp_path, safe_serialization=True)
    checkpoint = qualify_adapter_checkpoint(
        tmp_path,
        expected_layers=[0, 1],
        expected_modules=_lora_config()["target_modules"],
        expected_rank=2,
        expected_alpha=4,
        expected_dropout=0.05,
        expected_parameters=trainable,
        expected_tensors=trainable_tensors,
    )
    assert checkpoint["pass"] is True

    del model
    reloaded = PeftModel.from_pretrained(_tiny_model(), tmp_path)
    reload_report = qualify_reloaded_adapter(
        reloaded,
        tmp_path,
        expected_layers=[0, 1],
        expected_modules=_lora_config()["target_modules"],
        expected_rank=2,
        expected_alpha=4,
        expected_dropout=0.05,
        expected_parameters=trainable,
        expected_tensors=trainable_tensors,
    )
    assert reload_report["pass"] is True
