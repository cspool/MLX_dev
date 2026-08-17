import torch
from peft import get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from mlxsim.llama_fft import (
    CompressedLlamaAttention,
    audit_trainable_parameters,
    install_compressed_attention,
    make_lora_config,
)


def _tiny_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    config._attn_implementation = "sdpa"
    return LlamaForCausalLM(config)


def test_compressed_llama_attention_preserves_shape_and_gradients() -> None:
    model = _tiny_model()
    installed = install_compressed_attention(
        model, layer_indices=[1], chunk_length=8, compression_ratio=0.75
    )
    assert installed == [1]
    assert isinstance(model.model.layers[1].self_attn, CompressedLlamaAttention)
    input_ids = torch.randint(0, 128, (1, 16))
    logits = model(input_ids=input_ids, use_cache=False).logits
    assert logits.shape == (1, 16, 128)
    logits.float().square().mean().backward()
    assert model.model.layers[1].self_attn.q_proj.weight.grad is not None


def test_peft_targets_only_registered_layer_and_lora_parameters() -> None:
    model = _tiny_model()
    install_compressed_attention(model, layer_indices=[1], chunk_length=8, compression_ratio=0.75)
    lora = {
        "rank": 8,
        "alpha": 16,
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
        "layers_to_transform": [1],
        "layers_pattern": "layers",
    }
    model = get_peft_model(model, make_lora_config(lora))
    report = audit_trainable_parameters(model, expected_layers=[1], maximum_fraction=0.5)
    assert report["pass"] is True
    assert report["trainable_parameters"] > 0
    assert report["unexpected_non_lora_parameters"] == []
    assert report["outside_expected_layers"] == []


def test_compressed_attention_rejects_cache() -> None:
    model = _tiny_model()
    install_compressed_attention(model, layer_indices=[1], chunk_length=8, compression_ratio=0.75)
    input_ids = torch.randint(0, 128, (1, 16))
    try:
        model(input_ids=input_ids, use_cache=True)
    except RuntimeError as error:
        assert "does not support KV cache" in str(error)
    else:
        raise AssertionError("compressed attention unexpectedly accepted KV cache")
