import torch
from transformers import BertConfig, BertForQuestionAnswering

from mlxsim.bert_structured import (
    CompressedBertSelfAttention,
    inject_structured_bert_layers,
    structured_parameter_summary,
)
from mlxsim.structured import chunked_fft_compress


def _tiny_bert() -> BertForQuestionAnswering:
    config = BertConfig(
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return BertForQuestionAnswering(config)


def test_injected_model_forward_backward_and_shapes() -> None:
    torch.manual_seed(3)
    model = _tiny_bert()
    reports = inject_structured_bert_layers(
        model,
        modified_last_k_layers=1,
        compression_ratio=0.5,
        chunk_length=16,
        block_size=8,
        fit_steps=2,
        fit_learning_rate=0.01,
        fit_seed_base=100,
    )
    assert len(reports) == 3
    assert isinstance(model.bert.encoder.layer[1].attention.self, CompressedBertSelfAttention)
    inputs = torch.randint(0, 101, (2, 23))
    mask = torch.tensor([[1] * 23, [1] * 15 + [0] * 8])
    output = model(
        input_ids=inputs,
        attention_mask=mask,
        start_positions=torch.tensor([2, 3]),
        end_positions=torch.tensor([4, 5]),
    )
    assert output.start_logits.shape == (2, 23)
    assert output.end_logits.shape == (2, 23)
    output.loss.backward()
    factors = model.bert.encoder.layer[1].attention.self.query.factors
    assert factors.grad is not None
    assert torch.isfinite(factors.grad).all()


def test_mask_marks_fully_padded_chunks_invalid() -> None:
    model = _tiny_bert()
    original = model.bert.encoder.layer[0].attention.self
    module = CompressedBertSelfAttention(
        original, compression_ratio=0.5, chunk_length=8, block_size=8
    )
    hidden = torch.randn(1, 16, 32)
    _, context = chunked_fft_compress(
        hidden, chunk_length=8, compression_ratio=0.5, dim=1
    )
    source_mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    source_mask[..., :8] = True
    compressed = module.compressed_attention_mask(source_mask, context)
    assert compressed.shape == (1, 1, 8, 8)
    assert compressed[..., :4].all()
    assert not compressed[..., 4:].any()


def test_parameter_summary_has_expected_density() -> None:
    model = _tiny_bert()
    inject_structured_bert_layers(
        model,
        modified_last_k_layers=2,
        compression_ratio=0.5,
        chunk_length=16,
        block_size=8,
        fit_steps=1,
        fit_learning_rate=0.01,
        fit_seed_base=200,
    )
    summary = structured_parameter_summary(model)
    assert summary["structured_projection_count"] == 6
    assert summary["weight_density"] == 0.75
