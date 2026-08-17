# BERT structured-layer distillation basis for H38

Search cutoff: 2026-08-17. This note records only the primary sources used to
choose H38 before any H38 model forward or target-facing output.

## Patient Knowledge Distillation (PKD)

Sun et al., *Patient Knowledge Distillation for BERT Model Compression*,
EMNLP-IJCNLP 2019, DOI
[`10.18653/v1/D19-1441`](https://aclanthology.org/D19-1441/).

PKD reports that matching multiple intermediate teacher layers provides richer
transfer than using only the teacher's final output. It studies both last-layer
and skip-layer mappings. H38 does not remove BERT layers, so it uses the simpler
same-index mapping at every MLX-modified layer.

## TinyBERT

Jiao et al., *TinyBERT: Distilling BERT for Natural Language Understanding*,
Findings of EMNLP 2020, DOI
[`10.18653/v1/2020.findings-emnlp.372`](https://aclanthology.org/2020.findings-emnlp.372/).

TinyBERT uses task-specific Transformer distillation in addition to ordinary
task supervision, transferring both prediction and intermediate Transformer
knowledge. This supports a second, uniformly applied task-specific stage after
H15's hard-label fine-tuning. H38 does not use TinyBERT's data augmentation or
pretend to reproduce its smaller-layer architecture.

## Empirical objective comparison

Wang et al., *How to Distill your BERT: An Empirical Study on the Impact of
Weight Initialisation and Distillation Objectives*, ACL 2023, DOI
[`10.18653/v1/2023.acl-short.157`](https://aclanthology.org/2023.acl-short.157/).

The task-specific comparison finds intermediate transfer useful and reports
attention transfer as robust across initialization choices. Direct attention
matrix matching is not well-defined for H15 because its student attention is
computed at compressed sequence length whereas the teacher remains at full
length. H38 therefore matches the same-length post-layer hidden states, records
this deviation explicitly, and does not invent a target-dependent resampling
rule for attention matrices.

## Frozen H38 interpretation

H15's error and training loss grow monotonically with the number of replaced
layers, while its per-projection fit MSE stays nearly flat. The independently
motivated test is therefore one uniform task-specific distillation stage for
all five H15 checkpoints, combining:

1. the normal SQuAD start/end hard-label loss;
2. temperature-scaled start/end distribution KL from the frozen dense H11
   teacher; and
3. valid-token, unit-normalized hidden-state MSE at every modified layer.

This is a literature-grounded inferred reconstruction, not an author-recipe
claim. It cannot change the structured operator, `s`, `B`, `L`, layer sets,
checkpoint selection, or loss weights after observing H38 outcomes.
