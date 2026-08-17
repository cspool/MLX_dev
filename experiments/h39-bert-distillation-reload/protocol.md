# H39 protocol: full-validation reload audit of H38 checkpoints

## Hypothesis

All five H38 `model.safetensors` artifacts can be reconstructed independently
from their frozen bytes and reproduce run043's full SQuAD F1 and exact-match
values within 0.02 percentage point.

H39 is a checkpoint-persistence integrity audit. It performs no training and
does not change H38's rejected hypothesis or validation-ineligible evidence
classification.

## Frozen sources

Bind the immutable 21,758-byte run043 result (`599e43b2...6f76`), its exact
H38 config and source script, the five saved checkpoint byte counts/SHA-256
values recorded by run043, the BERT config/tokenizer, and the complete SQuAD
1.1 validation file. Require run043 `audit_integrity=true`, status `rejected`,
source commit `8124867...`, and settings `[1,3,6,9,12]`.

## Reconstruction and evaluation

For each k independently:

1. instantiate a fresh BERT question-answering model from the frozen config;
2. reconstruct the exact H15/H38 compressed-attention topology for the last k
   layers (`s=0.5`, `L=32`, `B=32`, Q/K/V only);
3. normalize exactly 50 legacy LayerNorm `.gamma/.beta` suffixes to
   `.weight/.bias`, reject collisions, and require strict state-dict loading;
4. require `3k` structured projections and 31.25% replaced-weight density;
5. evaluate all 10,570 SQuAD questions using the same 384/128 tokenization,
   BF16/TF32 prediction path, answer post-processing, and official scorer; and
6. compare each F1/EM value to run043 with an absolute tolerance of 0.02 point.

No teacher forward, optimizer, loss, training data, augmentation, target-based
selection, or checkpoint modification is permitted.

## Decision and stopping rule

H39 is supported only if all ten reloaded metrics pass the 0.02-point gate and
all source/hash/topology/runtime checks pass. A checkpoint/source/schema failure
makes H39 inconclusive; a metric mismatch with intact sources rejects it.
Execute one five-setting audit and serialize one result. Do not choose a looser
tolerance or resave/retrain a failed checkpoint after observing run044.
