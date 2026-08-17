# H29 protocol: compressed-Llama source identifiability and causality

## Hypothesis

The hash-qualified supplied MLX manuscript is insufficient to identify a
unique executable compressed-Llama experiment: every necessary domain has at
least one undisclosed field, and the literal chunk transform admits
source-consistent interpretations with different values and no shared
all-position autoregressive causality.

H29 is a source-exposed identifiability audit. Its outcome cannot count as a
native model-quality validation and cannot select a new compressed recipe from
Fig. 15/16 residuals.

## Frozen sources

Bind the audit to the 85,855-byte supplied Markdown manuscript (SHA-256
`5785eb81...7745`) and the 76,206-byte Fig. 7 raster (SHA-256
`9265cdea...20d`). Independently hash the exact line spans for semantic FFT
(115-140), hierarchical BSMM (141-149), and LLM evaluation (350-353). Abort if
any byte or segment changes.

## Necessary-field audit

The config freezes ten fields required to reproduce a compressed Fig. 15(c,d)
model. They cover five independent domains:

1. exact base-model revision;
2. exact modified layers;
3. spectrum input/aggregation, fixed threshold, and every per-layer `L`;
4. real/complex coefficient retention plus RoPE, mask, residual, and symmetric
   decompression placement;
5. a causal training/likelihood rule;
6. dense-to-hierarchical-BSMM initialization; and
7. LoRA plus task-evaluation hyperparameters and split/objective semantics.

The omission judgments are manual semantic readings, frozen before runner
implementation and bound to the complete source hashes. H29 passes only if at
least one omitted field remains in each of model identity, layer plan, FFT
graph, BSMM, and training/evaluation.

## Executable ambiguity witnesses

Use one deterministic real 32-element signal and `s=0.75` in FP64.

- Apply the repository's real-preserving chunk Fourier resampler and symmetric
  decompressor. Perturb only input index 31 by +1. Earlier restored positions
  must change above `1e-12`, witnessing that one shared full-chunk forward pass
  is not causal for all teacher-forced token positions.
- Compare that real-preserving interpretation with the manuscript's literal
  wording: complex FFT, keep the first 24 coefficients, then a 24-point iFFT.
  The two shortened outputs must differ above `1e-12`; the literal prefix
  interpretation must retain a nonzero imaginary component above `1e-12`.
  The paper does not state which conjugate/real convention resolves this.
- Check the layer-plan ambiguity arithmetically. “More than 60%” of 32 layers
  allows 462,411,533 unconstrained subsets of size 20-32. Even restricting each
  of 20 layers to the ten power-of-two lengths at N=512 leaves at least
  `10^20` chunk assignments before threshold/input choices.

These are identifiability witnesses, not candidate recipes. Do not run any
Llama logits or compare either interpretation with a paper accuracy target.

## Gate and consequence

H29 is supported only when all source hashes, all five missing-domain gates,
all numerical ambiguity witnesses, and both combinatorial canaries pass. A
supported result establishes that H20 cannot be repaired into an author-faithful
compressed reproduction by choosing another hidden parameter from its residual.
It does not establish that MLX's internal implementation is incorrect.

The next native work should use an independently identifiable public baseline
(for example the remaining InternLM2/Ada-LEval original bars) unless authors
release the compressed graph, factors, and training/evaluation manifest.
