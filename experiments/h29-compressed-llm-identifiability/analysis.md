# H29 result: the public compressed-Llama recipe is underidentified

The source-bound audit supports H29: all ten registered necessary fields are
undisclosed, with at least one omission in each of model identity, layer plan,
FFT graph, hierarchical BSMM, and training/evaluation. The executable
ambiguity witnesses also pass. This is evidence about public identifiability,
not a model-quality result.

| Audit quantity | Result |
|---|---:|
| Necessary fields audited | 10 |
| Undisclosed fields | 10 |
| Missing domains / required domains | 5 / 5 |
| Admissible subsets of at least 20/32 layers | 462,411,533 |
| Minimum `L` assignments for 20 layers | `10^20` |
| Earlier restored positions changed by final-token perturbation | 29 / 31 |
| Maximum earlier absolute change | 0.2464525754 |
| Two-interpretation maximum difference | 1.7762786010 |
| Literal-prefix maximum imaginary magnitude | 1.7753490182 |

The official report is
`artifacts/results/compressed-llm-identifiability-run033.json` (13,268 bytes,
SHA-256 `e89f742561750352b9b5b3f34512bb5e92a6237df657b83d43ce767ecf509bbe`).

## Source qualification

The complete 85,855-byte supplied manuscript, the 76,206-byte Fig. 7 raster,
and all three independently hashed manuscript spans pass their byte and
SHA-256 checks. The spans bind the judgments to Sections III-A/B and VII-A:
the semantic FFT description, hierarchical BSMM description, and compressed
LLM evaluation description. A source change therefore invalidates rather than
silently updates this audit.

The paper does specify the high-level algorithm: chunk Q/K/V along the token
dimension, keep `sL` frequency coefficients, apply an `sL`-point inverse FFT,
symmetrically decompress, cache completed compressed chunks during decode,
factor each hidden-dimension tile, modify more than 60% of layers, and refine
compressed layers with LoRA. Those statements are retained as constraints.

They do not identify the exact checkpoint revision; modified layers; spectrum
input, aggregation, fixed threshold, or per-layer `L`; real/complex retention
and normalization; RoPE, mask, residual, and decompression placement; causal
teacher-forcing rule; dense-to-BSMM initialization; or the LoRA/task-evaluation
recipe.

## Numerical ambiguity and causality

For the registered real 32-element FP64 signal at `s=0.75`, the repository's
real-preserving Fourier resampler and the literal “complex FFT, retain first 24,
24-point iFFT” reading differ by as much as 1.77628. The literal result has an
imaginary component up to 1.77535, so an unstated conjugate or real-projection
rule is necessary.

Under symmetric compression/decompression, perturbing only input position 31
changes 29 of the 31 earlier restored positions, by as much as 0.24645. This
does not prevent valid end-of-prompt inference, where the whole prompt is past
context. It does prove that one shared full-chunk teacher-forced pass cannot
provide causal likelihoods at every earlier token position—the failure exposed
by H20.

## Consequence and scope

H29 is supported as a source-exposed identifiability audit and remains
validation-ineligible for Fig. 15/16 accuracy or perplexity. It does not claim
that the authors' internal implementation is incomplete or noncausal.

It does establish that selecting another layer subset, `L`, conjugate rule,
placement, factor initializer, or LoRA schedule from known plot residuals would
be a new inferred fit, not recovery of the paper's implementation. H20 cannot
be upgraded into a full compressed-model reproduction without new author
artifacts. Native work should move to independently identifiable public
baselines while this source gap remains.
