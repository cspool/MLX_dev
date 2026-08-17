# H12 protocol: Fig. 15/16 quality-target digitization

## Question

Can the supplied Fig. 15/16 rasters, their printed annotations, and the accompanying prose recover every plotted model-quality target with an explicit uncertainty bound?

## Status before execution

Exploratory target-recovery audit. The plots have already been visually inspected, so this is not held-out evidence for the MLX algorithm. The derived values become immutable acceptance targets for later training/evaluation runs.

## Frozen inputs

- Fig. 15 image SHA-256: `a7b9aa55d963b212f5bff3ffeb580e62d636b4af0c342cac859509691648f18b`.
- Fig. 16 image SHA-256: `0111f75cf7d8c3b1f38f83be07411a5f88e034161036f0b9bba65f8eff38b92c`.
- All bar endpoints, axis calibrations, and printed annotations are frozen in `artifacts/targets/quality_digitization_pixels.yaml` before the derivation script is implemented.
- Fig. 15 accuracy panels share 51 raster pixels per percentage point. Values without a printed absolute label are derived by pixel displacement from a printed value in the same series.
- Fig. 15(d) uses `(y=182, perplexity=0)` and `(y=45, perplexity=8)`.
- Fig. 16 accuracy panels use 48 raster pixels per percentage point. Printed Llama2 and InternLM2 values remain reported values rather than digitized estimates.

The original-perplexity endpoints in Fig. 15(d) and the three annotated Llama2 endpoints in Fig. 16 are retained only as independent raster-to-label cross-checks. Their printed annotations remain the acceptance targets.

## Plot semantics

- Fig. 15(a) bars are accuracy margins; printed original top-1/top-5 scores provide absolute anchors.
- Fig. 15(b) bars are margins relative to the all-12-layer compressed model. The prose fixes its absolute F1/EM at `87.7-1.3=86.4` and `79.1-1.75=77.35`.
- Fig. 15(c) has two bars per task: original plus `s=0.75` for WinoGrande, and original plus `s=0.5` for each Ada-LEval length. The three-entry legend describes the union of series, not three bars in every group.
- Fig. 15(d) has original, `s=0.75`, and `s=0.5` for every perplexity task.
- Fig. 16 uses block sizes `[16, 32, 64]`.

## Acceptance gate

- The derivation must be executable from the frozen coordinate manifest.
- Reconstructed printed cross-checks must agree within 0.15 accuracy percentage point or 0.12 perplexity.
- Every visible quality bar must be represented once, with `reported` versus `digitized` provenance retained.
- No result from this audit counts as model execution, training, or validation of structured layers.

## Failure policy

Ambiguous series identity or a failed printed-value cross-check remains target-only/unknown. Do not choose a value based on later model residuals.
