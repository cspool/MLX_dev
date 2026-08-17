# H12 result: complete Fig. 15/16 quality-target recovery

## Outcome

Supported as an exploratory target-recovery hypothesis. Run 015 represents all 53 visible model-quality bars and passes all eight independent raster-to-text cross-checks.

| Audit | Result | Gate |
|---|---:|---:|
| Source image hashes | 2/2 pass | exact |
| Visible quality bars represented | 53/53 | complete |
| Cross-checks | 8/8 pass | complete |
| Maximum accuracy-coordinate discrepancy | 0.0922 percentage point | <=0.15 |
| Maximum perplexity-coordinate discrepancy | 0.0450 | <=0.12 |

## Recovered semantics

- Fig. 15(c) does not omit a third bar accidentally. WinoGrande pairs the original with `s=0.75`; the three Ada-LEval groups pair the original with `s=0.5`.
- Fig. 15(d) contains all three original/`s=0.75`/`s=0.5` perplexity bars.
- Fig. 15(b)'s original and all-12-layer endpoints reconcile with the reported 1.3-F1/1.75-EM losses within the frozen raster uncertainty.
- Fig. 16's three printed Llama2 values independently reproduce their pixel displacements within 0.034 percentage point.

## Evidence boundary

This run recovers immutable acceptance targets. It does not execute ViT, BERT, Llama2, InternLM2, FFT compression, hierarchical BSMM, or LoRA, and is therefore not model-quality validation. Later training runs must use these values without altering them from residuals.
