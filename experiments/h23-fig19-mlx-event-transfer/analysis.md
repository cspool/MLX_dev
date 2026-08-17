# H23 result: the direct event-model transfer fails mainly on 2D FFT attention

H23 is **rejected**. The formal artifact retains
`prior_target_exposure=true`, so it is an exploratory reproducibility package
rather than validation evidence.

| Sequence | Component | Target (ms) | Event model (ms) | Relative error |
|---:|---|---:|---:|---:|
| 128 | attention | 0.5587 | 0.7406 | 32.6% |
| 128 | FFN | 1.6760 | 1.3878 | 17.2% |
| 256 | attention | 1.0056 | 1.5811 | 57.2% |
| 256 | FFN | 2.3464 | 2.5655 | 9.3% |
| 512 | attention | 2.0112 | 3.0238 | 50.4% |
| 512 | FFN | 4.5810 | 4.7794 | 4.3% |
| 1024 | attention | 5.0279 | 6.0420 | 20.2% |
| 1024 | FFN | 10.6145 | 9.2073 | 13.3% |

Attention MAPE/max error is 40.08%/57.23%, with zero of four points passing.
FFN MAPE/max is 11.03%/17.19%, with the 256 and 512 points passing. Across all
eight components, MAPE/max is 25.56%/57.23%.

The reconstructed totals are 2.128/4.147/7.803/15.249 ms versus
2.235/3.352/6.592/15.642 ms. The 128 and 1024 totals pass, while 256 and 512
fail; total MAPE/max is 12.34%/23.71%. Partial total agreement cannot override
the registered component gate.

## Mechanistic interpretation

The no-fit H2 simulator transfers much better than the official external FABNet
model, especially for global BSMM FFN. Its main remaining Fig. 19 discrepancy is
the 2D FFT path. H23 deliberately models the hidden- and token-axis transforms
as two separately launched workloads, so it pays two launch boundaries and an
off-chip store/load between axes. The paper describes 2D FFT as one attention
operator on a multi-layer execution substrate, making an array-resident,
cross-axis composite the next mechanism-level test. A follow-up may remove only
those structurally redundant boundaries; it must not fit a scale to these
residuals.
