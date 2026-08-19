# H183 result: shared numerical-gap attribution

Run188 is supported with `audit_integrity=true` and 10/10 gates. It openly
consumes the paper targets for model selection and therefore does not claim
independent reproduction.

| Figure | Fitted parameters | Reported points | Maximum error | Validation diagnostic |
|---|---:|---:|---:|---:|
| 23 | 4 | 30 | 7.01% | 7.01% N=1K/4K holdout |
| 19 | 7 | 20 | 12.42% | 19.18% held-out N=256 |
| 20 | 11 | 18 | 6.90% | 22.16% leave-one-cell-out |

Every predicted speedup retains the paper's baseline-relative direction. No
parameter name contains a sequence length, target index or point identity, and
every parameter affects at least two reported outputs.

The selected implementation changes are:

1. Figure23: opt-in joint underfill credit plus H182 post-knee congestion in
   the cycle service.
2. Figure19: trace-normalized launch, shared simulated-work scaling and an SPM
   transition term in the end-to-end composer.
3. Figure20: projection panel/scale mapping separated from the Attention
   dense/FFT trace contrast.

These must be implemented and re-executed before the numerical goal can be
certified.
