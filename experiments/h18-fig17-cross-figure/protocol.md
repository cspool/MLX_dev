# H18 protocol: Fig. 3 to Fig. 17 H100 prefill consistency

## Question

Do the paper's own Fig. 3 H100 component throughputs, combined with its
disclosed `s=0.5` and `B=32` work reductions, predict all five corrected Fig. 17
prefill-eager speedups within 10%?

## Status before execution

Exploratory cross-figure consistency audit. Fig. 17 has already been inspected
and recovered by H17, so H18 is not held out. No H100 execution occurs. The
purpose is to test whether published profiles are sufficient to reconstruct the
native claim without fitting a timing coefficient.

## Frozen inputs

- Llama2-7B: `D=4096`, 32 Q heads, 32 KV heads, and 32 layers.
- Twenty layers are modified, the smallest integer count exceeding the paper's
  stated 60% coverage. All five sequence lengths are 512, 1K, 2K, 4K, and 8K.
- Fig. 3 achieved throughput at N=512/8K, in GFLOP/s:
  - dense QKV: 492,000 / 770,000;
  - dense attention: 395,000 / 503,000;
  - BSMM: 10,450 / 12,100;
  - FFT: 12,100 / 14,400.
- Intermediate throughputs use linear interpolation in `log2(N)` and clamp
  outside the two endpoints. These values were frozen from run011, not selected
  from Fig. 17 residuals.
- Corrected Fig. 17 prefill-eager targets are the immutable H17 values.

## Prediction

For each dense layer, compute time is

`T_dense = FLOPs_QKV/P_QKV + FLOPs_attn/P_attn`.

For a modified layer, use Eq. 2's QKV fraction
`2*log2(32)/32 = 0.3125` and the semantic-compression attention fraction
`s^2 = 0.25`:

`T_struct = 0.3125*FLOPs_QKV/P_BSMM + 0.25*FLOPs_attn/P_attn(sN)`.

FFT time is deliberately omitted, making this an optimistic bound for the
published operator set. The 32-layer time is the sum of 12 dense and 20
modified layers. Work unchanged between models is omitted. If structured QKV
is already slower than dense QKV under Fig. 3 throughput, adding FFT can only
worsen the prediction; adding identical positive work to both models moves a
sub-unity speedup toward 1 but cannot make it exceed 1.

## Acceptance gate

All five predicted phase speedups must have relative error at most 10% against
the Fig. 17 prefill-eager targets. The report also emits component times and an
`identifiable_from_public_profiles` flag.

## Failure policy

Do not fit throughput, modified-layer count, hidden size, or an unreported
constant to Fig. 17. Rejection means Fig. 3 plus the disclosed operator counts
do not identify Fig. 17; it is not evidence that a different unpublished CUDA
implementation or checkpoint could not obtain the plotted result.
