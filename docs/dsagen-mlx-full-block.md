# Full structured Transformer-block proxy

H48 composes the open MLX mechanisms into one folded 28-tag schedule. It covers
pre-attention RMSNorm, QKV hierarchical BSMM, RoPE, FFT/truncation/iFFT,
attention score and softmax, SV normalization, output projection, the attention
residual, gated FFN1, SiLU, FFN2, and the final residual/store.

Four logical CDC lanes are folded over a 4x4 DSAGEN mesh. Intermediate values
move only through explicit adjacent-tag xfer events; V is relayed through the
score/max/exponentiation chain, while long-lived residuals are reloaded at
macro boundaries. This prevents the compiler from hiding a long-range edge in
a synthetic delay. The bounded active window remains four tags.

Inside each PE, instructions select separate FMA, FMAX, FEXP, FDIV, FRSQRT,
shuffle, multiply, and add resources with operation-specific latency and
initiation interval. This is the intended hybrid interpretation: DSAGEN owns
spatial execution, while GPGPU-Sim's public scoreboard, operand-collector,
register-bank, pipelined-FU, and load/store concepts constrain PE-local hazards.
No warp, SIMT, CTA, or GPU coherence semantics are used.

The formal proxy executes 1,352 dynamic instructions: 840 compute, 472 xfer,
24 load, and 16 store. Its fixed backend completes in 393 overlay cycles and
observes 320 early cross-tag issues. The real-memory backend conserves all
40 requests/responses, records 24 source-attributed DDR reads and 16 completed
stores, and passes guest data checks. These are mechanism results only; they do
not claim the authors' unpublished schedule or Figure 18--25 accuracy.
