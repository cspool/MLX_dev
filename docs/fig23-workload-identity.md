# Figure 23 transformer-block identity

## Outcome

H122 run127 supports the negative identifiability hypothesis with
`audit_integrity=true` and 10/10 gates. The Figure 23 paragraph fixes N=
512/1K/2K/4K/8K, D=512, batch=8, SIMD8/32 and 4x4/8x8 meshes, but none of the
13 fields required to instantiate the reported transformer block is specified
for this experiment.

Missing fields include component order, structured/dense mix, projection and
FFN shapes, attention/FFT-CMP choice, B/L/s, FFN dimension, head layout,
elementwise work, memory boundaries, launch interval and mesh-specific active
window.

## H64 proxy boundary

H64's 20 configurations are internally exact target-free scaling runs, but
they compile only `compile_fig10_mapping("bsmm", 512)` with fixed memory and
active window three. Its five baseline lane-work rows exactly equal one BSMM
transform over `N * batch` lanes:

`output_lane_work = N * 8 * 512 * log2(512)`.

The proxy has no FFT-CMP, attention, output projection, FFN1, FFN2 or
elementwise component. H90/H91 demonstrate the extra fields required by a
complete layer contract, but their Llama2 D=4096 values cannot be substituted
for Figure 23's D=512 workload.

## Consequence

The earlier H46/H65/H70 speedup transfers remain useful mechanism/proxy
evidence, not strict Figure 23 reproduction. No further simulator timing change
may target their residuals until an author workload manifest supplies the
missing block composition and schedule.

Figure 23 remains incomplete and active completion stays 0/8. Work moves to
Figure 24, where exact MLX operator paths already exist and the remaining gap
is the matched open-GPU-simulator denominator.

Evidence is in
[run127](../artifacts/results/fig23-workload-identity-run127.json), with the
frozen plan in
[H122 protocol](../experiments/h122-fig23-workload-identity/protocol.md).
