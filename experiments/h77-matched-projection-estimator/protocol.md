# H77 protocol: matched Figure 20 projection estimator

H77 estimates the six Figure 20 structured projection kernels with complete
logical FMA work: QKV, FFN1, and FFN2 at N=256/8192.

MLX uses the frozen B32 column-port template. Trip=4/8 anchors fit its steady
cycle slope and trip=16 validates it. One full-design SIMD32 trip represents
four times H73's SIMD8 FMA work. Xavier uses the two execution-driven H57 BSMM
anchors to fit cycles per represented FMA. H75 supplies matched logical work.

H77 reads no Figure 20 target. Attention remains excluded because its matched
work has two components (FFT compression and compressed attention) while H57
has only one FFT proxy.

The immutable output is
`artifacts/results/matched-projection-estimator-run082.json`.
