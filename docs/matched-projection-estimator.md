# Matched Figure 20 projection estimator

H77 covers QKV, FFN1, and FFN2 at N=256/8192 with complete logical FMA work.

The MLX B32 column-port schedule fits trip4/8 as `cycles = 5 + 50*trip` and
predicts two independent trip16 runs exactly. Full-design SIMD32 represents
7,680 FMA equivalents per trip. Xavier cycles/FMA are fitted from H57's two
execution-driven BSMM anchors. H75 supplies the full logical work.

All six projection estimates converge near 2.02x sparse-CUDA speedup after
expanding to billions of FMA equivalents. QKV, FFN1, and FFN2 retain their own
logical operations, bytes, and output footprints even though the current cycle
anchor shares the B32 schedule.

Attention remains explicitly uncovered: its logical workload contains FFT
compression and compressed attention, while H57 provides only an FFT anchor.

The immutable target-free result is
`artifacts/results/matched-projection-estimator-run082.json`.
