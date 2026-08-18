# Execution-driven AGX Orin GPU proxy

H54 runs the exact H51 CUDA 11.8 PTX kernels under an AGX Orin resource proxy.
The public RTX3070 SM86 timing template is retained; only NVIDIA-documented
resources change: 16 SMs, 1.3-GHz core/interconnect/L2 proxy clocks, and a
1.6-GHz DRAM command clock corresponding to LPDDR5-6400 on the unchanged
256-bit/16-partition memory interface.

Detailed-mode cycles are 5,593 for vector-add, 23,730 for four-stage BSMM,
25,902 for four-stage FFT, and 13,444 for SWA-16. Checksums and instruction/CTA
counts exactly match H51. RTX3090 is only 4--10% faster on these registered
kernels because the configured 5,000-cycle launch latency dominates; this is a
measured proxy property, not a fitted Figure 20/24 factor.

The configuration is not vendor-validated and cannot by itself validate the
paper's Orin bars. It supplies a transparent execution-driven denominator for
the next no-fit cross-simulator transfer.

H123 later holds exact QKV FMA work, code, data, binary and configuration fixed
while changing only CTA shape; see
[fig24-gpu-schedule-ambiguity.md](fig24-gpu-schedule-ambiguity.md). Cycles vary
by 6.149% despite equal simulated instruction counts, proving that arithmetic
identity alone does not recover the authors' Orin schedule.

H124 freezes block128 and tests repeat folding in
[fig24-qkv-orin-folding.md](fig24-qkv-orin-folding.md). q1/q2 anchors remain
pre-saturation and fail q8 for all three QKV stage counts; full denominators are
withheld pending larger anchors.

H125 extends through q32 in
[fig24-qkv-orin-steady-state.md](fig24-qkv-orin-steady-state.md). q16 is stable,
but q32 crosses a common cache/working-set boundary and runs about one-third
slower than pre-cache extrapolation. A post-cache fold is required.

H126 validates that post-cache regime in
[fig24-qkv-orin-postcache.md](fig24-qkv-orin-postcache.md). q32/q64 predicts all
q128 templates within 2.47%, yielding 21 exact-FMA QKV estimates under the
explicit block128 proxy label.
