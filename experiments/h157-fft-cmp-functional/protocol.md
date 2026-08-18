# H157 protocol: same-input FFT-CMP payload

## Hypothesis

The scalar completion state can represent complex values as paired real/imag
registers and execute a complete semantic FFT-CMP chunk—forward radix-2 FFT,
low-frequency truncation, short inverse real FFT and amplitude correction—over
tagged spatial PEs, while matching an independent NumPy implementation and
conserving every operation, transfer and byte.

## Frozen semantics

The paper-analysis knowledge note and the repository's previously audited
inferred operator agree on this sequence: chunked real FFT, retain the low
frequency prefix, then inverse-transform at the compressed length. This is an
inferred executable contract, not a claim of recovered author code.

Freeze two real chunks of length four and compression `s=0.5`. Bit-reversed
stage-0 pairs `(0,2)` and `(1,3)` feed a full radix-2 FFT. Stage 1 computes all
four complex bins, retains bins 0 and 1, and transfers their paired real/imag
components to a final PE. That PE applies the same even-Nyquist doubling and
`2/4` amplitude scaling as `fourier_resample_real`, producing two real values
per chunk. NumPy independently evaluates `rfft`, prefix resize, `irfft(n=2)`
and amplitude scaling.

## Exact conservation contract

Across two chunks the schedule must execute 80 completion-path operations:
eight loads, 44 compute instructions (14 add, 14 fma, 16 mul), 24 transfers and
four stores. This corresponds to 30 scalar multiplications, 28 additions, 12
memory requests/96 bytes and 24 events. Five PEs and three tags route 40 hops,
of which exactly 24 are skip hops and 16 are unit hops.

## Acceptance gates

1. Frozen H156 functional-chain and H120 target-free multiport evidence qualify
   and remain supported with integrity.
2. Compilation deterministically materializes the frozen five-block/three-tag
   FFT→truncate→short-iFFT schedule without paper performance targets.
3. Static arithmetic, pipeline, memory, event and route counts exactly equal
   the registered conservation contract.
4. Debug, optimized and ASan/UBSan builds execute with byte-identical
   summaries/traces and empty sanitizer stderr.
5. All four compressed outputs match independent NumPy within 1e-12.
6. Retained F0/F1 real/imag registers match NumPy's complex FFT bins before
   compression; all 80 numeric updates are finite and error-free.
7. Every source component reaches its specified PE/tag/register, all 24 events
   fire, and the 24 skip/16 unit route split is exact.
8. Enabled and disabled modes have identical cycles and every nonfunctional
   timing/event/route/stall statistic.
9. The eight frozen H120 FFT rows remain same-work, strictly improved and at
   least 1.2x; H156 plus H157 yield 2/6 functional operator coverage.
10. The result claims FFT-CMP functional coverage and existing multiport trend
    only; it does not claim recovered author code or a new paper ratio.

The immutable result will be
`artifacts/results/fft-cmp-functional-run162.json`.
