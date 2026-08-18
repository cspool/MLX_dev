# H118 protocol: exact Figure 22 coupled workloads

## Hypothesis

The exact 4x4/SIMD8 Figure 10 BSMM and FFT mappings can execute directly at all
eight Figure 22 sizes through the current bounded-context and historical
DMA/two-half-SPM clock, while conserving the frozen H62 work graph and
producing qualified compute/load/store/xfer PE counters without any Figure 22
utilization target.

H118 is a source-identity and execution experiment. It does not claim that the
resulting utilization matches the paper.

## Frozen architecture boundary

- H62 supplies the 16 exact Figure 10 loop/tag/event/route workloads: BSMM and
  FFT at N=64..8192, on the 4x4 mesh with SIMD8 and 16-byte vectors.
- The active window remains H62's independently frozen three-tag inference,
  consistent with Figure 9's simultaneous load/current-compute/previous-forward
  description. It is not selected from Figure 22 residuals.
- H105's 2018 DPU fixture supplies 32 instruction slots, eight active blocks
  and 256 operand contexts per PE. Every H62 active footprint must fit.
- H109 supplies four bounded iteration contexts for latency-4/II-1 FMA.
- H106 supplies the 8 MiB, 32-bank, two-half non-stop SPM/DMA controller,
  64 B/cycle bandwidth and zero-setup lower bound. H113/H114 qualify its live
  coupling to `dpu_pipelined`.
- Initial input and final output DMA bytes are fixed from the H62 workload,
  not from utilization: two input vectors and one output vector per N output,
  each vector 16 bytes. Intermediate CDC loads/stores remain on-chip requests.

The paper-analysis note says only that small-kernel launch overhead is around
17% and falls below 12%; it does not expose launch timing or the counter
interval. H118 therefore leaves launch cycles null. The primary utilization
denominator is preselected as `end_to_end_cycles * 16`, matching the current
coupled-counter convention; `overlay_cycles * 16` is emitted as a labeled
diagnostic only.

## Execution

Transform each frozen H62 document only by enabling `dpu_pipelined`, adding the
frozen DPU capacities/contexts, and selecting `dpu_memory`. Execute every full
size directly—no q folding or full-size extrapolation. Run each of 16 configs
twice optimized and once under ASan and UBSan: 64 total executions.

## Acceptance gates

1. H62/H106/H109/H113/H114/H117 bytes and statuses qualify; H117 specifically
   retains 80/80 passing cycle/compute/load/store/xfer holdouts.
2. Exactly 16 H62 workload identities are consumed and no Figure 22 target or
   target-analysis file is opened by the compiler, runner or auditor.
3. Recompiling H62 from source is byte-identical to its frozen documents before
   the explicitly allowed DPU/backend/metadata transform.
4. Operator, N, blocks, tags, PE placement, trip counts, instructions, FU
   operations, events, routes and dynamic pipeline counts remain exact.
5. Every config uses mesh 4x4, SIMD8, 16-byte requests, active window three,
   32 instruction slots, eight active blocks, 256 operands and four iteration
   contexts; active instruction footprints remain at most 32.
6. The memory contract has one resident tile, `2*N*16` input bytes, `N*16`
   output bytes, all H62 external stores as its release count, and every request
   inside one 4 MiB half.
7. All 64 executions finish with no stderr; optimized replays and both
   sanitizer summaries are byte-identical per config.
8. Dynamic instructions and per-pipeline issue/completion counts equal H62;
   all boundary events and route hops conserve exactly.
9. Every external request receives one completion; load/store request counts,
   released/drained tiles, off-chip bytes and ownership violations match the
   frozen memory contract exactly.
10. End-to-end cycles are positive and nondecreasing with N within each
    operator; all four productive PE counters are nonnegative and bounded by
    `overlay_cycles * 16`.
11. Primary and diagnostic utilizations are finite and in [0,1], remain
    separately labeled, and no launch correction, counter multiplier, family
    factor or target-derived selection is present.
12. H118 changes no C++ simulator source, Figure 25 result, or active 0/8
    completion count.

Support requires all 12 gates. The immutable target-free result will be
`artifacts/results/fig22-coupled-workloads-run123.json`. Only a separately
pre-registered H119 may join these frozen outputs to Figure 22 targets.
