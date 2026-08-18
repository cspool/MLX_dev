# Open hybrid simulator substrate

## Outcome

The currently executable second-development substrate is DSAGEN/dsa-gem5 for
the MLX spatial fabric and tagged-block PE. H104's author-lineage audit now
shows that the unpublished original is more likely a closed SimICT-hosted
ICT/Ricore DPU/DFU simulator than a DSAGEN fork. H105 begins that corrected open
surrogate by adding a target-free historical DPU execution contract to the
gem5-integrated overlay, while reusing DSAGEN/Assassyn mechanisms selectively.
Accel-Sim/GPGPU-Sim remains the separate GPU baseline.

MLX is spatial between PEs, but its PE is more programmable than a fixed
systolic MAC. The resulting split is intentional:

- DSAGEN owns the event clock, physical graph, streams, ports, banked
  scratchpad, static schedules, and inter-PE data motion.
- The MLX extension adds immutable per-layer instruction blocks, tag
  readiness/windows, pipeline/FU-aware issue, and skip-hop packet timing inside
  that spatial clock.
- The MLX extension implements Fig. 9's tagged instruction buffers, loop and
  bookkeeping state, lower-tag arbitration, and independent xfer/load/store/
  compute pipelines. Static intra-block order replaces dynamic scoreboarding.
- GPGPU-Sim does not define MLX PE semantics. Its warp, SIMT, CTA, operand-
  collector, scoreboard, register-bank, and cache-coherence behavior belongs
  only to the GPU comparison backend.

The historical H40 source-selection record is in
[`open_hybrid_v1.yaml`](../configs/simulators/open_hybrid_v1.yaml). Its
GPGPU-derived PE-hazard hypothesis is superseded by the paper audit in
[`mlx_pe_semantics_correction_v1.yaml`](../configs/simulators/mlx_pe_semantics_correction_v1.yaml).

## Executed upstream gates

The Accel-Sim binary was built from `c5296df` with GPGPU-Sim v4.2.1 at
`68e1cd3`. Its official QV100 Rodinia backprop trace completed two kernels at
14,903 cumulative cycles, 9,290,080 instructions, and 512 CTAs.

The DSAGEN stack was built from `273e141`, including dsa-gem5 `1e5d2c3`, the
DSA scheduler `d0dd816`, and LLVM `2f17ed0`. The official `vecadd.c` was
scheduled onto the public PE16 mesh ADG and ran in gem5 syscall-emulation mode.
It completed 256 CGRA instances and 1,024 mapped DFG instructions, read
16,384 bytes, wrote 8,192 bytes, and passed the application's numerical sanity
check.

Run both existing binaries again with a new output label:

```bash
OPEN_SIM_RUN_LABEL=rerun001 scripts/run_open_simulator_smokes.sh all
```

Run the read-only source/binary/evidence audit:

```bash
.venv/bin/python scripts/audit_open_hybrid_simulators.py --verify-existing
```

## Host-compatibility fixes

Only build/runtime compatibility was changed before the upstream timing gates:

- GPGPU-Sim's generated-header chmod tolerates the shared workspace ACL.
- CUDA 11.8 is used side-by-side because CUDA 13 removed APIs required by
  GPGPU-Sim v4.2.1.
- Old gem5's fatal-signal alternate stack accepts modern dynamic `SIGSTKSZ`.
- RISC-V syscall 278 uses gem5's deterministic random generator, matching the
  later upstream `getrandom` implementation needed by modern static glibc.
- DSAGEN's official custom GNU assembler path is used. Its LLVM TableGen
  generator names the immediate operand `$simm12` instead of the `RVInstS`
  field `$imm12`; using LLVM's integrated assembler silently encoded register
  numbers as masks. The patched Chipyard binutils path produces the verified
  masks 98, 229, and 1220.

These fixes do not alter modeled PE, stream, memory, or network latency.

## Current boundary

The upstream execution gates are complete, and the source-integrated MLX
overlay now implements tag blocks, programmable-PE resources, skip-hop
semantics, counted cross-layer events, radix-2 CDC compilation, DSAGEN
scratchpad callbacks, real MinorCPU LSQ/L1/L2/DDR traffic, and a 28-tag reduced
full-Transformer-block proxy. See [`dsagen-mlx-overlay.md`](dsagen-mlx-overlay.md),
[`dsagen-mlx-cdc-memory.md`](dsagen-mlx-cdc-memory.md),
[`dsagen-mlx-dma-memory.md`](dsagen-mlx-dma-memory.md), and
[`dsagen-mlx-full-block.md`](dsagen-mlx-full-block.md), and
[`mlx-pe-paper-contract.md`](mlx-pe-paper-contract.md). The corrective Figure
22/23 replay is reported in
[`paper-static-fig22-23.md`](paper-static-fig22-23.md). The complete Figure 22
target/counter correction is reported in
[`fig22-resource-counters.md`](fig22-resource-counters.md). It shows that the
previous global-any-PE busy counter is not a valid PE-normalized utilization
metric. The target-free Figure 10 loop/template reconstruction is documented in
[`fig10-mapping.md`](fig10-mapping.md), and its full resource transfer in
[`fig10-fig22-transfer.md`](fig10-fig22-transfer.md). The unpublished
scratchpad/counter boundary is recorded in
[`fig22-data-supply-evidence.md`](fig22-data-supply-evidence.md). Target-free
SIMD/mesh scaling is documented in
[`fig10-scalability.md`](fig10-scalability.md), with its frozen Figure 23
comparison in [`fig10-fig23-transfer.md`](fig10-fig23-transfer.md). The exact
standalone reproduction of DSAGEN scratchpad timing is documented in
[`standalone-dsagen-spad.md`](standalone-dsagen-spad.md), and its scalability
limit in [`spad-scalability.md`](spad-scalability.md) and
[`spad-fig23-transfer.md`](spad-fig23-transfer.md). The Fig.9-derived memory-port
candidate is documented in [`multiport-spad.md`](multiport-spad.md), with its
frozen comparison in
[`multiport-fig23-transfer.md`](multiport-fig23-transfer.md). Physical per-FU
counters required by Figure 25 are documented in
[`fu-counters.md`](fu-counters.md), with the corrected transfer in
[`fma-fig25-transfer.md`](fma-fig25-transfer.md). The same counters now cover
all 42 Figure 24 MLX cases in
[`fig24-fu-rerun.md`](fig24-fu-rerun.md), with the cross-simulator audit in
[`fu-fig24-transfer.md`](fu-fig24-transfer.md). Figure 20 matched-shape gaps are
quantified in [`fig20-workload-identity.md`](fig20-workload-identity.md). The
large-repeat estimator required for matched shapes is validated in
[`repeat-folding.md`](repeat-folding.md), and the matched Figure 20 projection
estimator in
[`matched-projection-estimator.md`](matched-projection-estimator.md). Its frozen
Figure 20 comparison is rejected in
[`matched-projection-fig20-transfer.md`](matched-projection-fig20-transfer.md),
showing that matched aggregate work alone is insufficient. The remaining
attention work mismatch is now made explicit in
[`attention-work-signature.md`](attention-work-signature.md). Its exact-work
variable-depth FFT execution and rejected small-anchor estimator are in
[`matched-fft-cycle-estimator.md`](matched-fft-cycle-estimator.md). Its
target-free steady-state repair passes in
[`fft-steady-state-folding.md`](fft-steady-state-folding.md). The remaining
compressed-attention reduction is implemented in
[`grouped-attention-cdc.md`](grouped-attention-cdc.md). Full SIMD32 composition,
NoC packet flow, and four-port DSAGEN SRAM timing pass in
[`combined-attention-memory.md`](combined-attention-memory.md). The remaining
Figure 20 boundary is a matched Xavier FFT-CMP plus compressed-Attention
execution. Its first 32-run attempt is execution-valid but its small-anchor
folding is rejected in
[`xavier-matched-attention.md`](xavier-matched-attention.md); saturated GPU
anchors are still required. Figures 18--21 and 24--25 also require broader
shape/device coverage. Paper result bars were not inputs to the invariant runs.

The Xavier follow-up trajectory is closed in
[`xavier-final-attention.md`](xavier-final-attention.md), and the resulting
eight-cell Figure 20 ledger is in
[`fig20-matched-evidence-closure.md`](fig20-matched-evidence-closure.md): six
numerical failures and two execution-incomplete Attention cells.

Figure 21's next boundary is localized in
[`fig21-workload-identity.md`](fig21-workload-identity.md): the analytical work
is correct, but the real source path lacks the complete batch-8, five-shape,
32-layer structured/dense graph.

H91 supplies the five exact batch-8 layer contracts and replayable structured-
Attention graphs in [`fig21-layer-contract.md`](fig21-layer-contract.md); timed
projection/dense/elementwise blocks and 24+8 folding remain next.

H92 supplies all timed non-Attention paths in
[`fig21-timed-paths.md`](fig21-timed-paths.md); five batch-8 Attention models
and the final layer fold remain.

H93-H95 complete the MLX layer fold, and H96 closes its target evidence in
[`fig21-evidence-closure.md`](fig21-evidence-closure.md). Memory passes 9/10,
GEMM share fails 0/5, and five speedups remain unavailable without Xavier
Tensor execution.

Figure 19's new source path is scoped in
[`fig19-source-identity.md`](fig19-source-identity.md): its mapping is exact,
but needs plain forward-FFT and global-BSMM compilers rather than H81/H92
direct reuse.

H98 implements those paths and H99 rejects their target transfer at 0/12 in
[`fig19-source-transfer.md`](fig19-source-transfer.md). No residual correction
is admitted.

Figure 24/25 proxy identity is quantified in
[`fig24-25-work-identity.md`](fig24-25-work-identity.md): all 66 proxies cover
less than `6.11e-5` of at least one required FU workload despite many matching
stage counts.

H101 replaces those proxies with all 48 exact batch-32 paths in
[`fig24-25-exact-paths.md`](fig24-25-exact-paths.md). H102 then applies the
paper-derived 16-PE spatial loop in
[`fig24-25-full-mesh-paths.md`](fig24-25-full-mesh-paths.md): every work,
replay, holdout, and physical-counter gate passes, and QKV utilization rises
from about 25% to at least 99.30% without changing work or FU latency. The
frozen Figure 25 comparison in
[`full-mesh-fma-fig25-transfer.md`](full-mesh-fma-fig25-transfer.md) is still
a 1/24 physical-occupancy diagnostic. It is not the paper's roofline metric and
therefore does not constitute a Figure 25 transfer. H104's broader author-team
survey is recorded in
[`../literature/mlx-author-simulator-lineage.md`](../literature/mlx-author-simulator-lineage.md).

H105 implements the first historically grounded DPU layer in
[`simict-dpu-contract.md`](simict-dpu-contract.md): FRFO readiness, explicit
task/block/instance identity, instruction/operand/active-block capacities, and
independent physical NoC planes. Its 12/12 gates include H52 event-level and
full-gem5 569-cycle regressions. It consumes no paper performance target and
does not change the current 0/18 full-paper reproduction certificate.

H106 adds the source-derived DDR/DMA/two-half-SPM ownership layer in
[historical-dpu-memory.md](historical-dpu-memory.md). It conserves tile,
relative-address, bank and off-chip traffic through the same overlay, while
keeping every undisclosed DRAM/DMA/SPM latency explicit. Its synthetic
non-stop mechanism passes 12/12 gates, but its 37-versus-59-cycle comparison is
not a paper result. Full-work tile/residency compilation is still required
before Figure 25 operational intensity can be evaluated.

H107 completes that target-free OI prerequisite for all 48 exact batch-32 paths
in [full-mesh-memory-residency.md](full-mesh-memory-residency.md). FFT-CMP OI
is 17.33–25.33 FLOP/B, QKV-BSMM is 142.75–1527.05 FLOP/B, and explicit
windowed-KV streaming lowers SWA from optimistic 64/128 to 25.6/51.2 FLOP/B.
MLX bandwidth, achieved performance and roofline utilization remain null, so
no Figure 25 cell is claimed.

H108 composes compute and DMA bounds in
[compute-dma-overlap.md](compute-dma-overlap.md), but also discovers a
source-level throughput defect: one inflight instruction state per tagged block
turns latency-4/II-1 FMA trips into effective II=4. H102's high physical-FMA
residence is therefore not high issue throughput. Figure 25 work must pause
until multi-iteration in-flight execution is implemented.

H109 implements and validates that correction in
[pipelined-block-contexts.md](pipelined-block-contexts.md). A latency-4/II-1
eight-trip FMA now issues in cycles 0–7 and completes in 4–11 with four bounded
contexts, while every legacy result remains byte-identical.

H110 performs the target-free recompile in
[pipelined-full-mesh-paths.md](pipelined-full-mesh-paths.md). All 96 corrected
cycle holdouts pass and QKV issue utilization reaches 97.78%–99.79%, but the
registered joint hypothesis is rejected because 16 FFT physical-residence
holdouts fail. The issue/cycle correction is usable; the failed residence fold
is not.

H111 uses only that valid subset to rebuild H108 in
[corrected-compute-dma-overlap.md](corrected-compute-dma-overlap.md). All 240
matched sensitivity points are strictly faster and pass 12/12 gates. The
manuscript still supplies no numeric MLX bandwidth, so the grid remains an
unselected envelope.

H112 freezes that grid before joining Figure 25 in
[fig25-corrected-bandwidth-matrix.md](fig25-corrected-bandwidth-matrix.md).
All five bandwidth rows and the per-point oracle pass 0/24. QKV/SWA remain far
too close to ideal issue/bandwidth roofs, proving that a scalar bandwidth is
not the missing simulator parameter. The next substrate change must couple
`dpu_pipelined` contexts directly to the historical DMA/SPM ownership clock.

H113 validates that live combination in
[coupled-pipelined-dpu-memory.md](coupled-pipelined-dpu-memory.md). Six scenarios
and 36 four-build executions pass exact ownership, request/response, tile
ordering, context, bank-pressure and parent-regression gates. The remaining
boundary is compiling all 48 exact paths into this coupled clock.

H114 closes that boundary in
[coupled-full-mesh-paths.md](coupled-full-mesh-paths.md). All 48 paths, 192
configs, 480 executions and 96 cycle holdouts pass. Full coupled FMA issue is
25.22%–28.61% for FFT, 83.53%–98.43% for QKV and 55.02%–71.00% for SWA. These
are target-free outputs; work pauses before a new Figure 25 join.

H115 performs the frozen join in
[fig25-coupled-transfer.md](fig25-coupled-transfer.md). Two long-SWA cells pass,
but FFT remains uniformly low and QKV uniformly high; full Figure 25 is
rejected at 2/24 and 43.71% MAPE. No residual factor follows.

H116 audits run119's physical counters target-free in
[coupled-resource-counter-folding.md](coupled-resource-counter-folding.md).
QKV/SWA counters are stable, while all FFT FMA-residence holdouts fail; the
next simulator work is an FFT-only steady-state extension.

H117 performs that extension in
[fft-coupled-counter-steady-state.md](fft-coupled-counter-steady-state.md).
Sixteen q64/q128 FFT configs and 48 executions pass every integrity gate. All
80 cycle/compute/load/store/xfer holdouts pass; only seven FMA-residence
holdouts fail. This is sufficient to freeze the coupled pipeline-counter
semantics for Figure 22, but not to reinterpret or repair Figure 25.

H118 then executes the exact SIMD8 Figure 22 workload set in
[fig22-coupled-workloads.md](fig22-coupled-workloads.md). Sixteen full-size
paths and 64 four-build runs conserve all source and memory contracts. The run
also fixes two cross-configuration simulator defects—active-window instruction
residency and aligned sub-bank requests—with reversible patches and frozen
parent regressions. Its utilization is target-free pending H119.

H119 freezes that output before the Figure 22 comparison in
[fig22-coupled-transfer.md](fig22-coupled-transfer.md). Only 3/64 segments pass;
compute/store are low and load is high for every workload. This rejects the
single shared H106 request queue as a complete Figure 22 data-supply model but
does not invalidate H118's execution correctness.

H120 replaces only that attachment with H69's diagram-derived port topology in
[fig22-coupled-multiport.md](fig22-coupled-multiport.md). Total banks and issue
width remain 32, but four independent queues accelerate every exact path by
1.76x–2.75x. Default one-port and all historical regressions remain exact; no
paper target is consumed.

H121 joins the frozen output in
[fig22-multiport-transfer.md](fig22-multiport-transfer.md) and rejects Figure 22
at 4/64. Port topology improves compute occupancy but exposes that the paper's
load counter cannot be reconstructed from disclosed semantics. Further
Figure 22 counter variants are stopped pending author evidence.

H122 audits Figure 23 in
[fig23-workload-identity.md](fig23-workload-identity.md). The paper fixes shapes
and hardware configurations but omits all 13 complete-block identity fields;
H64 executes one BSMM only. Existing scaling curves remain proxy evidence and
do not justify further target-driven simulator changes.

H123 exercises the open GPU side directly in
[fig24-gpu-schedule-ambiguity.md](fig24-gpu-schedule-ambiguity.md). Three
GPGPU-Sim Orin runs conserve exact QKV FMA work and simulated instructions, but
CTA shape alone creates 6.149% cycle spread. Figure 24 now requires separately
frozen GPU schedules per operator family, not work-normalized micro-proxies.

H124 begins that family-specific path in
[fig24-qkv-orin-folding.md](fig24-qkv-orin-folding.md). All 12 exact block128
runs pass execution gates, but q1/q2 folding fails q8 for B16/B32/B64. The GPU
evidence is valid; the small-anchor extrapolation is not.

H125 in [fig24-qkv-orin-steady-state.md](fig24-qkv-orin-steady-state.md)
passes q16 but finds a synchronized q32 cache cliff for all QKV stage counts.
This validates the need for regime-aware GPU simulation while keeping every
full denominator null.

H126 closes the post-cache QKV path in
[fig24-qkv-orin-postcache.md](fig24-qkv-orin-postcache.md). Three q128 holdouts
pass at 2.27%–2.47%, and 21 exact-FMA Orin proxy estimates become eligible
without paper targets. Their author-kernel identity remains explicitly false.

H127 freezes those estimates before the direct Figure 24 QKV join in
[fig24-qkv-coupled-transfer.md](fig24-qkv-coupled-transfer.md). All 21 ratios
are 6.09x–7.18x versus 0.58x–1.36x targets. This closes the staged-memory GPU
proxy route without extending it to FFT/SWA.

H128 returns to the MLX-only Figure 19 paths in
[fig19-coupled-paths.md](fig19-coupled-paths.md). Forty-eight exact configs and
192 four-build runs pass; seven FFN paths fold, while all FFT paths and one
two-tile FFN need larger target-free anchors.

H129 closes those folds in
[fig19-coupled-steady-state.md](fig19-coupled-steady-state.md). Ten q64/q128
holdouts pass at 1.23% MAPE, including exact power-of-two multi-tile FFN
scaling, so all 12 current-coupled components are frozen before target access.

H130 joins them in [fig19-coupled-transfer.md](fig19-coupled-transfer.md).
Current coupling cuts the old error sharply, but all 12 component/total values
remain high and the full figure is rejected. Published component sums forbid an
overlap repair without new author evidence.

H131 audits Figure 18 in
[fig18-workload-identity.md](fig18-workload-identity.md). Its N/D/s labels do
not identify 12 workload fields or six simulator/tapeout provenance fields, so
neither reduced nor full current paths may be selected from the bars.

H132 packages the current strict scope in
[active-simulator-completion.md](active-simulator-completion.md): 0/8 complete,
with two identity gaps, three numerical rejections and three execution gaps.

H133 begins closing Figure 20's Xavier Attention gap in
[xavier-fft-regime.md](xavier-fft-regime.md). Two stable FFT 64K holdouts pass
within 2.03%, releasing exact full FFT proxy components without target access.

H134 completes QK/SV/softmax in
[xavier-attention-components.md](xavier-attention-components.md). Three new
holdouts pass at 2.13% MAPE and all six non-FFT shape-components become
eligible, enabling a target-free Xavier total next.

H135 composes complete totals in
[xavier-attention-composition.md](xavier-attention-composition.md). The frozen
transparent-proxy Attention speedups are 3.454x and 3.152x before target access.

H136 completes the target join in
[fig20-attention-completion.md](fig20-attention-completion.md). Figure 20 passes
all eight user-directed trend cells, while the retained strict diagnostic is
only 1/8; the active primary count becomes 1/8 without rewriting the numerical
failures.

H137 refreshes all eight figures in
[active-simulator-trend-completion.md](active-simulator-trend-completion.md).
It certifies primary 1/8 and strict 0/8, and separates three ready trend audits
from two execution gaps and two identity/provenance gaps.

H138 completes Figure 19's qualitative audit in
[fig19-trend-completion.md](fig19-trend-completion.md). All latency curves have
the same rank/order trend, and the independent open FABNet/current MLX ratios
show 1.378x-1.616x improvement; the primary count becomes 2/8.

H139 rejects Figure 22's qualitative audit in
[fig22-trend-completion.md](fig22-trend-completion.md). None of eight resource
curves reaches the frozen rank threshold, confirming a counter/timing-semantics
gap rather than a merely strict numerical tolerance.

H140 rejects Figure 25 in
[fig25-trend-completion.md](fig25-trend-completion.md). FFT-CMP preserves the
four-case ordering, but five QKV/SWA curves flatten or reorder under the current
roofline path; active completion remains 2/8.

H141 replaces Figure 23's single-BSMM proxy with the target-free complete-block
sweep in [fig23-complete-block.md](fig23-complete-block.md). Forty configs and
120 three-build executions conserve work exactly and show robust SIMD, mesh and
joint scaling before target access.

H142 completes the qualitative join in
[fig23-complete-block-trend.md](fig23-complete-block-trend.md). Both robustness
windows pass all 15 scaling comparisons, raising primary completion to 3/8;
the 23/30 strict diagnostic and representative-schedule caveat remain explicit.

H143 audits Figure 21's remaining denominator in
[fig21-xavier-coverage.md](fig21-xavier-coverage.md). The existing Xavier config
enables tensor units but has never executed WMMA/MMA; sparse projection and
structured-Attention proxies are rejected as dense-model substitutes.

H144 tests functional-PTX WMMA in
[fig21-xavier-wmma.md](fig21-xavier-wmma.md). Genuine WMMA PTX reaches kernel
enqueue but crashes the functional simulator before metrics, so no projection
estimate is emitted and the next route is trace-driven Accel-Sim.

H145 tests live SASS capture in
[fig21-xavier-wmma-trace.md](fig21-xavier-wmma-trace.md). The frozen NVBit
tracer is unsupported by the current driver before application execution, so
the trace-driven route must use a transparent source-derived HMMA microtrace.

The active objective is now limited to simulator-dependent hardware performance
in Figures 18–25, as recorded in
[simulator-experiment-scope.md](simulator-experiment-scope.md). Accuracy,
perplexity, training, algorithm-only and standalone native-GPU results no
longer drive simulator changes.
