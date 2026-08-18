# Open hybrid simulator substrate

## Outcome

The currently executable second-development substrate is DSAGEN/dsa-gem5 for
the MLX spatial fabric and tagged-block PE. H104's author-lineage audit now
shows that the unpublished original is more likely a closed SimICT-hosted
ICT/Ricore DPU/DFU simulator than a DSAGEN fork. The next open surrogate should
therefore use gem5 component timing plus historically reconstructed DPU
PE/memory/NoC behavior, while reusing DSAGEN/Assassyn mechanisms selectively.
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
