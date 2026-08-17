# Open hybrid simulator substrate

## Outcome

The executable second-development substrate is DSAGEN/dsa-gem5 for the
spatial fabric, with Accel-Sim/GPGPU-Sim retained as a separately executable
GPU baseline and as a source-level reference for programmable-PE resource
hazards. This is an engineering reconstruction, not a claim that the
unpublished MLX simulator was derived from either repository.

MLX is spatial between PEs, but its PE is more programmable than a fixed
systolic MAC. The resulting split is intentional:

- DSAGEN owns the event clock, physical graph, streams, ports, banked
  scratchpad, static schedules, and inter-PE data motion.
- The MLX extension will add immutable per-layer instruction blocks, tag
  readiness/windows, resource-aware issue, and skip-hop packet timing inside
  that spatial clock.
- GPGPU-Sim's scoreboard, operand collector, register banks, pipelined SIMD
  units, and load/store unit constrain the new PE resource abstraction.
- GPU warps, SIMT reconvergence, CTA residency, and GPU cache coherence are
  excluded from the MLX PE.

The exact source revisions and machine-auditable mechanism map are in
[`open_hybrid_v1.yaml`](../configs/simulators/open_hybrid_v1.yaml).

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

The upstream execution gates are complete. The next code layer is deliberately
small and auditable: an MLX overlay in `dsa-gem5/src/cpu/minor/ssim` that
retains DSAGEN memory/stream timing and introduces only the missing tag-block,
programmable-PE resource, and skip-hop semantics. Paper result bars are not
inputs to this substrate selection or smoke validation.
