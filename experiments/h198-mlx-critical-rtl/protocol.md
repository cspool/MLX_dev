# H198 protocol: synthesizable MLX critical-module RTL

## Hypothesis

The public MLX microarchitecture contract is sufficient to reconstruct a
synthesizable PE boundary whose configuration/data networks, tag control,
buffers, register file and heterogeneous SIMD functional units execute the
paper's BSMM, FFT-CMP and SWA instruction classes. Functional correctness and
structural synthesis must precede PPA fitting.

## Registered architecture

- Full: 4x4 mesh, SIMD32, FP16 data, 32 instructions/PE, 16 tag slots.
- Reduced: 4x4 mesh, SIMD8; no vector-shuffle, divide or high-precision path.
- Four decoupled issue classes: load, store, compute and xfer.
- Stateless signed residual-hop routing with skip steps 2 then 1.
- 64-bit data-network packet, matching the simulator's 8-byte/cycle link.
- Quarter-SIMD FMAX/FEXP/FDIV service width in the full PE.
- Reconstructed assumptions absent from the paper: 16 architectural SIMD
  registers and two-read/one-write RTL ports. These may be revised only in a
  later pre-result PPA protocol, not hidden as paper facts.

## RTL and software scope

1. `mlx_config_network`: serial configuration word reception and indexed
   instruction delivery.
2. `mlx_data_network`: deterministic unit/skip residual-hop consumption and
   destination-register payload forwarding.
3. `mlx_tag_buffer`: resident/ready/done state, trip count and instruction
   frontier for sixteen tagged blocks.
4. `mlx_control_logic`: lower-tag arbitration across load/store/compute/xfer
   frontiers with independent issue readiness.
5. `mlx_register_file`: SIMD-striped storage with registered writes and two
   reads, including full/reduced parameterization.
6. `mlx_fu`: SIMD FP16 add/multiply/FMA/max plus full-only shuffle/divide and
   quarter-width exp/div/max service behavior.
7. `mlx_pe_top`: integrates the six Table-II PE components without SRAM,
   off-chip PHY or the RISC-V host.
8. A spatial assembler compiles registered BSMM, FFT-CMP and SWA YAML programs
   to 64-bit instruction hex and emits a lineage manifest.

## Acceptance gates

1. H197 and all paper/contract inputs qualify; no author RTL is claimed.
2. All seven RTL modules are synthesizable and lint-clean in Verilator.
3. Icarus and Verilator execute full and reduced testbenches with identical
   final checksums, route traces, tag completion and operation counts.
4. The assembler emits all three workloads reproducibly; every binary
   instruction maps back to one YAML source operation.
5. Full RTL executes load/FMA/add/max/exp/div/shuffle/xfer/store; reduced RTL
   executes the registered reduced subset and rejects removed operations.
6. FP16 normal/zero arithmetic tests match NumPy golden values for the
   registered vectors; unsupported NaN/Inf/subnormal behavior is explicit.
7. Tag arbitration selects the lowest ready tag, four pipeline classes overlap,
   and residual distances 1/2/3 route as [1]/[2]/[2,1].
8. Full/reduced VCDs are non-empty and contain activity in all instantiated
   critical modules.
9. Yosys elaborates full and reduced PE tops with no inferred latch, unresolved
   cell or zero-sized component; per-module cell/area statistics are positive.
10. RTL, assembler and tests contain no Table-II area/power target values.

H198 makes no <=15% PPA claim. The immutable result will be
`artifacts/results/mlx-critical-rtl-run203.json`.
