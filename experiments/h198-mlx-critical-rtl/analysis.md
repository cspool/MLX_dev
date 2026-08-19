# H198 result: MLX critical RTL functionally complete

Run203 is supported with `audit_integrity=true` and 10/10 gates.

- Seven critical modules plus full/reduced wrappers are synthesizable.
- The spatial assembler emits 18 lineage-complete instructions for BSMM,
  FFT-CMP and SWA.
- Icarus and Verilator agree across eight runs (four workload/variant pairs).
- Four VCDs contain activity in configuration network, data network, tag
  buffer, control, register file and functional unit.
- FP16 registered normal/zero vectors pass FMA, add, max, exp, divide and
  shuffle checks. NaN, Inf arithmetic, subnormal and true fused-single-rounding
  behavior remain explicitly unsupported in this reconstruction.
- Ten Yosys/ABC tops have positive Nangate45 area with no inferred latch.

Initial full-PE component areas are not yet paper-calibrated: config network
16,839.662 um2, data network 709.422, control 298.186, tag buffer 3,057.404,
register file 74,205.488 and FU 110,527.522, for an integrated PE of
205,449.888 um2. The reduced integrated PE is 66,767.596 um2. These figures
are target-free diagnostics; they expose large relative-layout gaps that H199
must resolve before any 12-nm/Table-II claim.
