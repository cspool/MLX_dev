# H200 protocol: structural and activity correction for MLX RTL PPA

## Hypothesis

H199's component errors primarily reflect reconstructed RTL structure and a
short configuration-dominated activity trace, not a missing technology scale.
The following architecture-level corrections should move all Table-II area and
power rows toward the 15% gate while retaining H198 functionality.

## Pre-registered changes

1. Register file: revise the unpublished 16-entry assumption to a four-entry
   active working set, retaining two reads/one write and SIMD32/SIMD8 widths.
2. Data network: implement six unit/skip ingress links, each with an eight-entry
   64-bit buffer, plus deterministic skip-hop selection.
3. Control: register per-tag/per-pipeline age and issue state instead of a
   purely combinational loop-only arbiter.
4. Tag buffer: add 32-bit tagged-block template metadata per resident tag.
5. Reduced FU: use a separate add/mul/FMA/max datapath so divide/exp/shuffle
   logic is structurally absent, rather than only made unreachable by opcode.
6. Configuration storage remains 32x64 as required by the paper; only its bulk
   reset is removed so synthesis may infer compact storage behavior.
7. Activity: repeat BSMM, FFT-CMP and SWA operations for 128 steady-state
   cycles with 90% compute, registered RF/data activity and configuration only
   at program boundaries. Reduced power uses the repeated BSMM subset.

## Measurement and acceptance

- Re-run H198 functional/assembler checks before PPA.
- Use the same Nangate45 Yosys/OpenROAD flow and the same single full-PE area
  and power scales as H199; no component scale is allowed.
- Require positive VCD annotations, finite power, and finite 1-ns timing.
- Require every six component, PE, array and reduced area/power error <=15%.
- Preserve explicit target-informed, non-Synopsys, non-12-nm and non-silicon
  labels.

The immutable result will be
`artifacts/results/mlx-rtl-ppa-structural-correction-run205.json`.
