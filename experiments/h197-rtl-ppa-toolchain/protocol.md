# H197 protocol: RTL/PPA contract and open-tool qualification

## Purpose

Qualify the executable RTL-to-PPA path before implementing MLX modules. The
paper's private Synopsys Design Compiler executable and 12-nm Liberty/physical
kit are unavailable, so H197 must not claim method-identical reproduction. It
establishes a transparent open equivalent: Icarus/Verilator functional RTL,
Yosys/ABC mapping, and OpenROAD/OpenSTA timing plus VCD-driven power using the
Nangate45 reference library.

## Frozen evidence

- MLX paper Markdown: 85,855 bytes, SHA-256
  `5785eb81b28741a3806ca42749d7b556bbcd2404e1622ae644a32bae2ade7745`.
- Paper targets: 14,931 bytes, SHA-256
  `c4a22ab8052ff8cb223a729b8ce07320c6826cfee0d37e1a664a8761e018b41e`.
- OpenROAD-flow-scripts commit
  `6101364b2d7909dd797e1e3e7f80695401cfa4e4`.
- Nangate45 typical Liberty: 6,692,032 bytes, SHA-256
  `8d540a4d4cf6d09d27c87ad067857a9c0c2eeb023ab7a56e058cd3113db4e9b1`.
- Nangate45 tech/macro LEFs: SHA-256
  `834a79295054cd4209178d1bade67c353863c47bb4b3c22ee38b862b7cec37f2`
  and `840b01e500826096d1edcc752350834da647fdbf360798f243f8122b52b357c3`.
- Platform license: SHA-256
  `0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594`.

## Registered paper contract

1. Verilog RTL, 12 nm and 1 GHz; synthesis software is Synopsys DC.
2. Full-design power is post-silicon measured; reduced-design power is
   post-synthesis estimated.
3. Table II reports six PE components, PE sum, 16-PE array, and SIMD8 reduced
   design; all area and power values will be registered for later comparison.
4. The full design is 4x4/SIMD32; the reduced design is SIMD8 and removes
   shuffle, divide and high-precision floating-point resources.

## Acceptance gates

1. Paper and target bytes qualify and the four paper-contract statements are
   found directly in the frozen text/targets.
2. Yosys, ABC, Icarus, Verilator and OpenROAD execute and their versions are
   recorded; the ORFS commit and Nangate library/LEF/license hashes qualify.
3. A synthesizable sequential smoke RTL passes the same self-checking test in
   Icarus and Verilator and emits a non-empty VCD.
4. Yosys/ABC maps the smoke RTL to Nangate45 cells and reports positive cell
   count and Liberty area; the mapped netlist is emitted.
5. OpenROAD links the mapped netlist, constrains a 1-ns clock, reads the VCD,
   and reports finite non-negative internal, switching, leakage and total
   power plus timing checks.
6. Generated logs/netlist/VCD/metrics are hashed and replayable from one CLI.
7. The result explicitly labels Nangate45 as a non-fabricable 45-nm reference
   and denies Synopsys/12-nm equivalence; H198 must implement real MLX RTL
   before any paper-value error claim.

The immutable result will be
`artifacts/results/rtl-ppa-toolchain-run202.json`.
