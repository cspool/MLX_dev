# H175 protocol: MLX full-operator end-to-end functionality

## Hypothesis

Prepending actual RMSNorm and RoPE numerical stages to H171's dynamically linked
structured block closes the remaining Figure-21 functional operator gap without
changing the already validated BSMM/FFT/Attention/SWA/elementwise semantics.

## Preprocess schedule

For each of H171's two width-4 input vectors:

1. load four raw values;
2. copy each value to a distinct RF bank, then compute sum of squares, add
   epsilon and execute `frsqrt`;
3. normalize all four values;
4. store the normalized vector;
5. reload it and apply two real-valued RoPE pair rotations using mul/FMA;
6. store directly to the original BSMM input addresses.

The eight original BSMM input seeds are removed. Existing tags shift by two and
the first BSMM tag depends on RoPE completion. Existing address-ready events,
instructions, routes and numerical payloads otherwise remain unchanged.

## Exact contract

The preprocess adds four blocks, 82 operations, 32 memory requests and 256
bytes. The completed MLX graph has 15 tags, 58 blocks, 548 operations, 194
memory requests/1552 bytes, 97 events and 139 route hops.

## Acceptance gates

1. H171, Xavier functionality and H174 performance estimate qualify.
2. Enabled/disabled documents compile deterministically and differ only in the
   functional enable flag.
3. Raw inputs replace all eight original BSMM seeds; every BSMM input load is
   produced by RoPE stores.
4. RMSNorm and RoPE schedules contain exact registered operations and values.
5. All original H171 instruction/memory/event/route work is preserved after a
   two-tag shift; added work matches the preprocess contract exactly.
6. Three builds complete enabled/disabled runs with identical timing and clean
   sanitizers.
7. RMSNorm, RoPE, all five original component boundaries and eight final values
   match an independent from-origin NumPy recomputation within 1e-12.
8. All 548 operations are finite and error-free.
9. H173 Xavier and H175 MLX together cover the Figure-21 end-to-end operator
   inventory in actual simulation.
10. No paper performance target or fitted timing parameter is consumed.

Before any accepted execution, the first launch was rejected by the simulator's
same-bank double-read validator because a scalar square read the same register
twice. The registered schedule therefore includes one explicit `shuffle` copy
per input and multiplies the original/copy from distinct banks. This adds eight
operations globally and changes no mathematical value.

The immutable result will be
`artifacts/results/mlx-full-operator-e2e-functional-run180.json`.
