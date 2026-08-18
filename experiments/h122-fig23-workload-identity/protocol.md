# H122 protocol: Figure 23 workload identity

## Hypothesis

The paper and frozen source evidence do not uniquely identify the batch-8,
D=512 transformer-block workload behind Figure 23, and H64's one-BSMM fixed-
memory scaling path is therefore a proxy rather than complete block execution.

H122 is target-free and reads no Figure 23 speedup values.

## Audit

Verify the paper-disclosed N, D, batch, SIMD and mesh values directly. Then
check whether the paper fixes each workload field needed to instantiate a
transformer block: component order, structured/dense mix, QKV/output/FFN
shapes, attention/FFT-CMP choice, B/L/s, FFN dimension, heads/layout,
elementwise work, memory boundaries, launch interval and active window.

Independently inspect H64's compile manifest/source contract. It must resolve
to one `compile_fig10_mapping("bsmm", 512)` transform repeated over N*batch
lanes, with fixed memory and no FFT-CMP, attention, output, FFN or elementwise
component. H90/H91 demonstrate which additional fields a complete layer
contract normally requires; their Llama2 D=4096 values are methodology
evidence, not substituted Figure 23 parameters.

## Acceptance gates

1. All frozen files and H64/H90/H91 parent statuses qualify.
2. The local paper contains every disclosed Figure 23 shape/hardware field.
3. All 13 required identity fields are classified from source text, with
   absent fields retained as `not_reported` rather than inferred.
4. At least one required identity field is absent; exact workload identity is
   therefore false.
5. H64 covers exactly 20 configs and its five baseline lane-work rows match the
   one-BSMM formula for N=512..8192.
6. H64 compiler metadata identifies BSMM width 512, fixed memory, active window
   three and the four hardware configurations.
7. H64 has none of the six complete-block component classes listed in the
   config; `h64_full_transformer_block` is false.
8. H90/H91 are used only to enumerate complete-contract fields; no D=4096
   operation or timing value is transferred into D=512.
9. Auditor/test contain no Figure 23 target/result input, speedup fit or
   residual correction.
10. H122 changes no simulator source or active 0/8 completion count.

Support means the negative identifiability hypothesis is proven. The immutable
result will be `artifacts/results/fig23-workload-identity-run127.json`.
