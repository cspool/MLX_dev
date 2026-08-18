# Figure 21 workload identity

H90 separates logical work identity from source-integrated execution identity.

The H6 analytical proxy correctly constructs batch=8 Llama2 work at
N=128/256/512/1024/2048. For QKV, Attention, output projection, FFN1, and
FFN2, every structured and dense operation total exactly matches a fresh
workload derivation. Its failure is therefore not an arithmetic shape bug.

No current real simulator run covers the complete contract:

- H48 covers phase names with trip=2 but explicitly does not claim the
  authors' schedule or a paper shape;
- H77 covers batch=1 QKV/FFN at N=256/8192 and omits output projection;
- H83 covers batch=1 structured Attention at N=256/8192 only;
- no source-integrated run executes the dense eight layers, exact elementwise
  work, or the 24 structured + 8 dense layer composition.

Thus `matched_source_execution_available` is false even though the analytical
work model is explicit. The next implementation must generalize the H83 graph
to all five batch-8 shapes, add output projection and dense/elementwise paths,
and fold the complete 32-layer mix before Figure 21 timing can be revisited.

The first correction is complete in
[`fig21-layer-contract.md`](fig21-layer-contract.md): all five batch-8 work
signatures and structured-Attention u=1 graphs replay exactly, while the
remaining timed component paths stay explicit.

The immutable result is
`artifacts/results/fig21-workload-identity-run095.json`.
