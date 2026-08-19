# H194 result: five objectives complete with a scoped holdout boundary

Run199 is supported with `audit_integrity=true` and 11/11 acceptance gates.
The certificate directly freezes and checks H189--H193, then records a fresh
Ruff pass and 478 passed/0 failed/17 warnings from full pytest.

The first full run produced 473 passed/5 failed. All five failures were old
patch-stack audits: reversing physical timing left two separator boundaries
that did not belong to the older latency patch. The correction changes only
temporary audit trees, collapses the boundaries before reverse replay, restores
them before forward replay, and still requires byte-exact round trips. No patch,
simulator source, workload result or numerical parameter changed.

The five-objective completion does not claim that every independent holdout is
within 15%. H193 remains rejected, unrefit and visible at 46/48 points with all
36 directions correct. Both failures remain Figure20 N=4096 Attention, which is
the registered two-endpoint log-N interpolation/crossover limitation.
