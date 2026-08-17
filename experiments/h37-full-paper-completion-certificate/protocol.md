# H37 protocol: machine-auditable full-paper completion certificate

## Question and hypothesis

Every numeric experiment row in the frozen full-paper inventory can be bound to
immutable targets/results and assigned exactly one evidence-preserving status:
`reproduced_within_10pct`, `attempt_rejected`,
`calibration_replay_only`, or `publicly_blocked`. A fresh CPU-only execution of
all 13 lightweight `mlxsim reproduce --figure all` sections should preserve the
registered pass/fail/replay boundaries.

H37 tests certificate completeness, not the false claim that every author
experiment has been reproduced. The certificate must explicitly emit
`all_paper_experiments_reproduced_within_10pct: false` unless every one of the
18 rows independently satisfies the full measurement gate.

## Frozen inventory and evidence

Bind the exact 18-row `docs/experiment_inventory.md`, the target manifest, five
simulator hardware/calibration inputs, the corrected H33 public-artifact result,
and every decisive historical result listed in the config. Each file is frozen
by byte count and SHA-256; JSON evidence additionally carries semantic pointer
assertions. A changed or missing file makes the certificate inconclusive.

The row/status counts are frozen before the fresh suite:

- 1 `reproduced_within_10pct` row (Fig. 22);
- 7 `attempt_rejected` rows;
- 3 `calibration_replay_only` rows (Figs. 23-25); and
- 7 `publicly_blocked` rows.

`reproduced_within_10pct` means the open surrogate's complete registered
numeric gate passes; it does not mean the unpublished MLX simulator/RTL was
recovered. `attempt_rejected` requires an adequate registered attempt whose
all-point 10% gate failed. `calibration_replay_only` requires explicit
target-guided fit/replay and cannot count as validation. `publicly_blocked`
requires at least one necessary public input, implementation, trace, hardware,
or recipe to be absent; subsidiary arithmetic/baseline successes do not change
the row status.

## Fresh lightweight suite

After committing this protocol, run `reproduce("all")` once through the H37
runner. It executes the 13 registered sections using only the checked-out CPU
simulator, local configs, and target manifests. It performs no network access,
GPU inference/training, image inspection, external publisher request, or
third-party simulator run. Save the complete suite JSON and bind it into the
certificate by byte count and SHA-256.

The fresh suite must retain all registered semantic assertions: disclosed
Fig. 15/16 compute equations pass while training remains unreproduced; Fig. 2/3
arithmetic/roofline checks pass while native profiling remains false; Fig. 18,
20, and 21 retain failed gates; Fig. 22's two utilization series pass; Fig. 23
passes numerically but remains a consumed calibration target; and Figs. 24/25
remain explicit calibration replays. All non-baseline H2 ablations must have
cycle regression at least one.

## Integrity and decision gates

H37 is supported only when all of the following hold:

1. the inventory has exactly the frozen 18 unique paper labels in order;
2. all frozen source/evidence hashes and semantic assertions pass;
3. every item has complete Boolean status facts, at least one evidence binding,
   only registered suite-section references, and the invariant required by its
   status;
4. category counts are exactly 1/7/3/7 and all four categories are present;
5. the tracked worktree is clean at formal launch and both formal outputs are
   absent;
6. the fresh suite contains exactly the 13 frozen sections and passes every
   registered assertion; and
7. the certificate reports the actual global reproduction verdict rather than
   treating rejection, replay, target recovery, or public blockage as a pass.

Any schema/source/suite failure makes H37 inconclusive. A complete certificate
with a false global reproduction verdict supports H37 because that is the
pre-registered evidence claim.

## Stopping rule

Run one formal lightweight suite and one certificate serialization. Do not
rerun a failed section with altered calibration, targets, status labels, or
evidence selection. Afterward, verification may only re-hash the immutable
suite/certificate and reevaluate the frozen assertions; it must not overwrite
them. Remaining blocked/rejected experiment rows reopen only with independently
new author inputs, exact artifacts, required hardware measurements, or a new
pre-registered hypothesis not selected from these residuals.
