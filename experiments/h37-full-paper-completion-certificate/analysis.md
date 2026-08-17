# H37 analysis: certificate complete, global reproduction goal false

## Immutable result

- Run: `run_042`
- Source commit: `f8be0f5aa47fb08893428731e68b2b1ce997d267`
- Config: `configs/analysis/full_paper_completion_v1.yaml`
- Config bytes: 20,484
- Config SHA-256: `20d29ad77833dd61c4ab10131f5a49e015a9ef616207e417de91c9fb2c2d1a74`
- Fresh suite: `artifacts/results/full-paper-lightweight-suite-run042.json`
- Suite bytes: 449,350
- Suite SHA-256: `07d47946a391c5b1192eba4fa914a23ba36f2a0ccce33424a3d7d6f229a3877e`
- Certificate: `artifacts/results/full-paper-completion-certificate-run042.json`
- Certificate bytes: 70,922
- Certificate SHA-256: `7182c5dd1612680f85912c6e96bf14e78b871a4d560f0f3610a6c0dc7de6b071`
- Audit integrity: `true`
- H37 certificate hypothesis: `supported`
- Global all-paper reproduction verdict: `false`

The formal launch found a clean tracked worktree and absent outputs. All 40
frozen files pass their byte/hash checks and every registered JSON-pointer
assertion. The inventory contains exactly the 18 expected unique labels in
order; all item schemas, evidence references, status invariants, and 1/7/3/7
counts pass.

## Fresh suite

The single CPU-only `reproduce("all")` execution emitted all 13 registered
sections. All 23 semantic assertions pass, as do the exact section set/count,
mapping schema, H2 baseline presence, and non-baseline cycle-regression gates.

The suite preserves the important evidence boundaries:

- Fig. 22's BSMM and chunk-FFT utilization series pass with 3.33% and 7.10%
  maximum relative error;
- Fig. 23's three scaling series pass numerically at 1.07%-3.20% maximum error,
  but remain target-consumed calibration evidence;
- Figs. 24/25 remain explicit validation-ineligible calibration replays;
- Fig. 15/16 disclosed compute equations pass while `training_reproduced` is
  false;
- Fig. 2/3 arithmetic/roofline checks pass while native profiling remains
  false; and
- Fig. 18, Fig. 20 speed, and Fig. 21 speed retain failed 10% gates.

## Completion classification

Exactly one row, Fig. 22, is classified `reproduced_within_10pct` for the open
surrogate. Seven adequate attempts are `attempt_rejected`, three fitted rows
are `calibration_replay_only`, and seven rows are `publicly_blocked`. Thus 17
of 18 rows are not full experiment reproductions, and no exact MLX author
artifact is used.

This does not erase successful subclaims: all numeric targets are acquired,
the Fig. 15/16 operation-count equations pass, several original/dense quality
baselines pass, table arithmetic is consistent, and the local simulator is
executable. It prevents those narrower results from being promoted into the
user's requested all-experiment 10% result.

## Terminal evidence boundary

The certificate is complete; the reproduction objective is not achieved from
public evidence. H33-H36 independently show that no exact artifact, qualifying
lineage full text, recoverable first-party representation, or legible supplied
figure identifier can fill the missing author inputs. Further calibration or
residual-selected recipe/simulator variants would be target fitting rather than
reproduction.

`make audit-completion` performs a read-only requalification of every frozen
evidence file, the suite hash, config hash, suite assertions, and certificate
summary. It passed immediately after formal serialization. Reopen a blocked or
rejected row only when new author artifacts/recipes/traces, required native
hardware measurements, or an independently motivated pre-registered method
becomes available.
