# H188 protocol: final numerical-convergence and toolchain certificate

## Hypothesis

The expanded goal is complete when Figures23/19/20 are all numerically within
15%, preserve the paper's baseline-relative directions, are grounded in a
native RTX4090 trace and shared rather than pointwise parameters, and the model
workloads reach both MLX simulator formats through the replayable H187
toolchain. The full repository must then pass fresh verification.

## Acceptance gates

1. H182--H187 pass frozen byte/hash/status/integrity qualification.
2. H182 records the registered RTX4090/SM89 UUID, 38 cases, 361 positive timing
   samples and no paper targets during trace execution.
3. H183 uses 4/7/11 shared parameters, no point-keyed coefficient, finite fits
   and explicit target-informed/non-independent labeling.
4. H184 executes 40 configs/120 builds with 40 raw-cycle/work matches; all 30
   Figure23 points and directions pass 15%.
5. H185 retains four raw-cycle and trace groups; all 20 Figure19 points and all
   four MLX-over-FABNet directions pass 15%.
6. H186 retains sixteen raw execution/trace rows; all sixteen bars, two
   geomeans and sixteen directions pass 15%.
7. H187 validates three graphs/fourteen nodes/twelve units, complete lineage,
   twelve lowering replays and 24 native execution replays.
8. Maximum errors remain <=15% simultaneously and all required point counts
   are complete.
9. The certificate openly excludes independent validation, the authors'
   unpublished compiler, RTL, power and area claims.
10. Goal/handoff/source files qualify and result ordering is H182..H188.
11. Fresh Ruff passes over scripts/src/tests.
12. Fresh full pytest reports 467 passed, zero failed and 17 known warnings.

The immutable result will be
`artifacts/results/numerical-convergence-goal-run193.json`.
