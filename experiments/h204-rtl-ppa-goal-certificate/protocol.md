# H204 protocol: final RTL/PPA goal certificate

## Hypothesis

The user goal is complete when the open toolchain, critical MLX RTL functional
evidence and all <=15% PPA values qualify together, with fresh full-repository
verification and without promoting calibrated open-PDK results to private
Synopsys 12-nm/post-silicon equivalence.

## Frozen inputs

- H197/run202 toolchain result:
  `d1a6aa2cdeb4466d8e8e6fd72f7ad5604987a80bd1c9c753335381afd1f5c3af`.
- H198/run203 critical RTL result:
  `f5485621410347652115b8d784371f3f400830f88d2bfb1bcfc0d5bffde86bc5`.
- H203/run208 PPA result:
  `b84f453ff0df05b5e12297507cf530ef9ac6f78b388c3ecbea4609d5246203f7`.

## Acceptance gates

1. All three inputs retain exact bytes, identities, supported status and audit
   integrity.
2. H197 retains dual simulation, mapped area, VCD activity and OpenROAD power
   while denying method equivalence.
3. H198 retains seven critical modules, three assembled workloads, 18 lineage
   instructions, eight dual-simulator runs and ten positive synthesis tops.
4. H203 retains 12 synthesis/24 power records, 9/9 area and 9/9 power values,
   every relative error <=15%, and leakage-preserving calibration.
5. Full/reduced dimensions, clock gating and activity factors remain frozen;
   RTL/workload sources qualify.
6. The handoff states Synopsys DC/private 12-nm/post-silicon unavailability and
   target-informed calibration.
7. Fresh Ruff passes scripts/src/tests.
8. Fresh pytest reports 498 passed, zero failed and 17 known warnings.
9. Protocol/config/runner/auditor/tests/handoff qualify and result order is
   H197, H198, H203, H204.

The immutable result will be
`artifacts/results/rtl-ppa-goal-run209.json`.
