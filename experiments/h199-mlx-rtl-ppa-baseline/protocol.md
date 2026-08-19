# H199 protocol: activity-based MLX RTL PPA baseline

## Hypothesis

One global area conversion and one global power conversion, each anchored only
to the reported full-PE aggregate, may transfer the target-free Nangate45 RTL
measurements to every Table-II component and the reduced design within 15%.
This is a target-exposed transfer diagnostic, not independent validation.

## Measurement separation

The runner may read H198 RTL, mapped workloads, VCDs and the pinned open PDK,
but not `paper_targets.yaml` or Table-II values. It synthesizes and analyzes six
full components over BSMM/FFT-CMP/SWA VCDs and six reduced components over the
reduced BSMM VCD. Only the auditor reads Table II, derives one area scale and
one power scale from the full-PE sums, then evaluates all rows.

## Frozen inputs

- H198/run203 result, SHA-256
  `f5485621410347652115b8d784371f3f400830f88d2bfb1bcfc0d5bffde86bc5`.
- H198 run/program manifests, SHA-256
  `d874ac23582e437f0faf9669a51d8a6131bcddccd96dfbf209ab007e2145ef42`
  and `338e2f1287b6d658bad6529db95a60fab5724410dfa6825fe2ad5843e1f1268a`.
- H197/run202 result, SHA-256
  `d1a6aa2cdeb4466d8e8e6fd72f7ad5604987a80bd1c9c753335381afd1f5c3af`.
- Nangate45 Liberty/LEF and paper target hashes remain those frozen by H197.

## Acceptance gates

1. All frozen inputs, source RTL, library and full/reduced VCD hashes qualify.
2. The runner source contains no target path or Table-II numeric value.
3. Twelve component/variant netlists map successfully with positive cell count
   and Liberty area; same RTL parameters replay H198 area exactly.
4. Each of the 20 component/workload OpenROAD jobs annotates positive activity
   and reports finite internal/switching/leakage/total power.
5. Clocked modules report finite 1-ns timing; all logs and manifests replay.
6. The auditor derives exactly one area and one power scale from the full PE;
   no per-component coefficient is allowed.
7. Full PE and 4x4 array arithmetic are exact after aggregate anchoring.
8. All six component area values, six component power values, and reduced
   area/power must be within 15% for the transfer hypothesis to be supported.
9. Raw and normalized results retain Nangate45, target-exposed and unavailable
   Synopsys/12-nm/post-silicon labels.

The immutable result will be
`artifacts/results/mlx-rtl-ppa-baseline-run204.json`.
