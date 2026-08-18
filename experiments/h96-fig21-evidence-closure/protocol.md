# H96 protocol: Figure 21 evidence closure

H96 exposes H95 only to the completed H25 Figure 21 targets.

- GEMM-time share compares H95's five dense-projection shares;
- dense and sparse memory compare H95's first-principles values;
- speedup remains `execution_incomplete` for all five N because Xavier
  dense-Tensor cycles are null.

Each available numeric point uses the 10% relative-error gate. The audit must
account for all 20 target values (5 speedup, 5 GEMM share, 10 memory), calculate
no synthetic Xavier denominator, and keep the complete Figure 21 verdict false.

The immutable output is
`artifacts/results/fig21-evidence-closure-run101.json`.
