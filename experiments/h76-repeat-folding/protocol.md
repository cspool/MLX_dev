# H76 protocol: repeat-folded cycle estimator

H76 validates an affine steady-state estimator for large repeated CDC work:

`cycles(repeats) = intercept + slope * repeats`.

For each hardware configuration and each independently frozen memory mechanism
(fixed H64, single-buffer H67, column-port H69), N=512/1024 are the two fit
anchors. N=2048/4096/8192 are held-out mechanism checks. No Figure 20/21/23
target is read.

Support requires every held-out cycle prediction within 5%, positive slope,
and exact affine conservation for instruction, event, route, and logical work
counts. This estimator may then replace billions of identical outer-loop
iterations while retaining explicitly modeled fill/drain intercepts.

The immutable output is
`artifacts/results/repeat-folding-run081.json`.
