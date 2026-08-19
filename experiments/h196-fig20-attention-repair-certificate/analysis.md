# H196 result: Figure20 Attention repair goal complete

Run201 is supported with `audit_integrity=true` and 9/9 gates. It freezes the
original five-objective H194 certificate and the H195 repair, then records a
fresh Ruff pass and 482 passed/0 failed/17 warnings from full pytest.

The two requested N=4096 errors are 2.39% for dense-TCU and 1.22% for
sparse-CUDA, versus 27.89% and 20.91% before repair. All 48 registered holdout
points now pass 15% and all 36 directions remain correct. H183 parameters were
not refit.

The certificate deliberately remains `validation_eligible=false`: H195 was
designed after observing H193's failures, N=4096 uses a synthetic two-endpoint
paper interpolation, and the trace device/workload proxy is not the authors'
Xavier implementation. The achieved result closes the requested registered
error metric while preserving that evidence boundary.
