# Figure 19 source-integrated transfer

H99 exposes H98's frozen component cycles to the Figure 19 MLX stacks.

All 12 component and total points fail the 10% gate. MAPE is 724.4% and maximum
error is 858.4%. The source-integrated tagged-block/FU/SRAM schedule is much
slower than both the raster and H23's analytical event model.

No overlap, frequency, launch, or boundary factor is fitted after target
access. The Figure 19 residual route is closed by the registered stopping rule.

The immutable result is
`artifacts/results/fig19-source-transfer-run104.json`.

H128/H129 later upgrade all paths to current coupled timing. Their frozen H130
join in [fig19-coupled-transfer.md](fig19-coupled-transfer.md) reduces MAPE from
724% to 180% but still passes 0/12, preserving the stopping rule.
