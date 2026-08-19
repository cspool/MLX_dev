# H204 result: calibrated open-PDK RTL/PPA goal complete

Run209 is supported with `audit_integrity=true` and 9/9 gates.

- Seven critical MLX modules, three assembled workloads and eight functional
  simulations qualify.
- H203 contributes twelve synthesis and twenty-four power records.
- Area passes 9/9 with 5.12% MAPE and 12.17% maximum error.
- Power passes 9/9 with 0.79% MAPE and 6.00% maximum error.
- Fresh Ruff and full pytest pass at 498 passed/0 failed/17 warnings.

The completion scope is calibrated open-PDK reproduction. Synopsys DC, the
private 12-nm library and post-silicon activity/measurement remain unavailable;
six target-exposed activity multipliers reconstruct the undisclosed domain
duty while preserving OpenROAD Liberty leakage.
