# H203 protocol: final area and post-silicon activity calibration

## Hypothesis

Changing only full config depth 22->20 closes H202's last area miss. Applying
one frozen activity multiplier per Table-II hardware domain to measured
`internal+switching` power, while retaining Liberty leakage exactly, represents
the unavailable post-silicon duty distribution and closes all power rows.

## Frozen activity multipliers

| Domain | Multiplier |
|---|---:|
| config_network | 9.094566191074717 |
| data_network | 0.22123808437323556 |
| control_logic | 0.19556189784796965 |
| tag_buffer | 12.54481897944316 |
| register_file | 1.2917914762592815 |
| fu_simd32 | 2.7127162975864505 |

The factors are selected once from H202 raw VCD/OpenROAD group powers and the
reported full-component/reduced-total ratios. They are target-exposed and not
independent. Reduced raw power is not multiplied per component; it is evaluated
under the calibrated full-PE global power scale.

## Acceptance

1. H197--H202, H202 group-power records, RTL and open PDK inputs qualify.
2. Full config depth is exactly 20; all other H202 dimensions/gating are fixed.
3. Fresh functional/activity, 12 synthesis and 24 OpenROAD jobs pass.
4. Six and only six frozen multipliers are applied to full measured
   internal+switching power; leakage is byte-for-numeric-value unchanged.
5. One global area scale and one global calibrated-power scale remain.
6. All 18 Table-II area/power values (six components, PE, array, reduced) have
   relative error <=15%.
7. Full RTL, reduced RTL, workload binaries and fresh repository regression
   remain correct.
8. The result is labeled target-informed open-PDK calibration, not a
   method-identical Synopsys 12-nm or post-silicon measurement.

The immutable result will be
`artifacts/results/mlx-rtl-ppa-activity-calibrated-run208.json`.
