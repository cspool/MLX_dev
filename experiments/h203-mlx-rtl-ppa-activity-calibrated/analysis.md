# H203 result: all registered RTL PPA values within 15%

Run208 is supported with `audit_integrity=true` and 10/10 gates.

- 12 fresh synthesis records and 24 fresh OpenROAD power records pass.
- All 9 area and 9 power values are within 15%.
- Area MAPE is 5.12%, maximum 12.17% (config network).
- Power MAPE is 0.79%, maximum 6.00% (config network).
- Reduced SIMD8 area/power errors are 8.70% and 0.19%.
- All activity calibration rows preserve measured Liberty leakage exactly.

The implementation evidence remains target-informed. One global 45-nm-to-paper
area scale, one global calibrated power scale and six domain activity
multipliers are used. The multipliers reconstruct the paper's undisclosed
post-silicon duty distribution from H202; they do not turn Nangate45/Yosys/
OpenROAD into Synopsys DC with the private 12-nm library or a silicon power
measurement.
