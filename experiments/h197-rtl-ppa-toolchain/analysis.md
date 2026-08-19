# H197 result: open RTL-to-PPA flow qualified

Run202 is supported with `audit_integrity=true` and 7/7 gates.

- Icarus 11.0 and Verilator 4.038 execute the same three arithmetic vectors and
  checksum successfully.
- Icarus emits a 1,723-byte VCD.
- Yosys/ABC maps the sequential smoke block to 1,939 Nangate45 cells with
  2,738.47 um2 Liberty area.
- OpenROAD annotates 132 VCD pin activities and reports 1.55 mW internal,
  0.205 mW switching, 0.055 mW leakage and 1.81 mW total power.
- The 1-ns timing check reports -0.415216 ns slack, proving that the flow is
  applying rather than ignoring the paper clock constraint.

This result qualifies tooling only. Synopsys DC, the private 12-nm library and
post-silicon full-design measurement remain unavailable; Nangate45 is a
non-fabricable open reference. H198 must implement and functionally validate
the actual MLX critical modules, including enough pipeline depth to meet the
1-GHz reference constraint, before any Table II numerical comparison.
