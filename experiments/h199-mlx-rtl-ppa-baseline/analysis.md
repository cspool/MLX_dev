# H199 result: one-factor PPA transfer is rejected

Run204 is rejected with `audit_integrity=true` and 9/10 gates. Twelve
component/variant netlists and twenty OpenROAD activity jobs are valid after two
measurement corrections: nested scopes are promoted to standalone port-only
VCDs, and a virtual 1-ns clock prevents combinational delta-cycle power from
becoming non-physical.

One full-PE aggregate area scale and one power scale reproduce the PE and 4x4
array anchors exactly, but do not transfer to the component breakdown:

| Component | Area error | Power error |
|---|---:|---:|
| Config network | 119.28% | 305.12% |
| Data network | 98.19% | 96.42% |
| Control logic | 93.65% | 99.72% |
| Tag buffer | 62.28% | 35.06% |
| Register file | 295.30% | 832.09% |
| FU SIMD32 | 13.06% | 82.55% |
| Reduced SIMD8 design | 224.52% | 363.78% |

Only FU area transfers. The RF is modeled as a 16-entry flip-flop/mux array and
is much too large, while the one-input stateless data router and combinational
tag arbiter are far smaller than the paper rows. The reduced FU also retains
unreachable full-lane divider/exp logic through its helper module, so feature
removal does not produce the reported reduction. H200 must change those RTL
structures before another numerical comparison; per-row scale factors remain
prohibited.
