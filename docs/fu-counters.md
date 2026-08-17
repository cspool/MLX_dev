# Physical FU-class utilization counters

H71 replaces the historical global-any-compute proxy with a physical counter:
one productive PE-cycle per distinct active `(PE, FU class)`. Multiple
instructions on the same PE/FU count once, while heterogeneous FU classes may
overlap.

All 24 Figure 25 operator/case configs execute twice under fixed and Fig.9
column-port memory. Every backend transform, instruction, pipeline, event,
route, memory, FU-class, capacity, and replay check passes without target
access.

The counter exposes why the old metric was misleading. Representative
column-port FMA utilizations are:

- FFT-CMP: 12.2–34.3%;
- QKV-BSMM families: 18.2–58.9%;
- SWA-W128: 21.5–42.2%;
- SWA-W256: 23.7–44.8%.

Global compute could remain active while only a subset of physical PE FMA units
were occupied. Figure 25 must therefore be re-audited with the FMA counter; the
previous 6/24 proxy result is not a roofline/FMA reproduction.

The immutable mechanism result is
`artifacts/results/fu-counters-run076.json`.
