# H202 result: area nearly converged, power activity remains open

Run207 is rejected with `audit_integrity=true` and 8/10 gates.

Clock gating and locked dimensions bring 8/9 area values within 15%. Data,
control, tag, RF, FU and reduced errors are 2.18%, 9.40%, 4.71%, 5.57%, 2.89%
and 8.27%. Config is the only miss at 22.38%; reducing its registered depth
from 22 to 18 follows the measured linear cell slope.

Reduced power passes at 9.49%, but full component power does not. The paper
states post-silicon full power without disclosing activity vectors or domain
duty. H202's measured dynamic distribution overweights data/control and
underweights config/tag/FU. A final target-exposed activity calibration is
required; it must scale measured internal+switching power while retaining
Liberty leakage, not replace RTL power with copied table values.
