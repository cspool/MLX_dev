# Figure 25 arithmetic-expanded transfer

H50 expands H49's representative operations using source-derived CDC counts,
without reading the Figure 25 surface. BSMM stages use four FMA and two adds per
pair; complex FFT stages use four FMA and six adds; SWA uses 32/128 vector FMA
groups and 4/8 KV-load waves derived from its W/Q tile shapes.

All 24 real-memory runs pass. The expansion improves the numerical transfer
from 0/24 to 8/24 cells within 10% and lowers MAPE from 46.9% to 16.2%. FFT and
BSMM are mostly close, while SWA-W128 and the InternLM2-4K column expose larger
residuals. The maximum error falls from 59.5% to 49.7%.

H50 remains rejected because the registered all-cell criterion fails. The
remaining mismatch cannot be repaired honestly with another arithmetic
multiplier: the simulator reports global compute-pipeline occupancy, whereas
the paper reports FMA throughput normalized to a compute-or-bandwidth roofline.
Further work must add the missing per-PE issued-FMA and byte-based roofline
counter, or obtain author raw counters; target-derived row corrections are not
permitted.
