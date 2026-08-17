# H62 protocol: source-derived Figure 10 mapping

## Objective

Replace H44's pair-aggregate workload mapping with a target-free compiler that
follows the loop nest and tagged block printed in Figure 10. H62 is a mechanism
and conservation test; it reads no Figure 22 utilization values.

## Paper-bound mapping

For width N on the 4x4 mesh:

- `i2=0..15` is spatially unrolled over 16 PEs;
- `i1=0..3` is time-multiplexed locally;
- `i0=0..N/64-1` is sequencer-driven;
- each PE therefore produces N/16 outputs per butterfly layer;
- the 64-output closed set has six radix-2 layers.

The BSMM block follows Fig. 10(d): two loads, `mul`, `fma`, then a terminal
operation. Non-boundary layers terminate with xfer. Every sixth layer and the
final layer terminate with a scratchpad store for the paper's inter-CDC I/O
shuffle. The next CDC begins with external scratchpad loads; loads within a CDC
use the load pipeline but consume array-local values. This requires an explicit
per-instruction `memory_external` declaration; old configs default to true.

Routing repeats within each 64-output CDC. Strides 1/2 map horizontally,
strides 4/8 map vertically by one/two hops as stated in the paper, and strides
16/32 are time-multiplexed self hops. FFT uses the same loop/routing skeleton
and its already registered `mul, add, add` complex-butterfly abstraction.

The Figure 22/23 simulator baseline records the reduced design's SIMD8. Batch
SIMD is omitted from the Figure 10 drawing and does not multiply instruction
instances in the timing graph. Each FP16 vector request is therefore 16 bytes.
Every active-window block footprint must remain within the disclosed 32
instructions per PE.

## Gates

- all 16 Figure 22 shapes compile twice byte-identically;
- output, operation, event, route, memory, and instruction counts satisfy the
  formulas above;
- old configs replay byte-identically after the new optional memory field;
- fixed-memory and dsa-gem5 scratchpad smokes complete with exact counts;
- compiler and runner contain no paper target input.

No numerical Figure 22 comparison is permitted in H62. That held-out transfer
is a subsequent experiment.

The immutable output is `artifacts/results/fig10-mapping-run067.json`.
