# H92 protocol: timed non-Attention Figure 21 paths

H92 implements the nine missing timed path families from H91:

- structured QKV, output, FFN1, and FFN2;
- dense QKV, output, FFN1, and FFN2;
- inferred elementwise work.

For projections, structured work is split across five B=32 tags and dense work
uses one GEMM tag. The H91 analytical FMA-equivalent convention is preserved;
no additional butterfly arithmetic is invented. Elementwise paths use H91's
MUL/ADD/FRSQRT/FEXP/FDIV/SHUFFLE counts in a fixed sequential FU order.

Each component's full per-lane compute trips and 64-byte load/store packet
counts are divided by their greatest common divisor to form one exact unit
topology. Pre-execution formula review rejects a shared topology across N:
weight loads are layer-constant while activation traffic scales with N. H92
therefore materializes all 45 shape×family unit topologies independently. One
load-complete event authorizes the first compute block, each FU/stage emits one
completion event, and the final event authorizes stores. Four H69 column SRAM
ports provide real H66 timing.

For every family q=4/8 fit affine cycles and q=16/32 are held out. Support
requires all 18 holdouts within 5%, exact instruction/FU/load/store scaling,
byte-identical double runs, and exact full work/bytes for all 45 paths. No
Figure 21 target is read.

The immutable output is
`artifacts/results/fig21-timed-paths-run097.json`.
