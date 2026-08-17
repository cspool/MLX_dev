# Paper-derived MLX PE contract

The MLX PE is programmable, but describing it as GPU-SM-like is misleading.
Fig. 9(c,d) shows a specialized spatial PE with a register file, tagged
instruction buffers, loop unit, bookkeeping entries, instruction scheduler,
configuration/data switches, and independent xfer/load/store/compute paths.
The compute path selects heterogeneous arithmetic units.

Sec. IV-C states that deterministic instructions inside a folded layer are
statically ordered. Hardware arbitrates only the frontier instructions of a
small active-tag window, prioritizing the smaller tag when blocks contend for
the same pipeline. The design explicitly avoids fine-grained instruction-level
hazard tracking and large dependency tables. Cross-layer readiness is carried
by tag events and explicit xfers.

Accordingly, `paper_static` models:

- one ordered frontier and at most one in-flight instruction per block;
- tag/event readiness and lower-tag arbitration;
- per-PE xfer/load/store/compute pipeline latency and initiation interval;
- heterogeneous compute-FU latency/II;
- skip-hop links, active-window limits, and memory backpressure/completion.

It does not impose dynamic register scoreboards, WAW arbitration between tags,
RF-bank conflicts, or RF-port stalls. The earlier model remains available as
`scoreboard_experimental` only for historical replay. GPGPU-Sim is completely
separate and models the Xavier/Orin/RTX comparison devices, including their
actual warp/SIMT/CTA machinery.
