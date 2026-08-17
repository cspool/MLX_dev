# Figure 25 after PE-semantics correction

H53 reruns H50's exact arithmetic-expanded 24-config surface after changing
only `pe_dependency_model` to `paper_static`. All instructions, stages, events,
trip counts, memory traffic, FU timing, and case mappings are byte-identical
after removing the registered dependency-model metadata.

Every detailed dsa-gem5 run passes and contains no register-scoreboard or
RF-bank/port stalls. The transfer nevertheless remains rejected: 6/24 cells
are within 10%, MAPE is 18.1%, and maximum error is 46.4%. Paper-static control
substantially improves SWA-W128 but raises long-case butterfly occupancy because
the previous inferred scoreboard serialized work that the paper schedules
statically.

This result confirms that architecture fidelity and numerical curve matching
are separate questions. The corrected PE model is retained even though the old
experimental scoreboard happened to produce a slightly lower aggregate MAPE.
Figure 25 still requires a true roofline-normalized FMA counter rather than the
current global compute-busy proxy.
