# H104 protocol: MLX author-team simulator lineage

## Question

Can the historical publications, patents, institutional pages, and official
software artifacts of MLX's eleven authors identify the accelerator family,
simulation stack, or implementation methodology that most plausibly underlies
the unpublished MLX simulator?

This broadens H34-H36 from a fixed candidate audit to an author-centric survey.
It does not assume that a shared author, institution, or generic dataflow term
establishes derivation.

## Frozen authors and sources

Search all manuscript authors with their stated affiliations: Haibin Wu,
Wenming Li, Zhihua Fan, Zirui Ma, Yuqun Liu, Tengfei Xia, Yanhuan Liu, Kunming
Zhang, Xiaochun Ye, Dongrui Fan, and Jian Weng; ICT/CAS, UCAS, Ricore, and
KAUST are disambiguators.

Use the T1→T2→T3 order:

1. Crossref/OpenAlex/arXiv and publisher records;
2. official institutional bibliographies, author/project pages, patents, and
   official GitHub organizations;
3. Semantic Scholar or search-engine discovery only when primary routes are
   insufficient, with snippets never promoted into technical evidence.

Search exact authors plus accelerator/dataflow/CGRA/processor/simulator/gem5/
cycle-accurate/RTL/FPGA/tapeout/compiler terms. Retain papers outside H34's
fixed candidate list when they predate MLX and have a verified author match.

## Evidence fields

For every deduplicated candidate record:

- title, year, venue, DOI/arXiv/patent ID, authors, affiliation, stable URL;
- exact MLX-author overlap and chronology;
- architecture family and PE/NoC/memory/compiler features;
- explicit simulation method: named framework, cycle/transaction/RTL level,
  configuration source, calibration/validation target, and modeled memory;
- implementation evidence: Verilog/Chisel/HLS, synthesis, FPGA, tapeout, or
  silicon measurement;
- artifact state: official code/repository URL, license, revision, buildability;
- evidence tier and whether the statement is primary, metadata-only, or an
  inference.

The focused candidates SimICT, DFU-E, M2-DFU, DFGAS, the ICCD-2023 transfer
paper, DSAGEN, Assassyn, DACO, and the two registered patents must remain in the
matrix, but the survey may add independently discovered author-matched work.

## Decision rules

- `explicit_lineage`: only an author/publisher/patent primary source states a
  derivation or shared implementation.
- `simulator_reuse_supported`: only a primary source names the simulator/code
  base and connects it to MLX or its stated parent.
- `family_candidate`: chronology plus at least three high-specificity technical
  matches spanning architecture and software/simulation; author overlap alone
  scores zero.
- `engineering_precedent`: a historical work exposes a reusable public method
  or artifact but lacks an MLX derivation link.

The output must preserve unresolved fields and contradictions. It may rank
candidates for second development, but may not label DSAGEN, Assassyn, SimICT,
or another project as MLX's source without the explicit gates above.

The immutable result is
`artifacts/results/mlx-author-simulator-lineage-run109.json`; the narrative is
`literature/mlx-author-simulator-lineage.md`.
