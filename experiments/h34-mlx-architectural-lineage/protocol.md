# H34 protocol: evidence-bounded MLX architectural-lineage audit

## Question and hypothesis

Primary records identify MLX's unnamed general-purpose taped-out parent at
least at the **architecture-family** level as the ICT DFU-E/M2-DFU line, while
SimICT supplies the explicitly cited simulation ancestry and DSAGEN/Assassyn
remain external semantic relatives. Exact parent-chip identity is a separate,
stricter claim and must remain unresolved unless an explicit link or unique
hardware fingerprints establish it.

H34 addresses origin, not artifact availability. H33 already found no
qualifying exact-paper implementation, and no related paper, patent, or
repository may be relabeled as MLX code.

## Frozen local evidence

Qualify the complete supplied manuscript at 85,855 bytes and SHA-256
`5785eb81b28741a3806ca42749d7b556bbcd2404e1622ae644a32bae2ade7745`.
Before fresh lineage search it states:

- MLX is a profile-driven specialized subset of an unnamed general-purpose
  dataflow design implemented in Verilog RTL and taped out.
- The design practice is 12 nm at 1 GHz; the full point uses SIMD32, a compact
  PE mesh, about 1 TOP/s, a 7.712-mm2 PE array, and 5.846-W array power, while
  the specialized reduced point removes units and narrows SIMD to eight.
- Deployment uses a RISC-V host, dataflow assembly, an LLVM-based C path, and a
  spatial assembler.
- The reduced 256-GOp/s design is tuned in the cycle-accurate simulator at
  citation [36], SimICT.
- The paper cites DFGAS [25], DFU-E [75], and the ICCD 2023 dataflow-transfer
  work [77], but never names any of them as the parent design. It cites DSAGEN
  as related spatial work and does not cite Assassyn.

Bind H33's 5,791-byte browser snapshot
(`3c1760c7ff539451d1875ff54d7415b9473d03d8c5ebfa56341948e0b1d4d605`)
only for the pre-existing M2-DFU/DFU-E/patent leads and its 88,807-byte
corrected report
(`462df17b15a8acdeee8820f59b476a60939a795d9efd69991d0fabfe0eccff09`)
for the no-artifact boundary.

## Frozen candidates and search

Audit SimICT; DFU-E; DFGAS; M2-DFU; *Alleviating Transfer Latency in Dataflow
Accelerator for DSP Applications*; patents CN202510992730.2 and
CN202511751816.2; and DSAGEN/Assassyn as negative controls. Execute every exact
title, identifier, and registered fingerprint query in the config. Follow DOI,
publisher, author-hosted full text, institutional bibliography, patent-family,
inventor, assignee, priority, citation, and supplemental links from those
records. Do not introduce a new candidate because it happens to resemble an
observed result.

Use primary sources for substantive features: the target manuscript,
publisher papers/abstracts, author-hosted paper copies, institutional records,
and official patent records. Crossref/OpenAlex may establish bibliographic
identity. Search snippets and third-party indexes are discovery-only; generated
summaries, citation mirrors, and unsourced reposts cannot satisfy a feature or
lineage gate.

For each retained primary response, record URL, final URL, retrieval time,
status, bytes, SHA-256, stable identifier, title, authors/inventors, date or
priority, venue/assignee, and access limitation. Do not execute related code or
download any payload over 25 MiB.

## Pre-registered evidence matrix

Compare only these feature classes, frozen from MLX before search:

1. **Chronology and ownership:** publication/priority predates MLX; ICT/CAS or
   Ricore ownership; overlapping hardware authors. Authorship alone scores
   nothing.
2. **Parent-hardware fingerprints:** taped-out Verilog design, 12 nm, 1 GHz,
   SIMD32, mesh dimensions, approximately 1 TOP/s, 7.712-mm2 PE-array area,
   and 5.846-W PE-array power. Record exact values and contradictions.
3. **Software interface:** RISC-V host, per-PE/dataflow assembly, LLVM compiler,
   spatial assembler, or equivalent binary-header configuration path.
4. **Execution substrate:** programmable spatial dataflow PEs, heterogeneous
   functional units, decoupled load/compute/transfer, explicit operand routing,
   instruction/data reuse, or compact instruction hierarchy.
5. **MLX-specific delta:** CDC folding, tagged blocks, bounded active-layer
   windows, skip-hop links, semantic FFT compression, and hierarchical BSMM.
   A parent is expected to lack some or all of these specialization features;
   their absence is not a contradiction unless the source claims identity.
6. **Simulation:** explicit SimICT use or derivation versus merely citing the
   framework.

Every match must carry a source identifier plus a short paraphrase or a
copyright-compliant excerpt. Generic words such as `dataflow`, `reuse`,
`RISC-V`, or `mesh` never count alone. Negative evidence means an inspected
source explicitly contradicts a field; silence is `not reported`.

## Decision gates

Report three distinct conclusions:

- **Exact parent-chip identity** is supported only by an explicit primary
  statement linking MLX to the candidate, or by a taped-out candidate matching
  at least four parent-hardware fingerprint fields including two exact numeric
  values, with no conflicting value.
- **Architecture-family attribution** is supported by an explicit family/
  derivation statement, or by prior tape-out/ownership plus at least three
  high-specificity matches across the software-interface and execution-
  substrate classes, with no material contradiction. Common authorship and
  generic terminology cannot be among the three.
- **Simulator ancestry** is supported only to the level stated by MLX or a
  primary candidate source. MLX's citation [36] can establish SimICT as the
  referenced simulation framework, but not source-code reuse or the hardware
  parent.

H34 is supported if the family-level hypothesis passes. It is rejected if
complete primary evidence identifies a different family or materially
contradicts DFU-E/M2-DFU. It is inconclusive if access gaps prevent either
decision. Regardless of H34 status, set exact identity and code provenance to
their own gates; never promote a family result into either claim.

## Stopping rule

Run one frozen audit. Do not select features after reading sources, infer an
exact chip from authorship, or search for another candidate from a residual.
If family attribution passes but exact identity does not, stop at that
two-level conclusion. A future exact claim requires an author statement,
artifact, patent family link, or independently matching hardware record.
