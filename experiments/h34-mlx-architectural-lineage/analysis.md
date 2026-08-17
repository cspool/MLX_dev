# H34 analysis: architectural lineage remains bounded by primary-text access

## Immutable result

- Run: `run_039`
- Source commit: `41655f7c6244fde919626799ab5b3f48da93a9e6`
- Result: `artifacts/results/mlx-architectural-lineage-run039.json`
- Result bytes: 211,568
- Result SHA-256: `adce1465fcf91b06b9b312e9b5db59630d3ae7d261236201708a4a60d91756af`
- Audit integrity: `true`
- Hypothesis status: `inconclusive`
- Validation eligible: `false`

The formal run issued 34 bounded requests in the registered T1-primary -> T1
metadata -> T2 cross-check order. Thirty-two transports succeeded and returned
11,032,394 bytes; no payload exceeded or was blocked by the 25-MiB cap. All
candidate, query, feature-class, source-order, DOI-deduplication, and
primary-evidence gates pass.

## Bibliographic and ownership result

Crossref and OpenAlex independently identify five DOI-bound lineage records;
Semantic Scholar independently confirms all five:

| Candidate | DOI | Year | H34 role |
|---|---|---:|---|
| SimICT | `10.1109/ISLPED.2013.6629308` | 2013 | cited simulator framework |
| DFU-E | `10.1109/TPDS.2025.3555329` | 2025 | parent-family candidate |
| DFGAS | `10.1145/3773768` | 2025 | scheduling-lineage candidate |
| ICCD transfer paper | `10.1109/ICCD58817.2023.00073` | 2023 | interconnect-lineage candidate |
| DSAGEN | `10.1109/ISCA45697.2020.00032` | 2020 | external control |

OpenAlex affiliations place DFU-E, DFGAS, the transfer paper, and SimICT in
the ICT/CAS research line. DFU-E overlaps seven MLX authors, DFGAS six, and the
transfer paper seven. These observations establish chronology, institutional
association, and bibliographic identity only; the protocol assigns zero
family-gate credit to authorship.

M2-DFU has no Crossref/OpenAlex identity or DOI at the cutoff. The frozen
browser discovery observed it only on Wenming Li's official UCAS bibliography
as a just-accepted 2026 TOCS paper. Because the formal UCAS request negotiated
the site's known 1.9-MiB JSON representation rather than its HTML page, that
browser observation remains discovery-only in run039.

## Primary-text access boundary

The identity channels succeeded, but the feature-eligible full texts needed by
the pre-registered gates did not:

- IEEE's DFU-E and SimICT stamp endpoints returned HTTP 202 with zero body and
  an HTML document redirect rather than a PDF.
- ACM's DFGAS PDF endpoint returned HTTP 403.
- M2-DFU has neither DOI nor accessible full text.
- The ICCD and ISLPED official program/proceedings PDFs were retrieved and
  parsed, but they are bibliography/program records, not substantive paper
  text.
- Crossref/OpenAlex abstracts describe DFU-E's multi-layer parallelism and
  custom PE/memory/NoC/software stack, DFGAS's block-level scheduling, and the
  ICCD paper's forwarding NoC and bandwidth reuse. The protocol explicitly
  permits those indexes to establish identity but forbids using index text as
  a primary technical-feature claim, so none is promoted into the matrix.

This produces explicit primary-full-text access gaps for DFU-E, DFGAS,
M2-DFU, and the ICCD transfer paper. Silence under those gaps is
`not_reported`, never a contradiction.

## Decision gates

### Architecture family

Neither DFU-E nor M2-DFU has an explicit primary derivation statement. DFU-E
passes the prior ICT/chronology condition, but has zero primary-text
software-interface or execution-substrate matches; M2-DFU lacks even a formal
metadata/full-text record. Neither can meet the required three
high-specificity matches spanning both classes. Architecture-family status is
therefore **inconclusive**, not rejected.

### Exact parent chip

No candidate has an explicit exact-parent link, four matching parent-hardware
fingerprints, or two exact numeric fingerprints in eligible primary text.
The exact taped-out parent remains **unresolved**. A family inference cannot be
promoted to chip identity.

### Simulator ancestry

The supplied primary manuscript contains both registered checks: the reduced
design is “tuned in our simulator [36]”, and reference [36] resolves to
SimICT. SimICT is therefore **supported at citation level only**. Neither
SimICT source-code reuse nor SimICT as the hardware parent is supported.

### Code provenance

H33's qualified zero-artifact result remains in force and no primary record
links any candidate repository to MLX. Code/RTL/simulator provenance is
**not supported**.

## Interpretation and next test

The strongest current origin statement is deliberately asymmetric: MLX is an
ICT/CAS work embedded in the same dated publication line as DFU-E/DFGAS and
the ICCD transfer mechanism, and it cites SimICT for simulator ancestry, but
the evidence does not yet identify the general-purpose parent family or exact
chip. The result does not license labeling DSAGEN, Assassyn, DFU-E, or M2-DFU
as MLX code.

Run039 itself exposes a non-target-derived next access test: Crossref records
publisher PDF URLs for DFU-E, SimICT, DSAGEN, and the transfer paper, while a
text/html-only request can avoid the UCAS JSON representation already seen.
A new protocol may test only these record-derived first-party routes and
content negotiation. It must retain all H34 lineage gates and cannot count
Crossref/OpenAlex technical abstracts as primary feature evidence.
