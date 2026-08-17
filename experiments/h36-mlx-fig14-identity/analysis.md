# H36 analysis: supplied Fig. 14 contains no legible origin identifier

## Immutable result

- Run: `run_041`
- Source commit: `c08841d0c6039b8dedbcbf9d6660e1ed427e251a`
- Observation: `artifacts/source-snapshots/mlx-fig14-visual-observation-run041.yaml`
- Observation bytes: 1,114
- Observation SHA-256: `087a7ab72771eb89f98f4304bb251e57cb237298bc4d5ccef0f335095b9dd9e4`
- Result: `artifacts/results/mlx-fig14-identity-run041.json`
- Result bytes: 5,058
- Result SHA-256: `930d6d7857689fd2d5a16301cca1a89e1e4fd7b56d59010c029503427f904549`
- Audit integrity: `true`
- H36 status: `rejected`

The visual pass inspected only the frozen 16,309-byte, 266x213 RGB JPEG at
original detail. The file, complete paper, H35 result, protocol, observation,
and source-commit bindings all pass. No second view, alternate rendering,
upscaling, sharpening, OCR sweep, or candidate image was used.

## Observation and gates

The raster is inspectable. It shows cyan outlines and traces, repeated outlined
interior regions, a cyan upper band, and small colored markings near the upper
left. No alphanumeric string can be transcribed reliably at the available
detail. The small upper-left and perimeter/interior details are therefore
recorded as too small, not guessed.

The frozen decision counts are:

- clear non-generic chip/project/family identifiers: 0;
- clear numeric parent-hardware values: 0;
- registered identifier labels: 0; and
- hardware-only exact-parent candidate labels: 0.

Neither the one-identifier path nor the two-numeric-value path passes. Because
the exact raster opens and the observation schema passes, this is a rejection,
not an inconclusive image-access result.

## Provenance boundary

Fig. 14 supplies no explicit text that changes H34/H35:

- architecture-family attribution remains `inconclusive`;
- exact taped-out parent identity remains `unresolved`;
- SimICT remains supported only as the framework cited by MLX; and
- candidate simulator/code/RTL provenance remains `not_supported`.

The layout cannot be compared to candidate floorplans as identity evidence.
The strongest evidence-bounded statement remains that MLX is an ICT/CAS work
in the same documented publication line as DFU-E, M2-DFU, DFGAS, and the ICCD
transfer paper, while no public primary record establishes architectural or
code derivation from any of them.

## Stopping rule

Close the supplied-figure origin route. Together, H33-H36 exhaust the frozen
public-artifact searches, record-derived publisher representations, accessible
lineage records, and the supplied raster's explicit-text channel. Do not try
layout matching, image enhancement, OCR parameter sweeps, or more URL/header
variants. Only genuinely new primary evidence—such as an author statement,
substantive full text, or exact-paper artifact—can reopen provenance.
