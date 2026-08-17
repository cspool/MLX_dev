# H33 run037 audit: inconclusive because of an identity-gate defect

`run037` is retained but excluded as an H33 hypothesis result. It found no
qualifying artifact, but its composite audit-integrity flag is false, so the
zero count cannot by itself reject H33.

The immutable report is
`artifacts/results/mlx-public-artifact-discovery-run037.json` (88,731 bytes,
SHA-256
`e7b1c50dab982d41d934203bb8e7929b8a1b89a87eb46bf71b3d6d824defb6ad`).
It records source/runtime commit
`3b833d73a4b174cda59f5b4772359373d13ed55c` and 184.055 seconds of wall
time.

## Valid observations

- Crossref and OpenAlex both match the exact title, DOI, and all 11 authors.
  Crossref exposes only an IEEE paper PDF link and no artifact relation;
  OpenAlex reports closed access and no repository full text or dataset.
- The official ISCA 2026 program and Jian Weng's author-controlled page both
  identify the exact paper. Semantic Scholar independently matches the DOI,
  title, and all authors and reports no open-access PDF.
- All six GitHub repository queries, all six GitLab project queries, all three
  Hugging Face catalog queries, Zenodo, and arXiv return zero exact-title
  candidates. The 25 `were` repositories and five Synthesys-Lab repositories
  were all inspected; none passes exact-paper identity.
- No GitLab or Hugging Face exact identity remains unresolved, and every
  candidate gate is replayable. DBLP timed out after three attempts and the
  Gitee search redirect returned HTTP 405; both are explicitly recorded, with
  the corresponding frozen web-index queries already completed.

## Why the run is inconclusive

The runner imposed an extra integrity check requiring three exact identity
matches among fetched official pages. Only the ISCA program and Jian Weng page
pass. The UCAS profile endpoint returns a stable 1,935,744-byte
`application/json` response that is syntactically truncated and contains no
MLX title in its fetched representation, even though the frozen browser-index
snapshot exposed the publication. DOI and IEEE requests resolve to empty
HTTP-202 challenge responses, as already expected.

The raw count of three official pages was not a paper-artifact qualification
gate and does not measure source diversity. A corrected run must freeze, before
fresh requests, explicit independent identity requirements: exact Crossref and
OpenAlex records, at least one exact official venue page, and at least one exact
author-controlled page. It must retain all original artifact gates, queries,
and unavailable-channel reporting.
