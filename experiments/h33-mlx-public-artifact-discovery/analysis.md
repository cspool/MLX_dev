# H33 result: no qualifying public MLX artifact was found

## Outcome

Corrected H33 run038 is **rejected**. As of the frozen 2026-08-17 cutoff,
none of the registered public channels exposes an anonymously retrievable,
author- or venue-linked artifact for the exact MLX paper that passes every
identity, stability, critical-domain, license/dependency, and noise-exclusion
gate.

The report is
`artifacts/results/mlx-public-artifact-discovery-run038.json` (88,807 bytes,
SHA-256
`462df17b15a8acdeee8820f59b476a60939a795d9efd69991d0fabfe0eccff09`).
It records source/runtime commit
`31a9be8142250956459aade703261779e6cb3cde`, 31 endpoint requests,
2,386,935 retained response bytes, and 47.653 seconds of wall time. Every
preflight, channel-coverage, candidate-replay, and source-integrity check is
true; `audit_integrity=true`, the qualifying count is zero, and the registered
hypothesis status is `rejected`.

Run037 remains immutable and excluded. Its only false integrity gate was the
defective raw count of official-page identities documented in
`analysis-run037.md`. Run038 was independently pre-registered and fetched
fresh responses under source-diverse identity classes before reaching this
decision.

## Exact-paper identity and metadata

Five independent parsed sources identify the paper:

- Crossref DOI `10.1109/ISCA66397.2026.00017` matches the complete normalized
  title and all 11 authors. Its only content link is the IEEE paper PDF for
  similarity checking; its relation map and artifact-link set are empty.
- OpenAlex work `W7172417674` matches the same identity and reports closed
  access, no repository full text, and no dataset.
- Semantic Scholar paper `1612f4dc2230af2b47073f84fdab2dd0121f5976`
  matches the DOI, title, and all authors and exposes no open-access PDF.
- The official ISCA 2026 program is the passing venue identity.
- Jian Weng's author-controlled page is the passing author identity. Its MLX
  entry has no paper, code, data, or supplemental link.

The ISCA artifact-evaluation page publishes the venue policy but no per-paper
artifact list. The DOI and IEEE landing requests return empty HTTP-202
challenge responses and are not promoted to identity or artifact evidence.
The UCAS direct endpoint returns the stable malformed JSON representation
described in run037; the browser-index snapshot remains discovery-only.

## Code, model, data, and archive search

All six GitHub repository searches and all six GitLab project searches return
zero results. The complete public listings contain 25 repositories under the
`were` profile and five under Synthesys-Lab; all 30 repository descriptions
and attempted README retrievals fail exact-paper identity, so none enters the
artifact gate. All Hugging Face model, dataset, and Space searches return
empty arrays. Zenodo and arXiv each return zero exact-title records. ModelScope
returns a search page with no exact title. The frozen general-web and
domain-restricted searches likewise surface no qualifying candidate and
explicitly reject unrelated Apple MLX projects, mirrors, title-only media,
generated summaries, and repositories that merely cite the work.

DBLP times out on its single optional request and Gitee redirects to an
HTTP-405 search response. Both failures are retained in the report; their
registered exact-title web-index queries completed without surfacing a
candidate. No required transport failed, no response was truncated, and no
exact GitLab or Hugging Face lead remains unevaluated.

## Consequence and scope

H33 unlocks none of the four blocked domains: architecture simulator/RTL/
mapping, structured model/operator/training configuration, exact dataset/
evaluator/checkpoint manifest, or native trace/raw measurement. H29's
compressed-model identifiability stop, H30-H32's Ada evaluator gap, and the
unpublished architecture-timing boundary therefore remain in force. No
source-free parameter, seed, cache, prompt, or scheduling sweep is justified
by this negative audit.

This is a timestamped public-availability conclusion, not proof that private
or future artifacts do not exist. It is validation-ineligible and reproduces
no numeric paper bar. M2-DFU, DFU-E, patents, SimICT, DSAGEN, Assassyn, and
DFGAS remain possible literature-lineage evidence only; common authors or
vocabulary cannot establish that any one is the MLX implementation.
