# H33 corrected protocol: source-diverse public-artifact audit

## Correction scope

`run037` is immutable and inconclusive. Its only false integrity gate was an
extra count requiring three exact matches among directly fetched official
pages. That count conflated source quantity with source diversity and failed
when the UCAS profile endpoint returned a stable, malformed JSON/SPA payload.
It was not one of H33's five artifact-qualification gates.

Run038 corrects only this audit-integrity defect before making fresh requests.
It keeps the exact H33 hypothesis, paper identity, browser snapshot, 12 web
queries, six repository queries, endpoint set, four critical domains, five
candidate gates, cutoff, and noise exclusions unchanged. No run037 response,
candidate count, repository name, or residual may add or remove a query.

Bind the excluded report at 88,731 bytes and SHA-256
`e7b1c50dab982d41d934203bb8e7929b8a1b89a87eb46bf71b3d6d824defb6ad`.
It must identify run037, source commit
`3b833d73a4b174cda59f5b4772359373d13ed55c`,
`audit_integrity=false`, `hypothesis_status=inconclusive`, and zero qualifying
artifacts. Run038 writes a distinct artifact and never overwrites run037.

## Frozen identity-integrity gate

Require four source-diverse facts independently:

1. Crossref matches the exact DOI, complete normalized title, and at least one
   registered author.
2. OpenAlex matches the same exact paper identity.
3. At least one fetched official venue-program page matches the exact title
   and at least one author.
4. At least one fetched author-controlled page matches the exact title and at
   least one author.

The ISCA program is the registered venue class. Jian Weng's and Wenming Li's
pages are the registered author-controlled classes; either may satisfy the
author class, while every URL is still fetched and hashed. DOI/publisher
challenge pages and the artifact-policy page are recorded but are not counted
as venue or author identity.

Required transports receive at most three attempts for transient timeout,
HTTP 408/425/429, or 5xx failures. Optional supplements receive one attempt;
their completed registered web-index query plus the direct attempt constitutes
coverage when no candidate was surfaced. All failures remain explicit.

## Candidate acceptance and stopping rule

The original conjunctive gates remain exact: paper identity, anonymous
retrieval, stable revision/release/DOI/hash, files in at least one critical
domain, recorded license/dependency status, and exclusion of Apple-MLX,
mirrors, summaries, citations, and title-only media. A surfaced exact GitLab or
Hugging Face identity that is not fully evaluated makes the audit
inconclusive. Crossref artifact relations or OpenAlex repository full text do
the same.

H33 is supported only if at least one candidate passes every gate. It is
rejected only when every corrected integrity check passes and no candidate
qualifies. Otherwise run038 is inconclusive. Execute once; do not tune a
query, timeout, source class, or gate after responses arrive, and do not run
newly discovered code or download large payloads.
