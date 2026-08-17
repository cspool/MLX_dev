# H33 protocol: post-acceptance MLX public-artifact discovery

## Hypothesis and purpose

At least one publicly retrievable, author- or venue-linked artifact for the
exact MLX paper now exposes a reproduction-critical domain that was missing
from the supplied manuscript: architecture simulator/RTL/mapping, structured
model/operator/training configuration, exact dataset/evaluator/checkpoint
manifest, or native benchmark traces.

H33 is a time-bounded source-discovery audit, not a performance experiment.
It follows H32 because seed and session-cap variants are exhausted. A positive
result may register a later executable hypothesis; H33 itself cannot validate
numeric paper bars. A negative result means only that the frozen public
channels contain no qualifying artifact as of 2026-08-17.

## Frozen identity

Require the complete supplied manuscript to match 85,855 bytes and SHA-256
`5785eb81b28741a3806ca42749d7b556bbcd2404e1622ae644a32bae2ade7745`.
Freeze the exact title, all 11 authors, ICT/CAS, UCAS, Ricore, and KAUST
affiliations, and the published ICT/Ricore/KAUST email handles from the local
source before external search.

The two lineage clues are also frozen before search: the manuscript calls the
chip a specialized subset of a general-purpose taped-out dataflow design, and
it cites SimICT at the reduced-design simulator comparison. Similar names,
shared authors, or architectural vocabulary alone never establish lineage.

## Search channels and queries

Execute every exact-title and title-fragment query in the registered config
against a general web index and the registered domain-restricted views. Query
the public APIs or search pages for GitHub, GitLab, Gitee, Zenodo, Hugging Face
models/datasets/spaces, ModelScope, Crossref, OpenAlex, DBLP, arXiv, ACM/IEEE,
the ISCA 2026 program/artifact pages, ICT/CAS, UCAS, Ricore, and KAUST. Follow
all direct code, data, supplemental, DOI, project, and author-homepage links
from an exact paper record.

For each machine-readable endpoint, record the request URL, retrieval time,
HTTP status, final URL, response byte count and SHA-256, result count, and
stable identifiers. For browser-only results, freeze query, rank, displayed
title, final URL, and source classification. Search result snippets are
discovery evidence only and cannot qualify an artifact.

Do not add new residual-motivated model, decoder, simulator, or training
queries after inspecting results. Spelling-normalized author searches and the
exact registered title fragments are allowed; generic `MLX` hits must be
rejected unless an exact-paper identity link exists.

## Qualification and acceptance

A candidate qualifies only when all of the following hold:

1. An author-controlled page, venue record, DOI metadata, or repository
   metadata explicitly identifies the exact title and at least one paper
   author.
2. The artifact is anonymously retrievable without credentials and has a
   stable repository revision, release identifier, DOI, or content hashes.
3. Inspection identifies concrete files in at least one critical domain:
   simulator/RTL/mapping; structured operators/model/training; exact
   dataset/evaluator/checkpoint; or native trace/raw measurement.
4. License and dependency status are recorded. Source without a visible
   license may be inspected and hashed but not copied or silently reused.
5. The candidate is not an unrelated Apple-MLX project, paper mirror,
   bibliography-only page, generated summary, or a repository that merely
   cites MLX.

H33 is supported if at least one candidate passes all five gates. It is
rejected if all frozen channels complete and none passes. If a channel is
unavailable, report it separately; H33 may remain inconclusive if the missing
channel is necessary to identify a surfaced candidate.

## Stopping rule

Do not execute newly discovered code or download large model/data payloads in
H33. First pin its revision/metadata and write a new protocol scoped to the
unlocked domain. If no artifact qualifies, stop speculative provenance and
retain SimICT, DSAGEN, Assassyn, DFU-E, DFGAS, and the taped-out-design family
only under their existing evidence labels.
