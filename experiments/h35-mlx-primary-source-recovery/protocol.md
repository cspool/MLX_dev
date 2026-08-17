# H35 protocol: first-party primary-source recovery for MLX lineage

## Question and hypothesis

The first-party document URLs already exposed by H34's immutable Crossref
records, together with explicit HTML content negotiation on the registered UCAS
institutional page, anonymously recover at least one feature-eligible primary
text for DFU-E or M2-DFU. Any recovered text may then resolve H34's separate
architecture-family or exact-parent gates; source recovery alone never proves
lineage.

H35 is an access-recovery experiment, not a new candidate search. It binds
run039 and does not query a search engine, discover a new architecture by
similarity, or treat an index abstract as technical evidence.

## Frozen inputs and routes

Bind the complete supplied MLX manuscript and H34 run039 by byte count and
SHA-256. Run039 has `audit_integrity=true`, source commit
`41655f7c6244fde919626799ab5b3f48da93a9e6`, and an inconclusive family result.

Request exactly the config's frozen routes:

1. the literal Crossref IEEE PDF URLs for DFU-E, the ICCD transfer paper,
   SimICT, and DSAGEN plus deterministic HTTPS scheme upgrades of those same
   URLs;
2. ACM's DOI landing, abstract, full-HTML, PDF, and ePDF representations for
   DFGAS DOI `10.1145/3773768`;
3. DOI landing pages for DFU-E, the transfer paper, and SimICT; and
4. Wenming Li's already registered UCAS page with an explicit HTML-only
   `Accept` header.

These are representation/follow-up routes from frozen identifiers, not fresh
search terms. Record request/final URL, exact headers, retrieval time, status,
content type/length, bytes, SHA-256, redirects, and errors. Retry only transient
failures, at most three attempts. Never download more than 25 MiB, execute
payloads, or store copyrighted full texts in the repository.

## Source qualification

A feature-eligible paper response must be anonymous, non-truncated, parse as
PDF or substantive publisher HTML, and contain the candidate's exact registered
title or alias. An official institutional HTML response may establish
bibliographic identity, chronology, projects, or an explicit relationship, but
publication-list adjacency and shared authorship remain ineligible feature
evidence. Crossref/OpenAlex/Semantic metadata and abstracts remain identity or
discovery context only.

For every eligible response, run the unchanged H34 feature expressions and
retain a source ID plus no more than 20 normalized words around each match.
Merge only H34's already qualified chronology/ownership observations; do not
import index-abstract technical claims.

## Unchanged lineage gates

- Exact parent requires an explicit primary link or at least four matching
  parent-hardware fingerprints including two exact numeric values, with no
  exact-chip conflict. The unbound mesh dimension does not count.
- Family attribution requires an explicit family/derivation statement, or
  prior tape-out/ownership plus at least three high-specificity matches spanning
  both software-interface and execution-substrate classes, with no explicit
  family conflict.
- Generic terminology and authorship score zero. Exact-chip process differences
  do not by themselves contradict an architecture family.
- SimICT citation ancestry, exact chip identity, family attribution, and code
  provenance remain separate verdicts.

## Decision and stopping rule

H35 source recovery is **supported** only if DFU-E or M2-DFU obtains at least
one newly qualified feature-eligible primary text. It is **rejected** if every
frozen DFU-E/M2-DFU paper route fails qualification while all audit-integrity
checks pass. It is **inconclusive** if the route set is not completely executed
or a required local/source binding fails.

Report the H34 family and exact-parent gates independently after adding the new
eligible primary text. A recovered paper that does not pass a lineage gate is
not negative lineage evidence. If no DFU-E/M2-DFU text is recovered, stop this
route family and classify the origin claim as public-access-limited unless a
future author/publisher release supplies new primary evidence. Do not add a URL,
header variant, candidate, or feature after reading run040 responses.
