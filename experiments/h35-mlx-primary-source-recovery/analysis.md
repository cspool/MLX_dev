# H35 analysis: first-party representation recovery is rejected

## Immutable result

- Run: `run_040`
- Source commit: `e5422fded921f06ee7bf9c6072dc4b60523aac2e`
- Result: `artifacts/results/mlx-primary-source-recovery-run040.json`
- Result bytes: 84,348
- Result SHA-256: `49944b8e0e66a6835a88f26784fca1552e8770a779aed06da31078e1c9e81ef8`
- Audit integrity: `true`
- H35 source-recovery status: `rejected`

All 17 frozen routes were attempted with their exact Accept/User-Agent headers.
Twelve transports succeeded and returned 510,039 bytes; no payload hit the
25-MiB limit. Every local binding, route-coverage, header, feature-class,
institutional-source exclusion, and inference-integrity check passes.

## Route outcomes

- All five ACM DFGAS representations (landing, abstract, full HTML, PDF, and
  ePDF) return HTTP 403.
- The literal HTTP and deterministic HTTPS versions of all four Crossref IEEE
  PDF URLs redirect to `xplorestaging.ieee.org/document/...` HTML pages. They
  return 47-56 kB of HTML, not a PDF, and contain no exact paper identity in
  parsable text.
- The DFU-E, transfer-paper, and SimICT DOI landings reach IEEE HTTP-202 shells
  with only 26 normalized words. They are neither substantive nor
  feature-eligible.
- Explicit HTML-only negotiation succeeds on the UCAS page: 74,572 response
  bytes, 60,242 extracted characters, and 2,995 normalized words. It exactly
  identifies DFU-E, M2-DFU (including the institutional spelling alias), and
  the ICCD transfer paper. Per protocol, this publication bibliography remains
  identity/chronology evidence only and never contributes architectural
  features.

No DFU-E or M2-DFU route yields a qualified feature-eligible primary text, so
the registered minimum of one is not met.

## Lineage verdicts

- **Architecture family:** remains `inconclusive`. The formal UCAS HTML result
  strengthens institutional bibliographic identity but supplies no explicit
  MLX derivation and no high-specificity software-plus-substrate feature set.
- **Exact parent chip:** remains `unresolved`; there is no explicit link or
  qualifying hardware-fingerprint set.
- **Simulator ancestry:** H34's `supported_at_citation_level` result is
  unchanged. Source-code reuse remains unsupported.
- **Code/RTL provenance:** remains `not_supported`.

The failed first-party representations are an access result, not negative
architectural evidence. Publisher denial, document shells, and an
institutional bibliography cannot show that DFU-E/M2-DFU are unrelated to
MLX; they only prevent a public provenance claim at this cutoff.

## Stopping rule and next source class

Stop HTTP/HTTPS, Accept-header, DOI-path, and publisher-representation variants
for these records. Additional variants would be post-response route search and
would violate H35. The strongest evidence-bounded origin statement remains:
MLX is an ICT/CAS work in the same documented publication line as DFU-E,
M2-DFU, DFGAS, and the transfer paper, while SimICT is citation [36]'s
framework; neither the architecture family, exact taped-out chip, nor code
provenance is publicly established.

One untested primary source already supplied by the user remains outside the
failed web-route class: the raster Fig. 14 floorplan referenced as the full
taped-out design. A separately pre-registered visual audit may test whether it
contains a legible chip/family identifier or additional exact hardware labels
lost in Markdown extraction. It must not infer identity from layout resemblance
and must stop if no explicit text is visible.
