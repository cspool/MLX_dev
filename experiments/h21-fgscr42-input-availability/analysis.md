# H21 analysis: public FGSCR-42 inputs are insufficient for native ViT reproduction

## Outcome

H21 is **rejected with an auxiliary-endpoint caveat**. Corrected `run_025`
finds both required gates false, so the missing-required-input fraction is
`2/2 = 1.0`. This is an input-identifiability result, not a numerical accuracy
validation and not a claim about the original authors' private data.

`run_024` is retained but excluded: an incorrect short-link `surl` caused both
password checks to fail before diagnostic requests could run. The corrected
endpoint contract was separately pinned and committed before `run_025`.

## Confirmatory observations

- The supplied-paper hash matches. Its complete ViT result paragraph contains
  neither an identifiable architecture nor split, preprocessing/augmentation,
  or optimization details.
- The official repository is at the frozen revision with exactly 25 commits,
  one branch, no tags, and four HEAD files (`README.md` plus three illustrative
  images). Scanning every historical tree finds no archive/manifest and no
  filename containing a split, label, or annotation marker.
- The independent index is at the pinned revision, contains the second Baidu
  link, and still has an empty Google Drive link. All nine public GitHub issues
  in the frozen snapshot are present; download/completeness and label requests
  corroborate but do not determine the verdict.
- All three Hugging Face dataset-catalog searches return zero matches.
- Both public extraction-code verifications return `errno=0`, issue the public
  share cookie, and expose dynamic signing data. Both batch-download requests
  return `errno=0` and an HTTPS probe target.
- Every one of the 36 registered one-byte probes (two shares, three PCS hosts,
  three user-agent classes, with/without the public share cookie) returns HTTP
  403 with PCS error 31064. Zero archive bytes are returned.
- No versioned source supplies the MLX experiment split. This independently
  falsifies the sufficiency hypothesis even if archive access later changes.

## Audit-integrity caveat

The result's composite `audit_integrity.pass` is false. The formal root-list
requests returned one transport failure and one `errno=2`, the individual
download parser observed no top-level value, and the registered ZIP-list calls
returned `errno=-6` rather than the exploratory `120`. These auxiliary endpoint
responses are not promoted to evidence about archive contents. They also do not
weaken the two decisive observations: all actual byte probes were denied, and
the exact split is absent from the paper and every versioned repository tree.

The runner serializes no extraction code, cookie, dynamic signature, download
URL, task string, or request identifier, and it downloads no large object.

## Consequence

Figures 15(a) and the ViT part of Figure 16 remain unreproduced natively. A new
ViT experiment would be an explicitly inferred benchmark on a user-chosen split,
not a reproduction of MLX. Native training should wait for an author-provided
archive/split/recipe or be registered under a separate sensitivity hypothesis.
