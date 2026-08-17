# H21 protocol: FGSCR-42 public-input availability for Figures 15(a) and 16

## Hypothesis

The official public FGSCR-42 release and an independently indexed mirror permit
anonymous byte retrieval of the 42-class image corpus and publish enough label
and deterministic train/test-split information to launch a native reproduction
of the ViT experiments in MLX Figures 15(a) and 16 without private credentials.

## Evidence classification

This is a **public-input availability audit**, not a model-quality experiment and
not evidence that any accuracy bar has been reproduced. It is validation-ineligible
for the project's numerical best-error metric. Source discovery and endpoint
reverse-engineering before registration were exploratory; the formal `run_024`
must execute only the frozen, read-only checks below and must not adapt its gates
after observing a response.

## Frozen paper and repository checks

- The supplied paper extraction has SHA-256
  `5785eb81b28741a3806ca42749d7b556bbcd2404e1622ae644a32bae2ade7745`.
  Its ViT discussion says only that the model is trained from scratch on the
  dataset cited as FGSCR-42. It does not state a ViT variant, image resolution,
  preprocessing/augmentation, optimizer, learning rate, schedule, epoch count,
  seed, or train/validation/test split.
- Official repository: `https://github.com/DYH666/FGSCR-42.git`, revision
  `ced49c37964c3c7c453602ba6e4ba2a812f67086`. Its tracked tree contains only
  `README.md` and three illustrative PNG files. The README hash is
  `3eefd5aef9f55103481d5cbad42a06a607596f09a865d2e4e110d2110c8e2822`,
  declares about 9,320 images and 42 categories, and points only to Baidu Pan.
  All 25 commits, all branches, and all tags must be searched for an archive,
  per-image labels, or a split manifest.
- Independent index: `https://github.com/JACYI/Dataset-for-Remote-Sensing`,
  revision `29e6aac03ff44f811e84073d0c5ae6abb381141e`. Its FGSCR-42 section exposes a
  second Baidu share while its Google Drive link is empty.
- The public GitHub issue snapshot is used only as corroborating evidence. In
  particular, issues 1, 6, 7, and 8 report download/access problems, issue 4
  reports an incomplete image count, and issues 3 and 9 request missing labels.
  No issue comment may be treated as a substitute for a versioned split.
- Three Hugging Face dataset-catalog searches (`FGSCR-42`, `FGSCR42`, and
  `fine-grained ship classification`) are frozen as discovery checks. A catalog
  match would still need byte, label, provenance, and split validation.

## Frozen anonymous Baidu checks

The public extraction codes from the two README files may be submitted. No
account login, personal credential, CAPTCHA bypass, or persistent authentication
state is allowed. Cookies, dynamic signatures, request identifiers, download
URLs, and desktop-client task strings must never be serialized.

For each share, `run_024` will:

1. verify the public extraction code and list the root metadata;
2. compare filename, file size, file identifier, share identifiers, and creation
   time with the frozen manifest;
3. request both the individual and batch shared-download routes;
4. if an HTTPS batch link is returned, make only one-byte Range probes against
   the three registered PCS hosts, three registered user agents, with and without
   the public share cookie (18 probes maximum);
5. request the online ZIP-member listing for both accepted path spellings.

The exploratory reference observation is that both shares describe the same
5,112,338,338-byte object (`FGSCR.zip` / `FGSCR_old.zip`, creation time
1608263618), individual download returns a Base64 desktop-client task rather
than bytes, batch Range probes return HTTP 403 / PCS error 31064, and ZIP listing
returns error 120. These observations are frozen as an audit-integrity check,
not silently converted into successful data access.

## Decision rule

Two necessary gates are evaluated independently:

1. **Corpus gate:** at least one anonymous route returns an actual archive byte
   and exposes a class-label organization consistent with 42 categories.
2. **Experiment-split gate:** a versioned source supplies an unambiguous split
   manifest (or exact deterministic split construction) for the MLX ViT run.

H21 is supported only if both gates pass. The primary diagnostic is the fraction
of these two required inputs that is missing. The formal audit itself passes when
all pinned source identities are verified, responses are sanitized, and the
observed access classification is internally consistent; an audit pass can and
is expected to coexist with an H21 rejection. No accuracy training may start on
an invented split under this hypothesis.
