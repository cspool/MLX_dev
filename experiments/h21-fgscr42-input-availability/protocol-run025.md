# H21 run025 addendum: correct Baidu share-init normalization

## Reason for a replacement run

`run_024` is retained as an invalid execution artifact. Both password checks
returned `errno=2` because the implementation passed the short-link routing
token (for example `1eXpl...`) directly as the API `surl`. That prevented the
public `BDCLND` share cookie from being issued and made every downstream Baidu
response non-diagnostic. Its `audit_integrity.pass=false` was therefore correct;
the run is excluded from the H21 verdict and from the research trajectory.

The corrected normalization is independently specified by the pinned public
BaiduPCS-Py implementation at revision
`e81e9b65c4b35fc8f7f2993a81e25e0bc24608db`:

1. parse a short link as `/s/1{token}`;
2. initialize `https://pan.baidu.com/share/init?surl={token}`;
3. submit `{token}`, without the short-link routing prefix `1`, to
   `/share/verify`;
4. include the init URL as `Referer`, plus `bdstoken=null`,
   `X-Requested-With: XMLHttpRequest`, and form content type.

This is an endpoint-contract repair, not a response-dependent change to the
corpus/split gates. Every source identity, expected file metadata, downstream
download/ZIP probe, sanitization rule, and H21 decision rule from the original
protocol remains unchanged. `run_025` is the only corrected formal run. It must
write a new artifact and must not overwrite `run_024`.
