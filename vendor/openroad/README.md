# Pinned OpenROAD GRT variant

`openroad-a008522d8-tile48-guard101-amd64.xz` is the compressed executable used
only for the H206 global-routing stage.  It is retained so the exact multi-hour
route can be replayed without relying on an unpublished local build directory.

- upstream: `https://github.com/The-OpenROAD-Project/OpenROAD.git`
- source commit: `a008522d88b669ac4c985609533cf5a3d2649222`
- patch: `patches/openroad/grt-tile48-guard101.patch`
- patch SHA-256: `ebc8dd97967bb782828fc18d331013cdc66ca3f88623f25a7dbc850b7c766f98`
- compressed SHA-256: `aecbe75154d214e939645272161290e980744693be742c5a0f09ca4a7f2c0dff`
- executable SHA-256: `2fe0b0a5a576a4d940487b7ada0d62931ac0fc055e85653c498a08cef7f9a21f`
- target: AMD64 Linux, linked against the same OpenSTA ABI as the pinned
  `v2.0-17598-ga008522d8` Debian 11 package

The patch changes the FastRoute tile from 15 to 48 routing pitches and raises
the internal pre-adjustment edge-usage sanity multiplier from 100 to 101.  It
does not relax detailed-route DRC.  `scripts/bootstrap_rtl_ppa_tools.sh`
verifies both hashes while installing the executable at the path consumed by
the PPA config.  Base detailed routing, RC extraction, STA, power, and DRC use
the unmodified official OpenROAD binary; only the explicit post-route local
repair uses the second variant documented below.

OpenROAD is distributed under the BSD 3-Clause license, with component-specific
licenses documented by the upstream project.

## POINT_EXT-safe local-repair variant

`openroad-a008522d8-tile48-guard101-drt-point-ext-amd64.xz` contains the same
source commit and FastRoute changes plus
`patches/openroad/drt-point-ext-orthogonal.patch`.  The DRT parser now performs
the existing orthogonal-corner split for both `POINT` and `POINT_EXT` records
and advances its current layer from each VIA decoder opcode.  These changes
prevent both false diagonal segments and stale-layer reconstruction during
post-route ODB re-entry; they do not change any geometry or DRC rule.  A
read-only import-probe mode reports per-layer wirelength and via totals before
PA/TA, allowing exact round-trip verification.

- source commit: `a008522d88b669ac4c985609533cf5a3d2649222`
- executable version: `a008522d8-tile48-guard101-drt-postroute-safe`
- DRT patch SHA-256: `3d6a1901c2f698992b0aab5e25c6f88170b71515dc3281c6e48c64b162de3724`
- compressed SHA-256: `33a027eae4570bbab52f7289b798788f81bb114ce243379c47dceaeaed969735`
- executable SHA-256: `c9ec6634b6f146d37e96485f49f48b494cd2ab629f4948f36d906dd18be1d3e4`
