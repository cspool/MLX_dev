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
PA/TA, allowing exact round-trip verification.  The optional local-repair
switch skips incremental iterations 1 and 2 after the initial marker scan;
those stock iterations preserve every pre-routed net and therefore cannot
change the imported violations.  Repair resumes directly with the stock DRC
rip-up strategy at iteration 3.

- source commit: `a008522d88b669ac4c985609533cf5a3d2649222`
- executable version: `a008522d8-tile48-guard101-drt-postroute-repair`
- DRT patch SHA-256: `b51349641493c1082fd8be15902cfe03689df20cddc57ef2335b5c67962f699f`
- compressed SHA-256: `72f754e5b9f4b93655a2a5c83e0443bb511752193f24967e15e883dcfbe1b263`
- executable SHA-256: `d43ecf4a09e1dbe25a38b6d4134d7d6ca059c305bae34cf3d9527a513cebcb67`
