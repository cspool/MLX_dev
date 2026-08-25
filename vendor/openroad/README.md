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
the PPA config.  Detailed routing, RC extraction, STA, power, and DRC continue
to use the unmodified official OpenROAD binary.

OpenROAD is distributed under the BSD 3-Clause license, with component-specific
licenses documented by the upstream project.
