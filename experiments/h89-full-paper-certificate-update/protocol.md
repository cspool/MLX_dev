# H89 protocol: update the full-paper completion certificate

H89 overlays the latest source-integrated evidence on H37's 18-row certificate
without rerunning target-guided legacy models.

Five rows are reconsidered:

- Figure 20 remains `attempt_rejected`, now using H88's 0 reproduced / 6
  numerical-failure / 2 execution-incomplete closure;
- Figure 22 changes from `reproduced_within_10pct` to `attempt_rejected`
  because H44's no-fit real-DSAGEN run fails the strict all-point gate at 15/16;
- Figure 23 changes from `calibration_replay_only` to `attempt_rejected`
  because H70's diagram-derived real-SRAM transfer passes only 7/15;
- Figures 24 and 25 change from `calibration_replay_only` to
  `attempt_rejected` because physical-counter transfers pass only 3/42 and
  0/24 respectively.

All other H37 statuses remain frozen. The expected updated counts are zero
reproduced, eleven rejected attempts, zero replay-only, and seven publicly
blocked. The global all-experiment 10% verdict must remain false.

The immutable output is
`artifacts/results/full-paper-completion-update-run094.json`.
