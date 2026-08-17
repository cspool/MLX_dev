# H42 protocol: compile MLX CDCs and drive DSAGEN scratchpad timing

## Classification

Confirmatory for radix-2 CDC structure, forward-only boundary events, and use
of DSAGEN's existing scratchpad timing path; exploratory for the concrete
small-shape block instruction template and placement policy omitted by MLX.

## Motivation

H41 validates MLX control, PE-resource, and skip-hop semantics inside
dsa-gem5, but its load/store completion is a configured latency and its test
blocks are hand-written. An end-to-end simulator needs a deterministic
operator-to-block compiler and real memory completion callbacks. It also needs
event-level cross-layer wakeup: waiting for a whole predecessor tag would
serialize stages and contradict the paper's active-window pipeline.

## Hypothesis

Small radix-2 BSMM and FFT closed-dependency components can be compiled into
H41's tagged-block JSON with exact source-derived counts and adjacent-stage
events, while an opt-in adapter sends overlay load/store operations through
DSAGEN's unmodified `RequestBuffer` and bank pipeline. The resulting dsa-gem5
execution should exhibit both real memory backpressure and successor-stage
issue before predecessor-tag completion, without changing either H41's fixed
backend or the environment-clean upstream regression.

## Frozen contract

The exact fixtures, counts, event semantics, source boundaries, and result
path are frozen in `configs/simulators/dsagen_mlx_cdc_memory_v1.yaml`.

- Each stage contains `n/2` radix-2 pair CDCs; there are `log2(n)` stages.
- BSMM parameter count is `2*n*log2(n)`. Its two-by-two pair update accounts
  for four scalar multiplies and two scalar adds.
- An FFT pair accounts for one complex multiply and two complex additions;
  the manifest expands this to four real multiplies and six real additions.
- Every pair emits a unique boundary event after its final transfer. The
  mapped consumer at stage `k+1` waits for the corresponding iteration count,
  not for all blocks in stage `k`.
- The compiler template is load-first, compute-middle, then store/xfer. This
  is an explicit reconstruction of Fig. 10(d), not recovered author code.
- `memory_backend=dsagen_spad` is the only mode allowed to claim real memory
  timing. The fixed backend remains H41's regression mode.

## Tests

1. Compile BSMM-8 and FFT-8 twice and require byte-identical JSON/manifest.
2. Audit all registered counts, block fields, addresses, event edges, routes,
   and static instruction groups.
3. Add an event-count microtrace with two loop iterations and prove iteration
   `i` cannot consume event `i+1`, while stage overlap occurs before full tag
   retirement.
4. Add a fake adapter test for token issue, delayed completion, queue-full
   stalls, and exact one-shot completion.
5. Incrementally build dsa-gem5, run both compiled operators with the real
   scratchpad adapter, and require issued request count equals completion count
   and the overlay reaches `done`.
6. Require at least one observed memory queue/bank delay beyond a fixed
   one-cycle path; a bypass that immediately acknowledges requests fails.
7. Repeat H41's fixed-backend integration and the no-`MLX_CONFIG` DSAGEN
   regression exactly.

## Pass criteria and stopping rule

All structural, callback, overlap, deterministic, and regression checks must
pass. No cycle total is compared to an MLX paper figure, and no latency is
tuned from a paper residual. If DSAGEN cannot safely distinguish adapter
responses from vector-port responses, reject the adapter boundary rather than
using a parallel synthetic memory queue. If event-level overlap violates
iteration order, reject H42 before scaling shapes.

## Immutable output

The sole formal output is
`artifacts/results/dsagen-mlx-cdc-memory-run048.json`; compiler outputs,
microtraces, builds, and gem5 logs are hash-qualified evidence inputs.
