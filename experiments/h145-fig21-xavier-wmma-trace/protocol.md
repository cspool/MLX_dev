# H145 protocol: target-free trace-driven Xavier WMMA projection

## Hypothesis

Accel-Sim's intended specialized-unit route can replay real WMMA SASS traces on
the frozen H56 Xavier timing configuration, validate repeat folding, and emit
five target-free dense projection estimates where H144 functional PTX failed.

## Capture and replay boundary

Reuse H144's exact 64-CTA WMMA source and repeats 16/32/64/128. Capture each
execution with the byte-frozen Accel-Sim NVBit tracer on available device 0,
RTX4090 UUID `GPU-316b42a1-49a5-f647-aa0c-05b853d289a8`, then postprocess to
`.traceg`. Require the application checksum and tensor SASS opcode in every
trace.

Replay each trace using byte-frozen `accel-sim.out`, H56's 1.377-GHz Xavier
GPGPU config and the SM7 trace config. The trace is SM89/Ada while the timing
model is SM70/Volta-derived; this cross-ISA schedule proxy must remain explicit
and is not author Xavier/cuBLAS identity. It is nevertheless preferable to
inventing scalar FMA timing because TensorCore opcodes use Accel-Sim's
specialized-unit path.

Fit exact FMA work for repeats16/32 and require repeats64/128 within 5%. Only a
passing fold may map H91's 32-layer dense QKV/output/FFN totals for five shapes.

## Acceptance gates

1. H56/H91/H144 and all tracer/simulator binaries/configs qualify by hash and
   required status/integrity.
2. Device 0 identity matches the frozen RTX4090 name/UUID and H144 failure class
   is the expected functional-PTX WMMA crash.
3. Four real-GPU captures pass checksum, produce kernelslist/traceg artifacts,
   contain 64 CTAs and at least one TensorCore SASS opcode.
4. Four Accel-Sim trace replays on the unmodified H56/SM7 configs exit normally
   with positive cycles, instructions and 64 CTAs.
5. Every capture/replay records exact 64*repeat*4096 FMA equivalents and immutable
   artifact hashes.
6. The repeats16/32 affine cycle model has positive slope/predictions.
7. Both repeats64/128 holdouts pass <=5% relative cycle error.
8. Five H91 32-layer projection totals map to finite positive Xavier cycles and
   seconds only after gate 7.
9. Source contains no Figure 21 target, target factor, efficiency fit or
   post-result trace/config selection.
10. Output remains labeled SM89-trace/SM70-timing transparent proxy; Figure 21
    and active completion remain 3/8 pending attention/elementwise composition.

If the first anchor cannot be captured or replayed, stop remaining jobs and
record a rejected result with zero estimates. The immutable result will be
`artifacts/results/fig21-xavier-wmma-trace-run150.json`.
