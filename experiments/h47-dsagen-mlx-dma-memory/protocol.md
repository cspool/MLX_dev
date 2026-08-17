# H47 protocol: route MLX overlay memory through DSAGEN cache/DRAM timing

## Classification

Mechanism-confirmatory simulator construction. This experiment has no MLX
paper performance target and cannot validate a paper figure. It tests the
missing off-chip-memory boundary of the open DSAGEN-based reconstruction.

## Hypothesis

An opt-in `dsagen_dma` backend can route MLX overlay loads and stores through
MinorCPU's existing LSQ, address translation, cache hierarchy, and configured
DDR controller without sharing a vector-port response queue. Loads complete
only on their cache response and stores complete only on their returned memory
response, while the existing fixed-latency and DSAGEN-scratchpad backends remain
unchanged.

## Frozen implementation boundary

`configs/simulators/dsagen_mlx_dma_memory_v1.yaml` freezes the source seams,
microtrace, paired control, and pass gates before implementation.

- Transfer index 126 is reserved for the MLX DMA adapter. Native DSAGEN uses
  input-port queues below that index and reserves 127/128 for memory writes and
  configuration; activation must reject an accelerator exposing 126 or more
  input ports.
- Requests use `LSQ::pushRequest`; the adapter may not instantiate a parallel
  cache, DRAM, or latency queue.
- A load token becomes visible to the overlay only when the ordered LSQ load
  response is complete.
- A store leaves the adapter transfer queue through DSAGEN's existing store
  buffer, but its token becomes visible only after `recvTimingResp` observes the
  completed write. Store-buffer insertion is not completion.
- Adapter request data are deterministic zero bytes. This phase validates
  timing and the real write path, not MLX numerical values.
- The guest binary owns aligned BSS regions. Addresses are resolved from its
  ELF symbol table after linking; no host pointer or invented physical address
  may be used.
- H47 sets `start_in_roi=true`, so warm-up accelerator ticks cannot consume the
  overlay before `begin_roi()` resets gem5 statistics. The default is false to
  preserve every existing overlay run.

## Frozen microtrace and paired control

The workload has 16 PE-local tagged blocks on a 4x4 mesh. Each block performs
four iterations of load, integer add, and store. Thus each run contains exactly
64 reads, 64 stores, and 64 compute operations. Reads address untouched,
cache-line-separated BSS locations; stores address a distinct initialized BSS
region. The same generated overlay is run once with `fixed` and once with
`dsagen_dma` using the same guest binary and gem5 memory configuration.

## Tests

1. Resolve both guest symbols with the RISC-V ELF tools and compile both overlay
   JSON files twice; require byte identity and in-range aligned addresses.
2. Unit-test parsing and fake-adapter completion for the new backend without a
   paper target.
3. Incrementally build dsa-gem5 and run the fixed/DMA pair under MinorCPU,
   32-KiB L1D, 512-KiB L2, and DDR4-2400.
4. Require exactly 128 issued and 128 completed DMA requests, split 64/64 by
   direction, with zero failed translations or responses.
5. Require the DMA guest's post-run store checksum to be zero and the fixed
   control's checksum to retain the initialized value.
6. Require positive paired deltas in data-cache accesses and at least one
   data-side DRAM read/burst. Require maximum DMA response latency to exceed
   the DSAGEN scratchpad adapter's one-cycle fast path.
7. Require request-queue backpressure or multiple simultaneously outstanding
   requests, proving that the adapter did not serialize before the LSQ.
8. Re-run H42's fixed/SPAD overlay checks and the 569-cycle environment-clean
   DSAGEN vecadd regression.

## Pass criteria and stopping rule

All structural, functional, cache/DRAM, completion, and regression gates must
pass. A missing data-side DRAM delta rejects the cold-memory claim. A write
acknowledged before its real response rejects the store-completion claim. Do
not tune any latency from an MLX figure; preserve failures and change the
mechanism only when source evidence identifies the cause.

## Immutable output

The sole formal output is
`artifacts/results/dsagen-mlx-dma-memory-run053.json`; generated configs,
build manifests, paired gem5 logs, statistics, and regression logs are
hash-qualified evidence inputs.

## Pre-run cache-conditioning amendment

Development attempts 1--6 are excluded from `run053`. Requestor-specific
occupancy proved that all 64 MLX lines entered the cache hierarchy, but the ELF
BSS lines were already L2-resident and therefore could not exercise DDR. Before
the first formal run, the guest is amended to read one byte from every 64-byte
line of a separate 2-MiB volatile BSS region before `begin_roi()`. This is four
times the configured 512-KiB L2 capacity. Both fixed and DMA runs execute the
identical conditioner, its checksum is checked, and the ROI reset excludes its
traffic. No overlay count, address, latency, or paper-derived parameter changes.

Attempt 7 showed that untouched BSS can retain demand-zero/shared-page
semantics in syscall-emulation mode. It is excluded as well. Before the formal
run, the guest writes byte value 1 to the exact 64 eight-byte read targets,
then performs the 2-MiB conditioner. This forces distinct writable backing and
freezes an independent read-data gate: the adapter byte sum must be 512.

The first long-wait formal candidate is retained but excluded: after successful
overlay completion, 500,000 host iterations obscured the requestor command
counters in gem5's final statistics. Since `accel_t::done()` now keeps the
guest's accelerator call blocked until every overlay token completes, no host
wait is required. The final build freezes it at zero so the post-reset window
contains the overlay plus only its immediate checksum and exit code.
