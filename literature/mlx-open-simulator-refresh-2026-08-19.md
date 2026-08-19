# MLX team/open-simulator refresh — 2026-08-19

## Scope and evidence rule

This refresh searches the MLX authors' ICT/CAS/UCAS and KAUST/PolyArch lines,
then inspects open simulator source as implementation precedent. Author overlap
and mechanism similarity do not establish MLX source reuse. Paper targets are
not used to choose parameters in this note.

## Author and institution update

MLX is primarily from the State Key Laboratory of Processors, Institute of
Computing Technology, Chinese Academy of Sciences and UCAS, with Jian Weng at
KAUST. The current ICT and author pages show a continuous dataflow program:

- DPU/SimICT and SmarCo hardware/simulator work;
- multi-batch DFU, DFU-E and the newly listed M2-DFU;
- DFGC/DFGAS scheduling and NoC control;
- ROMA/PANDA/TOSCA on memory and decentralized/semi-centralized scheduling;
- MLX, UniNL and AHASD for structured/nonlinear/speculative LLM execution.

Primary pages refreshed on 2026-08-19:

- https://www.ict.ac.cn/sourcedb/cn/jssrck/201810/t20181030_5151416.html
- https://fanzhihua-ict2024.github.io/
- https://shantianqin.github.io/
- https://people.ucas.ac.cn/~liwenming

The Zhihua Fan page now explicitly lists `M^2-DFU: Multi-Mode Dataflow
Architecture for Adaptive and High-Efficiency Data Processing` (TOCS 2026),
MLX, DFGAS, PANDA and TOSCA in the same program. This strengthens family
continuity but still provides no public M2-DFU/MLX simulator.

## Source-derived mechanisms

### Scheduling and NoC

The public DFGC presentation
(https://akaliu.github.io/files/iccd-presentation.pdf) exposes a PE block-state
record containing DFG ID, block ID, instruction ID, activation count,
acknowledgement count and status. A compiler estimates execution+routing
latency and generates a timestamp; higher timestamps receive higher packet
priority. Hardware retains round-robin control, per-direction queues and
autonomous routing. This is relevant to NoC arbitration, not evidence that
MLX's compute arbiter should replace its explicitly reported lower-tag priority
and round-robin tie breaking.

TOSCA (DOI `10.1145/3833878`) adds on-demand triggering, global workload
sensing, global-aware planning and low-latency remapping around locally
autonomous PEs. PANDA (DOI `10.1145/3721288`) combines adaptive prefetch with
decentralized scheduling. These are useful irregular-workload extensions but
MLX's regular CDC paths should remain statically placed and tag-elastic.

### Memory hierarchy and fusion

ROMA (DOI `10.1109/HPCC-DSS-SmartCity-DependSys60770.2023.00017`) separates
prefetchable traffic from execution with explicit prefetch/postback ISA
operations and dynamically partitions shared storage between SPM and cache.
Historical DPU papers separately disclose 64 GB/s external DMA at 1 GHz and a
32-bank, 256-bit-per-bank SPM. These are different bandwidth domains and must
not be represented by one scalar roofline bandwidth.

The same-team Attention patent CN119940434B/CN202510009132.9 discloses a
schedule boundary absent from the current Figure-21 composition:

- if `sequence_length * embedding_dimension` fits SPM capacity, pre-transposed
  inputs allow five Attention operations to execute as one kernel;
- otherwise, sequence blocks are streamed through SPM and the five operations
  are fused into two reusable kernels;
- the threshold is the SPM capacity.

This supports source-driven configuration/fill-drain and transpose-traffic
changes; it does not disclose MLX's exact Llama2 tile sizes or cycle constants.

## Open simulator source audit

| Simulator | Pinned revision | Reusable principle | Role / limitation |
|---|---|---|---|
| DSAGEN | `273e141a519d12138ee0fbc9743059d13e9b5a64` | gem5 host, spatial ADG, compiler/mapping, decoupled pipelines | Retain as executable MLX substrate; not provenance |
| Assassyn | `6a99ade0e9380c93d4817f7de51b7edd8a473dd2` | asynchronous stage activation, queues and credit flow | Semantic cross-check; no MLX mapper |
| DAM-RS | `8771934ce6074bea0d8d901cf787aae8983ae42e` | CSP contexts, bounded latency/response channels, max-context elapsed time, explicit DRAM latency/bandwidth | Closest open SimICT-style component engine; Rust rewrite would not itself fix mappings |
| NPUsim | `f0378abe89e16a1f8a78cd2ce3ed20c1ccae3f3f` | architecture-oriented components, separate schedule table, functional+cycle execution, DRAMsim3 | Strong memory/counter reference; CNN/systolic defaults do not model tags |
| STONNE | `22634b8b7668c2691bc2df8f612232d90839df20` | per-component `total_cycles`, operation/use counts, port and FIFO statistics | Best counter-definition reference; network topology differs from MLX |
| Accel-Sim | `c5296df152c99a28dd64e5d9560bd58a8fd2e774` | trace/PTX GPU timing and tuner-generated configs | Retain for GPU baselines; no official Xavier/Orin/RTX-3090 config was found |

DAM-RS is dual MIT/Apache-2.0, STONNE is MIT, and NPUsim uses a BSD-style
three-clause license. No code is copied by this audit.

## Counter implications

The three open implementations deliberately separate metric identities:

- STONNE reports component operations/use counts against that component's own
  total cycles and records each SRAM port separately.
- NPUsim distinguishes spatial PE-array utilization (`active PEs / physical
  PEs`) from MAC and buffer utilization rather than collapsing them.
- DAM defines graph elapsed time as the maximum context time and gives channels
  independent capacity, forward latency and response latency.

The current Figure-22 transfer instead divides latency-weighted productive PE
cycles by `end_to_end_cycles * 16` for every resource. That is one legitimate
metric, but it is not interchangeable with component busy time, FU operation
utilization, spatial occupancy or per-port utilization. A new counter ledger
must expose these identities side-by-side before any paper-target join.

## GPU-baseline boundary

Accel-Sim's documented tuner generates a device config from native
microbenchmarks and traces. The public tested configurations include Volta and
selected desktop GPUs, but the refreshed search found no first-party Xavier,
AGX Orin or RTX-3090 config/trace bundle. Cross-ISA synthetic HMMA traces remain
useful for service curves but cannot be relabeled as those devices. A strict
GPU baseline therefore needs either native-device capture/microbenchmarks or a
clearly labeled specification model with held-out real measurements.

## Ranked experiments

1. **Counter ledger:** add STONNE/NPUsim/DAM-compatible component, spatial,
   issue and per-port identities to existing H120 records; select no metric from
   Figure-22 residuals, then expose Figure 22 as held-out.
2. **Window/coverage scheduling:** sweep active tag windows permitted by the
   paper inequality `B_T*C >= T_load+T_xfer`, retaining lower-tag/RR compute
   arbitration; assess utilization before target access.
3. **ROMA/fusion boundary:** implement one-/two-kernel Attention composition
   based only on SPM capacity and pre-transposed inputs, then revisit Figure 21.
4. **GPU mapping:** use Accel-Sim tuner only if native Xavier/Orin/3090 access
   or traces become available; otherwise keep specification/synthetic results
   explicitly non-native.

