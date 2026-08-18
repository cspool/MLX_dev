# MLX author-team simulator and accelerator lineage

## Conclusion

The strongest evidence now points to a closed ICT/Ricore lineage rather than a
direct DSAGEN fork:

`SimICT -> DPU microarchitecture -> multi-batch DFU -> DFU-E/PANDA/DFGAS -> M2-DFU -> MLX`.

MLX explicitly resolves its simulator citation to SimICT. Historical papers
from the same ICT authors repeatedly implement cycle-accurate dataflow models
on SimICT, validate them against gem5 or RTL, and implement hardware in Verilog
with Synopsys tools. Official author pages identify the later DPU/DFU papers,
commercial DPU-s/HTC chips, and an internal SmarCo stack spanning simulator,
RTL, memory and runtime work.

This supports **SimICT as the simulator framework at citation/method level** and
the **DPU/DFU-E/M2-DFU family as the highest-confidence parent candidate**. It
does not establish the exact parent chip, a shared repository, or source-code
reuse. No public SimICT/DFU-E/M2-DFU simulator repository was found.

Immutable evidence: `artifacts/results/mlx-author-simulator-lineage-run109.json`.

## Historical chain

| Year | Work | Simulator / implementation evidence | Relevance to MLX |
|---:|---|---|---|
| 2013 | [SimICT](https://doi.org/10.1109/ISLPED.2013.6629308) | Component-based parallel performance/power framework with configurable target systems | MLX's reduced-design simulator citation resolves here |
| 2015 | BDSim | Same-team component-based parallel simulation framework for many-core/big-data evaluation | Possible engineering precedent; no MLX citation or code link |
| 2017 | [Efficient NoC router](https://jcst.ict.ac.cn/article/doi/10.1007/s11390-017-1703-5) | SimICT DPU model; PE-array NoC optimized for multicast/high injection/low latency | Predecessor for explicit PE routing |
| 2018 | [Pipelining loop optimization](https://jcst.ict.ac.cn/article/doi/10.1007/s11390-017-1748-5) | SimICT, 8x8 PEs, SPM, loop control and ready-work pipelining | Predecessor for loop-driven folded templates |
| 2018 | [Non-stop double buffering](https://jcst.ict.ac.cn/article/doi/10.1007/s11390-017-1747-6) | SimICT, instruction buffers and multiple blocks in flight | Critical missing memory/fill-drain semantics in the current surrogate |
| 2019 | [Instruction memory-conflict optimization](https://crad.ict.ac.cn/fileJSJYJYFZ/journal/article/jsjyjyfz/HTML/2019-12-2720.shtml) | SimICT plus matching Verilog; 1 GHz 4x4 PE array, ARM host, operand RAM, two 64-bit XY NoCs, 3 MB SPM | Closest public reduced DPU configuration |
| 2022 | [Look-ahead acknowledgment](https://jcst.ict.ac.cn/en/article/doi/10.1007/s11390-020-0555-6) | Cycle-accurate SimICT, calibrated against gem5; SimpleScalar-derived host, 4x4 PEs, instruction buffers, SPM, multiple meshes | Strong simulator-validation and transfer-subsystem precedent |
| 2023–24 | [Multi-batch DFU](https://doi.org/10.1145/3637906) | Unified scale-vector modes, reconfigurable clusters, time-multiplexed DFG-node stages and task model | Direct conceptual predecessor to M2-DFU |
| 2025 | [DFU-E](https://doi.org/10.1109/TPDS.2025.3555329) | Multi-layer task/block/instruction/data parallelism; custom PE/memory/NoC and software stack | Strong parent-family candidate with extensive MLX author overlap |
| 2025 | [PANDA](https://doi.org/10.1145/3721288) / DFGAS | Adaptive prefetch/decentralized scheduling and DFG-aware HW/SW scheduling | Likely memory/scheduler ancestry |
| 2026 | M2-DFU | Official author pages identify a general multi-mode DFU with core MLX authors; full text unavailable | Highest-ranked unnamed general-purpose parent candidate |
| 2026 | MLX | Specialized, profile-driven subset; cycle-accurate simulator, Verilog 12 nm, RISC-V host and compiler | New tagged-block/CDC/skip-hop specialization |

## Author and organization evidence

- [Wenming Li's UCAS page](https://people.ucas.ac.cn/~liwenming) lists the
  complete DPU/DFU publication chain, BDSim, M2-DFU, DFU-E, MLX, and patents
  for multi-layer fused dataflow execution, reuse, hybrid routing and FFT.
- A second [official profile](https://people.ucas.ac.cn/~0024137) reports the
  taped-out DPU-s, HTC-3000 and HTC-3500 chips and software/dataflow research.
- [Xiaochun Ye's ICT page](https://www.ict.ac.cn/sourcedb/cn/jssrck/201411/t20141115_4253437.html)
  identifies the DPU chip program and lists MLX and DFU-E.
- [Zhihua Fan's page](https://fanzhihua-ict2024.github.io/) places M2-DFU,
  DFU-E, DFGAS, PANDA and MLX in one continuous publication program.
- [Shantian Qin's page](https://shantianqin.github.io/) reports work in
  Ricore/SmarCo's Processor Architecture Group on PE/on-chip-memory/data-transfer
  simulation, SPM/cache RTL, and multi-application runtime scheduling.
- [Jian Weng's KAUST page](https://www.kaust.edu.sa/en/study/faculty/jian-weng)
  establishes a separate open full-stack spatial-accelerator line: DSAGEN,
  OverGen and compiler automation. [Assassyn](https://doi.org/10.1145/3695053.3731004)
  unifies simulation and RTL, but MLX does not cite it and no shared code is shown.

## Reconstructed simulator methodology

The public historical sources support the following methodology:

1. A componentized parallel discrete-event framework (SimICT) instantiates
   processor/host, PE, router/NoC, memory and power components.
2. Target DPU models are cycle-accurate. One team paper reports calibration to
   gem5 with error below 9%; other DPU work reports simulator-versus-RTL cycle
   deviation below 3%.
3. Host execution historically derives from SimpleScalar; later designs add a
   RISC-V host/control plane.
4. Hardware is implemented independently in Verilog, synthesized with Synopsys
   DC, and evaluated with VCS/PrimeTime; several DPU/HTC variants were taped out.
5. The modeled accelerator is not a GPU SM: it has instruction slots, operand
   RAM, fire/ready selection, repeated block instances, explicit PE routing,
   multiple physical NoCs, SPM, DMA/cache, and blocks in flight.
6. The recent internal stack adds PE/tensor, on-chip memory, normal/transpose
   transfer, runtime scheduling and RISC-V instruction support.

## Consequences for the open surrogate

The current repository should pivot as follows:

- Use **gem5 as the primary open system/component timing host**, because the
  historical SimICT models are explicitly calibrated against gem5.
- Recreate the historical DPU configurations before MLX specialization:
  2019 4x4 DPU, 2018 8x8 DPU, and 2022 4x4 look-ahead model.
- Model DRAM/cache/DMA/SPM and non-stop double buffering explicitly. Reusing
  DSAGEN's scratchpad as the whole MLX memory system is not historically
  supported and explains the over-saturated SWA result.
- Model task ID, block ID, instance, instruction slots, ready/fire selection,
  operand buffers, blocks in flight and multiple NoCs; add MLX tags/CDCs as the
  specialization on top.
- Retain **DSAGEN** for reusable spatial compiler/ADG mechanisms and
  **Assassyn** as an asynchronous simulation/RTL semantics reference, but no
  longer describe either as the most likely original simulator base.
- Retain **Accel-Sim/GPGPU-Sim** only for Xavier/Orin/RTX baselines.

## Evidence boundary

No primary source explicitly says “MLX is derived from M2-DFU/DFU-E,” and the
M2-DFU full text is unavailable. No public repository was found for SimICT,
BDSim, DFU-E, M2-DFU or the internal SmarCo simulator. The exact taped-out
parent and source-code provenance therefore remain unresolved.
