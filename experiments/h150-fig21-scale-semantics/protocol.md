# H150 protocol: target-free Figure 21 scale-semantics audit

## Hypothesis

H149's roughly 675x direction failure contains two independently identifiable
source-level scale errors: H92 underuses the full 4x4 SIMD32 PE array, while
H146 assigns one PTX WMMA's full work to each SASS-level HMMA trace instruction.
No target residual is needed to diagnose either error.

## MLX-side audit

The paper fixes the full design at a compact 4x4 mesh, SIMD32, 1 GHz and 1
TOp/s. H92 normalizes every component to only four lanes and uses
`paper_static`, active-window 2 paths. Inspect all 180 frozen run summaries for
physical/mapped PE count, maximum pipeline issues and dependency stalls. The
four-lane compute issue ceiling represents 256 GOp/s (4*32*2*1GHz), exactly the
paper's reduced-design peak rather than the 1-TOp/s full design used against
Xavier. H141 separately proves the current scoreboard can execute full-mesh
complete blocks with much higher concurrent issue.

H95's 24 structured + 8 dense transformer-layer sum is recorded as explicit
serialization, not silently corrected in H150. A successor must rebuild exact
H91 work on 16 physical lanes/current dependency semantics before deciding
whether cross-transformer-layer overlap is justified.

## Xavier-side audit

H146 generates Accel-Sim SASS-style `HMMA` trace instructions but labels each
as 4096 FMA equivalents, the work of one PTX 16x16x16 WMMA. Accel-Sim's frozen
Volta microbenchmark definition states `SASS_hmma_per_PTX_wmma=16`; therefore
one trace HMMA represents 4096/16=256 FMA equivalents. H146 overstates timed
work per trace instruction by exactly 16x and understates projected Xavier
cycles by the same factor.

## Acceptance gates

1. All frozen inputs qualify and result parents retain status/integrity.
2. Paper text and frozen expected contract agree on 4x4/SIMD32/1GHz/1TOp/s.
3. H92 has 45 models/180 runs, four normalized lanes, active-window 2 and
   paper-static dependency semantics.
4. H92 run summaries show at most four simultaneous pipeline issues and the
   four-lane peak is exactly 256 GOp/s versus the required 1 TOp/s.
5. H141 proves current scoreboard/full-mesh issue is available independently of
   Figure 21 targets.
6. H95 has five exact 24+8 additive serialized rows; no overlap is inferred.
7. H146 uses source-derived SASS-level HMMA traces and records 4096 FMA/HMMA.
8. Frozen Volta source fixes 16 SASS HMMA/PTX WMMA, deriving exactly 256
   FMA/trace-HMMA and a 16x Xavier cycle correction requirement.
9. The repair plan changes mechanisms/work semantics only, consumes no Figure
   21 target and applies no residual-derived factor.
10. H150 is diagnosis only; active completion remains 3/8.

Support means the diagnosis is correct, not that either corrected path has run.
The immutable result will be
`artifacts/results/fig21-scale-semantics-run155.json`.
