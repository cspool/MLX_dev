# Complete BERT physical-path numerical evidence

This package certifies only the completed original `physical` QA run. It does
not certify the concurrent paired path, Chipyard, author hybrid, GPU or RTL/PPA.

`physical-qa-evidence.tar.gz` contains 115 SHA-checked files: original program,
options, acceptance report, complete runtime report/log, executed source
snapshots, reference inventory/report, and six actual logits plus their numeric
baseline counterparts. Checkpoint weights and executables are not included.
Paths inside original reports retain their original provenance; they are not
rewritten to imply the archived source is the currently running system.

- SHA256: `f17ee353e8d99f129db968d38e654b9d408ffd9e866d8e142699d66bb3810c97`
- Archive size: 958,418 bytes.
- Coverage: 1,141 original calls, 3,046 lowered steps, all 3,838 backend windows.
- All six logits are bitwise equal to the completed microcode baseline.
- Answers: Zurich / Zurich / Oslo. Framework maximum absolute error: 9.5367431640625e-6.
- All 8,064,152,859 physical requests drained; all buffers released.
- Target cycles: 92,401,432,887 component + 7,800 readback = 92,401,440,687.

The physical executor orders sources sequentially while retaining window-local
concurrency. Its cycle result must not replace the still-pending concurrent
paired result in the formal MLX event-vs-E2E error calculation.

The packager independently rechecked full-model identity, output correctness,
actual file equality, source/input/library hashes, coverage, resource drain,
every copied file and every archive member. Three rejection tests additionally
reject running, failed and paired attempts before creating a package. The
manifest lists all original source paths, sizes and hashes.
