#pragma once
#include "tensor.h"

namespace mlx::tensor_model {
// Functional PE microcode only: no cycle/DMA completion is invented here.
// Fixed 16 x 64-byte RF and 8-KiB SPM; F32 mode uses 16 lanes per RF vector.
struct MatrixInstructionStats {
  uint64_t calls=0, tiles=0, instructions=0, inactive_row_instructions=0;
  uint64_t mul_lanes=0, add_lanes=0, memory_read_bytes=0, memory_write_bytes=0;
  uint64_t opcode_counts[14]{};
  unsigned max_rom_words=0, max_spm_bytes=0;
  Json::Value json() const;
};
void execute_matrix_program(const Json::Value &program, const Tensor &a, const Tensor &b,
                            const Tensor *bias, bool transposed_b, uint64_t a_batch,
                            uint64_t b_batch, uint64_t m, uint64_t n, uint64_t k,
                            Tensor &output, uint64_t output_batch, MatrixInstructionStats &stats);
} // namespace mlx::tensor_model
