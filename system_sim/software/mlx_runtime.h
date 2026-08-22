#pragma once

#include <stdint.h>

#include "rocc.h"

#define MLX_FUNCT_CONFIG 0
#define MLX_FUNCT_LAUNCH 1
#define MLX_FUNCT_WAIT 2
#define MLX_FUNCT_STATUS 3

typedef struct {
  uint8_t pe;
  uint8_t index;
  uint64_t word;
} mlx_program_entry_t;

static inline void mlx_fence(void) { asm volatile("fence" ::: "memory"); }

static inline uint64_t mlx_read_cycle(void) {
  uint64_t value;
  asm volatile("rdcycle %0" : "=r"(value));
  return value;
}

static inline void mlx_config(uint8_t target, uint8_t index, uint64_t value) {
  uint64_t address = ((uint64_t)target << 8) | index;
  ROCC_INSTRUCTION_SS(0, value, address, MLX_FUNCT_CONFIG);
}

static inline void mlx_launch(const void *input, void *output) {
  mlx_fence();
  ROCC_INSTRUCTION_SS(0, (uintptr_t)input, (uintptr_t)output, MLX_FUNCT_LAUNCH);
}

static inline uint64_t mlx_wait(void) {
  uint64_t status;
  ROCC_INSTRUCTION_DSS(0, status, 0, 0, MLX_FUNCT_WAIT);
  mlx_fence();
  return status;
}

static inline uint64_t mlx_status(uint8_t index) {
  uint64_t value;
  ROCC_INSTRUCTION_DSS(0, value, index, 0, MLX_FUNCT_STATUS);
  return value;
}
