#pragma once
#include "matrix_wire.h"
#include "vector_wire.h"
#include <stdint.h>

#define MLX_PAIR_WIRE_MAGIC UINT64_C(0x4d4c585041493031)
#define MLX_PAIR_MATRIX UINT64_C(1)
#define MLX_PAIR_VECTOR UINT64_C(2)
#define MLX_PAIR_WIRE_BYTES 8832

// Version 1: a whole producer source and same-shape pointwise consumer.
// Matrix producer slots encode batch zero; remaining broadcast batches are
// derived from the same checked tensor layouts. Timing is a device property.
typedef struct mlx_pair_wire {
  uint64_t magic,version,producer_kind,producer_bytes;
  uint64_t producer_source,consumer_source,event_slots,input_mask;
  uint64_t flags,reserved[23];
  uint64_t producer[536];
  mlx_vector_wire consumer;
} mlx_pair_wire;

#ifdef __cplusplus
static_assert(sizeof(mlx_pair_wire)==MLX_PAIR_WIRE_BYTES,"pair wire ABI size");
#else
_Static_assert(sizeof(mlx_pair_wire)==MLX_PAIR_WIRE_BYTES,"pair wire ABI size");
#endif
