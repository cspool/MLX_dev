#ifndef MLX_VECTOR_WIRE_H
#define MLX_VECTOR_WIRE_H
#include "../physical_host/control_runtime.h"

#define MLX_VECTOR_WIRE_MAGIC UINT64_C(0x4d4c585645433031)
#define MLX_VECTOR_WIRE_VERSION UINT64_C(1)
#define MLX_VECTOR_KEEP_DIM UINT64_C(1)
#define MLX_VECTOR_NARROW_INPUT UINT64_C(2)
#define MLX_VECTOR_LITERAL_F64 UINT64_C(4) /* Descriptor representation; execution converts to FP32. */
#define MLX_VECTOR_PHASES 12
enum mlx_vector_kind { MLX_VECTOR_ADD=1,MLX_VECTOR_MUL=2,MLX_VECTOR_POW2=3,
  MLX_VECTOR_RSQRT=4,MLX_VECTOR_SILU=5,MLX_VECTOR_COS=6,MLX_VECTOR_SIN=7,
  MLX_VECTOR_NEG=8,MLX_VECTOR_MEAN=9,MLX_VECTOR_SOFTMAX=10 };
/* Phase-index tables are bounded control descriptors, not additional PE ROM.
 * Their eventual hardware storage/implementation is a separate RTL obligation. */
typedef struct {
  uint64_t magic,version,kind,flags,width,rom_count,constant_count,phase_count,reserved[8];
  mlx_host_operand a,b;
  mlx_host_tensor output;
  uint64_t rom[32],constants[16],phase_lengths[MLX_VECTOR_PHASES];
  uint64_t phase_indices[MLX_VECTOR_PHASES][32],tail_reserved[2];
} mlx_vector_wire;
MLX_HOST_STATIC_ASSERT(sizeof(mlx_vector_wire)==4288,"vector wire size");
MLX_HOST_STATIC_ASSERT(offsetof(mlx_vector_wire,a)==128 && offsetof(mlx_vector_wire,rom)==720,"vector wire offsets");
#endif
