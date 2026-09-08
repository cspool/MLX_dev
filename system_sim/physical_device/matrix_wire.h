#ifndef MLX_MATRIX_WIRE_H
#define MLX_MATRIX_WIRE_H
#include "../physical_host/control_runtime.h"

/* Project wire encoding for the existing matrix execution profile. Not the
 * paper's unpublished encoding, and not tagged native-v2 RoCC compatibility. */
#define MLX_MATRIX_WIRE_MAGIC UINT64_C(0x4d4c584d41543031)
#define MLX_MATRIX_WIRE_VERSION UINT64_C(1)
#define MLX_MATRIX_TRANSPOSE_B UINT64_C(1)
#define MLX_MATRIX_HAS_BIAS UINT64_C(2)
typedef struct {
  uint64_t magic,version,flags,m,n,k,a_batch,b_batch,output_batch;
  uint64_t prologue_count,body_count,epilogue_count,reserved[4];
  mlx_host_tensor a,b,bias,output;
  uint64_t words[32];
} mlx_matrix_wire;
MLX_HOST_STATIC_ASSERT(sizeof(mlx_matrix_wire)==1088,"matrix wire size");
MLX_HOST_STATIC_ASSERT(offsetof(mlx_matrix_wire,a)==128 && offsetof(mlx_matrix_wire,words)==832,"matrix wire offsets");

#define MLX_MATRIX_REG_ID UINT64_C(0)
#define MLX_MATRIX_REG_DESCRIPTOR UINT64_C(8)
#define MLX_MATRIX_REG_LAUNCH UINT64_C(16)
#define MLX_MATRIX_REG_STATUS UINT64_C(24)
#define MLX_MATRIX_REG_CYCLES UINT64_C(32)
#define MLX_MATRIX_REG_ERROR UINT64_C(40)
#define MLX_MATRIX_STATUS_BUSY UINT64_C(1)
#define MLX_MATRIX_STATUS_DONE UINT64_C(2)
#define MLX_MATRIX_STATUS_ERROR UINT64_C(4)
#define MLX_MATRIX_DATA_OFFSET UINT64_C(0x1000)
#endif
