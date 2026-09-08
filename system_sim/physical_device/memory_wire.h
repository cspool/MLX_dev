#ifndef MLX_MEMORY_WIRE_H
#define MLX_MEMORY_WIRE_H
#include "../physical_host/control_runtime.h"

#define MLX_MEMORY_WIRE_MAGIC UINT64_C(0x4d4c584d454d3031)
#define MLX_MEMORY_SAME_DEVICE UINT64_C(1)
#define MLX_MEMORY_FORCE_COPY UINT64_C(2)
#define MLX_MEMORY_CONTIGUOUS UINT64_C(4)
#define MLX_MEMORY_PRESERVE_FORMAT UINT64_C(8)
#define MLX_MEMORY_STRICT_VIEW UINT64_C(16)
#define MLX_MEMORY_NONBLOCKING UINT64_C(32)
#define MLX_MEMORY_LITERAL_F64 UINT64_C(4)
enum mlx_memory_kind { MLX_MEMORY_EMBEDDING=1,MLX_MEMORY_WHERE=2,MLX_MEMORY_CAT=3,
  MLX_MEMORY_CAST=4,MLX_MEMORY_CAST_DEVICE=5,MLX_MEMORY_CONTIG=6,MLX_MEMORY_RESHAPE=7,
  MLX_MEMORY_TRANSPOSE=8,MLX_MEMORY_SLICE=9,MLX_MEMORY_SELECT=10,MLX_MEMORY_UNSQUEEZE=11,
  MLX_MEMORY_EXPAND=12,MLX_MEMORY_ALIAS=13,MLX_MEMORY_DROPOUT=14 };
typedef struct {
  uint64_t kind,scalar_dtype,scalar_bits,root_id;
  mlx_host_tensor tensor;
} mlx_memory_operand;
/* Bounded control/address metadata, not extra PE RF/SPM or opcodes. */
typedef struct {
  uint64_t magic,version,kind,mode,selector,flags,operand_count,argument_count;
  uint64_t word_count,output_root,param_count,reserved[5];
  uint64_t params[16],argument_indices[256];
  mlx_host_tensor output;
  mlx_memory_operand operands[64];
  uint64_t words[4],tail_reserved[6];
} mlx_memory_wire;
MLX_HOST_STATIC_ASSERT(sizeof(mlx_memory_wire)==15872,"memory wire size");
MLX_HOST_STATIC_ASSERT(offsetof(mlx_memory_wire,output)==2304 && offsetof(mlx_memory_wire,words)==15792,"memory wire offsets");
#endif
