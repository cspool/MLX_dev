#ifndef MLX_PHYSICAL_HOST_CONTROL_RUNTIME_H
#define MLX_PHYSICAL_HOST_CONTROL_RUNTIME_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
#define MLX_HOST_STATIC_ASSERT static_assert
extern "C" {
#else
#define MLX_HOST_STATIC_ASSERT _Static_assert
#endif

/* Host-memory ABI only. These opcodes select RISC-V C routines, not MLX PE
 * instructions or the existing tagged native-v2 RoCC command encoding. */
#define MLX_HOST_CONTROL_MAGIC UINT64_C(0x4d4c584843545231)
#define MLX_HOST_CONTROL_VERSION UINT64_C(1)
#define MLX_HOST_CONTROL_MASK_VERSION UINT64_C(2)
#define MLX_HOST_MAX_RANK 8
enum mlx_host_dtype { MLX_HOST_F16=0, MLX_HOST_F32=1, MLX_HOST_I64=2, MLX_HOST_BOOL=3 };
enum mlx_host_access { MLX_HOST_READ=1, MLX_HOST_WRITE=2 };
enum mlx_host_operand_kind { MLX_HOST_NONE=0, MLX_HOST_TENSOR=1, MLX_HOST_SCALAR=2 };
enum mlx_host_control_op { MLX_HOST_ARANGE=1, MLX_HOST_ADD=2, MLX_HOST_MUL=3, MLX_HOST_LE=4, MLX_HOST_ARGMAX=5,
  MLX_HOST_GE=6, MLX_HOST_BITWISE_AND=7, MLX_HOST_ALL=8, MLX_HOST_GUARD=9 };
enum mlx_host_status { MLX_HOST_OK=0, MLX_HOST_BAD_DESCRIPTOR=1, MLX_HOST_BAD_TYPE=2,
  MLX_HOST_BAD_SHAPE=3, MLX_HOST_BOUNDS=4, MLX_HOST_OVERLAP=5, MLX_HOST_UNSUPPORTED=6,
  MLX_HOST_FP_MODE=7, MLX_HOST_GUARD_FAILED=8 };
#define MLX_HOST_KEEP_DIM UINT64_C(1)

typedef struct {
  uint64_t base, bytes, offset, rank, dtype, access;
  uint64_t shape[MLX_HOST_MAX_RANK], stride[MLX_HOST_MAX_RANK];
} mlx_host_tensor;
typedef struct {
  uint64_t kind, scalar_dtype, scalar_bits, reserved;
  mlx_host_tensor tensor;
} mlx_host_operand;
typedef struct {
  uint64_t magic, version, opcode, flags, extent, reserved;
  mlx_host_operand a, b;
  mlx_host_tensor output;
} mlx_host_control_command;

MLX_HOST_STATIC_ASSERT(sizeof(mlx_host_tensor)==176,"host tensor ABI size");
MLX_HOST_STATIC_ASSERT(sizeof(mlx_host_operand)==208,"host operand ABI size");
MLX_HOST_STATIC_ASSERT(sizeof(mlx_host_control_command)==640,"host control ABI size");
MLX_HOST_STATIC_ASSERT(offsetof(mlx_host_control_command,a)==48 && offsetof(mlx_host_control_command,b)==256 && offsetof(mlx_host_control_command,output)==464,"host control ABI offsets");

/* Descriptors refer to ordinary CPU-addressable buffers. Validation does not
 * prove PMP/MMU permissions, RAM mapping, or accelerator cache coherence.
 * Invalid static descriptors are rejected before any output store. Version 2
 * registers GE/Boolean ALL/AND/GUARD; version 1 remains byte-compatible.
 * GUARD compares an actual one-element Boolean tensor to a Boolean scalar
 * specialization and returns GUARD_FAILED before publishing output on mismatch. */
enum mlx_host_status mlx_host_control_execute(const volatile mlx_host_control_command *command);
uint32_t mlx_host_half_to_float_bits(uint16_t bits);
/* Actual CPU readback into an ordinary caller-owned dense buffer. */
enum mlx_host_status mlx_host_tensor_readback(const volatile mlx_host_tensor *tensor,void *destination,uint64_t capacity_bytes);
#ifdef __cplusplus
}
#endif
#endif
