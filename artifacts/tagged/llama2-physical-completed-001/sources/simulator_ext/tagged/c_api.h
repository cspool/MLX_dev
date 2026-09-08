#ifndef MLX_TAGGED_C_API_H
#define MLX_TAGGED_C_API_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlx_tagged_model mlx_tagged_model;
/* Each handle owns its inputs. NULL timings selects defaults. No C++ exception
 * crosses this ABI. JSON pointers remain valid until the next JSON query on
 * the same handle, or destruction. Separate handles have independent state. */
mlx_tagged_model *mlx_tagged_create(const char *program_json, const char *timing_json,
                                   char *error_buffer, size_t error_capacity);
/* 0 = still running; 1 = complete; -1 = error. Completion polling is idempotent. */
int mlx_tagged_tick(mlx_tagged_model *model);
uint64_t mlx_tagged_cycles(const mlx_tagged_model *model);
const char *mlx_tagged_snapshot(mlx_tagged_model *model);
/* NULL before completion; does not implicitly advance the simulation. */
const char *mlx_tagged_result(mlx_tagged_model *model);
const char *mlx_tagged_error(const mlx_tagged_model *model);
void mlx_tagged_destroy(mlx_tagged_model *model);

#ifdef __cplusplus
}
#endif
#endif
