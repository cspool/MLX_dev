#ifndef MLX_GRAPH_RUNTIME_H
#define MLX_GRAPH_RUNTIME_H
#include "control_runtime.h"
#define MLX_GRAPH_MAGIC UINT64_C(0x4d4c584752503031)
enum mlx_graph_kind { MLX_GRAPH_VIEW=0,MLX_GRAPH_HOST=1,MLX_GRAPH_DEVICE=2 };
typedef struct { uint64_t source,destination,bytes,reserved; } mlx_graph_asset;
typedef struct {
  uint64_t kind,source_ordinal,source_id,batch_index,batch_count,command,bytes,reserved;
} mlx_graph_task;
typedef struct {
  uint64_t magic,version,source_count,task_count,asset_count,assets,tasks;
  uint64_t device_base,device_bytes,scratch_offset,scratch_bytes,poll_limit,completion,reserved[3];
} mlx_graph_program;
typedef struct {
  uint64_t status,last_task,completed_sources,host_calls,device_calls,view_elisions,asset_bytes,reserved;
} mlx_graph_result;
MLX_HOST_STATIC_ASSERT(sizeof(mlx_graph_asset)==32 && sizeof(mlx_graph_task)==64 && sizeof(mlx_graph_program)==128 && sizeof(mlx_graph_result)==64,"graph host ABI layout");
int mlx_graph_execute(const volatile mlx_graph_program *program,volatile mlx_graph_result *result);
#endif
