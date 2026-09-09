#ifndef MLX_GRAPH_RUNTIME_H
#define MLX_GRAPH_RUNTIME_H
#include "control_runtime.h"
#define MLX_GRAPH_MAGIC UINT64_C(0x4d4c584752503031)
enum mlx_graph_kind { MLX_GRAPH_VIEW=0,MLX_GRAPH_HOST=1,MLX_GRAPH_DEVICE=2,MLX_GRAPH_PAIR=3 };
// Version 2: program.reserved[0] points to source_count source records.
// Pair task.reserved is consumer ordinal + 1; zero for all other tasks.
// Version 1 continues to require all reserved fields zero and ordered tasks.
// Version 3 keeps source_count as executable lowered-stage slots. reserved[1]
// is original group count, reserved[2] points to writable group records;
// source.reserved is group ordinal+1. Groups partition every stage exactly once.
typedef struct { uint64_t source_id,dependencies,dependency_count,reserved; } mlx_graph_source;
typedef struct { uint64_t source_id,stage_begin,stage_count,completed_stages; } mlx_graph_source_group;
typedef struct { uint64_t source,destination,bytes,reserved; } mlx_graph_asset;
typedef struct {
  uint64_t kind,source_ordinal,source_id,batch_index,batch_count,command,bytes,reserved;
} mlx_graph_task;
typedef struct {
  uint64_t magic,version,source_count,task_count,asset_count,assets,tasks;
  uint64_t device_base,device_bytes,scratch_offset,scratch_bytes,poll_limit,completion,reserved[3];
} mlx_graph_program;
typedef struct {
  uint64_t status,last_task,completed_sources,host_calls,device_calls,view_elisions,asset_bytes;
  union { uint64_t reserved; uint64_t completed_source_groups; };
} mlx_graph_result;
MLX_HOST_STATIC_ASSERT(sizeof(mlx_graph_source)==32 && sizeof(mlx_graph_asset)==32 && sizeof(mlx_graph_task)==64 && sizeof(mlx_graph_program)==128 && sizeof(mlx_graph_result)==64,"graph host ABI layout");
MLX_HOST_STATIC_ASSERT(sizeof(mlx_graph_source_group)==32,"source group ABI layout");
int mlx_graph_execute(const volatile mlx_graph_program *program,volatile mlx_graph_result *result);
#endif
