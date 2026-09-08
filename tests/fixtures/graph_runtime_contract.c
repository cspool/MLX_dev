#include "graph_runtime.h"
#define BASE UINT64_C(0x100000000)
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)))=0;
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)))=0;
static volatile uint64_t completion[2];
static volatile mlx_graph_result result;
static volatile mlx_graph_task task;
static volatile mlx_graph_asset asset;
static volatile mlx_host_control_command bad_control;
static const uint64_t payload=UINT64_C(0x123456789abcdef0);
static volatile mlx_graph_program program={.magic=MLX_GRAPH_MAGIC,.version=1,.source_count=1,.task_count=1,
 .assets=(uintptr_t)&asset,.tasks=(uintptr_t)&task,.device_base=BASE,.device_bytes=65536,.scratch_offset=4096,.scratch_bytes=16384,.poll_limit=1000,.completion=(uintptr_t)completion};
static void reset(void){
  program.magic=MLX_GRAPH_MAGIC;program.source_count=1;program.task_count=1;program.asset_count=0;program.reserved[0]=0;
  task.kind=MLX_GRAPH_VIEW;task.source_ordinal=0;task.source_id=7;task.batch_index=0;task.batch_count=1;task.command=0;task.bytes=0;task.reserved=0;
}
int main(void){
  reset();if(mlx_graph_execute(&program,&result)||result.completed_sources!=1||completion[0]!=8||result.view_elisions!=1)return 1;
  reset();program.magic=0;if(mlx_graph_execute(&program,&result)!=1)return 2;
  reset();task.kind=3;if(mlx_graph_execute(&program,&result)!=4||completion[0])return 3;
  reset();task.source_ordinal=1;if(mlx_graph_execute(&program,&result)!=4)return 4;
  reset();task.batch_index=1;if(mlx_graph_execute(&program,&result)!=4)return 5;
  reset();task.batch_count=0;if(mlx_graph_execute(&program,&result)!=4)return 6;
  reset();task.source_id=UINT64_MAX;if(mlx_graph_execute(&program,&result)!=4)return 7;
  reset();program.source_count=2;if(mlx_graph_execute(&program,&result)!=4||result.completed_sources!=1||completion[0]!=8||completion[1])return 8;
  reset();task.kind=MLX_GRAPH_HOST;task.command=(uintptr_t)&bad_control;task.bytes=sizeof(bad_control);if(mlx_graph_execute(&program,&result)!=0x101||result.host_calls)return 9;
  reset();task.kind=MLX_GRAPH_DEVICE;task.command=(uintptr_t)&payload;task.bytes=8;if(mlx_graph_execute(&program,&result)!=4||result.device_calls)return 10;
  reset();program.asset_count=1;asset.source=(uintptr_t)&payload;asset.bytes=8;asset.destination=BASE+65532;if(mlx_graph_execute(&program,&result)!=3)return 11;
  reset();program.asset_count=1;asset.destination=BASE+4096;if(mlx_graph_execute(&program,&result)!=3)return 12;
  reset();program.asset_count=1;asset.destination=BASE+0x6000;if(mlx_graph_execute(&program,&result)||result.asset_bytes!=8||*(volatile uint64_t *)(uintptr_t)(BASE+0x6000)!=payload)return 13;
  reset();program.reserved[0]=1;if(mlx_graph_execute(&program,&result)!=1)return 14;
  reset();program.task_count=0;if(mlx_graph_execute(&program,&result)!=4||result.completed_sources)return 15;
  return 0;
}
