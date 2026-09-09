#include "graph_runtime.h"
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)))=0;
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)))=0;
static uint64_t complete[3],dependencies[]={0,1};
static volatile mlx_graph_source sources[3];
static volatile mlx_graph_source_group groups[2];
static volatile mlx_graph_task tasks[3];
static volatile mlx_graph_result result;
static volatile mlx_graph_program program={
  .magic=MLX_GRAPH_MAGIC,.version=3,.source_count=3,.task_count=3,.tasks=(uintptr_t)tasks,
  .device_base=UINT64_C(0x100000000),.device_bytes=1048576,.scratch_offset=4096,.scratch_bytes=16384,
  .poll_limit=100,.completion=(uintptr_t)complete,
  .reserved={(uintptr_t)sources,2,(uintptr_t)groups}
};
static void reset(void){
  program.version=3;program.task_count=3;program.reserved[1]=2;
  for(unsigned i=0;i<3;++i){
    sources[i].source_id=10+i;sources[i].dependencies=i?(uintptr_t)&dependencies[i-1]:0;
    sources[i].dependency_count=i?1:0;sources[i].reserved=i==2?2:1;
    tasks[i].kind=MLX_GRAPH_VIEW;tasks[i].source_ordinal=i;tasks[i].source_id=10+i;
    tasks[i].batch_index=0;tasks[i].batch_count=1;tasks[i].command=tasks[i].bytes=tasks[i].reserved=0;
  }
  groups[0].source_id=50;groups[0].stage_begin=0;groups[0].stage_count=2;groups[0].completed_stages=0;
  groups[1].source_id=51;groups[1].stage_begin=2;groups[1].stage_count=1;groups[1].completed_stages=0;
}
int main(void){
  for(unsigned test=0;test<14;++test){
    reset();unsigned expected=test?1:0;
    if(test==1){program.task_count=1;expected=4;}
    if(test==2){program.task_count=2;expected=4;}
    if(test==3){tasks[1].source_ordinal=2;tasks[1].source_id=12;expected=4;}
    if(test==4){tasks[1].source_ordinal=0;tasks[1].source_id=10;expected=4;}
    if(test==5)groups[0].stage_count=0;
    if(test==6)groups[1].stage_begin=1;
    if(test==7)groups[1].stage_count=2;
    if(test==8)sources[1].reserved=2;
    if(test==9)groups[1].source_id=50;
    if(test==10)program.version=2;
    if(test==11)program.version=4;
    if(test==12)program.reserved[1]=4;
    if(test==13)sources[0].reserved=0;
    if(mlx_graph_execute(&program,&result)!=(int)expected)return 100+test;
    if(!test&&(result.completed_sources!=3||result.completed_source_groups!=2||groups[0].completed_stages!=2||groups[1].completed_stages!=1))return 150;
    if((test==1||test==3||test==4)&&(result.completed_sources!=1||result.completed_source_groups||groups[0].completed_stages!=1||groups[1].completed_stages))return 151;
    if(test==2&&(result.completed_sources!=2||result.completed_source_groups!=1||groups[0].completed_stages!=2||groups[1].completed_stages))return 152;
    if(result.device_calls||result.host_calls)return 153;
  }
  return 0;
}
