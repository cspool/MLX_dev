#include "graph_runtime.h"
#include "../physical_device/matrix_wire.h"
#include "../physical_device/vector_wire.h"
#include "../physical_device/memory_wire.h"
#ifdef MLX_GRAPH_CLOCKED_ROCC
#include "../clocked_rocc/host_runtime.h"
#endif

static void fence(void){__asm__ volatile("fence iorw, iorw":::"memory");}
static uint64_t read64(uint64_t address){uint64_t value;__asm__ volatile("ld %0,0(%1)":"=r"(value):"r"(address):"memory");return value;}
static void write64(uint64_t address,uint64_t value){__asm__ volatile("sd %0,0(%1)"::"r"(value),"r"(address):"memory");}
static uint64_t device_id(uint64_t base){
#ifdef MLX_GRAPH_CLOCKED_ROCC
  (void)base;return mlx_clocked_status(14)==MLX_CLOCKED_ROCC_MAGIC?MLX_MATRIX_WIRE_MAGIC:0;
#else
  return read64(base+MLX_MATRIX_REG_ID);
#endif
}
static uint64_t device_status(uint64_t base){
#ifdef MLX_GRAPH_CLOCKED_ROCC
  (void)base;return mlx_clocked_status(0);
#else
  return read64(base+MLX_MATRIX_REG_STATUS);
#endif
}
static void device_launch(uint64_t base,uint64_t descriptor,uint64_t bytes){
#ifdef MLX_GRAPH_CLOCKED_ROCC
  (void)base;mlx_clocked_launch(descriptor,bytes);
#else
  (void)bytes;write64(base+MLX_MATRIX_REG_DESCRIPTOR,descriptor);write64(base+MLX_MATRIX_REG_LAUNCH,1);
#endif
}
static int within(uint64_t address,uint64_t bytes,uint64_t base,uint64_t size){return address>=base&&address-base<=size&&bytes<=size-(address-base);}
static void copy_bytes(uint64_t destination,uint64_t source,uint64_t bytes){
  while(bytes>=8 && !(source%8) && !(destination%8)){write64(destination,read64(source));destination+=8;source+=8;bytes-=8;}
  const volatile unsigned char *input=(const volatile unsigned char *)(uintptr_t)source;volatile unsigned char *output=(volatile unsigned char *)(uintptr_t)destination;
  for(uint64_t index=0;index<bytes;++index)output[index]=input[index];
}
static int fail(volatile mlx_graph_result *result,uint64_t status){result->status=status;fence();return (int)status;}

int mlx_graph_execute(const volatile mlx_graph_program *p,volatile mlx_graph_result *r){
  if(!p||!r || (uintptr_t)p%8 || (uintptr_t)r%8)return 1;
  r->status=r->last_task=r->completed_sources=r->host_calls=r->device_calls=r->view_elisions=r->asset_bytes=r->reserved=0;
  if(p->magic!=MLX_GRAPH_MAGIC||p->version!=1||!p->poll_limit||p->device_base%4096||p->device_bytes>UINT64_MAX-p->device_base)return fail(r,1);
  for(unsigned i=0;i<3;++i)if(p->reserved[i])return fail(r,1);
  if(p->scratch_offset<MLX_MATRIX_DATA_OFFSET||p->scratch_offset%8||p->scratch_offset>p->device_bytes||p->scratch_bytes>p->device_bytes-p->scratch_offset)return fail(r,1);
  if(p->asset_count>UINT64_MAX/sizeof(mlx_graph_asset)||p->task_count>UINT64_MAX/sizeof(mlx_graph_task)||p->assets%8||p->tasks%8||p->assets>UINT64_MAX-p->asset_count*sizeof(mlx_graph_asset)||p->tasks>UINT64_MAX-p->task_count*sizeof(mlx_graph_task))return fail(r,1);
  if((p->asset_count&&!p->assets)||(p->task_count&&!p->tasks))return fail(r,1);
  if(p->source_count>UINT64_MAX/8||p->completion%8||p->completion>UINT64_MAX-p->source_count*8||(p->source_count&&!p->completion))return fail(r,1);
  volatile uint64_t *completion=(volatile uint64_t *)(uintptr_t)p->completion;for(uint64_t i=0;i<p->source_count;++i)completion[i]=0;
  if(device_id(p->device_base)!=MLX_MATRIX_WIRE_MAGIC)return fail(r,2);
  const volatile mlx_graph_asset *assets=(const volatile mlx_graph_asset *)(uintptr_t)p->assets;
  uint64_t scratch=p->device_base+p->scratch_offset;
  for(uint64_t i=0;i<p->asset_count;++i){
    const volatile mlx_graph_asset *a=&assets[i];
    if(a->reserved||a->source>UINT64_MAX-a->bytes||(a->bytes&&!a->source)||!within(a->destination,a->bytes,p->device_base+MLX_MATRIX_DATA_OFFSET,p->device_bytes-MLX_MATRIX_DATA_OFFSET))return fail(r,3);
    if(a->bytes&&a->destination<scratch+p->scratch_bytes&&scratch<a->destination+a->bytes)return fail(r,3);
    copy_bytes(a->destination,a->source,a->bytes);r->asset_bytes+=a->bytes;
  }
  fence();const volatile mlx_graph_task *tasks=(const volatile mlx_graph_task *)(uintptr_t)p->tasks;uint64_t next_source=0,next_batch=0,batches=0,source_id=0;
  for(uint64_t index=0;index<p->task_count;++index){
    const volatile mlx_graph_task *t=&tasks[index];r->last_task=index;
    if(t->reserved||t->source_ordinal!=next_source||t->source_ordinal>=p->source_count||t->source_id==UINT64_MAX||t->batch_index!=next_batch||!t->batch_count||t->batch_index>=t->batch_count)return fail(r,4);
    if(!next_batch){batches=t->batch_count;source_id=t->source_id;}else if(batches!=t->batch_count||source_id!=t->source_id)return fail(r,4);
    if(t->kind==MLX_GRAPH_VIEW){
      if(t->command||t->bytes||t->batch_count!=1)return fail(r,4);
      ++r->view_elisions;
    }else if(t->kind==MLX_GRAPH_HOST){
      if(t->batch_count!=1||t->command%8||!t->command||t->bytes!=sizeof(mlx_host_control_command))return fail(r,4);
      fence();unsigned status=mlx_host_control_execute((const volatile mlx_host_control_command *)(uintptr_t)t->command);fence();
      if(status)return fail(r,UINT64_C(0x100)+status);
      ++r->host_calls;
    }else if(t->kind==MLX_GRAPH_DEVICE){
      if(t->command%8||!t->command||t->bytes>p->scratch_bytes||t->command>UINT64_MAX-t->bytes)return fail(r,4);
      uint64_t magic=read64(t->command);uint64_t bytes=magic==MLX_MATRIX_WIRE_MAGIC?sizeof(mlx_matrix_wire):magic==MLX_VECTOR_WIRE_MAGIC?sizeof(mlx_vector_wire):magic==MLX_MEMORY_WIRE_MAGIC?sizeof(mlx_memory_wire):0;
      if(!bytes||bytes!=t->bytes)return fail(r,4);
      if(device_status(p->device_base)&MLX_MATRIX_STATUS_BUSY)return fail(r,5);
      copy_bytes(scratch,t->command,t->bytes);fence();device_launch(p->device_base,scratch,t->bytes);fence();
      uint64_t status=MLX_MATRIX_STATUS_BUSY;
      for(uint64_t spin=0;spin<p->poll_limit;++spin){status=device_status(p->device_base);if(!(status&MLX_MATRIX_STATUS_BUSY))break;}
      if(status&MLX_MATRIX_STATUS_BUSY)return fail(r,6);
      if(status!=MLX_MATRIX_STATUS_DONE)return fail(r,7);
      fence();++r->device_calls;
    }else return fail(r,4);
    if(++next_batch==batches){completion[next_source]=source_id+1;next_batch=0;++next_source;++r->completed_sources;}
  }
  if(next_batch||next_source!=p->source_count)return fail(r,4);
  fence();return 0;
}
