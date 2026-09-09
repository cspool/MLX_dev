#include "graph_runtime.h"
#include "../physical_device/matrix_wire.h"
#include "../physical_device/vector_wire.h"
#include "../physical_device/memory_wire.h"
#include "../physical_device/pair_wire.h"
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
static int ready(const volatile mlx_graph_source *sources,const volatile uint64_t *completion,uint64_t ordinal,uint64_t except){
  const volatile mlx_graph_source *s=&sources[ordinal];const volatile uint64_t *deps=(const volatile uint64_t *)(uintptr_t)s->dependencies;
  for(uint64_t i=0;i<s->dependency_count;++i){uint64_t d=deps[i];if(d!=except && completion[d]!=sources[d].source_id+1)return 0;}
  return 1;
}
static int finish_group(const volatile mlx_graph_source *sources,volatile mlx_graph_source_group *groups,
                        uint64_t ordinal,volatile mlx_graph_result *result){
  volatile mlx_graph_source_group *g=&groups[sources[ordinal].reserved-1];
  if(g->completed_stages>=g->stage_count)return 0;
  if(++g->completed_stages==g->stage_count)++result->completed_source_groups;
  return 1;
}

int mlx_graph_execute(const volatile mlx_graph_program *p,volatile mlx_graph_result *r){
  if(!p||!r || (uintptr_t)p%8 || (uintptr_t)r%8)return 1;
  r->status=r->last_task=r->completed_sources=r->host_calls=r->device_calls=r->view_elisions=r->asset_bytes=r->reserved=0;
  if(p->magic!=MLX_GRAPH_MAGIC||(p->version<1||p->version>3)||!p->poll_limit||p->device_base%4096||p->device_bytes>UINT64_MAX-p->device_base)return fail(r,1);
  int v2=p->version>=2,v3=p->version==3;
  for(unsigned i=v3?3:v2?1:0;i<3;++i)if(p->reserved[i])return fail(r,1);
  if(p->scratch_offset<MLX_MATRIX_DATA_OFFSET||p->scratch_offset%8||p->scratch_offset>p->device_bytes||p->scratch_bytes>p->device_bytes-p->scratch_offset)return fail(r,1);
  if(p->asset_count>UINT64_MAX/sizeof(mlx_graph_asset)||p->task_count>UINT64_MAX/sizeof(mlx_graph_task)||p->assets%8||p->tasks%8||p->assets>UINT64_MAX-p->asset_count*sizeof(mlx_graph_asset)||p->tasks>UINT64_MAX-p->task_count*sizeof(mlx_graph_task))return fail(r,1);
  if((p->asset_count&&!p->assets)||(p->task_count&&!p->tasks))return fail(r,1);
  if(p->source_count>UINT64_MAX/8||p->completion%8||p->completion>UINT64_MAX-p->source_count*8||(p->source_count&&!p->completion))return fail(r,1);
  volatile uint64_t *completion=(volatile uint64_t *)(uintptr_t)p->completion;for(uint64_t i=0;i<p->source_count;++i)completion[i]=0;
  const volatile mlx_graph_source *sources=(const volatile mlx_graph_source *)(uintptr_t)p->reserved[0];
  volatile mlx_graph_source_group *groups=(volatile mlx_graph_source_group *)(uintptr_t)p->reserved[2];
  if(v2){
    if(!sources||p->reserved[0]%8||p->source_count>UINT64_MAX/sizeof(*sources)||p->reserved[0]>UINT64_MAX-p->source_count*sizeof(*sources))return fail(r,1);
    for(uint64_t i=0;i<p->source_count;++i){
      const volatile mlx_graph_source *s=&sources[i];
      if((v3?(!s->reserved||s->reserved>p->reserved[1]):s->reserved)||s->source_id==UINT64_MAX||s->dependency_count>i||s->dependencies%8||s->dependency_count>UINT64_MAX/8||s->dependencies>UINT64_MAX-s->dependency_count*8||(s->dependency_count&&!s->dependencies))return fail(r,1);
      const volatile uint64_t *deps=(const volatile uint64_t *)(uintptr_t)s->dependencies;
      for(uint64_t j=0;j<s->dependency_count;++j)if(deps[j]>=i||(j&&deps[j]<=deps[j-1]))return fail(r,1);
    }
  }
  if(v3){
    uint64_t count=p->reserved[1],cursor=0;
    if(!count||count>p->source_count||!groups||p->reserved[2]%8||count>UINT64_MAX/sizeof(*groups)||p->reserved[2]>UINT64_MAX-count*sizeof(*groups))return fail(r,1);
    for(uint64_t i=0;i<count;++i){
      volatile mlx_graph_source_group *g=&groups[i];
      if(g->source_id==UINT64_MAX||(i&&g->source_id<=groups[i-1].source_id)||g->stage_begin!=cursor||!g->stage_count||g->stage_count>p->source_count-cursor)return fail(r,1);
      for(uint64_t j=0;j<g->stage_count;++j)if(sources[cursor+j].reserved!=i+1)return fail(r,1);
      cursor+=g->stage_count;g->completed_stages=0;
    }
    if(cursor!=p->source_count)return fail(r,1);
  }
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
    if((t->reserved&&(!v2||t->kind!=MLX_GRAPH_PAIR))||t->source_ordinal>=p->source_count||t->source_id==UINT64_MAX||t->batch_index!=next_batch||!t->batch_count||t->batch_index>=t->batch_count)return fail(r,4);
    if(!next_batch){
      if(v2){next_source=t->source_ordinal;if(completion[next_source]||sources[next_source].source_id!=t->source_id||!ready(sources,completion,next_source,UINT64_MAX))return fail(r,4);}
      batches=t->batch_count;source_id=t->source_id;
    }else if(batches!=t->batch_count||source_id!=t->source_id)return fail(r,4);
    if(t->source_ordinal!=next_source)return fail(r,4);
    uint64_t consumer=UINT64_MAX,consumer_id=0;
    if(t->kind==MLX_GRAPH_VIEW){
      if(t->command||t->bytes||t->batch_count!=1)return fail(r,4);
      ++r->view_elisions;
    }else if(t->kind==MLX_GRAPH_HOST){
      if(t->batch_count!=1||t->command%8||!t->command||t->bytes!=sizeof(mlx_host_control_command))return fail(r,4);
      fence();unsigned status=mlx_host_control_execute((const volatile mlx_host_control_command *)(uintptr_t)t->command);fence();
      if(status)return fail(r,UINT64_C(0x100)+status);
      ++r->host_calls;
    }else if(t->kind==MLX_GRAPH_DEVICE||t->kind==MLX_GRAPH_PAIR){
      if(t->command%8||!t->command||t->bytes>p->scratch_bytes||t->command>UINT64_MAX-t->bytes)return fail(r,4);
      uint64_t magic=read64(t->command);uint64_t bytes=magic==MLX_MATRIX_WIRE_MAGIC?sizeof(mlx_matrix_wire):magic==MLX_VECTOR_WIRE_MAGIC?sizeof(mlx_vector_wire):magic==MLX_MEMORY_WIRE_MAGIC?(t->bytes==sizeof(mlx_memory_wire_v2)?sizeof(mlx_memory_wire_v2):sizeof(mlx_memory_wire)):0;
      if(t->kind==MLX_GRAPH_PAIR){
        if(!v2||!t->reserved||t->reserved>p->source_count||t->batch_count!=1)return fail(r,4);
        consumer=t->reserved-1;
        if(consumer<=next_source||completion[consumer]||!ready(sources,completion,consumer,next_source)||magic!=MLX_PAIR_WIRE_MAGIC||t->bytes!=sizeof(mlx_pair_wire))return fail(r,4);
        const volatile mlx_pair_wire *wire=(const volatile mlx_pair_wire *)(uintptr_t)t->command;
        consumer_id=sources[consumer].source_id;
        if(wire->version!=1||wire->producer_source!=source_id||wire->consumer_source!=consumer_id)return fail(r,4);
        bytes=sizeof(mlx_pair_wire);
#ifndef MLX_GRAPH_CLOCKED_ROCC
        return fail(r,4); // The legacy Spike MMIO device has no pair capability.
#endif
      }
      if(!bytes||bytes!=t->bytes)return fail(r,4);
      if(device_status(p->device_base)&MLX_MATRIX_STATUS_BUSY)return fail(r,5);
      copy_bytes(scratch,t->command,t->bytes);fence();device_launch(p->device_base,scratch,t->bytes);fence();
      uint64_t status=MLX_MATRIX_STATUS_BUSY;
      for(uint64_t spin=0;spin<p->poll_limit;++spin){status=device_status(p->device_base);if(!(status&MLX_MATRIX_STATUS_BUSY))break;}
      if(status&MLX_MATRIX_STATUS_BUSY)return fail(r,6);
      if(status!=MLX_MATRIX_STATUS_DONE)return fail(r,7);
      fence();++r->device_calls;
    }else return fail(r,4);
    if(++next_batch==batches){
      completion[next_source]=source_id+1;next_batch=0;++r->completed_sources;
      if(v3&&!finish_group(sources,groups,next_source,r))return fail(r,4);
      if(consumer!=UINT64_MAX){completion[consumer]=consumer_id+1;++r->completed_sources;if(v3&&!finish_group(sources,groups,consumer,r))return fail(r,4);}
      if(!v2)++next_source;
    }
  }
  if(next_batch||r->completed_sources!=p->source_count||(v3&&r->completed_source_groups!=p->reserved[1]))return fail(r,4);
  fence();return 0;
}
