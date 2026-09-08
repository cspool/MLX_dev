#include "host_runtime.h"
#include "graph_runtime.h"
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)))=0;
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)))=0;
extern const unsigned char command_blob[],payload_blob[];
static uint64_t completion[5];
static volatile mlx_graph_result result;
static const volatile mlx_graph_asset assets[]={{0}};
static const volatile mlx_graph_task tasks[]={{.kind=2,.source_ordinal=0,.source_id=0,.batch_index=0,.batch_count=1,.bytes=4288,.command=(uintptr_t)(command_blob+0)},{.kind=2,.source_ordinal=2,.source_id=2,.batch_index=0,.batch_count=1,.bytes=4288,.command=(uintptr_t)(command_blob+4288)},{.kind=3,.source_ordinal=1,.source_id=1,.batch_index=0,.batch_count=1,.bytes=8832,.command=(uintptr_t)(command_blob+8576),.reserved=4},{.kind=1,.source_ordinal=4,.source_id=4,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+17408)}};
static const uint64_t dependencies_0[]={0};
static const uint64_t dependencies_1[]={0};
static const uint64_t dependencies_2[]={0};
static const uint64_t dependencies_3[]={1,2};
static const uint64_t dependencies_4[]={3};
static const volatile mlx_graph_source source_table[]={{.source_id=0,.dependencies=(uintptr_t)dependencies_0,.dependency_count=0},{.source_id=1,.dependencies=(uintptr_t)dependencies_1,.dependency_count=1},{.source_id=2,.dependencies=(uintptr_t)dependencies_2,.dependency_count=0},{.source_id=3,.dependencies=(uintptr_t)dependencies_3,.dependency_count=2},{.source_id=4,.dependencies=(uintptr_t)dependencies_4,.dependency_count=1}};
static const volatile mlx_graph_program program={.magic=MLX_GRAPH_MAGIC,.version=2,.source_count=5,.task_count=4,.asset_count=0,.assets=(uintptr_t)assets,.tasks=(uintptr_t)tasks,.device_base=UINT64_C(6459228160),.device_bytes=UINT64_C(1048576),.scratch_offset=4096,.scratch_bytes=16384,.poll_limit=UINT64_C(1000000000000),.completion=(uintptr_t)completion,.reserved={(uintptr_t)source_table,0,0}};
static const unsigned char expected_0[]={0,192,0,196,0,198,0,200};
static const unsigned char expected_1[]={0,0,0,0,0,0,0,0};
struct check {uint64_t base,offset,rank,width,count,shape[8],stride[8];const unsigned char *expected;};
static const volatile struct check checks[]={{.base=UINT64_C(6459294016),.offset=0,.rank=2,.width=2,.count=4,.shape={1,4},.stride={4,1},.expected=expected_0},{.base=UINT64_C(6459293824),.offset=0,.rank=1,.width=8,.count=1,.shape={1},.stride={1},.expected=expected_1}};
static const uint64_t source_ids[]={1,2,3,4,5};
int main(void){
if(mlx_graph_execute(&program,&result))return 10;
if(result.status||result.completed_sources!=5||result.host_calls!=1||result.view_elisions!=0)return 11;
if(result.device_calls!=3||result.asset_bytes!=0)return 14;
for(unsigned i=0;i<5;++i)if(completion[i]!=source_ids[i])return 12;
for(unsigned row=0;row<sizeof(checks)/sizeof(checks[0]);++row){
const volatile struct check *c=&checks[row];for(uint64_t flat=0;flat<c->count;++flat){uint64_t index=flat,at=c->offset;for(unsigned d=(unsigned)c->rank;d-->0;){at+=(index%c->shape[d])*c->stride[d];index/=c->shape[d];}const volatile unsigned char *actual=(const volatile unsigned char *)(uintptr_t)(c->base+at*c->width);for(unsigned b=0;b<c->width;++b)if(actual[b]!=c->expected[flat*c->width+b])return 13;}
}
mlx_clocked_pass();return 0;
}
