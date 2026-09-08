#include "host_runtime.h"
#include "graph_runtime.h"
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)))=0;
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)))=0;
extern const unsigned char command_blob[],payload_blob[];
static uint64_t completion[4];
static volatile mlx_graph_result result;
static const volatile mlx_graph_asset assets[]={{0}};
static const volatile mlx_graph_task tasks[]={{.kind=0,.source_ordinal=0,.source_id=0,.batch_index=0,.batch_count=1,.bytes=0,.command=0},{.kind=3,.source_ordinal=1,.source_id=1,.batch_index=0,.batch_count=1,.bytes=8832,.command=(uintptr_t)(command_blob+0),.reserved=3},{.kind=1,.source_ordinal=3,.source_id=3,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+8832)}};
static const uint64_t dependencies_0[]={0};
static const uint64_t dependencies_1[]={0};
static const uint64_t dependencies_2[]={1};
static const uint64_t dependencies_3[]={2};
static const volatile mlx_graph_source source_table[]={{.source_id=0,.dependencies=(uintptr_t)dependencies_0,.dependency_count=0},{.source_id=1,.dependencies=(uintptr_t)dependencies_1,.dependency_count=1},{.source_id=2,.dependencies=(uintptr_t)dependencies_2,.dependency_count=1},{.source_id=3,.dependencies=(uintptr_t)dependencies_3,.dependency_count=1}};
static const volatile mlx_graph_program program={.magic=MLX_GRAPH_MAGIC,.version=2,.source_count=4,.task_count=3,.asset_count=0,.assets=(uintptr_t)assets,.tasks=(uintptr_t)tasks,.device_base=UINT64_C(6459228160),.device_bytes=UINT64_C(1048576),.scratch_offset=4096,.scratch_bytes=16384,.poll_limit=UINT64_C(1000000000000),.completion=(uintptr_t)completion,.reserved={(uintptr_t)source_table,0,0}};
static const unsigned char expected_0[]={0,0,112,189,0,0,32,190,0,0,130,190,0,0,180,190,0,0,230,190,0,0,12,191,0,0,37,191,0,0,32,190,0,0,255,190,0,0,87,191,0,64,151,191,0,0,195,191,0,192,238,191,0,64,13,192,0,0,130,190,0,0,87,191,0,128,182,191,0,192,0,192,0,64,38,192,0,192,75,192,0,64,113,192,0,0,62,191,0,0,87,191,0,0,112,191,0,128,132,191,0,0,145,191,0,128,157,191,0,0,170,191,0,32,35,192,0,0,57,192,0,224,78,192,0,192,100,192,0,160,122,192,0,64,136,192,0,48,147,192,0,96,139,192,0,32,158,192,0,224,176,192,0,160,195,192,0,96,214,192,0,32,233,192,0,224,251,192,0,128,182,191,0,0,195,191,0,128,207,191,0,0,220,191,0,128,232,191,0,0,245,191,0,192,0,192,0,32,158,192,0,16,169,192,0,0,180,192,0,240,190,192,0,224,201,192,0,208,212,192,0,192,223,192,0,80,7,193,0,176,16,193,0,16,26,193,0,112,35,193,0,208,44,193,0,48,54,193,0,144,63,193,0,0,7,192,0,64,13,192,0,128,19,192,0,192,25,192,0,0,32,192,0,64,38,192,0,128,44,192,0,176,234,192,0,160,245,192,0,72,0,193,0,192,5,193,0,56,11,193,0,176,16,193,0,40,22,193,0,240,72,193,0,80,82,193,0,176,91,193,0,16,101,193,0,112,110,193,0,208,119,193,0,152,128,193,0,0,180,190,0,64,151,191,0,192,0,192,0,224,53,192,0,0,107,192,0,16,144,192,0,160,170,192,0,0,230,190,0,0,195,191,0,64,38,192,0,0,107,192,0,224,151,192,0,64,186,192,0,160,220,192,0,0,12,191,0,192,238,191,0,192,75,192,0,16,144,192,0,64,186,192,0,112,228,192,0,80,7,193,0,48,197,192,0,192,223,192,0,80,250,192,0,112,10,193,0,184,23,193,0,0,37,193,0,72,50,193,0,0,255,192,0,176,16,193,0,224,33,193,0,16,51,193,0,64,68,193,0,112,85,193,0,160,102,193,0,104,28,193,0,128,49,193,0,152,70,193,0,176,91,193,0,200,112,193,0,240,130,193,0,124,141,193,0,144,63,193,0,216,76,193,0,32,90,193,0,104,103,193,0,176,116,193,0,252,128,193,0,160,135,193,0,208,119,193,0,128,132,193,0,24,141,193,0,176,149,193,0,72,158,193,0,224,166,193,0,120,175,193,0,8,152,193,0,148,162,193,0,32,173,193,0,172,183,193,0,56,194,193,0,196,204,193,0,80,215,193,0,68,142,193,0,232,148,193,0,140,155,193,0,48,162,193,0,212,168,193,0,120,175,193,0,28,182,193,0,16,184,193,0,168,192,193,0,64,201,193,0,216,209,193,0,112,218,193,0,8,227,193,0,160,235,193,0,220,225,193,0,104,236,193,0,244,246,193,0,192,0,194,0,6,6,194,0,76,11,194,0,146,16,194};
static const unsigned char expected_1[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
struct check {uint64_t base,offset,rank,width,count,shape[8],stride[8];const unsigned char *expected;};
static const volatile struct check checks[]={{.base=UINT64_C(6459295104),.offset=0,.rank=4,.width=4,.count=168,.shape={2,4,3,7},.stride={84,21,7,1},.expected=expected_0},{.base=UINT64_C(6459294400),.offset=0,.rank=3,.width=8,.count=24,.shape={2,4,3},.stride={12,3,1},.expected=expected_1}};
static const uint64_t source_ids[]={1,2,3,4};
int main(void){
if(mlx_graph_execute(&program,&result))return 10;
if(result.status||result.completed_sources!=4||result.host_calls!=1||result.view_elisions!=1)return 11;
if(result.device_calls!=1||result.asset_bytes!=0)return 14;
for(unsigned i=0;i<4;++i)if(completion[i]!=source_ids[i])return 12;
for(unsigned row=0;row<sizeof(checks)/sizeof(checks[0]);++row){
const volatile struct check *c=&checks[row];for(uint64_t flat=0;flat<c->count;++flat){uint64_t index=flat,at=c->offset;for(unsigned d=(unsigned)c->rank;d-->0;){at+=(index%c->shape[d])*c->stride[d];index/=c->shape[d];}const volatile unsigned char *actual=(const volatile unsigned char *)(uintptr_t)(c->base+at*c->width);for(unsigned b=0;b<c->width;++b)if(actual[b]!=c->expected[flat*c->width+b])return 13;}
}
mlx_clocked_pass();return 0;
}
