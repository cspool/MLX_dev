#include "host_runtime.h"
#include "graph_runtime.h"
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)))=0;
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)))=0;
extern const unsigned char command_blob[],payload_blob[];
static uint64_t completion[45];
static volatile mlx_graph_result result;
static const volatile mlx_graph_asset assets[]={{.source=(uintptr_t)(payload_blob+0),.destination=UINT64_C(2164326400),.bytes=32},{.source=(uintptr_t)(payload_blob+32),.destination=UINT64_C(2164326464),.bytes=32},{.source=(uintptr_t)(payload_blob+64),.destination=UINT64_C(2164326528),.bytes=8},{.source=(uintptr_t)(payload_blob+72),.destination=UINT64_C(2164326592),.bytes=0}};
static const volatile mlx_graph_task tasks[]={{.kind=2,.source_ordinal=0,.source_id=0,.batch_index=0,.batch_count=1,.bytes=15872,.command=(uintptr_t)(command_blob+0)},{.kind=2,.source_ordinal=1,.source_id=1,.batch_index=0,.batch_count=1,.bytes=15872,.command=(uintptr_t)(command_blob+15872)},{.kind=0,.source_ordinal=2,.source_id=2,.batch_index=0,.batch_count=1,.bytes=0,.command=0},{.kind=2,.source_ordinal=3,.source_id=3,.batch_index=0,.batch_count=1,.bytes=4288,.command=(uintptr_t)(command_blob+31744)},{.kind=3,.source_ordinal=4,.source_id=4,.batch_index=0,.batch_count=1,.bytes=8832,.command=(uintptr_t)(command_blob+36032),.reserved=6},{.kind=2,.source_ordinal=6,.source_id=6,.batch_index=0,.batch_count=1,.bytes=4288,.command=(uintptr_t)(command_blob+44864)},{.kind=1,.source_ordinal=7,.source_id=7,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+49152)},{.kind=0,.source_ordinal=8,.source_id=8,.batch_index=0,.batch_count=1,.bytes=0,.command=0},{.kind=1,.source_ordinal=9,.source_id=9,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+49792)},{.kind=1,.source_ordinal=10,.source_id=10,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+50432)},{.kind=1,.source_ordinal=11,.source_id=11,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+51072)},{.kind=2,.source_ordinal=12,.source_id=12,.batch_index=0,.batch_count=1,.bytes=15872,.command=(uintptr_t)(command_blob+51712)},{.kind=1,.source_ordinal=13,.source_id=13,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+67584)},{.kind=0,.source_ordinal=14,.source_id=14,.batch_index=0,.batch_count=1,.bytes=0,.command=0},{.kind=2,.source_ordinal=15,.source_id=15,.batch_index=0,.batch_count=1,.bytes=15872,.command=(uintptr_t)(command_blob+68224)},{.kind=2,.source_ordinal=16,.source_id=16,.batch_index=0,.batch_count=1,.bytes=15872,.command=(uintptr_t)(command_blob+84096)},{.kind=0,.source_ordinal=17,.source_id=17,.batch_index=0,.batch_count=1,.bytes=0,.command=0},{.kind=2,.source_ordinal=18,.source_id=18,.batch_index=0,.batch_count=1,.bytes=4288,.command=(uintptr_t)(command_blob+99968)},{.kind=3,.source_ordinal=19,.source_id=19,.batch_index=0,.batch_count=1,.bytes=8832,.command=(uintptr_t)(command_blob+104256),.reserved=21},{.kind=2,.source_ordinal=21,.source_id=21,.batch_index=0,.batch_count=1,.bytes=4288,.command=(uintptr_t)(command_blob+113088)},{.kind=1,.source_ordinal=22,.source_id=22,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+117376)},{.kind=0,.source_ordinal=23,.source_id=23,.batch_index=0,.batch_count=1,.bytes=0,.command=0},{.kind=1,.source_ordinal=24,.source_id=24,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+118016)},{.kind=1,.source_ordinal=25,.source_id=25,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+118656)},{.kind=1,.source_ordinal=26,.source_id=26,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+119296)},{.kind=2,.source_ordinal=27,.source_id=27,.batch_index=0,.batch_count=1,.bytes=15872,.command=(uintptr_t)(command_blob+119936)},{.kind=1,.source_ordinal=28,.source_id=28,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+135808)},{.kind=0,.source_ordinal=29,.source_id=29,.batch_index=0,.batch_count=1,.bytes=0,.command=0},{.kind=2,.source_ordinal=30,.source_id=30,.batch_index=0,.batch_count=1,.bytes=15872,.command=(uintptr_t)(command_blob+136448)},{.kind=2,.source_ordinal=31,.source_id=31,.batch_index=0,.batch_count=1,.bytes=15872,.command=(uintptr_t)(command_blob+152320)},{.kind=0,.source_ordinal=32,.source_id=32,.batch_index=0,.batch_count=1,.bytes=0,.command=0},{.kind=2,.source_ordinal=33,.source_id=33,.batch_index=0,.batch_count=1,.bytes=4288,.command=(uintptr_t)(command_blob+168192)},{.kind=3,.source_ordinal=34,.source_id=34,.batch_index=0,.batch_count=1,.bytes=8832,.command=(uintptr_t)(command_blob+172480),.reserved=36},{.kind=2,.source_ordinal=36,.source_id=36,.batch_index=0,.batch_count=1,.bytes=4288,.command=(uintptr_t)(command_blob+181312)},{.kind=1,.source_ordinal=37,.source_id=37,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+185600)},{.kind=0,.source_ordinal=38,.source_id=38,.batch_index=0,.batch_count=1,.bytes=0,.command=0},{.kind=1,.source_ordinal=39,.source_id=39,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+186240)},{.kind=1,.source_ordinal=40,.source_id=40,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+186880)},{.kind=1,.source_ordinal=41,.source_id=41,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+187520)},{.kind=2,.source_ordinal=42,.source_id=42,.batch_index=0,.batch_count=1,.bytes=15872,.command=(uintptr_t)(command_blob+188160)},{.kind=1,.source_ordinal=43,.source_id=43,.batch_index=0,.batch_count=1,.bytes=640,.command=(uintptr_t)(command_blob+204032)},{.kind=0,.source_ordinal=44,.source_id=44,.batch_index=0,.batch_count=1,.bytes=0,.command=0}};
static const uint64_t dependencies_0[]={0};
static const uint64_t dependencies_1[]={0};
static const uint64_t dependencies_2[]={1};
static const uint64_t dependencies_3[]={2};
static const uint64_t dependencies_4[]={3};
static const uint64_t dependencies_5[]={4};
static const uint64_t dependencies_6[]={5};
static const uint64_t dependencies_7[]={0};
static const uint64_t dependencies_8[]={7};
static const uint64_t dependencies_9[]={8};
static const uint64_t dependencies_10[]={9};
static const uint64_t dependencies_11[]={7,10};
static const uint64_t dependencies_12[]={6,11};
static const uint64_t dependencies_13[]={12};
static const uint64_t dependencies_14[]={13};
static const uint64_t dependencies_15[]={14};
static const uint64_t dependencies_16[]={1,15};
static const uint64_t dependencies_17[]={16};
static const uint64_t dependencies_18[]={17};
static const uint64_t dependencies_19[]={18};
static const uint64_t dependencies_20[]={19};
static const uint64_t dependencies_21[]={20};
static const uint64_t dependencies_22[]={0};
static const uint64_t dependencies_23[]={22};
static const uint64_t dependencies_24[]={23};
static const uint64_t dependencies_25[]={24};
static const uint64_t dependencies_26[]={22,25};
static const uint64_t dependencies_27[]={21,26};
static const uint64_t dependencies_28[]={27};
static const uint64_t dependencies_29[]={28};
static const uint64_t dependencies_30[]={29};
static const uint64_t dependencies_31[]={16,30};
static const uint64_t dependencies_32[]={31};
static const uint64_t dependencies_33[]={32};
static const uint64_t dependencies_34[]={33};
static const uint64_t dependencies_35[]={34};
static const uint64_t dependencies_36[]={35};
static const uint64_t dependencies_37[]={0};
static const uint64_t dependencies_38[]={37};
static const uint64_t dependencies_39[]={38};
static const uint64_t dependencies_40[]={39};
static const uint64_t dependencies_41[]={37,40};
static const uint64_t dependencies_42[]={36,41};
static const uint64_t dependencies_43[]={42};
static const uint64_t dependencies_44[]={43};
static const volatile mlx_graph_source source_table[]={{.source_id=0,.dependencies=(uintptr_t)dependencies_0,.dependency_count=0},{.source_id=1,.dependencies=(uintptr_t)dependencies_1,.dependency_count=1},{.source_id=2,.dependencies=(uintptr_t)dependencies_2,.dependency_count=1},{.source_id=3,.dependencies=(uintptr_t)dependencies_3,.dependency_count=1},{.source_id=4,.dependencies=(uintptr_t)dependencies_4,.dependency_count=1},{.source_id=5,.dependencies=(uintptr_t)dependencies_5,.dependency_count=1},{.source_id=6,.dependencies=(uintptr_t)dependencies_6,.dependency_count=1},{.source_id=7,.dependencies=(uintptr_t)dependencies_7,.dependency_count=0},{.source_id=8,.dependencies=(uintptr_t)dependencies_8,.dependency_count=1},{.source_id=9,.dependencies=(uintptr_t)dependencies_9,.dependency_count=1},{.source_id=10,.dependencies=(uintptr_t)dependencies_10,.dependency_count=1},{.source_id=11,.dependencies=(uintptr_t)dependencies_11,.dependency_count=2},{.source_id=12,.dependencies=(uintptr_t)dependencies_12,.dependency_count=2},{.source_id=13,.dependencies=(uintptr_t)dependencies_13,.dependency_count=1},{.source_id=14,.dependencies=(uintptr_t)dependencies_14,.dependency_count=1},{.source_id=15,.dependencies=(uintptr_t)dependencies_15,.dependency_count=1},{.source_id=16,.dependencies=(uintptr_t)dependencies_16,.dependency_count=2},{.source_id=17,.dependencies=(uintptr_t)dependencies_17,.dependency_count=1},{.source_id=18,.dependencies=(uintptr_t)dependencies_18,.dependency_count=1},{.source_id=19,.dependencies=(uintptr_t)dependencies_19,.dependency_count=1},{.source_id=20,.dependencies=(uintptr_t)dependencies_20,.dependency_count=1},{.source_id=21,.dependencies=(uintptr_t)dependencies_21,.dependency_count=1},{.source_id=22,.dependencies=(uintptr_t)dependencies_22,.dependency_count=0},{.source_id=23,.dependencies=(uintptr_t)dependencies_23,.dependency_count=1},{.source_id=24,.dependencies=(uintptr_t)dependencies_24,.dependency_count=1},{.source_id=25,.dependencies=(uintptr_t)dependencies_25,.dependency_count=1},{.source_id=26,.dependencies=(uintptr_t)dependencies_26,.dependency_count=2},{.source_id=27,.dependencies=(uintptr_t)dependencies_27,.dependency_count=2},{.source_id=28,.dependencies=(uintptr_t)dependencies_28,.dependency_count=1},{.source_id=29,.dependencies=(uintptr_t)dependencies_29,.dependency_count=1},{.source_id=30,.dependencies=(uintptr_t)dependencies_30,.dependency_count=1},{.source_id=31,.dependencies=(uintptr_t)dependencies_31,.dependency_count=2},{.source_id=32,.dependencies=(uintptr_t)dependencies_32,.dependency_count=1},{.source_id=33,.dependencies=(uintptr_t)dependencies_33,.dependency_count=1},{.source_id=34,.dependencies=(uintptr_t)dependencies_34,.dependency_count=1},{.source_id=35,.dependencies=(uintptr_t)dependencies_35,.dependency_count=1},{.source_id=36,.dependencies=(uintptr_t)dependencies_36,.dependency_count=1},{.source_id=37,.dependencies=(uintptr_t)dependencies_37,.dependency_count=0},{.source_id=38,.dependencies=(uintptr_t)dependencies_38,.dependency_count=1},{.source_id=39,.dependencies=(uintptr_t)dependencies_39,.dependency_count=1},{.source_id=40,.dependencies=(uintptr_t)dependencies_40,.dependency_count=1},{.source_id=41,.dependencies=(uintptr_t)dependencies_41,.dependency_count=2},{.source_id=42,.dependencies=(uintptr_t)dependencies_42,.dependency_count=2},{.source_id=43,.dependencies=(uintptr_t)dependencies_43,.dependency_count=1},{.source_id=44,.dependencies=(uintptr_t)dependencies_44,.dependency_count=1}};
static const volatile mlx_graph_program program={.magic=MLX_GRAPH_MAGIC,.version=2,.source_count=45,.task_count=42,.asset_count=4,.assets=(uintptr_t)assets,.tasks=(uintptr_t)tasks,.device_base=UINT64_C(2164260864),.device_bytes=UINT64_C(1048576),.scratch_offset=4096,.scratch_bytes=16384,.poll_limit=UINT64_C(1000000000000),.completion=(uintptr_t)completion,.reserved={(uintptr_t)source_table,0,0}};
static const unsigned char expected_0[]={160,83,0,60,255,251,255,251};
static const unsigned char expected_1[]={0,0,0,0,0,0,0,0};
static const unsigned char expected_2[]={64,80,0,64,0,66,255,251};
static const unsigned char expected_3[]={0,0,0,0,0,0,0,0};
static const unsigned char expected_4[]={64,78,170,64,85,67,0,60};
static const unsigned char expected_5[]={0,0,0,0,0,0,0,0};
struct check {uint64_t base,offset,rank,width,count,shape[8],stride[8];const unsigned char *expected;};
static const volatile struct check checks[]={{.base=UINT64_C(2164326784),.offset=0,.rank=2,.width=2,.count=4,.shape={1,4},.stride={4,1},.expected=expected_0},{.base=UINT64_C(2164326656),.offset=0,.rank=1,.width=8,.count=1,.shape={1},.stride={1},.expected=expected_1},{.base=UINT64_C(2164326848),.offset=0,.rank=2,.width=2,.count=4,.shape={1,4},.stride={4,1},.expected=expected_2},{.base=UINT64_C(2164326720),.offset=0,.rank=1,.width=8,.count=1,.shape={1},.stride={1},.expected=expected_3},{.base=UINT64_C(2164326976),.offset=0,.rank=2,.width=2,.count=4,.shape={1,4},.stride={4,1},.expected=expected_4},{.base=UINT64_C(2164326912),.offset=0,.rank=1,.width=8,.count=1,.shape={1},.stride={1},.expected=expected_5}};
static const uint64_t source_ids[]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45};
int main(void){
if(mlx_graph_execute(&program,&result))return 10;
if(result.status||result.completed_sources!=45||result.host_calls!=15||result.view_elisions!=9)return 11;
if(result.device_calls!=18||result.asset_bytes!=72)return 14;
for(unsigned i=0;i<45;++i)if(completion[i]!=source_ids[i])return 12;
for(unsigned row=0;row<sizeof(checks)/sizeof(checks[0]);++row){
const volatile struct check *c=&checks[row];for(uint64_t flat=0;flat<c->count;++flat){uint64_t index=flat,at=c->offset;for(unsigned d=(unsigned)c->rank;d-->0;){at+=(index%c->shape[d])*c->stride[d];index/=c->shape[d];}const volatile unsigned char *actual=(const volatile unsigned char *)(uintptr_t)(c->base+at*c->width);for(unsigned b=0;b<c->width;++b)if(actual[b]!=c->expected[flat*c->width+b])return 13;}
}
mlx_clocked_pass();return 0;
}
