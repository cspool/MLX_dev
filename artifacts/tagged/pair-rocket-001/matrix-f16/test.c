#include "host_runtime.h"
#include "control_runtime.h"
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)))=0;
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)))=0;
extern const unsigned char pair_image[],control_image[];
static void copy(uint64_t address,const unsigned char *source,uint64_t bytes){volatile unsigned char *out=(volatile unsigned char *)(uintptr_t)address;for(uint64_t i=0;i<bytes;++i)out[i]=source[i];}
static int equal(uint64_t address,const unsigned char *expected,uint64_t bytes){const volatile unsigned char *in=(const volatile unsigned char *)(uintptr_t)address;for(uint64_t i=0;i<bytes;++i)if(in[i]!=expected[i])return 0;return 1;}
static const unsigned char initial_0[]={0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60,0,60};
static const unsigned char initial_1[]={0,52,0,52,0,52,0,52,0,60,0,60,0,60,0,60,0,68,0,68,0,68,0,68,0,52,0,52,0,52,0,52,0,60,0,60,0,60,0,60,0,68,0,68,0,68,0,68,0,52,0,52,0,52,0,52,0,60,0,60,0,60,0,60,0,68,0,68,0,68,0,68,0,52,0,52,0,52,0,52,0,60,0,60,0,60,0,60,0,68,0,68,0,68,0,68,0,52,0,52,0,52,0,52,0,60,0,60,0,60,0,60,0,68,0,68,0,68,0,68,0,52,0,52,0,52,0,52,0,60,0,60,0,60,0,60,0,68,0,68,0,68,0,68,0,52,0,52,0,52,0,52};
static const unsigned char expected_0_0[]={0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60,0,68,0,76,0,60};
static const unsigned char expected_0_1[]={0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60,0,56,0,52,0,60};
static const unsigned char *const expected[][2]={{expected_0_0,expected_0_1}};
static const uint64_t expected_tokens[][ 5 ]={{0,0,0,0,0}};
int main(void){
if(mlx_clocked_status(14)!=MLX_CLOCKED_ROCC_MAGIC)return 10;
copy(UINT64_C(2164326400),initial_0,sizeof(initial_0));
copy(UINT64_C(2164326464),initial_1,sizeof(initial_1));
copy(UINT64_C(2164264960),pair_image,8832);
for(unsigned step=0;step<1;++step){
if(mlx_clocked_submit(UINT64_C(2164264960),8832)!=2)return 12;
if(!equal(UINT64_C(2164326656),expected[step][0],190))return 13;
if(!equal(UINT64_C(2164326848),expected[step][1],190))return 13;
if(mlx_host_control_execute((const volatile mlx_host_control_command *)(const void *)control_image))return 14;
const volatile uint64_t *tokens=(const volatile uint64_t *)(uintptr_t)UINT64_C(2165243904);
for(unsigned i=0;i<5;++i)if(tokens[i]!=expected_tokens[step][i])return 15;
}
mlx_clocked_pass();return 0;
}
