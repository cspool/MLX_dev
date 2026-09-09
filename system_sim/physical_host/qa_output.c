#include "qa_output.h"
#if !defined(__riscv) || __riscv_xlen != 64
#error "QA system output must execute on RV64"
#endif
extern volatile uint64_t tohost,fromhost;

int mlx_host_qa_capture(const volatile mlx_host_tensor views[4],uint64_t n,
                        float *start,float *end,uint8_t *mask,int64_t *offsets,mlx_qa_span *span){
  if(!views||!n||n>UINT64_MAX/16)return 1;
  for(unsigned i=0;i<4;++i){
    const volatile mlx_host_tensor *v=&views[i];
    if(v->dtype!=(i<2?MLX_HOST_F32:i==2?MLX_HOST_BOOL:MLX_HOST_I64))return 1;
    if(i<2){if(v->rank!=2||v->shape[0]!=1||v->shape[1]!=n)return 1;}
    else if(i==2){if(v->rank!=1||v->shape[0]!=n)return 1;}
    else if(v->rank!=2||v->shape[0]!=n||v->shape[1]!=2)return 1;
  }
  if(mlx_host_tensor_readback(&views[0],start,n*4)||mlx_host_tensor_readback(&views[1],end,n*4)
     ||mlx_host_tensor_readback(&views[2],mask,n)||mlx_host_tensor_readback(&views[3],offsets,n*16))return 1;
  return mlx_qa_select_span(start,end,mask,n,30,span);
}

struct text {char bytes[512];unsigned size;};
static void flush(struct text *text){
  if(!text->size)return;
  volatile uint64_t call[8] __attribute__((aligned(64)))={64,1,(uintptr_t)text->bytes,text->size,0,0,0,0};
  while(tohost){}
  __asm__ volatile("fence":::"memory");tohost=(uintptr_t)call;
  while(!fromhost){}
  fromhost=0;__asm__ volatile("fence":::"memory");text->size=0;
}
static void character(struct text *t,char c){if(t->size==sizeof(t->bytes))flush(t);t->bytes[t->size++]=c;}
static void string(struct text *t,const char *s){while(*s)character(t,*s++);}
static void integer(struct text *t,uint64_t n){char digits[20];unsigned size=0;do{digits[size++]=(char)('0'+n%10);n/=10;}while(n);while(size)character(t,digits[--size]);}
static void field(struct text *t,const char *name,uint64_t value){character(t,' ');string(t,name);character(t,'=');integer(t,value);}
static void logits(const char *role,uint64_t forward,uint64_t n,const float *data){
  struct text t={{0},0};string(&t,"MLX_QA_LOGITS");field(&t,"forward",forward);string(&t," role=");string(&t,role);string(&t," words=");
  static const char hex[]="0123456789abcdef";
  for(uint64_t i=0;i<n;++i){union{float f;uint32_t u;}v;v.f=data[i];if(i)character(&t,',');for(unsigned j=8;j-->0;)character(&t,hex[(v.u>>(4*j))&15]);}
  character(&t,'\n');flush(&t);
}
void mlx_host_qa_emit(uint64_t forward,uint64_t n,const float *start,const float *end,
                     const int64_t *offsets,const mlx_qa_span *span){
  struct text t={{0},0};union{double f;uint64_t u;}score;score.f=span->score;
  logits("start_logits",forward,n,start);logits("end_logits",forward,n,end);
  string(&t,"MLX_QA_RESULT");field(&t,"forward",forward);field(&t,"start",span->start);field(&t,"end",span->end);
  field(&t,"start_byte",(uint64_t)offsets[2*span->start]);field(&t,"end_byte",(uint64_t)offsets[2*span->end+1]);
  field(&t,"score_bits",score.u);field(&t,"candidates",span->candidates);character(&t,'\n');flush(&t);
}
void mlx_host_qa_timing(uint64_t begin,uint64_t graph_end,uint64_t post_end){
  struct text t={{0},0};string(&t,"MLX_QA_CPU_CYCLES");field(&t,"begin",begin);field(&t,"graph_end",graph_end);field(&t,"post_end",post_end);character(&t,'\n');flush(&t);
}
