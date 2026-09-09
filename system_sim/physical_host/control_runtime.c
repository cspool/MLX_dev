#include "control_runtime.h"
#include <float.h>
#if !defined(__riscv) || __riscv_xlen != 64
#error "This runtime must execute on RV64, not a host tensor fallback"
#endif
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The host control ABI requires little-endian RV64"
#endif
_Static_assert(sizeof(uintptr_t)==8 && sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24,"host numeric ABI");

struct view {
  uint64_t base, bytes, offset, count, rank, dtype, width;
  uint64_t shape[8], stride[8];
};
struct operand {uint64_t kind, dtype, bits;struct view tensor;};

static unsigned width(uint64_t dtype){return dtype==MLX_HOST_F16?2:dtype==MLX_HOST_F32?4:dtype==MLX_HOST_I64?8:dtype==MLX_HOST_BOOL?1:0;}
static int zero(const volatile void *object,size_t bytes){const volatile unsigned char *p=object;for(size_t i=0;i<bytes;++i)if(p[i])return 0;return 1;}
static enum mlx_host_status tensor(struct view *v,const volatile mlx_host_tensor *t,unsigned access){
  if(t->rank>8 || (t->access&~UINT64_C(3)) || (t->access&access)!=access)return MLX_HOST_BAD_DESCRIPTOR;
  v->width=width(t->dtype);if(!v->width)return MLX_HOST_BAD_TYPE;
  if(t->base%v->width || t->bytes%v->width || t->bytes>UINT64_MAX-t->base || (t->bytes&&!t->base))return MLX_HOST_BOUNDS;
  v->base=t->base;v->bytes=t->bytes;v->offset=t->offset;v->rank=t->rank;v->dtype=t->dtype;v->count=1;
  for(unsigned d=0;d<8;++d){
    if(d<v->rank){
      if(t->shape[d]>INT64_MAX || t->stride[d]>INT64_MAX || (t->shape[d]&&v->count>UINT64_MAX/t->shape[d]))return MLX_HOST_BAD_SHAPE;
      v->shape[d]=t->shape[d];v->stride[d]=t->stride[d];v->count*=v->shape[d];
    }else if(t->shape[d] || t->stride[d])return MLX_HOST_BAD_DESCRIPTOR;
  }
  if(v->count){
    uint64_t limit=v->bytes/v->width,at=v->offset;if(at>=limit)return MLX_HOST_BOUNDS;
    for(unsigned d=0;d<v->rank;++d){uint64_t n=v->shape[d]-1,s=v->stride[d];if(s&&n>(limit-1-at)/s)return MLX_HOST_BOUNDS;at+=n*s;}
  }
  return MLX_HOST_OK;
}
static enum mlx_host_status operand(struct operand *v,const volatile mlx_host_operand *arg){
  if(arg->reserved)return MLX_HOST_BAD_DESCRIPTOR;
  v->kind=arg->kind;
  if(arg->kind==MLX_HOST_TENSOR){
    if(arg->scalar_dtype||arg->scalar_bits)return MLX_HOST_BAD_DESCRIPTOR;
    enum mlx_host_status status=tensor(&v->tensor,&arg->tensor,MLX_HOST_READ);if(status)return status;v->dtype=v->tensor.dtype;
  }else if(arg->kind==MLX_HOST_SCALAR){
    if(!zero(&arg->tensor,sizeof(arg->tensor)))return MLX_HOST_BAD_DESCRIPTOR;
    v->dtype=arg->scalar_dtype;v->bits=arg->scalar_bits;unsigned size=width(v->dtype);if(!size)return MLX_HOST_BAD_TYPE;
    if((size<8 && v->bits>>(size*8)) || (v->dtype==MLX_HOST_BOOL&&v->bits>1))return MLX_HOST_BAD_DESCRIPTOR;
  }else return MLX_HOST_BAD_DESCRIPTOR;
  return MLX_HOST_OK;
}
static int contiguous(const struct view *v){
  if(!v->count)return 1;
  uint64_t step=1;
  for(unsigned d=(unsigned)v->rank;d-->0;){if(v->shape[d]>1 && v->stride[d]!=step)return 0;if(v->shape[d]&&step>UINT64_MAX/v->shape[d])return 0;step*=v->shape[d]?v->shape[d]:1;}
  return 1;
}
static int overlap(const struct view *a,const struct view *b){return a->bytes&&b->bytes&&a->base<b->base+b->bytes&&b->base<a->base+a->bytes;}
static uint64_t address(const struct view *v,uint64_t index){
  uint64_t at=v->offset;
  for(unsigned d=(unsigned)v->rank;d-->0;){at+=(index%v->shape[d])*v->stride[d];index/=v->shape[d];}
  return v->base+at*v->width;
}
static uint64_t broadcast(uint64_t flat,const struct view *out,const struct view *in){
  uint64_t index=0,step=1;
  for(unsigned d=(unsigned)out->rank;d-->0;){uint64_t at=flat%out->shape[d];flat/=out->shape[d];
    if(d+in->rank>=out->rank){uint64_t size=in->shape[d+in->rank-out->rank];if(size!=1)index+=at*step;step*=size;}}
  return index;
}
static uint64_t load(uint64_t at,unsigned bytes){
  uint64_t result;
  switch(bytes){
    case 8:__asm__ volatile("ld %0, 0(%1)":"=r"(result):"r"(at):"memory");break;
    case 4:__asm__ volatile("lwu %0, 0(%1)":"=r"(result):"r"(at):"memory");break;
    case 2:__asm__ volatile("lhu %0, 0(%1)":"=r"(result):"r"(at):"memory");break;
    default:__asm__ volatile("lbu %0, 0(%1)":"=r"(result):"r"(at):"memory");break;
  }
  return result;
}
static void store(uint64_t at,unsigned bytes,uint64_t value){
  if(bytes==8)__asm__ volatile("sd %0, 0(%1)"::"r"(value),"r"(at):"memory");
  else __asm__ volatile("sb %0, 0(%1)"::"r"(value),"r"(at):"memory");
}
static uint64_t value(const struct operand *arg,const struct view *out,uint64_t index){
  if(arg->kind==MLX_HOST_SCALAR)return arg->bits;
  uint64_t raw=load(address(&arg->tensor,broadcast(index,out,&arg->tensor)),(unsigned)arg->tensor.width);
  return arg->dtype==MLX_HOST_BOOL?raw!=0:raw;
}
uint32_t mlx_host_half_to_float_bits(uint16_t h){
  uint32_t sign=(uint32_t)(h&0x8000)<<16,e=(h>>10)&31,f=h&1023;
  if(e==31)return sign|(f?UINT32_C(0x7fc00000):UINT32_C(0x7f800000));
  if(e)return sign|((e+112)<<23)|(f<<13);
  if(!f)return sign;
  unsigned shift=0;while(!(f&1024)){f<<=1;++shift;}
  return sign|((113-shift)<<23)|((f&1023)<<13);
}
static float floating(uint64_t bits,uint64_t dtype){union{uint32_t u;float f;}v;v.u=dtype==MLX_HOST_F16?mlx_host_half_to_float_bits((uint16_t)bits):(uint32_t)bits;return v.f;}
static unsigned nan_bits(uint64_t bits,uint64_t dtype){return dtype==MLX_HOST_F16?((bits&0x7c00)==0x7c00&&(bits&1023)):((bits&UINT32_C(0x7f800000))==UINT32_C(0x7f800000)&&(bits&UINT32_C(0x7fffff)));}
static uint64_t less(uint64_t a,uint64_t b){return (a^UINT64_C(0x8000000000000000))<(b^UINT64_C(0x8000000000000000));}
static uint64_t float_less(float a,float b){uint64_t result;__asm__ volatile("flt.s %0, %1, %2":"=r"(result):"f"(a),"f"(b));return result;}
static uint64_t float_le(float a,float b){uint64_t result;__asm__ volatile("fle.s %0, %1, %2":"=r"(result):"f"(a),"f"(b));return result;}
static int integer(uint64_t dtype){return dtype==MLX_HOST_I64||dtype==MLX_HOST_BOOL;}

enum mlx_host_status mlx_host_control_execute(const volatile mlx_host_control_command *cmd){
  if(!cmd || (uintptr_t)cmd%8 || (uintptr_t)cmd>UINT64_MAX-sizeof(*cmd) || cmd->magic!=MLX_HOST_CONTROL_MAGIC || (cmd->version!=1&&cmd->version!=2) || cmd->reserved)return MLX_HOST_BAD_DESCRIPTOR;
  if(cmd->version==2&&cmd->opcode<=MLX_HOST_ARGMAX)return MLX_HOST_BAD_DESCRIPTOR;
  if(cmd->opcode<MLX_HOST_ARANGE||cmd->opcode>(cmd->version==1?MLX_HOST_ARGMAX:MLX_HOST_GUARD))return MLX_HOST_UNSUPPORTED;
  if((cmd->opcode==MLX_HOST_ARGMAX?(cmd->flags&~MLX_HOST_KEEP_DIM):cmd->flags) || (cmd->opcode!=MLX_HOST_ARANGE&&cmd->extent))return MLX_HOST_BAD_DESCRIPTOR;
  struct view out;struct operand a,b;enum mlx_host_status status=tensor(&out,&cmd->output,MLX_HOST_WRITE);if(status)return status;
  if(out.bytes&&out.base<(uintptr_t)cmd+sizeof(*cmd)&&(uintptr_t)cmd<out.base+out.bytes)return MLX_HOST_OVERLAP;
  if(out.offset||!contiguous(&out))return MLX_HOST_BAD_SHAPE;
  if(cmd->opcode==MLX_HOST_ARANGE){
    if(!zero(&cmd->a,sizeof(cmd->a))||!zero(&cmd->b,sizeof(cmd->b)))return MLX_HOST_BAD_DESCRIPTOR;
    if(out.dtype!=MLX_HOST_I64)return MLX_HOST_BAD_TYPE;
    if(cmd->extent>INT64_MAX||out.rank!=1||out.shape[0]!=cmd->extent)return MLX_HOST_BAD_SHAPE;
    __asm__ volatile("fence rw, rw":::"memory");
    for(uint64_t index=0;index<out.count;++index)store(out.base+index*8,8,index);
  }else{
    status=operand(&a,&cmd->a);if(status)return status;
    if(a.kind==MLX_HOST_TENSOR&&overlap(&out,&a.tensor))return MLX_HOST_OVERLAP;
    if(cmd->opcode==MLX_HOST_ALL||cmd->opcode==MLX_HOST_GUARD){
      if(a.kind!=MLX_HOST_TENSOR||a.dtype!=MLX_HOST_BOOL||out.dtype!=MLX_HOST_BOOL)return MLX_HOST_BAD_TYPE;
      if(out.rank)return MLX_HOST_BAD_SHAPE;
      if(cmd->opcode==MLX_HOST_ALL){
        if(!zero(&cmd->b,sizeof(cmd->b)))return MLX_HOST_BAD_DESCRIPTOR;
        __asm__ volatile("fence rw, rw":::"memory");
        uint64_t result=1;
        /* Read every logical input, including strided/expanded views. Empty
         * Boolean ALL is true; this is not a cached framework branch value. */
        for(uint64_t i=0;i<a.tensor.count;++i)result&=load(address(&a.tensor,i),1)!=0;
        store(out.base,1,result);
      }else{
        if(a.tensor.count!=1)return MLX_HOST_BAD_SHAPE;
        status=operand(&b,&cmd->b);if(status)return status;
        if(b.kind!=MLX_HOST_SCALAR||b.dtype!=MLX_HOST_BOOL)return MLX_HOST_BAD_TYPE;
        __asm__ volatile("fence rw, rw":::"memory");
        uint64_t actual=load(address(&a.tensor,0),1)!=0;
        if(actual!=b.bits)return MLX_HOST_GUARD_FAILED;
        store(out.base,1,actual);
      }
    }else if(cmd->opcode==MLX_HOST_ARGMAX){
      if(!zero(&cmd->b,sizeof(cmd->b)))return MLX_HOST_BAD_DESCRIPTOR;
      if(a.kind!=MLX_HOST_TENSOR||a.dtype==MLX_HOST_BOOL||out.dtype!=MLX_HOST_I64)return MLX_HOST_BAD_TYPE;
      if(!a.tensor.rank||!a.tensor.shape[a.tensor.rank-1])return MLX_HOST_BAD_SHAPE;
      unsigned rank=(unsigned)a.tensor.rank;uint64_t n=a.tensor.shape[rank-1];int keep=(cmd->flags&MLX_HOST_KEEP_DIM)!=0;
      if(out.rank!=rank-!keep)return MLX_HOST_BAD_SHAPE;
      for(unsigned d=0;d<out.rank;++d)if(out.shape[d]!=(keep&&d==rank-1?1:a.tensor.shape[d]))return MLX_HOST_BAD_SHAPE;
      if(!integer(a.dtype)){uint64_t frm;__asm__ volatile("frrm %0":"=r"(frm));if(frm)return MLX_HOST_FP_MODE;}
      __asm__ volatile("fence rw, rw":::"memory");
      for(uint64_t row=0;row<out.count;++row){
        uint64_t best=load(address(&a.tensor,row*n),(unsigned)a.tensor.width),best_index=0;
        for(uint64_t column=1;column<n;++column){
          uint64_t next=load(address(&a.tensor,row*n+column),(unsigned)a.tensor.width),take;
          if(integer(a.dtype))take=less(best,next);
          else{uint64_t compare=float_less(floating(best,a.dtype),floating(next,a.dtype));take=compare||(nan_bits(next,a.dtype)&&!nan_bits(best,a.dtype));}
          if(take){best=next;best_index=column;}
        }
        store(out.base+row*8,8,best_index);
      }
    }else{
      status=operand(&b,&cmd->b);if(status)return status;
      if(b.kind==MLX_HOST_TENSOR&&overlap(&out,&b.tensor))return MLX_HOST_OVERLAP;
      int comparison=cmd->opcode==MLX_HOST_LE||cmd->opcode==MLX_HOST_GE;
      int boolean_and=cmd->opcode==MLX_HOST_BITWISE_AND;
      int integral=integer(a.dtype);if(integral!=integer(b.dtype)||(!comparison&&!integral))return MLX_HOST_BAD_TYPE;
      if(boolean_and&&(a.kind!=MLX_HOST_TENSOR||b.kind!=MLX_HOST_TENSOR||a.dtype!=MLX_HOST_BOOL||b.dtype!=MLX_HOST_BOOL))return MLX_HOST_BAD_TYPE;
      if(out.dtype!=((comparison||boolean_and)?MLX_HOST_BOOL:MLX_HOST_I64))return MLX_HOST_BAD_TYPE;
      unsigned ar=a.kind==MLX_HOST_TENSOR?(unsigned)a.tensor.rank:0,br=b.kind==MLX_HOST_TENSOR?(unsigned)b.tensor.rank:0,rank=ar>br?ar:br;
      if(out.rank!=rank)return MLX_HOST_BAD_SHAPE;
      for(unsigned d=0;d<rank;++d){uint64_t as=d+ar>=rank?a.tensor.shape[d+ar-rank]:1,bs=d+br>=rank?b.tensor.shape[d+br-rank]:1;
        if(as!=bs&&as!=1&&bs!=1)return MLX_HOST_BAD_SHAPE;
        if(out.shape[d]!=(as==1?bs:as))return MLX_HOST_BAD_SHAPE;
      }
      if(!integral){uint64_t frm;__asm__ volatile("frrm %0":"=r"(frm));if(frm)return MLX_HOST_FP_MODE;}
      __asm__ volatile("fence rw, rw":::"memory");
      for(uint64_t index=0;index<out.count;++index){
        uint64_t av=value(&a,&out,index),bv=value(&b,&out,index),result;
        if(cmd->opcode==MLX_HOST_ADD)result=av+bv;
        else if(cmd->opcode==MLX_HOST_MUL)result=av*bv;
        else if(boolean_and)result=av&bv;
        else if(cmd->opcode==MLX_HOST_GE)result=integral?!less(av,bv):float_le(floating(bv,b.dtype),floating(av,a.dtype));
        else result=integral?!less(bv,av):float_le(floating(av,a.dtype),floating(bv,b.dtype));
        store(out.base+index*out.width,(unsigned)out.width,result);
      }
    }
  }
  __asm__ volatile("fence rw, rw":::"memory");return MLX_HOST_OK;
}
