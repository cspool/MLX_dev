#include "rv64_leaf.h"
#include <cfenv>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace mlx::control_model {
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
int64_t signed_bits(uint64_t v){int64_t out;std::memcpy(&out,&v,8);return out;}
int64_t sext(unsigned value,unsigned width){uint64_t raw=value;if(value&(1u<<(width-1)))raw|=~((1ULL<<width)-1);return signed_bits(raw);}
uint32_t single(uint64_t raw){return raw>>32==0xffffffffu?uint32_t(raw):0x7fc00000u;}
bool nan(uint32_t raw){return (raw&0x7f800000u)==0x7f800000u&&(raw&0x7fffffu);}
bool snan(uint32_t raw){return nan(raw)&&!(raw&0x400000u);}
unsigned classify(uint32_t raw){
  unsigned sign=raw>>31,exponent=(raw>>23)&255,fraction=raw&0x7fffff;
  if(exponent==255){if(fraction)return 1u<<(fraction&0x400000?9:8);return 1u<<(sign?0:7);}
  if(!exponent)return 1u<<(fraction?(sign?2:5):(sign?3:4));
  return 1u<<(sign?1:6);
}
float as_float(uint32_t raw){float f;std::memcpy(&f,&raw,4);return f;}
} // namespace
void RV64::set_float(unsigned reg,float value){check(reg<32,"invalid floating register");uint32_t raw;std::memcpy(&raw,&value,4);f[reg]=0xffffffff00000000ULL|raw;}
float RV64::get_float(unsigned reg)const{check(reg<32,"invalid floating register");return as_float(single(f[reg]));}
void RV64::run(const Json::Value &words){
  check(words.isArray()&&words.size()<=32,"invalid RV64 leaf program extent");
  int64_t pc=0;unsigned steps=0;
  while(pc<int64_t(words.size()*4)){
    check(pc>=0&&pc%4==0&&++steps<=256,"RV64 leaf branch/step limit violation");const auto &item=words[unsigned(pc)/4];check(item.isUInt(),"RV64 word must be 32-bit");
    unsigned word=item.asUInt(),opcode=word&127,rd=(word>>7)&31,funct3=(word>>12)&7,a=(word>>15)&31,b=(word>>20)&31,funct7=word>>25;
    uint64_t result=0;bool integer_write=true;int64_t next=pc+4;x[0]=0;
    if(opcode==0x13){auto immediate=uint64_t(sext(word>>20,12));
      if(funct3==0)result=x[a]+immediate;else if(funct3==4)result=x[a]^immediate;else if(funct3==7)result=x[a]&immediate;else throw std::runtime_error("unsupported RV64 immediate opcode");
    }else if(opcode==0x33){
      if(funct7==1&&funct3==0)result=x[a]*x[b];
      else if(funct7==0x20&&funct3==0)result=x[a]-x[b];
      else{check(funct7==0,"unsupported RV64 register opcode");switch(funct3){case 0:result=x[a]+x[b];break;case 2:result=signed_bits(x[a])<signed_bits(x[b]);break;case 3:result=x[a]<x[b];break;case 4:result=x[a]^x[b];break;case 6:result=x[a]|x[b];break;case 7:result=x[a]&x[b];break;default:throw std::runtime_error("unsupported RV64 ALU operation");}}
    }else if(opcode==0x63){
      check(funct3==0||funct3==1,"unsupported RV64 branch");integer_write=false;bool equal=x[a]==x[b];
      unsigned immediate=((word>>31)&1)<<12|((word>>7)&1)<<11|((word>>25)&63)<<5|((word>>8)&15)<<1;
      if(equal==(funct3==0)){next=pc+sext(immediate,13);++branches_taken;}
      check(next>=0&&next<=int64_t(words.size()*4)&&next%4==0,"RV64 branch leaves the bounded leaf program");
    }else if(opcode==0x53){
      auto left=single(f[a]),right=single(f[b]);
      if(funct7==0x50){check(funct3<=2,"invalid FP comparison encoding");bool unordered=nan(left)||nan(right);if((funct3!=2&&unordered)||snan(left)||snan(right))fflags|=16;
        result=!unordered&&(funct3==0?as_float(left)<=as_float(right):funct3==1?as_float(left)<as_float(right):as_float(left)==as_float(right));
      }else if(funct7==0x70){check(b==0&&funct3==1,"unsupported FP class/move encoding");result=classify(left);}
      else if(funct7==0x10){check(funct3==0,"unsupported FP sign injection encoding");f[rd]=0xffffffff00000000ULL|(left&0x7fffffff)|(right&0x80000000u);integer_write=false;}
      else if(funct7==0x68){check(b==2&&funct3==0&&std::fegetround()==FE_TONEAREST,"only FCVT.S.L RNE is registered");auto value=signed_bits(x[a]);float converted=float(value);set_float(rd,converted);if(static_cast<long double>(converted)!=static_cast<long double>(value))fflags|=1;integer_write=false;}
      else throw std::runtime_error("unsupported RV64 floating opcode");
    }else throw std::runtime_error("unsupported RV64 instruction in controller leaf");
    if(integer_write&&rd)x[rd]=result;
    x[0]=0;pc=next;++retired;
  }
}
} // namespace mlx::control_model
