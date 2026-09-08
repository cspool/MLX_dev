#include "tensor.h"
#include "mlx_tagged_simulator.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mlx::tensor_model {
void require(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
DType dtype(const std::string &name) {
  if (name=="f16") return DType::F16;
  if (name=="f32") return DType::F32;
  if (name=="i64") return DType::I64;
  if (name=="bool") return DType::Bool;
  throw std::invalid_argument("unsupported tensor dtype: "+name);
}
std::string dtype_name(DType type) {
  return type==DType::F16?"f16":type==DType::F32?"f32":type==DType::I64?"i64":"bool";
}
unsigned element_bytes(DType type) { return type==DType::F16?2:type==DType::F32?4:type==DType::I64?8:1; }
Shape shape(const Json::Value &raw) {
  require(raw.isArray() && raw.size()<=8, "invalid tensor rank");
  Shape result;
  for (const auto &value:raw) { require(value.isInt64(), "noninteger tensor dimension"); result.push_back(value.asInt64()); }
  return result;
}
uint64_t elements(const Shape &s) {
  uint64_t count=1;
  for (auto size:s) {
    require(size>=0, "negative tensor dimension");
    require(!size || count<=std::numeric_limits<uint64_t>::max()/uint64_t(size), "tensor size overflow");
    count*=size;
  }
  return count;
}
Shape strides(const Shape &s) {
  Shape out(s.size()); int64_t step=1;
  for (size_t i=s.size();i-->0;) {
    require(s[i]>=0 && step<=INT64_MAX/std::max<int64_t>(s[i],1),"tensor stride overflow");
    out[i]=step; step*=std::max<int64_t>(s[i],1);
  }
  return out;
}
double scalar(const Json::Value &value) {
  if (value.isNumeric()) return value.asDouble();
  if (value.isBool()) return value.asBool()?1:0;
  if (value.isObject() && value.isMember("float_literal")) return std::stod(value["float_literal"].asString());
  throw std::invalid_argument("not a numeric scalar");
}
bool Tensor::contiguous() const {
  if(!numel())return true;
  uint64_t step=1;
  for (size_t i=sizes.size();i-->0;) {
    if (sizes[i]>1 && (steps[i]<0 || uint64_t(steps[i])!=step)) return false;
    if(step>UINT64_MAX/uint64_t(std::max<int64_t>(sizes[i],1)))return false;
    step*=uint64_t(std::max<int64_t>(sizes[i],1));
  }
  return true;
}
uint64_t Tensor::position(uint64_t flat) const {
  require(flat<numel(), "tensor element out of bounds");
  require(sizes.size()==steps.size() && offset>=0, "invalid tensor view layout");
  const uint64_t limit=storage->bytes/element_bytes(type);
  uint64_t at=offset;
  require(at<limit, "tensor view exceeds its storage");
  if (contiguous()) { require(flat<=limit-1-at,"tensor view exceeds its storage");at+=flat; }
  else for (size_t axis=sizes.size();axis-->0;) {
    const uint64_t coordinate=flat%uint64_t(sizes[axis]);flat/=sizes[axis];
    require(steps[axis]>=0 && (!steps[axis] || coordinate<=(limit-1-at)/uint64_t(steps[axis])),"tensor view exceeds its storage");
    at+=coordinate*uint64_t(steps[axis]);
  }
  return at;
}
static float half(uint16_t value) {
  unsigned exp=(value>>10)&31;
  if (exp && exp!=31) {
    uint32_t bits=uint32_t(value&0x8000)<<16 | (exp+112)<<23 | uint32_t(value&1023)<<13;
    float out; std::memcpy(&out,&bits,4); return out;
  }
  return tagged::half_to_float(value);
}
float Tensor::number(uint64_t flat) const {
  const uint8_t *p=storage->data+position(flat)*element_bytes(type);
  if (type==DType::F16) { uint16_t v; std::memcpy(&v,p,2); return half(v); }
  if (type==DType::F32) { float v; std::memcpy(&v,p,4); return v; }
  if (type==DType::I64) { int64_t v; std::memcpy(&v,p,8); return float(v); }
  return *p!=0;
}
int64_t Tensor::integer(uint64_t flat) const {
  const uint8_t *p=storage->data+position(flat)*element_bytes(type);
  if (type==DType::I64) { int64_t v; std::memcpy(&v,p,8); return v; }
  if (type==DType::Bool) return *p!=0;
  return int64_t(number(flat));
}
void Tensor::set_number(uint64_t flat, float value) {
  require(storage->writable!=nullptr, "attempted to write immutable model weights");
  uint8_t *p=storage->writable+position(flat)*element_bytes(type);
  if (type==DType::F16) { auto bits=tagged::float_to_half(value); std::memcpy(p,&bits,2); }
  else if (type==DType::F32) std::memcpy(p,&value,4);
  else if (type==DType::I64) { auto i=int64_t(value); std::memcpy(p,&i,8); }
  else *p=value!=0;
}
void Tensor::set_integer(uint64_t flat, int64_t value) {
  if (type!=DType::I64 && type!=DType::Bool) { set_number(flat,float(value)); return; }
  require(storage->writable!=nullptr, "attempted to write immutable storage");
  auto p=storage->writable+position(flat)*element_bytes(type);
  if (type==DType::I64) std::memcpy(p,&value,8); else *p=value!=0;
}
__attribute__((target("avx,f16c"))) static void convert_half(const uint16_t *in, float *out, size_t n) {
  size_t i=0;
  for (;i+8<=n;i+=8) _mm256_storeu_ps(out+i,_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(in+i))));
  for (;i<n;++i) out[i]=half(in[i]);
}
std::vector<float> Tensor::floats() const {
  std::vector<float> out(numel());
  if (out.empty()) return out;
  if (type==DType::F16 && contiguous() && __builtin_cpu_supports("f16c")) {
    const auto *data=reinterpret_cast<const uint16_t*>(storage->data)+offset;
    require(offset>=0 && (uint64_t(offset)+numel())*2<=storage->bytes,"contiguous tensor out of bounds");
    convert_half(data,out.data(),out.size());
  } else if (type==DType::F32 && contiguous()) {
    require(offset>=0 && (uint64_t(offset)+numel())*4<=storage->bytes,"contiguous tensor out of bounds");
    std::memcpy(out.data(),storage->data+offset*4,out.size()*4);
  } else for (uint64_t i=0;i<numel();++i) out[i]=number(i);
  return out;
}
Tensor Tensor::allocate(DType type, Shape sizes) {
  const uint64_t n=elements(sizes);
  require(n<=uint64_t(8)*1024*1024*1024/element_bytes(type), "tensor exceeds staging allocation limit (8 GiB)");
  const uint64_t bytes=n*element_bytes(type);
  auto data=std::make_shared<std::vector<uint8_t>>(bytes);
  auto storage=std::make_shared<Storage>(); storage->owner=data;
  storage->data=storage->writable=data->data(); storage->bytes=bytes;
  return Tensor{type,sizes,strides(sizes),0,storage};
}
Tensor Tensor::materialize(DType target) const {
  auto result=allocate(target,sizes);
  if(target==type){
    for(uint64_t i=0;i<numel();++i)std::memcpy(result.storage->writable+i*element_bytes(type),storage->data+position(i)*element_bytes(type),element_bytes(type));
    return result;
  }
  for (uint64_t i=0;i<numel();++i) {
    if (target==DType::Bool) result.set_integer(i,type==DType::I64?integer(i)!=0:number(i)!=0.0f);
    else if (target==DType::I64) result.set_integer(i,integer(i));
    else result.set_number(i,number(i));
  }
  return result;
}
Tensor Assets::load(const Json::Value &spec) {
  DType type=dtype(spec["dtype"].asString()); Shape sizes=shape(spec["shape"]);
  if (spec["kind"]=="literal") {
    auto result=Tensor::allocate(type,sizes); uint64_t index=0;
    auto flatten=[&](auto &&self,const Json::Value &value)->void {
      if (value.isArray()) for (const auto &child:value) self(self,child);
      else {
        require(index<result.numel(), "too many literal elements");
        if (type==DType::I64) result.set_integer(index++,value.asInt64());
        else result.set_number(index++,float(scalar(value)));
      }
    };
    flatten(flatten,spec["values"]); require(index==result.numel(),"literal element count mismatch"); return result;
  }
  require(spec["kind"]=="mapped_file", "unsupported asset binding");
  const auto name=spec["path"].asString();
  if (!files.count(name)) {
    int fd=open(name.c_str(),O_RDONLY); require(fd>=0,"cannot open weight asset: "+name);
    struct stat info{}; if (fstat(fd,&info)) { close(fd); throw std::runtime_error("cannot stat weight file"); }
    void *address=mmap(nullptr,info.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    require(address!=MAP_FAILED,"cannot map weight file");
    auto owner=std::shared_ptr<void>(address,[bytes=info.st_size](void *p){munmap(p,bytes);});
    auto storage=std::make_shared<Storage>(); storage->owner=owner; storage->data=static_cast<uint8_t*>(address);
    storage->bytes=info.st_size; files[name]=storage;
  }
  const uint64_t count=elements(sizes);
  require(count<=UINT64_MAX/element_bytes(type),"mapped tensor byte size overflow");
  uint64_t at=spec["byte_offset"].asUInt64(), bytes=count*element_bytes(type);
  require(spec["bytes"].isUInt64() && spec["bytes"].asUInt64()==bytes,"mapped tensor byte count mismatch");
  require(at<=files[name]->bytes && bytes<=files[name]->bytes-at,"weight asset exceeds file");
  auto view=std::make_shared<Storage>(*files[name]); view->data+=at; view->bytes=bytes;
  return Tensor{type,sizes,strides(sizes),0,view};
}
} // namespace mlx::tensor_model
