// Validation-only exact FP16 dot products. Never linked into MLX execution.
#include <json/json.h>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using I=__int128_t;using U=__uint128_t;
namespace {
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
int64_t units24(uint16_t h){
  unsigned exponent=(h>>10)&31,fraction=h&1023;require(exponent!=31,"exact FP16 diagnostic rejects NaN/Inf inputs");
  int64_t value=exponent?int64_t(1024+fraction)<<(exponent-1):fraction;
  return h&0x8000?-value:value;
}
U rne(U n,unsigned shift){
  require(shift>0&&shift<128,"invalid exact rounding shift");U q=n>>shift,remainder=n&((U(1)<<shift)-1),half=U(1)<<(shift-1);
  return q+(remainder>half||(remainder==half&&(q&1)));
}
uint16_t round48(I value){
  if(!value)return 0; // Exact cancellation rounds to +0 in the RN profile.
  uint16_t sign=value<0?0x8000:0;U n=value<0?U(-value):U(value);
  if(n<(U(1)<<34))return sign|uint16_t(rne(n,24));
  unsigned top=0;for(U x=n;x>>1;x>>=1)++top;
  U mantissa=rne(n,top-10);if(mantissa==2048){mantissa=1024;++top;}
  unsigned exponent=top-33;if(exponent>=31)return sign|0x7c00;
  return sign|uint16_t(exponent<<10)|uint16_t(mantissa-1024);
}
I decimal(const std::string &text){
  require(!text.empty(),"empty exact integer");bool negative=text[0]=='-';size_t at=negative?1:0;require(at<text.size(),"empty signed exact integer");U n=0,limit=U(1)<<120;
  for(;at<text.size();++at){char c=text[at];require(c>='0'&&c<='9',"nondecimal exact integer");unsigned d=c-'0';require(n<=(limit-d)/10,"exact integer exceeds diagnostic range");n=n*10+d;}
  return negative?-I(n):I(n);
}
I float_units48(float value){
  uint32_t bits;std::memcpy(&bits,&value,4);unsigned exponent=(bits>>23)&255;
  if(!(bits&0x7fffffff))return 0;
  require(exponent>0&&exponent<255,"invalid FP32 accumulation value");U n=(bits&0x7fffff)|0x800000;int shift=int(exponent)-102;
  if(shift>=0){require(shift<100,"FP32 accumulation exceeds diagnostic range");n<<=shift;}
  else{require(-shift<128&&!(n&((U(1)<<(-shift))-1)),"FP32 result left the exact input lattice");n>>=-shift;}
  return bits&0x80000000?-I(n):I(n);
}
std::vector<int64_t> input(const Json::Value &spec,uint64_t count){
  require(count>0&&count<=(UINT64_C(1)<<26)&&spec["offset"].isUInt64(),"invalid diagnostic tensor extent");
  auto file=std::filesystem::path(spec["file"].asString());uint64_t offset=spec["offset"].asUInt64(),size=std::filesystem::file_size(file);
  require(offset<=size&&count<=(size-offset)/2&&offset<=uint64_t(std::numeric_limits<std::streamoff>::max()),"diagnostic tensor exceeds source file");
  std::ifstream stream(file,std::ios::binary);stream.seekg(std::streamoff(offset));std::vector<uint16_t> raw(count);stream.read(reinterpret_cast<char*>(raw.data()),std::streamsize(count*2));require(bool(stream),"cannot read diagnostic input");
  std::vector<int64_t> converted;converted.reserve(count);for(auto h:raw)converted.push_back(units24(h));return converted;
}
uint64_t dimension(const Json::Value &job,const char *key){require(job[key].isUInt64(),"diagnostic dimension is not unsigned");auto n=job[key].asUInt64();require(n>0&&n<=(UINT64_C(1)<<20),"diagnostic dimension exceeds bounded int128 arithmetic");return n;}
void output(const std::filesystem::path &path,const std::vector<uint16_t> &values){std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*2));require(bool(out),"cannot write exact diagnostic output");}
}
int main(int argc,char **argv){try{
  require(argc==3,"usage: mlx-exact-f16-dot job.json output-directory");uint16_t endian=1;require(*reinterpret_cast<uint8_t*>(&endian)==1&&sizeof(float)==4&&std::numeric_limits<float>::is_iec559,"diagnostic requires little-endian IEEE FP32");
  require(std::fegetround()==FE_TONEAREST,"FP32 diagnostic requires round-to-nearest mode");Json::Value job;std::ifstream in(argv[1]);in>>job;require(bool(in),"cannot read exact diagnostic job");
  std::filesystem::path directory(argv[2]);require(!std::filesystem::exists(directory),"choose a fresh exact diagnostic output");std::filesystem::create_directories(directory);
  Json::Value report;report["classification"]="exact_integer_fp16_dot_reference_not_model_or_hardware_execution";
  if(job["mode"]=="round"){
    require(job["values"].isArray()&&job["values"].size()<=100000,"invalid rounding diagnostic input");report["rounded"]=Json::arrayValue;
    for(const auto &value:job["values"]){require(value.isString(),"exact integer must be a decimal string");report["rounded"].append(unsigned(round48(decimal(value.asString()))));}
  }else{
    require(job["mode"]=="linear_f16_no_bias","unregistered exact diagnostic mode");auto m=dimension(job,"m"),n=dimension(job,"n"),k=dimension(job,"k");
    require(m<=((UINT64_C(1)<<26)/k)&&n<=((UINT64_C(1)<<26)/k)&&m<=((UINT64_C(1)<<26)/n),"diagnostic matrix exceeds memory budget");
    auto a=input(job["a"],m*k),b=input(job["b"],n*k);std::vector<uint16_t> exact(m*n),ascending(m*n);
    for(uint64_t row=0;row<m;++row)for(uint64_t col=0;col<n;++col){I sum=0;float acc=0;
      for(uint64_t at=0;at<k;++at){int64_t av=a[row*k+at],bv=b[col*k+at];sum+=I(av)*I(bv);float af=float(av)*0x1p-24f,bf=float(bv)*0x1p-24f;float product=af*bf;acc=acc+product;}
      exact[row*n+col]=round48(sum);ascending[row*n+col]=round48(float_units48(acc));
    }
    output(directory/"exact.f16.bin",exact);output(directory/"kasc.f16.bin",ascending);
    report["m"]=Json::UInt64(m);report["n"]=Json::UInt64(n);report["k"]=Json::UInt64(k);report["output_elements"]=Json::UInt64(m*n);report["exact_products"]=Json::UInt64(m*n*k);
    report["integer_fraction_bits"]=48;report["max_dimension"]=1<<20;report["accumulation"]="signed_int128_exact_then_binary16_RN_ties_to_even";report["comparison_accumulation"]="separate_FP32_mul_add_ascending_K";
  }
  report["full_model_execution_verified"]=false;report["mlx_system_verified"]=false;report["inference_performance_eligible"]=false;
  std::ofstream out(directory/"report.json");out<<report<<'\n';require(bool(out),"cannot save exact diagnostic report");std::cout<<"EXACT_F16_DIAGNOSTIC_COMPLETE\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
