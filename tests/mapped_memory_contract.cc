#include "mapped_memory.hh"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using mlx::physical_device::MappedMemory;
namespace {
void check(bool value,const char *message) {
  if(!value)throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
  bool rejected=false;
  try{action();}catch(const std::runtime_error &){rejected=true;}
  check(rejected,"invalid mapped memory action accepted");
}
}

int main() {
  try {
    constexpr uint64_t capacity=UINT64_C(16)<<30,high=(UINT64_C(1)<<32)+4096;
    rejects([]{MappedMemory invalid(0);});
    rejects([]{MappedMemory invalid(4097);});
    MappedMemory memory(capacity);
    check(memory.size()==capacity&&memory.snapshot()["resident_pages"].asUInt64()==0,"capacity eagerly allocated");
    uint64_t value=UINT64_C(0xfedcba9876543210),other=UINT64_C(0x123456789abcdef0),read=0;
    memory.write(4096,&value,8);memory.write(high,&other,8);
    memory.read(4096,&read,8);check(read==value,"low address changed");
    memory.read(high,&read,8);check(read==other,"high address truncated");
    check(memory.snapshot()["resident_pages"].asUInt64()==2&&memory.snapshot()["writer_pages"].asUInt64()==0,"CPU store allocated device stamps");
    rejects([&]{memory.require_written(high,8,1);});

    std::vector<uint8_t> input(4131),output(input.size(),0xa5);
    for(size_t i=0;i<input.size();++i)input[i]=uint8_t(i*17);
    memory.write(high+4081,input.data(),input.size(),7);
    memory.read(high+4081,output.data(),output.size());check(output==input,"cross-page data changed");
    memory.require_written(high+4081,input.size(),7);
    rejects([&]{memory.require_written(high+4081,input.size()+1,7);});
    rejects([&]{memory.require_written(high+4080,input.size(),7);});
    rejects([&]{memory.require_written(high+4081,input.size(),8);});
    memory.write(high+4081,&other,8,8);
    memory.require_written(high+4081,8,8);
    rejects([&]{memory.require_written(high+4081,9,8);});
    memory.require_written(high+4089,input.size()-8,7);
    memory.write(high+4081,&value,8); // CPU stores cannot satisfy a new launch.
    memory.require_written(high+4081,8,8); // Preserve historical stamps, as before.
    rejects([&]{memory.require_written(high+4081,8,9);});
    rejects([&]{memory.require_written(high,0,0);});

    // Failed reads must not partly overwrite the caller, including a missing
    // next page and an uninitialized byte within an allocated page.
    MappedMemory partial(8192);
    partial.write(4092,&value,4);
    std::vector<uint8_t> guard(8,0xa5),original=guard;
    rejects([&]{partial.read(4092,guard.data(),8);});check(guard==original,"missing page partially copied");
    partial.write(4100,&value,4);
    rejects([&]{partial.read(4092,guard.data(),8);});check(guard==original,"invalid byte partially copied");
    partial.write(4096,&other,4);partial.read(4092,guard.data(),8);

    const auto before=memory.snapshot();
    rejects([&]{memory.write(capacity-4,&value,8);});
    rejects([&]{memory.read(capacity-4,&read,8);});
    rejects([&]{memory.write(UINT64_MAX,&value,8);});
    rejects([&]{memory.read(0,&read,UINT64_MAX);});
    rejects([&]{memory.require_written(capacity,1,7);});
    rejects([&]{memory.write(high,nullptr,1);});
    rejects([&]{memory.read(high,nullptr,1);});
    memory.write(capacity,nullptr,0);memory.read(capacity,nullptr,0);
    check(before==memory.snapshot(),"invalid or empty access allocated memory");
    memory.write(capacity-8,&value,8,UINT64_MAX);memory.read(capacity-8,&read,8);
    check(read==value,"capacity tail changed");memory.require_written(capacity-8,8,UINT64_MAX);

    // Differential randomized spans against an independent dense byte oracle.
    // A nonzero high base catches truncation; sparse holes stay uninitialized.
    MappedMemory random_memory(capacity);
    constexpr size_t extent=3*4096;
    std::vector<uint8_t> dense(extent),valid(extent);
    std::vector<uint64_t> writer(extent);
    std::mt19937 random(73129);
    for(unsigned step=0;step<3000;++step) {
      size_t at=random()%extent,count=std::min(size_t(random()%97),extent-at);
      uint64_t generation=random()%5;
      if(random()%2) {
        std::vector<uint8_t> bytes(count);for(auto &byte:bytes)byte=uint8_t(random());
        random_memory.write(high+at,bytes.data(),count,generation);
        for(size_t i=0;i<count;++i){dense[at+i]=bytes[i];valid[at+i]=1;if(generation)writer[at+i]=generation;}
      }else {
        std::vector<uint8_t> actual(count,0xa5),unchanged=actual;
        bool readable=std::all_of(valid.begin()+at,valid.begin()+at+count,[](uint8_t x){return x!=0;});
        if(readable){random_memory.read(high+at,actual.data(),count);check(std::equal(actual.begin(),actual.end(),dense.begin()+at),"random read differs");}
        else{rejects([&]{random_memory.read(high+at,actual.data(),count);});check(actual==unchanged,"random failed read partly copied");}
        bool produced=generation&&std::all_of(writer.begin()+at,writer.begin()+at+count,[&](uint64_t x){return x==generation;});
        if(produced)random_memory.require_written(high+at,count,generation);
        else rejects([&]{random_memory.require_written(high+at,count,generation);});
      }
    }
    Json::Value report;
    report["classification"]="cpp_sparse_backing_contract_not_model_or_system_validation";
    report["randomized_steps"]=3000;report["memory"]=memory.snapshot();
    report["full_model_execution_verified"]=false;report["mlx_system_verified"]=false;
    report["inference_performance_eligible"]=false;
    std::cout<<report<<'\n';
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
