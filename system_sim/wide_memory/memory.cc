#include "memory.hh"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <elf.h>
#include <fcntl.h>
#include <limits>
#include <openssl/evp.h>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace mlx::wide_memory {
namespace {
void check(bool value,const char *message){if(!value)throw std::runtime_error(message);}
uint64_t field(const Json::Value &row,const char *name){check(row[name].isUInt64(),"invalid preload unsigned field");return row[name].asUInt64();}
class File {
  int fd;
  struct stat initial{};
public:
  explicit File(const std::string &path){fd=open(path.c_str(),O_RDONLY|O_CLOEXEC);check(fd>=0,"cannot open memory image");if(fstat(fd,&initial)||!S_ISREG(initial.st_mode)||initial.st_size<0){close(fd);throw std::runtime_error("memory image is not a regular file");}}
  ~File(){close(fd);}
  uint64_t size()const{return uint64_t(initial.st_size);}
  void read(uint64_t at,void *data,size_t bytes)const{
    check(at<=size()&&bytes<=size()-at,"preload file range invalid");auto *out=static_cast<uint8_t*>(data);size_t done=0;
    while(done<bytes){auto n=pread(fd,out+done,bytes-done,off_t(at+done));if(n<0&&errno==EINTR)continue;check(n>0,"preload file read failed");done+=size_t(n);}
  }
  void verify()const{
    struct stat now{};check(!fstat(fd,&now)&&now.st_size==initial.st_size&&now.st_mtim.tv_sec==initial.st_mtim.tv_sec&&now.st_mtim.tv_nsec==initial.st_mtim.tv_nsec&&now.st_ctim.tv_sec==initial.st_ctim.tv_sec&&now.st_ctim.tv_nsec==initial.st_ctim.tv_nsec,"memory image changed during load");
  }
};
class Digest {
  std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(),EVP_MD_CTX_free};
public:
  Digest(){check(bool(ctx)&&EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"memory digest init failed");}
  void add(const void *data,size_t bytes){check(EVP_DigestUpdate(ctx.get(),data,bytes)==1,"memory digest update failed");}
  std::string finish(){unsigned char raw[EVP_MAX_MD_SIZE];unsigned bytes=0;check(EVP_DigestFinal_ex(ctx.get(),raw,&bytes)==1&&bytes==32,"memory digest final failed");std::string result;const char *digits="0123456789abcdef";for(unsigned i=0;i<bytes;++i){result+=digits[raw[i]>>4];result+=digits[raw[i]&15];}return result;}
};
}
Memory::Memory(uint64_t address,uint64_t bytes):base(address),capacity(bytes),backend(std::make_unique<mm_magic_t>()){
  check(base%4096==0&&capacity&&capacity%4096==0&&capacity<=UINT64_MAX-base&&capacity<=SIZE_MAX,"invalid wide memory physical extent");
  backend->init(size_t(capacity),8,64);check(backend->get_data()!=MAP_FAILED,"wide memory mapping failed");
}
uint64_t Memory::offset(uint64_t address,uint64_t bytes)const{
  check(address>=base&&address-base<=capacity&&bytes<=capacity-(address-base),"wide memory physical access outside configured range");return address-base;
}
uint64_t Memory::burst(uint64_t address,unsigned size,unsigned len)const{
  check(size<=3&&len<=255,"unsupported AXI burst size/length");auto width=UINT64_C(1)<<size;
  check(address%width==0,"unaligned AXI burst");auto bytes=(uint64_t(len)+1)*width;
  offset(address,bytes);offset(address&~UINT64_C(7),((bytes+(address&7)+7)/8)*8);return address-base;
}
Outputs Memory::eval()const{
  Outputs out;out.ar_ready=backend->ar_ready();out.aw_ready=backend->aw_ready();out.w_ready=backend->w_ready();out.r_valid=backend->r_valid();out.r_last=backend->r_last();out.r_id=unsigned(backend->r_id());out.r_response=unsigned(backend->r_resp());
  out.b_valid=backend->b_valid();out.b_id=unsigned(backend->b_id());out.b_response=unsigned(backend->b_resp());std::memcpy(&out.r_data,backend->r_data(),8);return out;
}
bool Memory::idle()const{return !writing&&!outstanding_reads&&!outstanding_writes&&!backend->r_valid()&&!backend->b_valid();}
void Memory::tick(const Inputs &in){
  auto out=eval();uint64_t ar=0,aw=0;
  if(in.reset){check(idle(),"wide memory reset requires drained transactions");}
  else{
    if(in.ar_valid)ar=burst(in.ar_address,in.ar_size,in.ar_len);
    if(in.aw_valid)aw=burst(in.aw_address,in.aw_size,in.aw_len);
    // mm_magic returns full-word read beats and increments their address by
    // eight. Single-beat narrow reads are valid, narrow read bursts are not.
    if(in.ar_valid)check((in.ar_burst==1||(in.ar_burst==0&&!in.ar_len))&&(in.ar_size==3||!in.ar_len),"unsupported AXI read burst profile");
    if(in.aw_valid)check(in.aw_burst==1||(in.aw_burst==0&&!in.aw_len),"unsupported AXI write burst profile");
    if(in.w_valid&&out.w_ready){
      check(writing&&write_remaining&&in.w_last==(write_remaining==1),"AXI write last/ownership mismatch");
      auto mask=((1u<<write_size)-1u)<<unsigned(write_address&7);check(!(in.w_strobe&~mask),"AXI strobe outside addressed beat");
    }
    if(in.ar_valid&&out.ar_ready){++ar_count;outstanding_reads+=uint64_t(in.ar_len)+1;read_bytes+=(uint64_t(in.ar_len)+1)*(UINT64_C(1)<<in.ar_size);max_read_address=std::max(max_read_address,in.ar_address);}
    if(in.aw_valid&&out.aw_ready){check(!writing,"overlapping AXI write bursts");++aw_count;++outstanding_writes;writing=true;write_address=in.aw_address;write_size=1u<<in.aw_size;write_remaining=in.aw_len+1;max_write_address=std::max(max_write_address,in.aw_address);}
    if(in.w_valid&&out.w_ready){++w_count;for(unsigned bit=0;bit<8;++bit)write_bytes+=(in.w_strobe>>bit)&1;write_address+=write_size;if(!--write_remaining)writing=false;}
    if(in.r_ready&&out.r_valid){check(outstanding_reads>0,"AXI read completion underflow");--outstanding_reads;++r_count;}
    if(in.b_ready&&out.b_valid){check(outstanding_writes>0,"AXI write completion underflow");--outstanding_writes;++b_count;}
  }
  auto data=in.w_data;backend->tick(in.reset,in.ar_valid,ar,in.ar_id,in.ar_size,in.ar_len,in.aw_valid,aw,in.aw_id,in.aw_size,in.aw_len,in.w_valid,in.w_strobe,&data,in.w_last,in.r_ready,in.b_ready);++cycle;
}
void Memory::inspect(uint64_t address,void *data,size_t bytes)const{std::memcpy(data,static_cast<const uint8_t*>(backend->get_data())+offset(address,bytes),bytes);}
void Memory::preload_segments(const Json::Value &segments){
  check(cycle==0&&idle()&&segments.isArray(),"preload is allowed only before memory clocks start");
  struct Region {Json::Value row;uint64_t address,file_offset,file_bytes,memory_bytes;};std::vector<Region> regions;
  std::vector<std::pair<uint64_t,uint64_t>> spans;std::set<std::string> names;
  for(const auto &done:initialized){spans.emplace_back(done["address"].asUInt64(),done["address"].asUInt64()+done["memory_bytes"].asUInt64());names.insert(done["name"].asString());}
  for(const auto &row:segments){
    check(row["path"].isString()&&row["name"].isString(),"preload path/name missing");
    check(!row["name"].asString().empty()&&names.insert(row["name"].asString()).second,"duplicate/empty preload name");
    Region region{row,field(row,"address"),field(row,"file_offset"),field(row,"file_bytes"),field(row,"memory_bytes")};
    check(region.file_bytes<=region.memory_bytes,"preload file bytes exceed memory extent");offset(region.address,region.memory_bytes);
    File file(row["path"].asString());check(region.file_offset<=file.size()&&region.file_bytes<=file.size()-region.file_offset,"preload source extent invalid");
    if(region.memory_bytes)spans.emplace_back(region.address,region.address+region.memory_bytes);
    regions.push_back(std::move(region));
  }
  std::sort(spans.begin(),spans.end());for(size_t i=1;i<spans.size();++i)check(spans[i-1].second<=spans[i].first,"overlapping memory initialization spans");
  std::vector<uint8_t> buffer(65536);
  for(const auto &region:regions){
    File file(region.row["path"].asString());auto *destination=static_cast<uint8_t*>(backend->get_data())+offset(region.address,region.memory_bytes);Digest source;
    for(uint64_t copied=0;copied<region.file_bytes;){auto count=size_t(std::min(uint64_t(buffer.size()),region.file_bytes-copied));file.read(region.file_offset+copied,buffer.data(),count);source.add(buffer.data(),count);std::memcpy(destination+copied,buffer.data(),count);copied+=count;}
    auto digest=source.finish();file.verify();
    if(region.row.isMember("sha256"))check(region.row["sha256"].isString()&&region.row["sha256"].asString()==digest,"preload input digest mismatch");
    std::memset(destination+region.file_bytes,0,size_t(region.memory_bytes-region.file_bytes));Digest stored;stored.add(destination,size_t(region.file_bytes));check(stored.finish()==digest,"preload destination digest mismatch");
    Json::Value row;row["name"]=region.row["name"];row["path"]=region.row["path"];row["address"]=Json::UInt64(region.address);row["file_offset"]=Json::UInt64(region.file_offset);row["file_bytes"]=Json::UInt64(region.file_bytes);row["memory_bytes"]=Json::UInt64(region.memory_bytes);row["sha256"]=digest;initialized.append(row);
  }
}
void Memory::preload_elf(const std::string &path){
  File file(path);Elf64_Ehdr header{};file.read(0,&header,sizeof(header));
  check(!std::memcmp(header.e_ident,ELFMAG,SELFMAG)&&header.e_ident[EI_CLASS]==ELFCLASS64&&header.e_ident[EI_DATA]==ELFDATA2LSB&&header.e_ident[EI_VERSION]==EV_CURRENT&&header.e_version==EV_CURRENT&&header.e_ehsize==sizeof(header)&&header.e_machine==EM_RISCV&&header.e_type==ET_EXEC&&header.e_entry%2==0,"unsupported preload ELF");
  check(header.e_phentsize==sizeof(Elf64_Phdr)&&header.e_phnum&&header.e_phnum<=128,"invalid ELF program header table");
  check(header.e_phoff<=file.size()&&uint64_t(header.e_phnum)*sizeof(Elf64_Phdr)<=file.size()-header.e_phoff,"ELF program headers exceed file");
  Json::Value segments{Json::arrayValue};bool entry=false;
  for(unsigned i=0;i<header.e_phnum;++i){Elf64_Phdr p{};file.read(header.e_phoff+uint64_t(i)*sizeof(p),&p,sizeof(p));if(p.p_type!=PT_LOAD)continue;check(p.p_filesz<=p.p_memsz,"ELF file segment exceeds memory size");if(!p.p_memsz)continue;
    check(p.p_paddr==p.p_vaddr,"preload ELF requires bare physical addresses");offset(p.p_paddr,p.p_memsz);
    entry|=(p.p_flags&PF_X)&&header.e_entry>=p.p_paddr&&header.e_entry-p.p_paddr<p.p_memsz;
    Json::Value row;row["name"]="elf:"+std::to_string(i);row["path"]=path;row["address"]=Json::UInt64(p.p_paddr);row["file_offset"]=Json::UInt64(p.p_offset);row["file_bytes"]=Json::UInt64(p.p_filesz);row["memory_bytes"]=Json::UInt64(p.p_memsz);segments.append(row);
  }
  check(entry,"ELF entry outside executable load segments");file.verify();preload_segments(segments);file.verify();
}
Json::Value Memory::report()const{
  Json::Value r;r["classification"]="wide_checked_axi_magic_memory_not_calibrated_dram_timing";r["base"]=Json::UInt64(base);r["bytes"]=Json::UInt64(capacity);r["cycle"]=Json::UInt64(cycle);r["ar_requests"]=Json::UInt64(ar_count);r["aw_requests"]=Json::UInt64(aw_count);
  r["read_beats"]=Json::UInt64(r_count);r["write_beats"]=Json::UInt64(w_count);r["write_responses"]=Json::UInt64(b_count);r["read_bytes_requested"]=Json::UInt64(read_bytes);r["write_bytes_strobed"]=Json::UInt64(write_bytes);r["idle"]=idle();r["initialized_segments"]=initialized;
  r["max_read_address"]=Json::UInt64(max_read_address);r["max_write_address"]=Json::UInt64(max_write_address);
  r["initialization_scope"]="host_file_copy_before_clock_not_cpu_or_dma_execution";r["mlx_system_verified"]=false;r["inference_performance_eligible"]=false;return r;
}
}
